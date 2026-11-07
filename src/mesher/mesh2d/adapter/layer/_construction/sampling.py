"""Bounded, source-density-aware sampling of a resampleable LINE front."""

from __future__ import annotations

from heapq import heapify, heappop, heappush
from math import ceil, floor

import numpy as np

from .._types import QuadLayerError
from .._line_policy import DEFAULT_MAX_TRANSITION_COUNT


def _infeasible(message: str, **context):
    return QuadLayerError(message, code="line_constraints_infeasible", **context)


def _interval_bounds(lengths, h, tol, ratio):
    minimum, maximum, forced = [], [], []
    h_min = float(ratio) * float(h)
    # Smaller positive ratios can underflow, or imply a count no NumPy front
    # can represent. Keep the bound finite without dividing by that tiny gap.
    max_count = int(np.iinfo(np.intp).max) - 1
    for index, length in enumerate(lengths):
        if not np.isfinite(length) or length <= tol:
            raise _infeasible("LINE anchor intervals must have positive length",
                              interval=index)
        short = length < h_min - tol
        low = 1 if short else max(1, ceil((float(length) - tol) / h))
        if short:
            high = 1
        elif h_min == 0 or float(length) + tol >= max_count * h_min:
            high = max_count
        else:
            high = floor((float(length) + tol) / h_min)
        if low > high:
            raise _infeasible("LINE anchor interval cannot satisfy the size band",
                              interval=index, length=float(length),
                              minimum_count=low, maximum_count=high)
        minimum.append(low)
        maximum.append(high)
        forced.append(short)
    return minimum, maximum, forced


def line_count_bounds(lengths, source_segments, h, tol,
                      max_transition_count=DEFAULT_MAX_TRANSITION_COUNT, min_size_ratio=0.5):
    """Count-only feasibility, usable before allocating any sampled points."""
    minimum, maximum, _ = _interval_bounds(lengths, h, tol, min_size_ratio)
    lower = max(sum(minimum),
                (source_segments + max_transition_count - 1) // max_transition_count)
    upper = sum(maximum)
    if not minimum or lower > upper:
        raise _infeasible("LINE transition limit and size band cannot both be satisfied",
                          minimum_count=lower, maximum_count=upper)
    return int(lower), int(upper)


def _forward_sweep(raw, length, h_min, h):
    """Clamp quantiles while retaining enough room for every remaining gap."""
    count = len(raw) - 1
    result = np.empty_like(raw)
    result[0], result[-1] = 0.0, length
    # Count bounds allow endpoint roundoff within the geometric tolerance.
    # Relaxing only to the mean gap makes those almost-exact cases feasible.
    lower_gap = min(h_min, length / count)
    upper_gap = max(h, length / count)
    for j in range(1, count):
        remaining = count - j
        lower = max(result[j - 1] + lower_gap, length - remaining * upper_gap)
        upper = min(result[j - 1] + upper_gap, length - remaining * lower_gap)
        result[j] = min(upper, max(lower, raw[j]))
    return result


class LineSampling:
    """Reuse density and samples across the constant number of seam states.

    Inputs are the canonical, ordered XYZ chains used by LINE construction.
    The constructor allocates only in proportion to source and anchor counts;
    ``sample`` is deliberately separate so callers can reject element budgets
    before allocating a potentially much larger front.
    """

    def __init__(self, source, anchors, h, tol, max_transition_count=DEFAULT_MAX_TRANSITION_COUNT,
                 min_size_ratio=0.5):
        self.source = np.asarray(source, dtype=float).copy()
        self.anchors = np.asarray(anchors, dtype=float).copy()
        self.h, self.tol = float(h), float(tol)
        self.max_transition_count = int(max_transition_count)
        self.min_size_ratio = float(min_size_ratio)
        self.h_min = self.min_size_ratio * self.h
        self.lengths = np.linalg.norm(np.diff(self.anchors[:, :2], axis=0), axis=1)
        self.lower_count, self.upper_count = line_count_bounds(
            self.lengths, len(self.source) - 1, h, tol,
            max_transition_count, min_size_ratio)
        self._minimum, self._maximum, self._forced = _interval_bounds(
            self.lengths, h, tol, min_size_ratio)
        delta = self.anchors[-1, :2] - self.anchors[0, :2]
        self._tangent = delta / np.linalg.norm(delta)
        self._anchor_positions = (self.anchors[:, :2] - self.anchors[0, :2]) @ self._tangent
        source_positions = (self.source[:, :2] - self.anchors[0, :2]) @ self._tangent
        if (np.any(np.diff(source_positions) <= tol)
                or np.any(np.diff(self._anchor_positions) <= tol)):
            raise _infeasible("LINE nodes must be strictly ordered along the line axis")
        if np.any(np.abs(np.diff(self._anchor_positions) - self.lengths) > tol):
            raise _infeasible("LINE outer anchors must be collinear")
        self._build_density(source_positions)
        # Both arrays are sorted; interpolation walks the density intervals.
        self._anchor_mass = np.interp(self._anchor_positions, self._positions, self._mass)
        self._demands = np.diff(self._anchor_mass)
        self.demand = float(self._mass[-1])
        self._samples: dict[int, tuple[np.ndarray, float]] = {}

    def _build_density(self, source_positions):
        length = self._anchor_positions[-1]
        interior = source_positions[(source_positions > 0) & (source_positions < length)]
        self._positions = np.concatenate(([0.0], interior, [length]))
        mass = np.empty(len(self._positions))
        mass[0] = 0.0
        index = int(np.searchsorted(source_positions, 0.0, side="right")) - 1
        baseline = 1.0 / self.h
        inverse_limit = 1 / self.max_transition_count
        for interval, (start, end) in enumerate(zip(self._positions[:-1],
                                                    self._positions[1:])):
            while index + 1 < len(source_positions) and source_positions[index + 1] <= start:
                index += 1
            rate = baseline
            if 0 <= index < len(source_positions) - 1:
                gap = source_positions[index + 1] - source_positions[index]
                rate = max(rate, inverse_limit / gap)
            mass[interval + 1] = mass[interval] + (end - start) * rate
        self._mass = mass

    @property
    def diagnostics(self):
        return {
            "line_sampling": "source_density",
            "max_transition_count": self.max_transition_count,
            "min_size_ratio": self.min_size_ratio,
            "density_demand": self.demand,
            "forced_short_intervals": [
                {"start": self.anchors[i, :2].tolist(),
                 "end": self.anchors[i + 1, :2].tolist(),
                 "length": float(self.lengths[i])}
                for i, forced in enumerate(self._forced) if forced
            ],
        }

    def counts(self, left_nodes, right_nodes):
        """At most the lower and upper parity-compatible demand neighbors."""
        parity = (len(self.source) - 1 + left_nodes + right_nodes) % 2
        lower = self.lower_count + (parity - self.lower_count) % 2
        upper = self.upper_count - (self.upper_count - parity) % 2
        if lower > upper:
            raise _infeasible("LINE size and transition constraints conflict with seam parity",
                              minimum_count=self.lower_count, maximum_count=self.upper_count,
                              parity=parity)
        target = min(upper, max(lower, self.demand))
        rounded = round(target)
        if abs(target - rounded) <= self.tol / self.h + 32 * np.finfo(float).eps * max(1, target):
            target = rounded
        below = floor(target)
        below -= (below - parity) % 2
        above = ceil(target)
        above += (parity - above) % 2
        return sorted({n for n in (below, above) if lower <= n <= upper})

    def _allocate(self, count):
        counts = self._minimum.copy()
        heap = [(2 * (n - self._demands[i]) + 1, i)
                for i, n in enumerate(counts) if n < self._maximum[i]]
        # Heap construction is linear in the mandatory interval count.
        heapify(heap)
        for _ in range(count - sum(counts)):
            _, index = heappop(heap)
            counts[index] += 1
            if counts[index] < self._maximum[index]:
                heappush(heap, (2 * (counts[index] - self._demands[index]) + 1, index))
        return counts

    def sample(self, count):
        """Allocate by demand, invert density, then average two bounded sweeps."""
        if count in self._samples:
            return self._samples[count]
        if not self.lower_count <= count <= self.upper_count:
            raise _infeasible("LINE outer count is outside the feasible range", count=count)
        counts = self._allocate(count)
        # A single interpolation of ordered masses avoids rescanning the source
        # profile for every mandatory interval.
        targets = np.empty(count + 1)
        cursor = 0
        for i, n in enumerate(counts):
            targets[cursor:cursor + n] = np.linspace(self._anchor_mass[i],
                                                    self._anchor_mass[i + 1], n + 1)[:-1]
            cursor += n
        targets[-1] = self._anchor_mass[-1]
        quantiles = np.interp(targets, self._mass, self._positions)
        sampled = np.empty((count + 1, 3))
        cursor = 0
        for i, n in enumerate(counts):
            length = float(self.lengths[i])
            raw = quantiles[cursor:cursor + n + 1] - self._anchor_positions[i]
            forward = _forward_sweep(raw, length, self.h_min, self.h)
            backward = length - _forward_sweep(length - raw[::-1], length,
                                               self.h_min, self.h)[::-1]
            positions = (forward + backward) / 2
            start, end = self.anchors[i:i + 2]
            sampled[cursor:cursor + n + 1] = start + (end - start) * (positions / length)[:, None]
            # Mandatory anchors are copied exactly, independent of roundoff.
            sampled[cursor], sampled[cursor + n] = start, end
            cursor += n
        self.validate_sample(sampled)
        gaps = np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1)
        error = float(np.mean((gaps / self.h - 1) ** 2))
        sampled.flags.writeable = False
        self._samples[count] = sampled, error
        return self._samples[count]

    def validate_sample(self, sampled):
        """Verify exact anchors and every spacing, allowing only forced shorts."""
        sampled = np.asarray(sampled)
        if (sampled.ndim != 2 or sampled.shape[1] != 3 or len(sampled) < 2
                or not np.all(np.isfinite(sampled))):
            raise _infeasible("LINE sampling produced invalid coordinates")
        projected = (sampled[:, :2] - self.anchors[0, :2]) @ self._tangent
        residual = sampled[:, :2] - self.anchors[0, :2] - projected[:, None] * self._tangent
        if np.any(np.linalg.norm(residual, axis=1) > self.tol):
            raise _infeasible("LINE sampling moved points off the outer line")
        gaps = np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1)
        if np.any(np.diff(projected) <= 0) or np.any(gaps > self.h + self.tol):
            raise _infeasible("LINE sampling violates ordering or maximum size")
        short = gaps < self.h_min - self.tol
        index, anchor_indices = 0, []
        for position, anchor in zip(self._anchor_positions, self.anchors):
            while index < len(sampled) and projected[index] < position - self.tol:
                index += 1
            if index == len(sampled) or not np.array_equal(sampled[index], anchor):
                raise _infeasible("LINE sampling did not retain a mandatory anchor")
            anchor_indices.append(index)
        if anchor_indices[0] != 0 or anchor_indices[-1] != len(sampled) - 1:
            raise _infeasible("LINE sampling changed the front endpoints")
        for interval, forced in enumerate(self._forced):
            if forced:
                start, end = anchor_indices[interval:interval + 2]
                if end != start + 1:
                    raise _infeasible("LINE sampling subdivided a forced short interval")
                short[start] = False
        if np.any(short):
            raise _infeasible("LINE sampling violates minimum size")


__all__ = ["LineSampling", "line_count_bounds"]
