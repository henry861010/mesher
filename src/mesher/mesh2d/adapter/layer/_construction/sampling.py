"""Bounded, source-density-aware sampling of a resampleable LINE front."""

from __future__ import annotations

from heapq import heapify, heappop, heappush

import numpy as np
from .....tolerance import subdivision_count

from .._limits import check_element_limit, section_element_lower_bound
from .._types import QuadLayerError
from .._line_policy import DEFAULT_MAX_SIZE_RATIO, DEFAULT_MAX_TRANSITION_COUNT


_MAX_SEGMENT_COUNT = int(np.iinfo(np.intp).max) - 1


def _infeasible(message: str, **context):
    return QuadLayerError(message, code="line_constraints_infeasible", **context)


def _spacing_limits(lengths, h, min_ratio, max_ratio, size_band_active):
    if size_band_active:
        return float(min_ratio) * float(h), float(max_ratio) * float(h)
    # Exact anchors and forward ordering already bound each gap by the
    # length of its anchor interval. This envelope imposes no size policy.
    return 0.0, max(lengths, default=0.0)


def _interval_bounds(lengths, tol, h_min, h_max):
    minimum, maximum, forced = [], [], []
    # Smaller positive ratios can underflow, or imply a count no NumPy front
    # can represent. Keep the bound finite without dividing by that tiny gap.
    for index, length in enumerate(lengths):
        if not np.isfinite(length) or length <= tol:
            raise _infeasible("LINE anchor intervals must have positive length",
                              interval=index)
        arithmetic_margin = 32 * np.finfo(float).eps * max(float(length), h_min)
        short = length < h_min - arithmetic_margin
        low = 1 if length <= h_max else subdivision_count(float(length), h_max)
        if short:
            high = 1
        elif h_min == 0 or float(length) >= _MAX_SEGMENT_COUNT * h_min:
            high = _MAX_SEGMENT_COUNT
        else:
            high = subdivision_count(float(length), h_min, rounding="floor")
        if low > high:
            raise _infeasible("LINE anchor interval cannot satisfy the size band",
                              interval=index, length=float(length),
                              minimum_count=low, maximum_count=high)
        minimum.append(low)
        maximum.append(high)
        forced.append(short)
    return minimum, maximum, forced


def line_count_bounds(lengths, source_segments, h, tol,
                      max_transition_count=DEFAULT_MAX_TRANSITION_COUNT, min_size_ratio=0.5,
                      max_size_ratio=DEFAULT_MAX_SIZE_RATIO, *, size_band_active=True):
    """Count-only feasibility, usable before allocating any sampled points."""
    lengths = tuple(lengths)
    limits = _spacing_limits(lengths, h, min_size_ratio, max_size_ratio, size_band_active)
    minimum, maximum, _ = _interval_bounds(lengths, tol, *limits)
    return _count_limits(minimum, maximum, source_segments, max_transition_count)


def _count_limits(minimum, maximum, source_segments, max_transition_count):
    lower = max(sum(minimum),
                (source_segments + max_transition_count - 1) // max_transition_count)
    upper = min(sum(maximum), _MAX_SEGMENT_COUNT)
    if not minimum or lower > upper:
        raise _infeasible("LINE transition limit and size band cannot both be satisfied",
                          minimum_count=lower, maximum_count=upper)
    return int(lower), int(upper)


def _forward_sweep(raw, length, h_min, h):
    """Clamp quantiles while retaining enough room for every remaining gap."""
    count = len(raw) - 1
    result = np.empty_like(raw)
    result[0], result[-1] = 0.0, length
    # Count bounds allow floating-point roundoff near exact integer ratios.
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
                 min_size_ratio=0.5, max_size_ratio=DEFAULT_MAX_SIZE_RATIO, *, size_band_active=True):
        self.source = np.asarray(source, dtype=float).copy()
        self.anchors = np.asarray(anchors, dtype=float).copy()
        self.h, self.tol = float(h), float(tol)
        self.max_transition_count = int(max_transition_count)
        self.min_size_ratio = float(min_size_ratio)
        self.max_size_ratio = float(max_size_ratio)
        self.size_band_active = size_band_active
        self.lengths = np.linalg.norm(np.diff(self.anchors[:, :2], axis=0), axis=1)
        self.h_min, self.h_max = _spacing_limits(
            self.lengths, h, min_size_ratio, max_size_ratio, size_band_active)
        self._minimum, self._maximum, self._forced = _interval_bounds(
            self.lengths, tol, self.h_min, self.h_max)
        self.lower_count, self.upper_count = _count_limits(
            self._minimum, self._maximum, len(self.source) - 1, max_transition_count)
        delta = self.anchors[-1, :2] - self.anchors[0, :2]
        self.tangent = delta / np.linalg.norm(delta)
        self.anchor_positions = (self.anchors[:, :2] - self.anchors[0, :2]) @ self.tangent
        self.source_positions = (self.source[:, :2] - self.anchors[0, :2]) @ self.tangent
        if (np.any(np.diff(self.source_positions) <= tol)
                or np.any(np.diff(self.anchor_positions) <= tol)):
            raise _infeasible("LINE nodes must be strictly ordered along the line axis")
        if np.any(np.abs(np.diff(self.anchor_positions) - self.lengths) > tol):
            raise _infeasible("LINE outer anchors must be collinear")
        self._build_density(self.source_positions)
        # Both arrays are sorted; interpolation walks the density intervals.
        self._anchor_mass = np.interp(self.anchor_positions, self._positions, self._mass)
        self._demands = np.diff(self._anchor_mass)
        self.demand = float(self._mass[-1])
        self._samples: dict[int, tuple[np.ndarray, float]] = {}

    def _build_density(self, source_positions):
        length = self.anchor_positions[-1]
        interior = source_positions[(source_positions > 0) & (source_positions < length)]
        self._positions = np.concatenate(([0.0], interior, [length]))
        # Only the first interval can start partway through a source segment.
        # Interior breakpoints are source nodes, so its successors advance by
        # one index each. The two tails outside the source use the nominal rate.
        first = int(np.searchsorted(source_positions, 0., side="right")) - 1
        indices = first + np.arange(len(self._positions) - 1)
        inside = (indices >= 0) & (indices < len(source_positions) - 1)
        rate = np.full(len(indices), 1. / self.h)
        widths = np.diff(source_positions)[indices[inside]]
        rate[inside] = np.maximum(rate[inside], (1 / self.max_transition_count) / widths)
        self._mass = np.r_[0., np.cumsum(np.diff(self._positions) * rate)]

    @property
    def diagnostics(self):
        return {
            "line_sampling": "source_density",
            "size_band_active": self.size_band_active,
            "max_transition_count": self.max_transition_count,
            "min_size_ratio": self.min_size_ratio,
            "max_size_ratio": self.max_size_ratio,
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
        below = subdivision_count(target, 1., rounding="floor")
        below -= (below - parity) % 2
        above = subdivision_count(target, 1.)
        above += (parity - above) % 2
        return sorted({n for n in (below, above) if lower <= n <= upper})

    def split_for_parity(self, left_nodes, right_nodes, limit):
        """Split one ordinary gap only if the size band forces the wrong parity."""
        count = self.lower_count
        source_segments = len(self.source) - 1
        if count != self.upper_count or (source_segments + count + left_nodes + right_nodes) % 2 == 0:
            return None
        lower = section_element_lower_bound(source_segments, count + 1, left_nodes, right_nodes)
        check_element_limit(lower, limit, stage="section_count_preflight", count_kind="lower_bound")
        ordinary, _ = self.sample(count)
        lengths = np.linalg.norm(np.diff(ordinary, axis=0), axis=1)
        longest = int(np.argmax(lengths))
        if lengths[longest] / 2 <= self.tol:
            return None
        index = longest + 1
        midpoint = (ordinary[longest] + ordinary[index]) / 2
        anchors = np.insert(ordinary, index, midpoint, axis=0)
        # The ordinary sample passed the size band; only the two new gaps
        # can acquire an additional minimum-spacing exception.
        return index, LineSampling(self.source, anchors, self.h, self.tol,
                                   self.max_transition_count, self.min_size_ratio, self.max_size_ratio)

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
            positions = quantiles[cursor:cursor + n + 1] - self.anchor_positions[i]
            if self.size_band_active:
                max_gap = min(self.h_max, length)
                forward = _forward_sweep(positions, length, self.h_min, max_gap)
                backward = length - _forward_sweep(length - positions[::-1], length, self.h_min, max_gap)[::-1]
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
        projected = (sampled[:, :2] - self.anchors[0, :2]) @ self.tangent
        residual = sampled[:, :2] - self.anchors[0, :2] - projected[:, None] * self.tangent
        if np.any(np.linalg.norm(residual, axis=1) > self.tol):
            raise _infeasible("LINE sampling moved points off the outer line")
        gaps = np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1)
        margin = spacing_error(sampled[:, :2], self.h)
        if np.any(np.diff(projected) <= 0) or np.any(gaps > self.h_max + margin):
            raise _infeasible("LINE sampling violates ordering or maximum size")
        short = gaps < self.h_min - margin
        index, anchor_indices = 0, []
        for position, anchor in zip(self.anchor_positions, self.anchors):
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


def spacing_error(points, h):
    ulps = np.abs(np.spacing(points))
    return (2 * np.hypot.reduce(ulps[:-1] + ulps[1:], axis=1)
            + 32 * np.finfo(float).eps * h)


def validate_spacing(points, h, ratio, forced, max_ratio=DEFAULT_MAX_SIZE_RATIO, *, size_band_active=True):
    """Measure committed gaps and enforce the active size band and its anchor exceptions."""
    points = np.asarray(points)[:, :2]
    gaps = np.hypot.reduce(np.diff(points, axis=0), axis=1)
    if not np.all(np.isfinite(gaps)) or np.any(gaps <= 0):
        raise _infeasible("committed LINE spacing must be finite and positive")
    if not size_band_active:
        return gaps
    margin = spacing_error(points, h)
    if np.any(gaps > float(h) * max_ratio + margin):
        raise _infeasible("committed LINE spacing exceeds maximum size",
                          maximum_spacing=float(gaps.max()), element_size=h, max_size_ratio=max_ratio)
    short = gaps < ratio * h - margin
    exceptions = {(tuple(interval["start"]), tuple(interval["end"])) for interval in forced}
    for i in np.flatnonzero(short):
        if (tuple(points[i]), tuple(points[i + 1])) not in exceptions:
            raise _infeasible("committed LINE spacing is below minimum size",
                              spacing=float(gaps[i]), minimum_size=ratio * h)
    return gaps
