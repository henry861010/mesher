"""Plan a LINE's unit cuts and free outer nodes in their common tangent frame."""

from __future__ import annotations

from bisect import bisect_left
from heapq import heapify, heappop, heappush
from math import ceil, floor, fsum, sqrt

import numpy as np

from .._limits import bounded_outer_counts
from .._types import QuadLayerError
from .sampling import LineSampling
from .strip import alignment_metrics, fixed_cut_plans


_EPS = 32 * np.finfo(float).eps
_MAX_COUNT = int(np.iinfo(np.intp).max)


class _ProjectedLine:
    """Scalar positions shared by every choice of this section's two end seams."""

    def __init__(self, sampling):
        self.tangent = sampling.tangent
        self.anchors = sampling.anchors
        self.contacts = sampling.anchor_positions
        self.x = sampling.source_positions.copy()
        self.n = len(self.x) - 1
        self.h, self.tol, self.cap = sampling.h, sampling.tol, sampling.max_transition_count
        self.size_band_active = sampling.size_band_active
        self.low, self.high = sampling.h_min, sampling.h_max
        self.x[0], self.x[-1] = self.contacts[[0, -1]]
        # An original anchor within coordinate tolerance owns that coordinate.
        upper = np.searchsorted(self.contacts, self.x).clip(0, len(self.anchors) - 1)
        lower = (upper - 1).clip(0)
        nearest = np.where(abs(self.x - self.contacts[lower]) <= abs(self.x - self.contacts[upper]),
                           lower, upper)
        matches = abs(self.x - self.contacts[nearest]) <= self.tol
        self.x[matches] = self.contacts[nearest[matches]]
        self.anchor_id = np.where(matches, nearest, -1)
        self.after = np.searchsorted(self.contacts, self.x, side="right")
        self.before = np.searchsorted(self.contacts, self.x, side="left")
        # End seams may be slanted enough to put source projections outside the
        # open front. They cannot become cuts; the source nodes stay in a unit.
        self.usable = (self.x > self.tol) & (self.x < self.contacts[-1] - self.tol)
        self.usable[[0, -1]] = True
        self.ordered = self.x.copy()
        self.ordered[0], self.ordered[-1] = -np.inf, np.inf

    def sweep(self, left, right, *, reverse):
        """Advance whole units, considering a bounded neighborhood per cut."""
        stations, ordered, usable = self.x, self.ordered, self.usable
        if reverse:
            stations, ordered, usable = -stations[::-1], -ordered[::-1], usable[::-1]
        result = []
        start = 0
        distances = (self.h, 2 * self.h, self.low, self.high, 2 * self.low, 2 * self.high)
        while start < self.n:
            window = min(self.n, start + 2 * self.cap)
            ends = {start + 1, start + 2, start + self.cap, window}
            for distance in distances:
                j = bisect_left(ordered, stations[start] + distance, start + 1, window + 1)
                ends.update((j - 2, j - 1, j, j + 1))
            options = []
            for end in ends:
                if not start < end <= window or not usable[end]:
                    continue
                i, j = (self.n - end, self.n - start) if reverse else (start, end)
                choice = self.allocation(i, j, left if i == 0 else 1, right if j == self.n else 1)
                if choice is not None:
                    cost, counts = choice
                    b = sum(counts)
                    options.append((cost / b, min(j - i, b), end, (i, j, counts)))
            if not options:
                return []
            _, _, start, unit = min(options)
            result.append(unit)
        return result[::-1] if reverse else result

    def pieces(self, i, j):
        """Mandatory subinterval endpoints, including these source projections."""
        return [self.x[i], *self.contacts[self.after[i]:self.before[j]], self.x[j]]

    def allocation(self, i, j, left_segments, right_segments):
        """Feasible outer subdivisions with even total unit boundary parity."""
        a = j - i
        parity = (a + left_segments + right_segments) % 2
        required = (a + self.cap - 1) // self.cap
        # The ordinary unit templates have a short tangential side of 1 or 2
        # edges. A section consisting of just one unit also needs a two-node
        # side for the existing builder.
        maximum = 2 if a > 2 else None
        if left_segments > 1 and right_segments > 1 and a > 1:
            maximum = 1
        inner_anchors = self.before[j] - self.after[i]
        if maximum is not None and inner_anchors + 1 > maximum:
            return None
        positions = self.pieces(i, j)
        targets, limits, counts = [], [], []
        for k, (start, end) in enumerate(zip(positions, positions[1:])):
            length = end - start
            if length <= self.tol:
                return None
            margin = _EPS * max(abs(start), abs(end), length, self.low)
            if length < self.low - margin:
                # Only an interval bounded by two mandatory points may be short.
                if ((k == 0 and self.anchor_id[i] < 0)
                        or (k + 2 == len(positions) and self.anchor_id[j] < 0)):
                    return None
                low = high = 1
            else:
                low = max(1, ceil((length - margin) / self.high))
                if maximum is not None and low > maximum:
                    return None
                high = (self._max_count(length, margin) if maximum is None
                        else min(maximum, self._max_count(length, margin)))
                if low > high:
                    return None
            target = length / self.h
            # The objective b*(target/b-1)^2 changes sign between b and
            # b+1 at target=sqrt(b*(b+1)). No loop over possible counts.
            base = max(1, floor(target))
            optimal = base + int(target > sqrt(base) * sqrt(base + 1))
            counts.append(min(high, max(low, optimal)))
            targets.append(target)
            limits.append((low, high))
        count = sum(counts)
        lo = max(required, sum(pair[0] for pair in limits))
        hi = sum(pair[1] for pair in limits)
        if maximum is not None:
            hi = min(hi, maximum)
        lo += (parity - lo) % 2
        hi -= (hi - parity) % 2
        if lo > hi:
            return None
        candidates = {max(lo, min(hi, count - (count - parity) % 2)),
                      max(lo, min(hi, count + (parity - count) % 2))}
        score, selected = min(self._adjust(counts, targets, limits, wanted)
                              for wanted in sorted(candidates))
        return score, selected

    def _max_count(self, length, margin):
        if self.low == 0.0 or self.low * _MAX_COUNT <= length + margin:
            return _MAX_COUNT
        return floor((length + margin) / self.low)

    @staticmethod
    def _adjust(counts, targets, limits, wanted):
        result = counts.copy()
        total = sum(result)
        if total != wanted:
            step = 1 if total < wanted else -1
            def marginal(index):
                n = result[index]
                return step * (1 - targets[index] ** 2 / (n * (n + step)))
            choices = [(marginal(k), k) for k, n in enumerate(result)
                       if limits[k][0] <= n + step <= limits[k][1]]
            heapify(choices)
            for _ in range(abs(wanted - total)):
                _, k = heappop(choices)
                result[k] += step
                if limits[k][0] <= result[k] + step <= limits[k][1]:
                    heappush(choices, (marginal(k), k))
        return sum(b * (t / b - 1) ** 2 for t, b in zip(targets, result)), result

    def simple_coarsening(self, gaps, left, right):
        """Balance k:1 units around the target, reserving exact suffix capacity.

        Common lower/upper bounds on k keep every source projection admissible
        despite modest variations in the source spacing. Lines with a wider
        variation can instead use the local bounds in projected_units.
        """
        length = self.contacts[-1]
        target = length / self.h
        if len(self.contacts) != 2 or target > self.n / 2 or gaps.min() <= self.tol:
            return None
        margin = _EPS * max(length, self.low)
        low = max(1, ceil((self.low - margin) / gaps.min()))
        high = self.cap if self.high >= length else min(self.cap, floor((self.high + margin) / gaps.max()))
        if low > high:
            return None
        parity = (self.n + left + right) % 2
        below = floor(target)
        below -= (below - parity) % 2
        candidates = {n for n in (below, below + 2) if 2 <= n <= self.n}
        if not self.size_band_active:
            # A short dense front can need many more edges than length / h.
            # Try balanced source groups at the cap floor before greedy cuts.
            minimum = (self.n + self.cap - 1) // self.cap
            if minimum > 1 and target < minimum:
                count = minimum + (parity - minimum) % 2
                if count <= self.n:
                    candidates.add(count)
        for count in sorted(candidates, key=lambda n: (abs(length / n - self.h), n)):
            # One exterior edge; internal sides each have one edge.
            parities = [left % 2, *([1] * (count - 2)), right % 2]
            lows = [low + (p - low) % 2 for p in parities]
            highs = [high - (high - p) % 2 for p in parities]
            total_low, total_high = sum(lows), sum(highs)
            if any(a > b for a, b in zip(lows, highs)) or not total_low <= self.n <= total_high:
                continue
            i, result = 0, []
            for k, (a, b) in enumerate(zip(lows, highs), 1):
                total_low -= a
                total_high -= b
                lo = max(i + a, self.n - total_high)
                hi = min(i + b, self.n - total_low)
                station = length * (k / count)
                index = bisect_left(self.x, station, lo, hi)
                upper = min(hi, index + (lo - index) % 2)
                lower = max(lo, upper - 2)
                j = lower if station - self.x[lower] <= self.x[upper] - station else upper
                result.append((i, j, [1]))
                i = j
            return result
        return None

    def projected_units(self, left, right):
        """Find a fully perpendicular layout over a bounded set of source spans."""
        n = self.n
        # A single row whose existing lengths are already close to h needs
        # only endpoint parity absorption. This is the common long-strip path.
        gaps = np.diff(self.x)
        simple = self.simple_coarsening(gaps, left, right)
        if simple:
            return simple
        # An ordinary coarsening unit takes at least two source edges per
        # outer edge. When even the neighboring pair exceeds the maximum,
        # only its endpoint caps need individual decisions.
        margin = _EPS * self.contacts[-1]
        if (n >= 5 and len(self.contacts) == 2
                and np.all(gaps >= self.low - margin) and np.all(gaps <= self.high + margin)
                and np.all(gaps[:-1] + gaps[1:] > self.high + margin)):
            ends = []
            for at_left, edge in ((True, left), (False, right)):
                options = []
                for a in (1, 2):
                    i, j = (0, a) if at_left else (n - a, n)
                    answer = self.allocation(i, j, edge if at_left else 1,
                                             1 if at_left else edge)
                    if answer is not None:
                        cost, counts = answer
                        options.append((a, cost, i, j, counts))
                if not options:
                    break
                _, _, i, j, counts = min(options)
                ends.append((i, j, counts))
            if len(ends) == 2:
                return [ends[0], *((i, i + 1, [1]) for i in range(ends[0][1], ends[1][0])),
                        ends[1]]

        # Both directions have exactly projected interfaces. A greedy sweep
        # from one end can leave an unnecessarily short unit at the opposite
        # end. Compare their complete outer spacings before constructing cells.
        plans = []
        reverse_first = right < left
        for reverse in (reverse_first, not reverse_first):
            units = self.sweep(left, right, reverse=reverse)
            if not units:
                continue
            errors = []
            count = 0
            for i, j, counts in units:
                bounds = self.pieces(i, j)
                for start, end, b in zip(bounds, bounds[1:], counts):
                    errors.append(b * ((end - start) / (b * self.h) - 1) ** 2)
                    count += b
            plans.append(((fsum(errors) / count, count), units))
        return min(plans, key=lambda option: option[0])[1] if plans else []

    def materialize(self, units):
        count = sum(sum(counts) for _, _, counts in units)
        positions = np.empty(count + 1)
        cuts = [(0, 0)]
        cursor = 0
        for i, j, counts in units:
            bounds = self.pieces(i, j)
            for start, end, subdivisions in zip(bounds, bounds[1:], counts):
                positions[cursor] = start
                if subdivisions > 1:
                    positions[cursor + 1:cursor + subdivisions] = (
                        start + (end - start) * (np.arange(1, subdivisions) / subdivisions))
                cursor += subdivisions
            cuts.append((j, cursor))
        positions[-1] = self.contacts[-1]
        points = self.anchors[0] + positions[:, None] * np.r_[self.tangent, 0.]
        points[np.searchsorted(positions, self.contacts)] = self.anchors
        return points, cuts


def _fit_cuts(profile, sampled, cuts):
    """Fit actual paired projections within the fixed count of each anchor gap."""
    positions = (sampled[:, :2] - profile.anchors[0, :2]) @ profile.tangent
    anchor_ids = np.searchsorted(positions, profile.contacts)
    source_ids, cut_ids = np.asarray(cuts, dtype=int).T
    target = np.zeros(len(sampled))
    target[cut_ids] = profile.x[source_ids]
    knots = np.union1d(anchor_ids, cut_ids)
    wanted = target[knots]
    ends = np.searchsorted(knots, anchor_ids)
    wanted[ends] = profile.contacts
    actual = wanted.copy()
    for a, b in zip(ends, ends[1:]):
        if b <= a + 1:
            continue
        e = knots[a:b + 1]
        start, end = actual[a], actual[b]
        mean = (end - start) / (e[-1] - e[0])
        lo, hi = min(profile.low, mean), max(profile.high, mean)
        def sweep(indices, preferences, length):
            values = np.empty(len(indices))
            values[0], values[-1] = 0., length
            for k in range(1, len(indices) - 1):
                step = indices[k] - indices[k - 1]
                left = max(values[k - 1] + step * lo, length - (indices[-1] - indices[k]) * hi)
                right = min(values[k - 1] + step * hi, length - (indices[-1] - indices[k]) * lo)
                values[k] = min(right, max(left, preferences[k]))
            return values
        direct = sweep(e, wanted[a:b + 1] - start, end - start)
        reverse = sweep(-e[::-1], end - wanted[a:b + 1][::-1], end - start)
        actual[a:b + 1] = .5 * (start + direct + end - reverse[::-1])
        actual[a], actual[b] = start, end
    moved = np.interp(np.arange(len(positions)), knots, actual)
    points = profile.anchors[0] + moved[:, None] * np.r_[profile.tangent, 0.]
    points[anchor_ids] = profile.anchors
    return points


def plan_resampled(source, anchors, left, right, h, tol, cap, min_ratio, max_ratio,
                   *, cache=None, max_added_elements=None, size_band_active=True):
    """Yield (outer XYZ, ordered cut pairs) without constructing any elements."""
    cache = {} if cache is None else cache
    if "sampling" not in cache:
        cache["sampling"] = LineSampling(source, anchors, h, tol, cap, min_ratio, max_ratio,
                                         size_band_active=size_band_active)
    sampling = cache["sampling"]
    count_args = (len(source) - 1, len(left), len(right), max_added_elements)
    demand_counts = sampling.counts(len(left), len(right))
    if max_added_elements is not None:
        # This parity-adjusted lower bound applies to every layout, including
        # projected layouts whose count differs from both demand neighbors.
        lower = sampling.lower_count + (demand_counts[0] - sampling.lower_count) % 2
        _, over_budget = bounded_outer_counts([lower], *count_args)
        if over_budget is not None:
            raise over_budget
    retained, limit_error = bounded_outer_counts(demand_counts, *count_args)
    if "projected_line" not in cache:
        cache["projected_line"] = _ProjectedLine(sampling)
    profile = cache["projected_line"]
    units = profile.projected_units(len(left) - 1, len(right) - 1)
    tried = []

    def usable(points, cuts):
        if any(cuts == existing and np.array_equal(points, coords) for coords, existing in tried):
            return False
        try:
            sampling.validate_sample(points)
        except QuadLayerError as error:
            # The scalar feasibility margin admits roundoff. Coordinates in
            # the actual frame still have to satisfy the requested band.
            if error.code != "line_constraints_infeasible":
                raise
            return False
        tried.append((points, cuts))
        return True

    if units:
        count = sum(sum(counts) for _, _, counts in units)
        accepted, rejected = bounded_outer_counts([count], *count_args)
        if accepted:
            points, cuts = profile.materialize(units)
            if usable(points, cuts):
                yield points, cuts
        elif limit_error is None:
            limit_error = rejected

    # The fallback handles competition between forced contacts and source
    # projections. It supplies feasible interval counts; coordinates remain
    # adjustable around the actual paired source indices.
    alternatives = []
    for count in retained:
        sampled, _ = sampling.sample(count)
        for cuts in fixed_cut_plans(source, sampled, left, right, tol, cap):
            points = _fit_cuts(profile, sampled, cuts)
            if not usable(points, cuts):
                # Slanted end seams can project several source cuts onto one
                # mandatory contact. The original density sample stays ordered.
                if size_band_active or not usable(sampled, cuts):
                    continue
                points = sampled
            mean, peak = alignment_metrics(source, points, cuts[1:-1], tol)
            gaps = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
            size_error = float(np.mean((gaps / h - 1) ** 2))
            alternatives.append(((peak, mean, size_error, count), points, cuts))
    for _, points, cuts in sorted(alternatives, key=lambda option: option[0]):
        yield points, cuts
    if not tried and limit_error is not None:
        raise limit_error
