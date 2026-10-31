"""Sample the outer edge first, then match LINE units without a global search."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np

from ._limits import (ElementLimitExceeded, bounded_outer_counts, check_element_limit,
                      outer_segment_counts)
from ._unit_plan import SectionPatch, UnitPlan, assemble_section, build_unit_patch, xyz_chain


def outer_count_candidates(source_segments: int, left_nodes: int, right_nodes: int,
                           length: float, h: float, mandatory_intervals: int) -> list[int]:
    """The closest admissible count below and above the target, at most two."""
    result = outer_segment_counts(source_segments, left_nodes, right_nodes,
                                  length, h, max(1, mandatory_intervals))
    return sorted(result, key=lambda count: (abs(length / count - h), count))


def sample_outer(anchors: np.ndarray, count: int, h: float,
                 tol: float) -> tuple[np.ndarray, float]:
    """Preserve mandatory points and optimally allocate a fixed segment count."""
    anchors = xyz_chain(anchors)
    lengths = np.linalg.norm(np.diff(anchors[:, :2], axis=0), axis=1)
    if np.any(lengths <= tol) or count < len(lengths):
        raise ValueError("outer intervals must have positive length and at least one segment")
    counts = np.ones(len(lengths), dtype=np.int64)
    heap = []
    for index, length in enumerate(lengths):
        heappush(heap, (h * h - length * length / 2, index))
    for _ in range(count - len(lengths)):
        _, index = heappop(heap)
        counts[index] += 1
        n = int(counts[index])
        heappush(heap, (h * h - lengths[index] ** 2 / (n * (n + 1)), index))
    points = [anchors[0].copy()]
    for start, end, n in zip(anchors[:-1], anchors[1:], counts):
        points.extend(
            np.r_[start[:2] + (end[:2] - start[:2]) * (i / n), 0.0]
            for i in range(1, int(n))
        )
        points.append(end.copy())
    error = float(np.sum(counts * (lengths / counts / h - 1) ** 2) / count)
    return np.asarray(points), error


@dataclass
class _Cut:
    long_index: int
    chain: np.ndarray


def _connector(source: np.ndarray, outer: np.ndarray, count: int) -> np.ndarray:
    if count == 2:
        return np.asarray((source, outer))
    return np.asarray((source, np.r_[.5 * (source[:2] + outer[:2]), 0.0], outer))


def _nearest_index(values: np.ndarray, target: float, low: int, high: int) -> int:
    """Nearest allowed projected node; ties choose the smaller chain index."""
    upper = min(high, max(low, int(np.searchsorted(values, target))))
    lower = max(low, upper - 1)
    return min((lower, upper), key=lambda index: (abs(values[index] - target), index))


def _plan_count(source: np.ndarray, outer: np.ndarray, left: np.ndarray,
                right: np.ndarray, size_error: float, tol: float,
                build_cache: dict | None = None,
                stats: dict | None = None,
                max_added_elements: int | None = None) -> SectionPatch | None:
    """Alternate from both ends, with three cut choices and one closing repair.

    The shorter chain advances exactly one segment per unit. Only invalid
    geometry triggers another build: no quality comparison changes a cut.
    Each sampled outer configuration needs at most 8 * min(n, m) unit trials.
    """
    if stats is None:
        stats = {"unit_trial_count": 0, "local_repair_count": 0}
    n, m = len(source) - 1, len(outer) - 1
    coarsening = n >= m
    short_count, long_count = min(n, m), max(n, m)
    axis = int(np.argmax(np.abs(outer[-1, :2] - outer[0, :2])))
    direction = 1 if outer[-1, axis] > outer[0, axis] else -1
    short, long = (outer, source) if coarsening else (source, outer)
    short_values, long_values = short[:, axis] * direction, long[:, axis] * direction
    if np.any(np.diff(short_values) <= tol) or np.any(np.diff(long_values) <= tol):
        raise ValueError("LINE nodes must be strictly ordered along the main axis")
    cuts = {0: _Cut(0, left), short_count: _Cut(long_count, right)}
    units: dict[int, UnitPlan] = {}
    quad_count = 0

    def indices(position, long_index):
        return (long_index, position) if coarsening else (position, long_index)

    def make_cut(position, index, fixed_position, fixed):
        i, j = indices(position, index)
        segments = abs(position - fixed_position) + abs(index - fixed.long_index)
        count = 2 if (len(fixed.chain) - 1 + segments) % 2 else 3
        return _Cut(index, _connector(source[i], outer[j], count))

    def trial(start, end, entry, exit_cut):
        i, j = indices(start, entry.long_index)
        ni, nj = indices(end, exit_cut.long_index)
        sides = (source[i:ni + 1], exit_cut.chain,
                 outer[j:nj + 1][::-1], entry.chain[::-1])
        stats["unit_trial_count"] += 1
        try:
            return build_unit_patch(sides, tol, _cache=build_cache)
        except ElementLimitExceeded:
            raise
        except ValueError:
            return None

    low, high = 0, short_count
    from_left = True
    while high - low > 1:
        remaining = high - low - 1
        position = low + 1 if from_left else high - 1
        minimum = cuts[low].long_index + (1 if from_left else remaining)
        maximum = cuts[high].long_index - (remaining if from_left else 1)
        nearest = _nearest_index(long_values, short_values[position], minimum, maximum)
        fixed_position = low if from_left else high
        fixed = cuts[fixed_position]
        for attempt, index in enumerate((nearest, nearest - 1, nearest + 1)):
            if not minimum <= index <= maximum:
                continue
            if attempt:
                stats["local_repair_count"] += 1
            cut = make_cut(position, index, fixed_position, fixed)
            unit = (trial(low, position, fixed, cut) if from_left
                    else trial(position, high, cut, fixed))
            if unit is not None:
                quad_count += len(unit.quads)
                check_element_limit(quad_count, max_added_elements, stage="section_builder")
                cuts[position] = cut
                units[low if from_left else position] = unit
                break
        else:
            return None
        if from_left:
            low = position
        else:
            high = position
        from_left = not from_left

    closing = trial(low, high, cuts[low], cuts[high])
    if closing is None and units:
        # Move only the last committed cut, by one node, rebuilding its two
        # owners. The farther connector remains fixed; there is no backtracking.
        neighbor_on_left = not from_left
        shared = low if neighbor_on_left else high
        far = low - 1 if neighbor_on_left else high + 1
        for index in (cuts[shared].long_index - 1, cuts[shared].long_index + 1):
            minimum = cuts[far if neighbor_on_left else low].long_index + 1
            maximum = cuts[high if neighbor_on_left else far].long_index - 1
            if not minimum <= index <= maximum:
                continue
            stats["local_repair_count"] += 1
            cut = make_cut(shared, index, far, cuts[far])
            neighbor = (trial(far, shared, cuts[far], cut) if neighbor_on_left
                        else trial(shared, far, cut, cuts[far]))
            if neighbor is None:
                continue
            closing = (trial(low, high, cut, cuts[high]) if neighbor_on_left
                       else trial(low, high, cuts[low], cut))
            if closing is not None:
                owner = far if neighbor_on_left else shared
                quad_count += len(neighbor.quads) - len(units[owner].quads)
                check_element_limit(quad_count + len(closing.quads), max_added_elements,
                                    stage="section_builder")
                cuts[shared] = cut
                units[far if neighbor_on_left else shared] = neighbor
                break
    if closing is None:
        return None
    check_element_limit(quad_count + len(closing.quads), max_added_elements,
                        stage="section_builder")
    units[low] = closing
    return assemble_section([units[i] for i in range(short_count)],
                            source, outer, left, right, size_error, tol)


def plan_line(source: np.ndarray, outer: np.ndarray, left: np.ndarray,
              right: np.ndarray, h: float, tol: float, *,
              best_only: bool = False,
              _diagnostics: dict | None = None,
              max_added_elements: int | None = None) -> list[SectionPatch]:
    """Return up to two feasible target-size candidates for a LINE section.

    Source and outer run in the same section-travel direction. Both seams run
    from source to outer. Source segments and every supplied anchor are fixed.
    Empty results mean these unit strategies could not mesh either count.
    With ``best_only``, stop at the first feasible count, ordered by size error
    then segment count. Quality never triggers additional LINE construction.
    """
    source, outer, left, right = map(xyz_chain, (source, outer, left, right))
    if not np.isfinite(h) or h <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("element size and tolerance must be positive and finite")
    # Section partitioning already orders the anchors. Collapse repeated
    # contacts without resampling or moving the surviving mandatory points.
    unique = [outer[0]]
    for point in outer[1:]:
        if np.linalg.norm(point[:2] - unique[-1][:2]) > tol:
            unique.append(point)
    outer = np.asarray(unique)
    if len(outer) < 2:
        raise ValueError("LINE outer boundary has no usable length")
    for actual, expected in ((left[0], source[0]), (left[-1], outer[0]),
                             (right[0], source[-1]), (right[-1], outer[-1])):
        if np.linalg.norm(actual[:2] - expected[:2]) > tol:
            raise ValueError("LINE source, outer and seam endpoints do not match")
    if any(np.any(np.linalg.norm(np.diff(chain[:, :2], axis=0), axis=1) <= tol)
           for chain in (source, outer, left, right)):
        raise ValueError("LINE boundary contains a zero-length segment")
    length = float(np.linalg.norm(np.diff(outer[:, :2], axis=0), axis=1).sum())
    counts = outer_count_candidates(len(source) - 1, len(left), len(right),
                                    length, h, len(outer) - 1)
    counts, limit_error = bounded_outer_counts(
        counts, len(source) - 1, len(left), len(right), max_added_elements)
    sampled_candidates = [(*sample_outer(outer, count, h, tol), count) for count in counts]
    sampled_candidates.sort(key=lambda item: (item[1], item[2]))
    candidates = []
    build_cache: dict = {}
    stats = {} if _diagnostics is None else _diagnostics
    stats.update(line_planner="outer_first", unit_trial_count=0,
                 local_repair_count=0, outer_count_attempts=0)
    for sampled, error, _ in sampled_candidates:
        stats["outer_count_attempts"] += 1
        try:
            patch = _plan_count(source, sampled, left, right, error, tol,
                                build_cache, stats, max_added_elements)
        except ElementLimitExceeded as error:
            limit_error = error
            continue
        if patch is not None:
            candidates.append(patch)
            if best_only:
                break
    for patch in candidates:
        patch.diagnostics.update(stats)
    if not candidates and limit_error is not None:
        raise limit_error
    return candidates


__all__ = ["plan_line", "outer_count_candidates", "sample_outer"]
