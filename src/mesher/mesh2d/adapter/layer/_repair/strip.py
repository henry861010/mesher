"""Tile an inward repair ring or elongated core with its frozen boundary chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._construction.units import SectionPatch, UnitPlan, assemble_section, build_unit_patch
from .._limits import ElementLimitExceeded, check_element_limit


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


def build_repair_strip(source: np.ndarray, outer: np.ndarray, left: np.ndarray,
                       right: np.ndarray, tol: float, *,
                       build_cache: dict | None = None,
                       max_added_elements: int | None = None) -> SectionPatch | None:
    """Alternate from both ends, with three cut choices and one closing repair.

    The shorter chain advances exactly one segment per unit. Only invalid
    geometry triggers another build: no quality comparison changes a cut.
    Each sampled outer configuration needs at most 8 * min(n, m) unit trials.
    """
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
        for index in (nearest, nearest - 1, nearest + 1):
            if not minimum <= index <= maximum:
                continue
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
                            source, outer, left, right, 0., tol)
