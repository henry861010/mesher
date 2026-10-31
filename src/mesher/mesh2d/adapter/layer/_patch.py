"""Shared four-sided direct Quad4 patch assembly."""
from __future__ import annotations

import numpy as np

from ..unit.build_unit_1_1_x_y import _validate_patch
from ._limits import ElementLimitExceeded, check_element_limit
from ._unit_plan import CoordinatePool, UnitPlan, build_unit_patch, merge_unit_plans


def _check_units(units, sides, tol):
    nodes, quads, _ = merge_unit_plans(units, tol)
    # Find the original corner chains in the merged, boundary-preserving pool.
    pool = CoordinatePool(tol)
    for point in nodes:
        pool.intern(point)
    ids = tuple(pool.chain(side) for side in sides)
    if len(pool.points) != len(nodes):
        raise ValueError("corner units omitted a boundary node")
    _validate_patch(nodes, quads, ids)


def _four_sided_units(sides, tol, max_added_elements=None, *, _cache=None) -> list[UnitPlan]:
    """Try the natural corner patch, then two-unit cuts on opposite chains.

    Cuts retain each original boundary segment. Their parity midpoint is one
    shared connector, and each child has a one-segment side for a unit builder.
    No triangular cells or global ring fallback are constructed here.
    """
    candidates = []
    limit_errors = []
    if any(len(side) == 2 for side in sides):
        try:
            unit = build_unit_patch(sides, tol, _cache=_cache)
            check_element_limit(len(unit.quads), max_added_elements, stage="section_builder")
            candidates.append([unit])
        except ElementLimitExceeded as error:
            limit_errors.append(error)
        except ValueError:
            pass
    for rotation in (0, 1):
        ordered = sides[rotation:] + sides[:rotation]
        bottom, right, top, left = ordered
        n, m = len(bottom) - 1, len(top) - 1
        if min(n, m) < 2:
            continue
        for i, j in sorted({(1, 1), (n - 1, m - 1)}):
            if np.linalg.norm(bottom[i, :2] - top[j, :2]) <= tol:
                continue
            connector = np.asarray((bottom[i], top[j]))
            # First child travels bottom[i:] -> right -> top[:j+1] -> cut.
            path_count = (n - i) + len(right) - 1 + j
            if path_count % 2 == 0:
                midpoint = np.r_[(bottom[i, :2] + top[j, :2]) / 2, 0.0]
                connector = np.asarray((bottom[i], midpoint, top[j]))
            first = (bottom[i:], right, top[:j + 1], connector[::-1])
            second = (bottom[:i + 1], connector, top[j:], left)
            try:
                units = [build_unit_patch(first, tol, _cache=_cache)]
                check_element_limit(len(units[0].quads), max_added_elements,
                                    stage="section_builder")
                second_unit = build_unit_patch(second, tol, _cache=_cache)
                check_element_limit(len(units[0].quads) + len(second_unit.quads),
                                    max_added_elements, stage="section_builder")
                units.append(second_unit)
                _check_units(units, sides, tol)
                candidates.append(units)
            except ElementLimitExceeded as error:
                limit_errors.append(error)
            except ValueError:
                continue
    if not candidates:
        if limit_errors:
            raise limit_errors[0]
        raise ValueError("corner has no valid direct unit assembly for these seams")
    return min(candidates, key=lambda units: (max(unit.quality for unit in units), len(units)))


