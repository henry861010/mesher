"""One coordinate identity for orthogonal constraints before grid construction."""
from __future__ import annotations

import numpy as np

from ...tolerance import (GeometryToleranceError, canonicalize_points,
                             resolve_tolerance)


def canonical_geometry(outer_outline, holes=(), pattern_lines=(), tolerance=None,
                       *, fixed_constraints=False):
    """Canonicalize all supports together, never independently by boundary.

    Existing mesh nodes do not enter this operation. Full adaptation may fix
    its already canonical outer/pattern constraints while moving generated
    fronts. A collapsed required edge is an unresolved feature, not a license
    to silently drop a material boundary or change its parity.
    """
    policy = resolve_tolerance(tolerance)
    loops = [np.array(value, dtype=float, copy=True) for value in (outer_outline, *holes)]
    for index, loop in enumerate(loops):
        if loop.ndim != 2 or loop.shape[1:] != (2,) or len(loop) < 4:
            raise ValueError("outlines must be finite XY arrays with at least four vertices")
        if len(loop) > 1 and np.array_equal(loop[0], loop[-1]):
            loops[index] = loop[:-1].copy()
    patterns = np.array(list(pattern_lines), dtype=float)
    if patterns.size == 0:
        patterns = np.empty((0, 2, 2), dtype=float)
    if patterns.ndim != 3 or patterns.shape[1:] != (2, 2):
        raise ValueError("pattern_lines must have shape (n, 2, 2)")
    sizes = [len(loop) for loop in loops]
    points = np.vstack((*loops, patterns.reshape(-1, 2)))
    policy.check_resolution(points, stage="geometry_canonicalization", source="orthogonal_constraints")
    fixed = None
    if fixed_constraints:
        fixed = np.zeros(len(points), dtype=bool)
        fixed[:sizes[0]] = True
        fixed[sum(sizes):] = True
    canonical = canonicalize_points(points, policy, fixed=fixed,
                                    stage="geometry_canonicalization", source="orthogonal_constraints")
    offset, result = 0, []
    for index, count in enumerate(sizes):
        loop = canonical[offset:offset + count].copy()
        original = points[offset:offset + count]
        offset += count
        # A caller can close a loop with a near-coincident copy of its start.
        if len(loop) > 4 and np.array_equal(loop[0], loop[-1]):
            loop = loop[:-1]
            original = original[:-1]
        collapsed = np.all(loop == np.roll(loop, -1, axis=0), axis=1)
        if np.any(collapsed):
            edge = int(np.flatnonzero(collapsed)[0])
            raise GeometryToleranceError(
                "a required boundary edge is unresolved at the requested tolerance",
                code="feature_below_resolution", stage="geometry_canonicalization",
                source="outer_outline" if index == 0 else f"hole[{index - 1}]",
                boundary_index=index - 1, edge_index=edge,
                measured_distance=float(np.linalg.norm(original[(edge + 1) % len(original)] - original[edge])),
                tolerance=policy.absolute)
        result.append(loop)
    patterns = canonical[offset:].reshape(-1, 2, 2).copy()
    return result[0], result[1:], patterns
