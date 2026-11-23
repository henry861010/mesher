"""Fill a reference rectangle using its fixed interior lines and four edge chains."""
from __future__ import annotations

import numpy as np

from ._constraints import on_outline
from ._types import ImprintLineError
from ._unit import build_unit


def _nearest(levels, fixed):
    right = np.minimum(fixed.searchsorted(levels), len(fixed) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(levels - fixed[left] <= fixed[right] - levels, fixed[left], fixed[right])


def _cell_rows(xy, ids):
    """Collapse surplus boundary vertices. Discard zero-area corners, pad Tri3s."""
    rows = np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]),
                    axis=-1).reshape(-1, 4)
    different = rows != np.roll(rows, 1, axis=1)
    counts = different.sum(axis=1)
    tris = rows[counts == 3][different[counts == 3]].reshape(-1, 3)
    quads = rows[counts == 4]
    # A collapsed corner can leave a flat angle. Use the interior diagonal
    # so both surviving elements are strictly convex.
    v = np.roll(xy[quads], -1, axis=1) - xy[quads]
    turn = v[..., 0] * np.roll(v[..., 1], -1, axis=1) - v[..., 1] * np.roll(v[..., 0], -1, axis=1)
    bad = np.any(turn <= 0, axis=1)
    split = []
    for q, turns in zip(quads[bad], turn[bad]):
        corner = (np.argmin(turns) + 1) % 4
        a, b, c, d = np.roll(q, -corner)
        split.extend(((a, b, c), (a, c, d)))
    if split:
        tris = np.concatenate((tris, split))
    return np.vstack((quads[~bad], tris[:, [0, 1, 2, 2]]))


def _protect_exterior(xy, cells, domain):
    """Wrap triangles owning a physical exterior edge with three local quads."""
    tri_ids = np.flatnonzero(cells[:, 2] == cells[:, 3])
    if not len(tri_ids):
        return xy, cells
    tris = cells[tri_ids, :3]
    edges = np.stack((tris, np.roll(tris, -1, axis=1)), axis=2)
    boundary = on_outline(xy[edges.reshape(-1, 2)], domain.loops[0]).reshape(-1, 3)
    selected = np.any(boundary, axis=1)
    if not selected.any():
        return xy, cells
    tris = tris[selected]
    corners = xy[tris]
    local = corners - corners[:, :1]
    inside = corners[:, :1] + (local + local.mean(axis=1, keepdims=True)) / 2
    inner = np.arange(len(xy), len(xy) + 3 * len(tris)).reshape(-1, 3)
    ring = np.stack((tris, np.roll(tris, -1, axis=1),
                     np.roll(inner, -1, axis=1), inner), axis=2)
    cells = np.vstack((np.delete(cells, tri_ids[selected], axis=0),
                       ring.reshape(-1, 4), inner[:, [0, 1, 2, 2]]))
    return np.vstack((xy, inside.reshape(-1, 2))), cells


def invalid_cells(xy, cells, tolerance):
    invalid = np.zeros(len(cells), dtype=bool)
    for n in (3, 4):
        group = (cells[:, 2] != cells[:, 3]) == (n == 4)
        vertices = xy[cells[group, :n]]
        v = np.roll(vertices, -1, axis=1) - vertices
        following = np.roll(v, -1, axis=1)
        turns = v[..., 0] * following[..., 1] - v[..., 1] * following[..., 0]
        invalid[group] = (~np.all(turns > 0, axis=1)
                          | np.any(np.linalg.norm(v, axis=2) <= tolerance, axis=1))
    return invalid


def mesh_cell(bounds, levels, sides, domain, budget, preceding):
    """Build a local Cartesian grid and terminate extra lines at its edge chains.

    Interior lattice intersections retain both coordinates. Each perimeter
    vertex maps monotonically to the nearest required contact on that side.
    Two adjacent cells therefore produce exactly the same sequence of edges.
    """
    fixed = []
    axes = []
    for axis in (0, 1):
        t = 1 - axis
        ends = (bounds[t], bounds[t + 2])
        fixed.extend(np.asarray(sorted({*ends, *sides[2 * axis + side]})) for side in (0, 1))
    for a in (0, 1):
        transverse = 1 - a
        low, high = fixed[2 * transverse:2 * transverse + 2]
        axes.append(np.array(sorted({*levels[a], *low, *high})))
    active = [i for i, values in enumerate(fixed) if len(values) > 2]
    if len(active) == 1:
        wall = active[0]
        axis = wall // 2
        if len(axes[axis]) == 2 and np.array_equal(axes[1 - axis], fixed[wall]):
            budget.check(preceding + (3 * len(fixed[wall]) - 4) // 2, "imprint_cell_rebuild")
            try:
                return build_unit(bounds, axis, 1 - wall % 2, fixed[wall],
                                  domain.tolerance, outline=domain.loops[0])
            except ImprintLineError as error:
                if error.code not in ("unit_below_resolution", "exterior_triangle"):
                    raise

    # With different chains on opposite sides, one interior column allows
    # both interfaces to keep their own contacts independently.
    for a in (0, 1):
        if len(axes[a]) == 2 and not np.array_equal(fixed[2 * a], fixed[2 * a + 1]):
            lo, hi = axes[a]
            mid = lo + (hi - lo) / 2
            if min(mid - lo, hi - mid) > domain.tolerance:
                axes[a] = np.array([lo, mid, hi])
    nx, ny = map(len, axes)
    budget.check(preceding + (nx - 1) * (ny - 1), "imprint_cell_grid")
    xy = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 2)
    ids = np.arange(len(xy)).reshape(ny, nx)
    for a in (0, 1):
        t = 1 - a
        for side, end in enumerate((0, -1)):
            edge = ids[:, end] if a == 0 else ids[end, :]
            positions = axes[t].searchsorted(_nearest(axes[t], fixed[2 * a + side]))
            edge[:] = edge[positions]
    cells = _cell_rows(xy, ids)
    xy, cells = _protect_exterior(xy, cells, domain)
    if np.any(invalid_cells(xy, cells, domain.tolerance)):
        raise ImprintLineError("local cell contains a flat corner or unresolved edge",
                               code="feature_below_resolution", stage="unit_construction",
                               bounds=bounds)
    used, inverse = np.unique(cells, return_inverse=True)
    return xy[used], inverse.reshape(-1, 4)
