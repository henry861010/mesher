"""Rectangular 1-1-1-N units with at most one parity triangle."""
from __future__ import annotations

import numpy as np

from ._types import ImprintLineError


def build_unit(bounds, axis, side, coordinates, tolerance):
    """Return independent XY nodes and padded connectivity; boundaries are fixed.

    Remaining depth is proportional to remaining dense-edge span, rather than
    halved at every ring. Thus many input points do not exponentially collapse
    generated edges. The topology otherwise follows adapter.unit's 111x unit.
    """
    transverse = 1 - axis
    width = bounds[transverse + 2] - bounds[transverse]
    depth = bounds[axis + 2] - bounds[axis]
    u = np.asarray(coordinates) - bounds[transverse]
    n = len(u)
    points = [[float(x), 0.] for x in u] + [[0., depth], [width, depth]]
    lower = list(range(n))
    tl, tr = n, n + 1
    elements = []
    while len(lower) > 3:
        left, right = points[lower[1]][0], points[lower[-2]][0]
        height = depth * ((right - left) / width)
        il, ir = len(points), len(points) + 1
        points.extend(([left, height], [right, height]))
        elements.extend(([lower[0], lower[1], il, tl], [tl, il, ir, tr],
                         [lower[-2], lower[-1], tr, ir]))
        lower = lower[1:-1]
        tl, tr = il, ir
    if len(lower) == 2:
        elements.append([*lower, tr, tl])
    else:
        a, b, c = lower
        alternatives = (
            [[a, b, tr, tl], [b, c, tr, tr]],
            [[a, b, tl, tl], [b, c, tr, tl]],
        )

        def score(cells):
            worst = 0.
            for cell in cells:
                ids = cell[:3] if cell[2] == cell[3] else cell
                xy = np.asarray([points[i] for i in ids])
                edges = np.roll(xy, -1, axis=0) - xy
                lengths = np.linalg.norm(edges, axis=1)
                turn = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
                worst = max(worst, float(lengths.max() / lengths.min()),
                            float(np.max(lengths * np.roll(lengths, -1) / turn)))
            return worst

        elements.extend(min(alternatives, key=score))
    local = np.asarray(points)
    xy = np.empty_like(local)
    xy[:, transverse] = bounds[transverse] + local[:, 0]
    inner = bounds[axis + (2 if side == 0 else 0)]
    xy[:, axis] = inner + (-1 if side == 0 else 1) * local[:, 1]
    # Restore input coordinates, including corners, without roundtrip drift.
    xy[:n, transverse] = coordinates
    xy[:n, axis] = inner
    xy[n:n + 2, transverse] = (bounds[transverse], bounds[transverse + 2])
    xy[n:n + 2, axis] = bounds[axis + (0 if side == 0 else 2)]
    cells = np.asarray(elements, dtype=np.int64)
    if (axis == 0 and side == 1) or (axis == 1 and side == 0):
        quad = cells[:, 2] != cells[:, 3]
        cells[quad] = cells[quad][:, [0, 3, 2, 1]]
        cells[~quad] = cells[~quad][:, [0, 2, 1, 1]]
    for count in (3, 4):
        selected = cells[(cells[:, 2] != cells[:, 3]) == (count == 4), :count]
        p = xy[selected]
        v = np.roll(p, -1, axis=1) - p
        lengths = np.linalg.norm(v, axis=2)
        nxt = np.roll(v, -1, axis=1)
        turns = v[..., 0] * nxt[..., 1] - v[..., 1] * nxt[..., 0]
        if np.any(lengths <= tolerance) or np.any(turns <= 0) or not np.all(np.isfinite(turns)):
            raise ImprintLineError("unit geometry is below resolution or inverted",
                                   code="invalid_unit", stage="unit_construction", bounds=bounds)
    return xy, cells
