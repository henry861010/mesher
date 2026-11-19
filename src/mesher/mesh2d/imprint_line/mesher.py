"""Standalone line-control imprinting; no process-flow integration."""
from __future__ import annotations

import numpy as np

from ..model import Mesh2D
from ._input import prepare
from ._planning import plan
from ._types import ImprintLineError


def _prepare_plan(element_size, line_pairs, pattern_lines=(), *, outer_outline,
                  min_transition_size, pattern_snap_distance, tolerance=None,
                  max_elements=1_000_000):
    args = prepare(element_size, line_pairs, pattern_lines, outer_outline,
                   min_transition_size, pattern_snap_distance, tolerance, max_elements)
    domain, bands, cores, patterns, size, minimum, snap, policy, budget = args
    result = plan(domain, bands, cores, patterns, size, minimum, snap, policy, budget)
    return domain, result, policy, budget


def _assemble(domain, result, policy, budget):
    rectangles = result.rectangles
    counts = [(3 * len(chain) - 4) // 2 for t in result.transitions for chain in t.chains]
    budget.check(len(rectangles) + sum(counts), "imprint_assembly")
    if len(rectangles) and np.any(rectangles[:, 2:] - rectangles[:, :2] <= policy.absolute):
        raise ImprintLineError("background contains an unresolved edge", code="feature_below_resolution",
                               stage="background_assembly")
    nodes = [rectangles[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)]
    cells = [np.arange(4 * len(rectangles), dtype=np.int64).reshape(-1, 4)]
    offset = len(nodes[0])
    for transition in result.transitions:
        for xy, elements in transition.units:
            nodes.append(xy)
            cells.append(elements + offset)
            offset += len(xy)
    xy, inverse = np.unique(np.concatenate(nodes), axis=0, return_inverse=True)
    if len(xy) > np.iinfo(np.int32).max:
        raise ImprintLineError("node count exceeds int32 capacity", code="node_limit", stage="assembly")
    elements = inverse[np.concatenate(cells)]
    # Deterministic IDs and element ordering, independent of pair order.
    order = np.lexsort(elements[:, ::-1].T)
    mesh = Mesh2D(xy, elements[order])
    del nodes, cells, xy, inverse, elements, order
    _validate(mesh, domain, result.constraints)
    return mesh


def _validate(mesh, domain, constraints=None):
    """Linear-size topology checks; no quadratic all-element intersection scan."""
    e = mesh.elements
    edges = np.stack((e, np.roll(e, -1, axis=1)), axis=-1).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    ordered = np.sort(edges, axis=1)
    unique, counts = np.unique(ordered, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ImprintLineError("nonmanifold mesh edge", code="invalid_topology", stage="validation")
    # With valid CCW cells, opposite directed incidence is necessary at every seam.
    directed = np.unique(edges, axis=0)
    if len(directed) != len(edges):
        raise ImprintLineError("duplicate directed mesh edge", code="invalid_topology", stage="validation")
    boundary = mesh.nodes[unique[counts == 1], :2]
    accepted = np.zeros(len(boundary), dtype=bool)
    outline = domain.loops[0]
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        fixed = 0 if a[0] == b[0] else 1
        varying = 1 - fixed
        accepted |= (np.all(boundary[:, :, fixed] == a[fixed], axis=1)
                     & np.all(boundary[:, :, varying] >= min(a[varying], b[varying]), axis=1)
                     & np.all(boundary[:, :, varying] <= max(a[varying], b[varying]), axis=1))
    if not np.all(accepted):
        raise ImprintLineError("unmatched interior edge or hanging node", code="invalid_topology",
                               stage="validation", edge=boundary[np.flatnonzero(~accepted)[0]].tolist())
    points = mesh.nodes[e, :2]
    local = points - points[:, :1]
    nxt = np.roll(local, -1, axis=1)
    areas = .5 * np.sum(local[:, :, 0] * nxt[:, :, 1] - local[:, :, 1] * nxt[:, :, 0], axis=1)
    allowance = 256 * np.finfo(float).eps * max(1, len(areas)) * domain.area
    if np.any(areas <= 0) or abs(float(areas.sum()) - domain.area) > allowance:
        raise ImprintLineError("mesh does not cover the exterior exactly", code="invalid_coverage", stage="validation")

    if constraints is not None:
        del edges, ordered, counts, directed, boundary, accepted, points, local, nxt, areas
        constraints.validate(mesh, unique)


def imprint_lines(element_size: float, line_pairs, pattern_lines=(), *, outer_outline,
                  min_transition_size: float, pattern_snap_distance: float,
                  tolerance=None, max_elements: int = 1_000_000) -> Mesh2D:
    """Create a new conforming mixed mesh from directional rectangular controls.

    ``direction='x'`` uses two vertical cross sections with matching Y spans;
    points contribute their Y coordinates. ``'y'`` is the transposed case.
    ``line1_expand_size`` and ``line2_expand_size`` are required finite,
    nonnegative scalar distances associated with line1 and line2 respectively.
    E is straight extension *before* the transition, not its depth.

    All geometry is orthogonal; the exterior is a simple loop without holes.
    Pattern endpoints are fixed at their original float64 coordinates, including
    during input normalization. Finite patterns are clipped to the exterior;
    their retained segments and endpoints are checked before return. Conflicting
    fixed supports are rejected rather than moved. ``pattern_snap_distance``
    moves generated boundaries only. Each transition occupies one coarse cell,
    has three two-node sides, and contains only quads except for one triangle
    when its dense side has an odd number of nodes. No single band is widened.
    Same-direction overlapping envelopes merge; blocked ends
    extend to the exterior if no admissible transition remains.

    ``min_transition_size`` limits transition depth along the expansion
    direction only, not line length or transverse cell width. Transverse widths
    must exceed the geometric tolerance. Both size parameters use world units. Require
    ``tolerance < min_transition_size <= element_size``; snap distance may be
    zero. ``tolerance`` follows the package policy (default 1e-3). All arrays are
    privately owned, and input mappings/arrays are never modified.
    ``max_elements`` bounds final cells and intermediate rectangular partitions.
    Sizing must make progress; incompatible size and resolution requirements
    raise a structured error. Unresolvable transition units are replanned before
    assembly, with dense lines continuing to the exterior if necessary.
    Runtime is output-sensitive, not bounded by the number of pairs alone.
    """
    return _assemble(*_prepare_plan(element_size, line_pairs, pattern_lines,
                                   outer_outline=outer_outline, min_transition_size=min_transition_size,
                                   pattern_snap_distance=pattern_snap_distance,
                                   tolerance=tolerance, max_elements=max_elements))
