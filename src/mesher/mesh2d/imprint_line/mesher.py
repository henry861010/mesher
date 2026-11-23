"""Standalone line-control imprinting; no process-flow integration."""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..model import Mesh2D
from ._constraints import on_outline
from ._input import prepare
from ._pattern_cut import plan as plan_pattern_cut
from ._planning import plan as plan_mesh_snap
from ._types import ImprintLineError


def _prepare_plan(element_size, line_pairs, pattern_lines=(), *, outer_outline,
                  min_transition_size, tag, tolerance=None, max_elements=1_000_000):
    if not isinstance(tag, str) or tag not in ("pattern_cut", "mesh_snap"):
        raise ValueError("tag must be 'pattern_cut' or 'mesh_snap'")
    args = prepare(element_size, line_pairs, pattern_lines, outer_outline,
                   min_transition_size, tolerance, max_elements)
    domain, bands, cores, patterns, size, minimum, policy, budget = args
    if tag == "pattern_cut":
        result = plan_pattern_cut(domain, bands, cores, patterns, size, minimum, policy, budget)
    else:
        result = plan_mesh_snap(domain, bands, cores, patterns, size, minimum, budget)
    return domain, result, policy, budget


def _assemble(domain, result, policy, budget):
    rectangles = result.rectangles
    counts = [len(elements) for _, elements in result.units]
    budget.check(len(rectangles) + sum(counts), "imprint_assembly")
    if len(rectangles) and np.any(rectangles[:, 2:] - rectangles[:, :2] <= policy.absolute):
        raise ImprintLineError("background contains an unresolved edge", code="feature_below_resolution",
                               stage="background_assembly")
    nodes = [rectangles[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)]
    cells = [np.arange(4 * len(rectangles), dtype=np.int64).reshape(-1, 4)]
    offset = len(nodes[0])
    for xy, elements in result.units:
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
    nonzero = edges[:, 0] != edges[:, 1]
    edges = edges[nonzero]
    ordered = np.sort(edges, axis=1)
    unique, first, counts = np.unique(ordered, axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ImprintLineError("nonmanifold mesh edge", code="invalid_topology", stage="validation")
    # With valid CCW cells, opposite directed incidence is necessary at every seam.
    directed = np.unique(edges, axis=0)
    if len(directed) != len(edges):
        raise ImprintLineError("duplicate directed mesh edge", code="invalid_topology", stage="validation")
    boundary_edges = unique[counts == 1]
    boundary = mesh.nodes[boundary_edges, :2]
    accepted = on_outline(boundary, domain.loops[0])
    if not np.all(accepted):
        raise ImprintLineError("unmatched interior edge or hanging node", code="invalid_topology",
                               stage="validation", edge=boundary[np.flatnonzero(~accepted)[0]].tolist())
    boundary_count = np.unique(boundary_edges).size
    if boundary_count % 2:
        raise ImprintLineError("exterior node count must be even", code="invalid_boundary_parity",
                               stage="validation", boundary_nodes=boundary_count)
    owners = np.flatnonzero(nonzero)[first[counts == 1]] // 4
    if np.any(e[owners, 2] == e[owners, 3]):
        raise ImprintLineError("exterior edges require Quad4 cells", code="exterior_triangle",
                               stage="validation")
    del nonzero, first, owners, boundary_edges
    points = mesh.nodes[e, :2]
    for axis in (0, 1):
        if domain.bounds[axis] < 0 < domain.bounds[axis + 2]:
            coordinates = points[:, :, axis]
            if np.any((coordinates.min(axis=1) < 0) & (coordinates.max(axis=1) > 0)):
                raise ImprintLineError("cell crosses a symmetry axis", code="constraint_not_preserved",
                                       stage="validation", source="symmetry_axes", source_index=axis)
            del coordinates
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
                  min_transition_size: float, tag: Literal["pattern_cut", "mesh_snap"],
                  tolerance=None, max_elements: int = 1_000_000) -> Mesh2D:
    """Create a new conforming mixed mesh from directional rectangular controls.

    ``direction='x'`` uses two vertical cross sections with matching Y spans;
    points contribute their Y coordinates. ``'y'`` is the transposed case.
    ``line1_expand_size`` and ``line2_expand_size`` are required finite,
    nonnegative scalar distances associated with line1 and line2 respectively.
    Each expand size is the minimum straight extension before the transition.

    All geometry is orthogonal; the exterior is a simple loop without holes.
    Its distinct node count is even and every exterior edge belongs to a Quad4.
    Both world axes (x=0 and y=0) are covered by conforming edges wherever
    they meet the domain. Their intersection is a shared node when present.
    Pattern endpoints are fixed at their original float64 coordinates, including
    during input normalization. Finite patterns are clipped to the exterior;
    their retained segments and endpoints are checked before return. Conflicting
    fixed supports are rejected.

    With ``tag="pattern_cut"``, the planner reserves adapter rectangles and cuts
    the surrounding grid along their boundaries. With ``tag="mesh_snap"``, each end extends to the first eligible reference
    mesh edge at or beyond its requested expansion. The adjacent outward layer
    must cover the band width and meet the minimum depth. All its cells are
    rebuilt in full. The reference grid comes from the outline, patterns,
    symmetry axes and element size. Adapters sharing a coarse cell contribute
    to one local mesh with conforming coarse-side edges.

    ``min_transition_size`` limits transition depth along the expansion
    direction only, not line length or transverse cell width. Transverse widths
    must exceed the geometric tolerance. Both size parameters use world units.
    Require ``tolerance < min_transition_size <= element_size``. ``tolerance``
    follows the package policy (default 1e-3). All arrays are privately owned,
    and input mappings/arrays are never modified.
    ``max_elements`` bounds final cells and intermediate rectangular partitions.
    Sizing must make progress; incompatible size and resolution requirements
    raise a structured error. Ends without room for the minimum transition
    depth carry their dense lines to the exterior.
    Runtime is output-sensitive, not bounded by the number of pairs alone.
    """
    return _assemble(*_prepare_plan(element_size, line_pairs, pattern_lines,
                                   outer_outline=outer_outline, min_transition_size=min_transition_size, tag=tag,
                                   tolerance=tolerance, max_elements=max_elements))
