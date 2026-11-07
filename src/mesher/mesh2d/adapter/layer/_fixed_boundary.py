"""Exact shared-boundary contracts for public and full-mesh fixed interfaces."""

from __future__ import annotations

import numpy as np

from ._types import UnmeshableDirectQuadLayout


def _failure(message, stage, **context):
    raise UnmeshableDirectQuadLayout(message, stage=stage,
                                    parameter="outline_o", **context)


def _cyclic_order(points, target, tol, *, stage):
    """Match every point and directed edge without inserting or dropping nodes."""
    if len(points) != len(target):
        _failure("the fixed boundary has extra or missing nodes", stage,
                 expected_nodes=len(target), actual_nodes=len(points))
    starts = np.flatnonzero(np.linalg.norm(target - points[0], axis=1) <= tol)
    if len(starts) != 1:
        _failure("the fixed boundary has no unique cyclic correspondence", stage)
    order = (np.arange(len(target)) + int(starts[0])) % len(target)
    distances = np.linalg.norm(points - target[order], axis=1)
    if np.any(distances > tol):
        _failure("the fixed boundary changed a node or directed segment", stage,
                 point_index=int(np.flatnonzero(distances > tol)[0]))
    return order


def prepare_fixed_boundary(plan, data):
    """Freeze the supplied world coordinates after checking section endpoints."""
    anchors, tol = data["outline_anchors"], data["tol"]
    source_count = len(data["frozen"])
    if source_count % 2 != len(anchors) % 2:
        _failure("source and fixed target boundaries must have matching parity",
                 "fixed_boundary_plan", source_nodes=source_count,
                 target_nodes=len(anchors))

    world = data["_source_mesh"].nodes[data["outline_anchor_ids"], :2].copy()

    # Corner/dense mapping must consume existing contacts, and patterns must
    # already meet anchors. Any unlisted contact would split a frozen edge.
    points = np.concatenate([section.outer_boundary[:-1]
                             for section in plan.sections
                             if len(section.outer_boundary) > 1])
    _cyclic_order(points, anchors, tol, stage="fixed_boundary_plan")
    data.update(fixed_outer_boundary=True, fixed_outline_world=world)


def fixed_boundary_nodes(nodes, outline, data, *, stage="fixed_boundary_validation"):
    """Return boundary IDs and their one-to-one canonical target indices."""
    ids = np.concatenate([np.asarray(chain, dtype=np.int64)[:-1] for chain in outline])
    if len(np.unique(ids)) != len(ids):
        _failure("the fixed boundary repeats a node", stage)
    order = _cyclic_order(nodes[ids, :2], data["outline_anchors"], data["tol"],
                          stage=stage)
    if not np.array_equal(ids, data["outline_anchor_ids"][order]):
        _failure("the fixed boundary changed a specified node ID", stage)
    return ids, order


def restore_fixed_boundary(world_nodes, normalized_nodes, outline, data):
    """Restore exact global interface coordinates without changing source IDs."""
    ids, order = fixed_boundary_nodes(normalized_nodes, outline, data)
    world_nodes[ids, :2] = data["fixed_outline_world"][order]
    world_nodes[ids, 2] = 0.0
