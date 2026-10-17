"""Final triangulation of nearly straight corners in the local imprint patch."""

import numpy as np
from numpy.typing import NDArray


def _unit_corner_edges(points):
    previous = np.roll(points, 1, axis=-2) - points
    following = np.roll(points, -1, axis=-2) - points
    with np.errstate(divide="ignore", invalid="ignore"):
        previous = previous / np.hypot(previous[..., 0], previous[..., 1])[..., None]
        following = following / np.hypot(following[..., 0], following[..., 1])[..., None]
    return previous, following


def _triangulate_flat_quads(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int32],
    *,
    flat_quad_angle: float = 170.0,
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    """Split CCW quads at the supplied interior-angle threshold, in degrees.

    ``flat_quad_angle`` is validated by the local imprint orchestrator.
    Choose the diagonal maximizing the smallest angle across both triangles.
    Only positive-area pairs qualify, so the split preserves the entire cell
    and its boundary, including a nearly collinear corner node. Coordinates
    and input arrays are never changed. Return connectivity and parent rows
    so the local stitching pass can retain element provenance.
    """
    sources = np.arange(len(elements), dtype=np.int64)
    quad_rows = np.flatnonzero(elements[:, 2] != elements[:, 3])
    if not len(quad_rows):
        return elements, sources

    previous, following = _unit_corner_edges(nodes[elements[quad_rows], :2])
    corner_cosines = np.sum(previous * following, axis=-1)
    flat = np.any(
        corner_cosines <= np.cos(np.deg2rad(flat_quad_angle)), axis=1,
    )
    flat_rows = quad_rows[flat]
    if not len(flat_rows):
        return elements, sources

    # Each diagonal supplies two triangles with the parent's winding.
    positions = np.array([[[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]]])
    candidates = elements[flat_rows][:, positions]
    previous, following = _unit_corner_edges(nodes[candidates, :2])
    corner_sines = (
        following[..., 0] * previous[..., 1]
        - following[..., 1] * previous[..., 0]
    )
    # For triangles, the smallest corner sine ranks the smallest angle.
    scores = np.min(corner_sines, axis=(2, 3))
    scores[~np.isfinite(scores)] = -np.inf
    best = np.argmax(scores, axis=1)
    if np.any(scores[np.arange(len(flat_rows)), best] <= 0.0):
        raise ValueError("cannot split a nearly straight quad into positive-area triangles")
    triangles = candidates[np.arange(len(flat_rows)), best]
    encoded = np.concatenate((triangles, triangles[..., 2:3]), axis=2)

    counts = np.ones(len(elements), dtype=np.int64)
    counts[flat_rows] = 2
    sources = np.repeat(sources, counts)
    output = elements[sources].copy()
    starts = (np.cumsum(counts) - counts)[flat_rows]
    output[starts] = encoded[:, 0]
    output[starts + 1] = encoded[:, 1]
    return output, sources
