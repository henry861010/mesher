"""Batched cleanup of elements collapsed by node equivalence."""

import numpy as np


_BATCH_SIZE = 65_536


def _nonzero_area(points):
    """Test XY area without imposing winding or an absolute area cutoff."""
    with np.errstate(over="ignore", invalid="ignore"):
        relative = points - points[:, :1]
    overflow = ~np.all(np.isfinite(relative), axis=(1, 2))
    if np.any(overflow):
        scaled = points[overflow]
        scaled = scaled / np.max(np.abs(scaled), axis=(1, 2))[:, None, None]
        relative[overflow] = scaled - scaled[:, :1]
    scale = np.max(np.abs(relative), axis=(1, 2))
    scale[scale == 0] = 1.0
    relative /= scale[:, None, None]
    following = np.roll(relative, -1, axis=1)
    area = np.sum(
        relative[..., 0] * following[..., 1]
        - relative[..., 1] * following[..., 0], axis=1
    )
    return area != 0.0


def _repair_collapsed_elements(nodes, elements):
    """Remove adjacent duplicates in batches and canonicalize retained Tri3s."""
    output = np.empty_like(elements)
    written = 0
    for begin in range(0, len(elements), _BATCH_SIZE):
        block = elements[begin : begin + _BATCH_SIZE]
        keep = np.ones(block.shape, dtype=bool)
        keep[:, 1:] = block[:, 1:] != block[:, :-1]
        positions = np.cumsum(keep, axis=1) - 1
        compact = np.full_like(block, -1)
        rows, columns = np.nonzero(keep)
        compact[rows, positions[rows, columns]] = block[rows, columns]
        counts = np.sum(keep, axis=1)
        row_ids = np.arange(len(block))
        closing = (counts > 1) & (compact[row_ids, counts - 1] == compact[:, 0])
        counts -= closing
        triangle = counts == 3
        compact[triangle, 3] = compact[triangle, 2]
        distinct_three = (
            (compact[:, 0] != compact[:, 1])
            & (compact[:, 0] != compact[:, 2])
            & (compact[:, 1] != compact[:, 2])
        )
        distinct_four = np.all(compact[:, :3] != compact[:, 3, None], axis=1)
        valid = distinct_three & (triangle | ((counts == 4) & distinct_four))
        candidates = compact[valid]
        if len(candidates):
            candidates = candidates[_nonzero_area(nodes[candidates, :2])]
            output[written : written + len(candidates)] = candidates
            written += len(candidates)
    return output[:written]


