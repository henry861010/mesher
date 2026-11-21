"""Display box footprints on their exact boundary-coordinate grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _normalize_boxes(boxes: ArrayLike) -> np.ndarray:
    try:
        values = np.asarray(boxes)
    except (TypeError, ValueError) as error:
        raise ValueError("boxes must be a numeric array with shape (N, 4)") from error
    if values.shape == (0,):
        return np.empty((0, 4), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4): [x1, y1, x2, y2]")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("boxes must contain real numeric coordinates")
    values = values.astype(np.float64)
    invalid = np.flatnonzero(~np.isfinite(values).all(axis=1))
    if invalid.size:
        raise ValueError(f"boxes[{invalid[0]}] must contain only finite coordinates")
    low = np.minimum(values[:, :2], values[:, 2:])
    high = np.maximum(values[:, :2], values[:, 2:])
    degenerate = np.flatnonzero(np.any(low == high, axis=1))
    if degenerate.size:
        raise ValueError(f"boxes[{degenerate[0]}] must have positive width and height")
    return np.concatenate((low, high), axis=1)


def _boundary_grid(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the union using corner updates, without a box-by-cell array."""
    x = np.unique(boxes[:, (0, 2)])
    y = np.unique(boxes[:, (1, 3)])
    ix = np.searchsorted(x, boxes[:, (0, 2)])
    iy = np.searchsorted(y, boxes[:, (1, 3)])
    counts = np.zeros((len(y), len(x)), dtype=np.int64)
    np.add.at(counts, (iy[:, 0], ix[:, 0]), 1)
    np.add.at(counts, (iy[:, 0], ix[:, 1]), -1)
    np.add.at(counts, (iy[:, 1], ix[:, 0]), -1)
    np.add.at(counts, (iy[:, 1], ix[:, 1]), 1)
    counts.cumsum(axis=0, out=counts)
    counts.cumsum(axis=1, out=counts)
    return x, y, counts[:-1, :-1] > 0


def plot_pattern2d(boxes: ArrayLike, *, ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Plot the union of boxes and every line of their boundary-coordinate grid.

    ``boxes`` is a numeric (N, 4) list or array of [x1, y1, x2, y2].
    Reversed endpoints are normalized without modifying the input. Coordinates
    are kept exactly as float64, with no tolerance merging or extra subdivision.
    Invalid shapes, nonfinite coordinates, and zero-area boxes raise ValueError.

    Covered cells are blue, uncovered cells white; overlaps use the same blue.
    All grid lines span the boxes' bounding rectangle, including empty cells.
    Empty input produces a blank labeled axes. Matplotlib is imported only on
    calling this function; PyVista is not required. Returns (figure, axes) and
    never calls show(), so callers can compose plots or save the figure.
    """
    normalized = _normalize_boxes(boxes)

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    figure = ax.figure
    covered_color = "#6baed6"
    if len(normalized):
        x, y, covered = _boundary_grid(normalized)
        ax.pcolormesh(x, y, covered, cmap=ListedColormap(["white", covered_color]),
                      vmin=0, vmax=1, shading="flat", edgecolors="none", zorder=1)
        segments = [((xi, y[0]), (xi, y[-1])) for xi in x]
        segments.extend(((x[0], yi), (x[-1], yi)) for yi in y)
        ax.add_collection(LineCollection(segments, colors="#666666", linewidths=.65,
                                         zorder=2))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.legend(handles=[Patch(facecolor=covered_color, label="Covered"),
                           Patch(facecolor="white", edgecolor="#666666", label="Uncovered")],
                  loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return figure, ax
