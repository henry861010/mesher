"""Visual examples of single-layer direct Quad4 extension.

From the repository root::

    venv/bin/python examples/extend_quad_layer.py
    venv/bin/python examples/extend_quad_layer.py --case pattern
    venv/bin/python examples/extend_quad_layer.py --case hole
    venv/bin/python examples/extend_quad_layer.py --no-show --plot output.png

The default opens a Matplotlib window and saves examples/output/extend_quad_layer.png.
Use the window toolbar to pan and zoom. Each case calls extend_quad_layer once.
Requires NumPy and Matplotlib (the project's visualization extra).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerResult, extend_quad_layer


@dataclass
class Example:
    name: str
    mesh: Mesh2D
    outline: list[list[int]]
    patterns: np.ndarray
    zoom: tuple[float, float, float, float]
    note: str


def _cell_mesh(cells: set[tuple[int, int]], spacing: float):
    """Build the existing Quad4 mesh, with shared vertex IDs."""
    coordinates = sorted({(x + dx, y + dy) for x, y in cells
                          for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))})
    ids = {point: i for i, point in enumerate(coordinates)}
    elements = [[ids[x, y], ids[x + 1, y], ids[x + 1, y + 1], ids[x, y + 1]]
                for x, y in sorted(cells)]
    return Mesh2D(np.asarray(coordinates, dtype=float) * spacing,
                  np.asarray(elements, dtype=np.int32)), ids


def _logical_edges(mesh: Mesh2D, loop: list[int]) -> list[list[int]]:
    """Group collinear boundary segments into logical outline edges."""
    points = mesh.nodes[loop, :2]
    vectors = np.roll(points, -1, axis=0) - points
    previous = np.roll(vectors, 1, axis=0)
    turns = previous[:, 0] * vectors[:, 1] - previous[:, 1] * vectors[:, 0]
    corners = np.flatnonzero(turns != 0).tolist()
    closed = loop + loop + [loop[0]]
    return [closed[start:(end if end > start else end + len(loop)) + 1]
            for start, end in zip(corners, corners[1:] + corners[:1])]


def _exterior_outline(mesh: Mesh2D) -> list[list[int]]:
    directed = {(int(a), int(b)) for cell in mesh.elements
                for a, b in zip(cell, np.roll(cell, -1))}
    clockwise = {b: a for a, b in directed if (b, a) not in directed}
    start = min(clockwise)
    loop = [start]
    while clockwise[loop[-1]] != start:
        loop.append(clockwise[loop[-1]])
    return _logical_edges(mesh, loop)


def make_example(case: str) -> Example:
    """Editable inputs: source geometry, outline, finite guides, and zoom area."""
    if case == "hole":
        mesh, ids = _cell_mesh({(x, y) for x in range(6) for y in range(6)
                                if not (2 <= x < 4 and 2 <= y < 4)}, 0.5)
        # Counterclockwise: leftward growth fills part of the hole.
        loop = [ids[p] for p in ((2, 2), (3, 2), (4, 2), (4, 3),
                                 (4, 4), (3, 4), (2, 4), (2, 3))]
        return Example("Hole / counterclockwise source", mesh, _logical_edges(mesh, loop),
                       np.empty((0, 2, 2)), (0.85, 2.15, 0.85, 2.15),
                       "The same single-layer operation grows into the hole.")

    # A refined version of the cross in the design sketch: 80 existing quads.
    mesh, _ = _cell_mesh({(x, y) for x in range(12) for y in range(12)
                          if 4 <= x < 8 or 4 <= y < 8}, 0.25)
    outline = _exterior_outline(mesh)
    if case == "basic":
        return Example("Cross / no pattern", mesh, outline, np.empty((0, 2, 2)),
                       (1.65, 3.40, 1.65, 2.50),
                       "Convex corners, concave corners, and straight strips in one layer.")

    patterns = np.asarray([
        [[1.2, 3.4], [1.8, 3.4]],       # Top expansion: 0.4 within 0.3 * 1.5.
        [[3.2, 1.2], [3.2, 1.8]],       # Right expansion: 0.2.
        [[3.07, 2.14], [3.16, 2.14]],   # Finite horizontal segment inside a corner.
        [[3.12, 2.06], [3.12, 2.24]],   # Finite vertical segment; shared X junction.
    ])
    return Example("Cross / finite patterns", mesh, outline, patterns,
                   (2.92, 3.26, 1.96, 2.34),
                   "Parallel guides change expansion; the finite X remains mesh edges.")


def _draw_outline(axis, nodes, outline, **style):
    indices = [node for chain in outline for node in chain[:-1]]
    points = nodes[indices + indices[:1], :2]
    axis.plot(*points.T, **style)


def _draw_mesh(axis, example, old_nodes, old_elements, result, *, before=False, detail=False):
    from matplotlib.collections import PolyCollection

    axis.add_collection(PolyCollection(old_nodes[old_elements, :2],
                                       facecolors="#e5e7eb", edgecolors="#9ca3af",
                                       linewidths=0.55, zorder=1))
    if not before:
        added = example.mesh.elements[len(old_elements):]
        axis.add_collection(PolyCollection(example.mesh.nodes[added, :2],
                                           facecolors="#dfedff", edgecolors="#386b9c",
                                           linewidths=0.75, zorder=2))
        _draw_outline(axis, example.mesh.nodes, result.outline,
                      color="#15916a", linewidth=1.9, zorder=4)
    _draw_outline(axis, old_nodes, example.outline, color="#db8b18",
                  linewidth=1.8, linestyle="--", zorder=5)
    for pattern in example.patterns:
        axis.plot(*pattern.T, color="#cf3535", linewidth=2.2,
                  linestyle="--" if before else "-", marker="o", markersize=3.8, zorder=6)
    if detail:
        # Visible vertices distinguish narrow Quad4 cells from triangles.
        axis.scatter(*example.mesh.nodes[:, :2].T, s=8, color="#244e7d", zorder=3)
        xmin, xmax, ymin, ymax = example.zoom
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
    else:
        points = example.mesh.nodes[:, :2]
        lower, upper = points.min(axis=0), points.max(axis=0)
        pad = 0.09 * max(upper - lower)
        axis.set_xlim(lower[0] - pad, upper[0] + pad)
        axis.set_ylim(lower[1] - pad, upper[1] + pad)
    axis.set_aspect("equal")
    axis.set_xlabel("x", fontsize=9)
    axis.set_ylabel("y", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)


def render(examples: list[Example], element_size: float, expand_ratio: float):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    figure, axes = plt.subplots(len(examples), 3, squeeze=False,
                                figsize=(14, 4.4 * len(examples)), layout="constrained")
    figure.suptitle(f"Single-layer direct Quad4 extension\n"
                   f"element_size = {element_size:g}    expand_ratio = {expand_ratio:g}",
                   fontsize=16, fontweight="bold")
    for row, example in zip(axes, examples):
        old_nodes, old_elements = example.mesh.nodes.copy(), example.mesh.elements.copy()
        # This is the algorithm call; all panels below use this actual result.
        result: QuadLayerResult = extend_quad_layer(
            example.mesh, example.outline, element_size,
            pattern_lines=example.patterns, expand_ratio=expand_ratio,
        )
        np.testing.assert_array_equal(example.mesh.nodes[:len(old_nodes)], old_nodes)
        np.testing.assert_array_equal(example.mesh.elements[:len(old_elements)], old_elements)
        assert np.all(example.mesh.element_types == 4)
        _draw_mesh(row[0], example, old_nodes, old_elements, result, before=True)
        _draw_mesh(row[1], example, old_nodes, old_elements, result)
        _draw_mesh(row[2], example, old_nodes, old_elements, result, detail=True)
        row[0].set_title(f"{example.name}\nInput: fixed mesh + source outline", fontsize=11)
        row[1].set_title(f"Output: +{result.diagnostics['added_elements']} Quad4 cells\n"
                         f"{result.diagnostics['path']} construction", fontsize=11)
        row[2].set_title("Detail: mesh edges and shared nodes", fontsize=11)
        xmin, xmax, ymin, ymax = example.zoom
        row[1].add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor="#6b46a1", linestyle=":",
                                   linewidth=1.8, zorder=7))
        print(f"{example.name}: {result.diagnostics['added_elements']} new Quad4 cells, "
              f"{result.diagnostics['added_nodes']} new nodes; {result.diagnostics['path']}")
        print(f"  {example.note}")
    handles = [Patch(facecolor="#e5e7eb", edgecolor="#9ca3af", label="Existing mesh"),
               Patch(facecolor="#dfedff", edgecolor="#386b9c", label="New Quad4 cells"),
               Line2D([], [], color="#db8b18", linestyle="--", linewidth=2, label="Fixed source"),
               Line2D([], [], color="#15916a", linewidth=2, label="New outline"),
               Line2D([], [], color="#cf3535", marker="o", linewidth=2, label="Finite patterns")]
    figure.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("all", "basic", "pattern", "hole"), default="all")
    parser.add_argument("--element-size", type=float, default=0.3)
    parser.add_argument("--expand-ratio", type=float, default=1.5)
    parser.add_argument("--plot", type=Path,
                        default=Path(__file__).resolve().parent / "output" / "extend_quad_layer.png")
    parser.add_argument("--no-show", action="store_true", help="Save PNG without opening a window.")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ("basic", "pattern", "hole") if args.case == "all" else (args.case,)
    figure = render([make_example(case) for case in cases], args.element_size, args.expand_ratio)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=170)
    print(f"Saved {args.plot.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
