"""Compare section geometry, unit assembly, and the final single-layer mesh.

From the repository root::

    venv/bin/python examples/extend_quad_layer.py
    venv/bin/python examples/extend_quad_layer.py --case pattern
    venv/bin/python examples/extend_quad_layer.py --case hole
    venv/bin/python examples/extend_quad_layer.py --case custom
    venv/bin/python examples/extend_quad_layer.py --no-show --plot output.png
    venv/bin/python examples/extend_quad_layer.py --no-show --diagnostics output.json

The default opens a Matplotlib window and saves examples/output/extend_quad_layer.png.
This file is a demo: the implementation lives in mesher.mesh2d.adapter.
Each case calls extend_quad_layer once, then plots its returned diagnostics.
Use the window toolbar to pan and zoom.
Requires NumPy and Matplotlib (the project's visualization extra).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import extend_quad_layer
from mesher.mesh2d.adapter.visualization import (
    diagnostic_report, plot_quad_layer, save_diagnostics,
)


@dataclass
class Example:
    name: str
    mesh: Mesh2D
    outline: list[list[int]]
    patterns: np.ndarray
    note: str
    new_outline: np.ndarray | None = None


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
    """Editable inputs: source geometry, outline, and finite pattern guides."""
    if case == "custom":
        mesh, _ = _cell_mesh({(x, y) for x in range(8) for y in range(6)}, 0.5)
        # A complete target polygon with asymmetric offsets and uneven anchors.
        # Start halfway along its top edge; the final edge closes implicitly.
        new_outline = np.asarray([
            [1.35, 3.6], [4.8, 3.6], [4.8, 0.65], [4.8, -0.4],
            [2.15, -0.4], [-0.2, -0.4], [-0.2, 1.2], [-0.2, 3.6],
        ])
        return Example("Rectangle / custom outline", mesh, _exterior_outline(mesh),
                       np.empty((0, 2, 2)),
                       "Pink circles mark the supplied points; target-size sampling adds the rest.",
                       new_outline=new_outline)

    if case == "hole":
        mesh, ids = _cell_mesh({(x, y) for x in range(12) for y in range(12)
                                if not (4 <= x < 8 and 4 <= y < 8)}, 0.25)
        # Counterclockwise: leftward growth fills part of the hole.
        points = ([(x, 4) for x in range(4, 8)]
                  + [(8, y) for y in range(4, 8)]
                  + [(x, 8) for x in range(8, 4, -1)]
                  + [(4, y) for y in range(8, 4, -1)])
        loop = [ids[p] for p in points]
        return Example("Hole / counterclockwise source", mesh, _logical_edges(mesh, loop),
                       np.empty((0, 2, 2)),
                       "The same single-layer operation grows into the hole.")

    # A refined version of the cross in the design sketch: 80 existing quads.
    mesh, _ = _cell_mesh({(x, y) for x in range(12) for y in range(12)
                          if 4 <= x < 8 or 4 <= y < 8}, 0.25)
    outline = _exterior_outline(mesh)
    if case == "basic":
        return Example("Cross / no pattern", mesh, outline, np.empty((0, 2, 2)),
                       "Convex corners, concave corners, and straight strips in one layer.")

    patterns = np.asarray([
        [[1.2, 3.4], [1.8, 3.4]],       # Top expansion: 0.4 within 0.3 * 1.5.
        [[3.2, 1.2], [3.2, 1.8]],       # Right expansion: 0.2.
        [[1.5, 3.0], [1.5, 3.4]],       # Full chord from an existing node to the front.
        [[3.0, 1.5], [3.2, 1.5]],       # Another complete section boundary.
    ])
    return Example("Cross / finite patterns", mesh, outline, patterns,
                   "Parallel guides set expansion; complete chords split LINE sections.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("all", "basic", "pattern", "hole", "custom"), default="all")
    parser.add_argument("--element-size", type=float, default=0.3)
    parser.add_argument("--expand-ratio", type=float, default=1.5)
    parser.add_argument("--plot", type=Path,
                        default=Path(__file__).resolve().parent / "output" / "extend_quad_layer.png")
    parser.add_argument("--no-show", action="store_true", help="Save PNG without opening a window.")
    parser.add_argument("--diagnostics", type=Path, help="Also save this run's diagnostics as JSON.")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ("basic", "pattern", "hole", "custom") if args.case == "all" else (args.case,)
    figure, axes = plt.subplots(len(cases), 3, squeeze=False,
                                figsize=(16, 5.1 * len(cases)), layout="constrained")
    figure.suptitle("Single-layer Quad4 extension: sections → units → mesh\n"
                   f"element_size = {args.element_size:g}    expand_ratio = {args.expand_ratio:g}",
                   fontsize=16, fontweight="bold")
    reports = []
    for row, case in zip(axes, cases):
        example = make_example(case)
        result = extend_quad_layer(
            example.mesh, example.outline, args.element_size,
            pattern_lines=example.patterns, expand_ratio=args.expand_ratio,
            new_outline=example.new_outline,
        )
        plot_quad_layer(result, example.outline, pattern_lines=example.patterns,
                        name=example.name, axes=row)
        if example.new_outline is not None:
            row[2].scatter(*example.new_outline.T, s=60, facecolors="none",
                           edgecolors="#c3346b", linewidths=1.7, zorder=8,
                           label="Supplied outline points")
            row[2].legend(loc="center", fontsize=8, framealpha=0.9)
        reports.append(diagnostic_report(result, name=example.name,
                                         element_size=args.element_size,
                                         expand_ratio=args.expand_ratio))
        diagnostics = result.diagnostics
        quality = diagnostics["quality"]
        print(f"{example.name}: {diagnostics['sections']}; "
              f"{diagnostics['unit_count']} units; "
              f"{diagnostics['added_elements']} new Quad4 cells, "
              f"{diagnostics['added_nodes']} new nodes")
        print(f"  {example.note}")
        print(f"  elapsed_seconds={diagnostics['elapsed_seconds']:.6f}; "
              f"max_aspect_ratio={quality['max_aspect_ratio']:.6g}; "
              f"min_scaled_jacobian={quality['min_scaled_jacobian']:.6g}; "
              f"worst_score={quality['worst_score']:.6g}")
        for section in diagnostics["section_details"]:
            print(f"  S{section['section_id']}: {section['kind']}, "
                  f"left seam {section['left_seam']}, right seam {section['right_seam']}")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=170)
    print(f"Saved {args.plot.resolve()}")
    if args.diagnostics is not None:
        save_diagnostics(reports, args.diagnostics)
        print(f"Saved {args.diagnostics.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
