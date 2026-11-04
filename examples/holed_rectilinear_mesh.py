"""Compare local grid cuts around holes with a global-projection reference.

Run from the repository root::

    PYTHONPATH=src python examples/holed_rectilinear_mesh.py --no-show
    PYTHONPATH=src python examples/holed_rectilinear_mesh.py --no-show \
        --plot /tmp/holed-grid.png --diagnostics /tmp/holed-grid.json

Only plotting requires Matplotlib. Both mesh generators and the reference
clipping use NumPy; the reference is a comparison fixture, not a second public
hole-meshing implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.generators import (
    generate_rectilinear_mesh,
    generate_rectilinear_mesh_with_holes,
)


def build_fixture():
    """Return a concave outer loop, two holes, and three short patterns."""
    outer = np.array(
        [[0, 0], [20, 0], [20, 8], [15, 8], [15, 10], [0, 10]],
        dtype=np.float64,
    )
    holes = [
        np.array([[5, 3], [8, 3], [8, 7], [5, 7]], dtype=np.float64),
        np.array([[12, 2], [14, 2], [14, 6], [12, 6]], dtype=np.float64),
    ]
    patterns = np.array(
        [[[1, y], [3, y]] for y in (4.2, 4.3, 4.4)], dtype=np.float64
    )
    return outer, holes, patterns


def _inside_polygon(points, outline):
    """Classify non-boundary points by horizontal ray crossing."""
    inside = np.zeros(len(points), dtype=bool)
    for start, end in zip(outline, np.roll(outline, -1, axis=0)):
        if start[1] == end[1]:
            continue
        crosses_y = (start[1] > points[:, 1]) != (end[1] > points[:, 1])
        crossing_x = start[0] + (points[:, 1] - start[1]) * (
            (end[0] - start[0]) / (end[1] - start[1])
        )
        inside ^= crosses_y & (points[:, 0] < crossing_x)
    return inside


def build_global_projection_reference(target_size, outer, holes, patterns):
    """Project every input coordinate globally, then remove exterior cells.

    Every orthogonal geometry edge lies on a mandatory grid line. Therefore
    centroid classification removes complete cells without cutting a cell.
    """
    coordinates = np.vstack([outer, *holes, patterns.reshape(-1, 2)])
    full_grid = generate_rectilinear_mesh(
        target_size, coordinates[:, 0], coordinates[:, 1], coordinate_tolerance=0.0
    )
    centers = full_grid.nodes[full_grid.elements, :2].mean(axis=1)
    keep = _inside_polygon(centers, outer)
    for hole in holes:
        keep &= ~_inside_polygon(centers, hole)
    retained = full_grid.elements[keep]
    used, inverse = np.unique(retained, return_inverse=True)
    return Mesh2D(
        nodes=full_grid.nodes[used],
        elements=inverse.reshape(-1, 4).astype(np.int32),
    )


def mesh_summary(mesh):
    points = mesh.nodes[mesh.elements, :2]
    lengths = np.linalg.norm(np.roll(points, -1, axis=1) - points, axis=2)
    return {
        "nodes": mesh.node_count,
        "elements": mesh.element_count,
        "minimum_edge_length": float(lengths.min()),
        "maximum_edge_length": float(lengths.max()),
        "maximum_aspect_ratio": float(
            np.max(lengths.max(axis=1) / lengths.min(axis=1))
        ),
    }


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _plot_mesh(axis, mesh):
    from matplotlib.collections import LineCollection

    edges = np.concatenate(
        [mesh.elements[:, [side, (side + 1) % 4]] for side in range(4)]
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    axis.add_collection(
        LineCollection(mesh.nodes[edges, :2], colors="#64748b", linewidths=0.65)
    )


def plot_comparison(outer, holes, patterns, result, reference, comparison):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    _plot_mesh(axes[1], result.mesh)
    _plot_mesh(axes[2], reference)
    for axis in axes:
        for index, outline in enumerate([outer, *holes]):
            closed = np.vstack([outline, outline[0]])
            axis.plot(closed[:, 0], closed[:, 1], color="#0f172a", linewidth=1.5)
            if index:
                axis.fill(outline[:, 0], outline[:, 1], color="#f1f5f9")
        for segment in patterns:
            axis.plot(
                segment[:, 0], segment[:, 1], color="#d97706", linewidth=2.3
            )
        axis.set_aspect("equal")
        axis.set_xlim(outer[:, 0].min() - 0.6, outer[:, 0].max() + 0.6)
        axis.set_ylim(outer[:, 1].min() - 0.6, outer[:, 1].max() + 0.6)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    for index, hole in enumerate(holes):
        center = hole.mean(axis=0)
        axes[0].text(*center, f"Hole {index + 1}", ha="center", va="center")
        node_ids = result.hole_outlines[index]
        nodes = result.mesh.nodes[node_ids, :2]
        axes[1].scatter(nodes[:, 0], nodes[:, 1], s=9, color="#059669", zorder=4)
        axes[1].text(
            *center,
            f"Hole {index + 1}\n{len(node_ids)} nodes",
            ha="center",
            va="center",
            fontsize=9,
        )
    for index, repair in enumerate(result.diagnostics["parity_repairs"]):
        segment = np.asarray(repair["segment"])
        axes[1].plot(
            segment[:, 0],
            segment[:, 1],
            "--",
            color="#dc2626",
            linewidth=2,
            zorder=5,
            label="Parity repair" if index == 0 else None,
        )
    if result.diagnostics["parity_repairs"]:
        axes[1].legend(loc="upper left", fontsize=8)
    axes[0].set_title("Inputs: short patterns stop at the first hole")
    for axis, key, title in (
        (axes[1], "local", "Local visibility cuts"),
        (axes[2], "global_projection_reference", "Global-projection reference"),
    ):
        summary = comparison[key]
        axis.set_title(
            f"{title}\n{summary['elements']} quads; "
            f"max aspect {summary['maximum_aspect_ratio']:.2f}"
        )
    figure.suptitle("Rectilinear Quad4 mesh around supplied holes")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", type=float, default=2.0)
    parser.add_argument(
        "--no-show", action="store_true", help="skip the interactive plot window"
    )
    parser.add_argument("--plot", type=Path, help="save the comparison plot")
    parser.add_argument("--diagnostics", type=Path, help="save diagnostics as JSON")
    args = parser.parse_args()

    outer, holes, patterns = build_fixture()
    result = generate_rectilinear_mesh_with_holes(
        args.target_size, outer, holes=holes, pattern_lines=patterns
    )
    reference = build_global_projection_reference(
        args.target_size, outer, holes, patterns
    )
    comparison = {
        "local": mesh_summary(result.mesh),
        "global_projection_reference": mesh_summary(reference),
    }
    for key, summary in comparison.items():
        print(
            f"{key}: {summary['nodes']} nodes, {summary['elements']} quads, "
            f"minimum edge={summary['minimum_edge_length']:.6g}, "
            f"maximum edge={summary['maximum_edge_length']:.6g}, "
            f"maximum aspect ratio={summary['maximum_aspect_ratio']:.6g}"
        )
    before = result.diagnostics["boundary_node_counts_before"]
    after = result.diagnostics["boundary_node_counts_after"]
    for index, node_ids in enumerate(result.hole_outlines):
        print(
            f"hole {index + 1}: {before[index + 1]} -> {len(node_ids)} "
            f"boundary nodes (even={len(node_ids) % 2 == 0})"
        )
    print(f"outer boundary nodes: {before[0]} -> {after[0]}")
    print(f"parity repair cuts: {len(result.diagnostics['parity_repairs'])}")

    if args.diagnostics is not None:
        report = {
            "target_edge_size": args.target_size,
            "outer_outline": outer,
            "holes": holes,
            "pattern_lines": patterns,
            "comparison": comparison,
            "diagnostics": result.diagnostics,
        }
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(
            json.dumps(report, indent=2, default=_json_value) + "\n",
            encoding="utf-8",
        )
        print(f"diagnostics: {args.diagnostics}")

    if args.plot is not None or not args.no_show:
        import matplotlib

        if args.no_show:
            matplotlib.use("Agg")
        figure = plot_comparison(outer, holes, patterns, result, reference, comparison)
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.plot, dpi=180)
            print(f"plot: {args.plot}")
        import matplotlib.pyplot as plt

        if not args.no_show:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
