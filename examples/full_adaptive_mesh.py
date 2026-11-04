"""Build multiple adaptive layers around separate focus regions and a ring.

Run from the repository root::

    PYTHONPATH=src python examples/full_adaptive_mesh.py
    PYTHONPATH=src python examples/full_adaptive_mesh.py --no-show
    PYTHONPATH=src python examples/full_adaptive_mesh.py --no-show --plot /tmp/full-adaptive.png
    PYTHONPATH=src python examples/full_adaptive_mesh.py --element-sizes 0.25 0.5

The ring's inner boundary grows inward; its remaining core is filled by the
same global generator used outside the focus regions. Only plotting requires
Matplotlib. ``--no-show`` without ``--plot`` runs with NumPy alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import generate_full_adaptive_mesh


def build_fixture():
    """Two disconnected focus regions, one with a hole, and a global rectangle."""
    ring = {(x, y) for x in range(12) for y in range(12)
            if not (2 <= x < 10 and 2 <= y < 10)}
    separate = {(x, y) for x in range(16, 20) for y in range(4, 8)}
    cells = ring | separate
    coordinates = sorted({(x + dx, y + dy) for x, y in cells
                          for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))})
    lookup = {point: index for index, point in enumerate(coordinates)}
    elements = [[lookup[x, y], lookup[x + 1, y], lookup[x + 1, y + 1], lookup[x, y + 1]]
                for x, y in sorted(cells)]
    mesh = Mesh2D(np.asarray(coordinates, dtype=float) * 0.5,
                  np.asarray(elements, dtype=np.int32))
    outer = np.array([[-2, -2], [12, -2], [12, 8], [-2, 8]], dtype=float)
    return mesh, outer


def _mesh_edges(axis, mesh, color="#64748b", linewidth=0.5):
    from matplotlib.collections import LineCollection

    edges = np.concatenate([mesh.elements[:, [side, (side + 1) % 4]] for side in range(4)])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    axis.add_collection(LineCollection(mesh.nodes[edges, :2], colors=color, linewidths=linewidth))


def plot_result(source, outer, result, sizes):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    _mesh_edges(axes[0], source, linewidth=0.8)
    _mesh_edges(axes[1], result.mesh)
    for axis in axes:
        axis.add_collection(PolyCollection(source.nodes[source.elements, :2],
                                           facecolors="#dbeafe", edgecolors="none", zorder=-1))
        closed = np.vstack((outer, outer[0]))
        axis.plot(*closed.T, color="#0f172a", linewidth=1.5)
        axis.set_aspect("equal")
        axis.set_xlim(outer[:, 0].min() - 0.4, outer[:, 0].max() + 0.4)
        axis.set_ylim(outer[:, 1].min() - 0.4, outer[:, 1].max() + 0.4)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    colors = plt.get_cmap("tab10")
    for index, layer in enumerate(result.layer_outlines):
        for boundary_index, outline in enumerate(layer):
            nodes = np.concatenate([np.asarray(edge)[:-1] for edge in outline])
            closed = result.mesh.nodes[np.r_[nodes, nodes[0]], :2]
            axes[1].plot(*closed.T, color=colors(index % 10), linewidth=1.6,
                         label=f"Layer {index + 1}: size {sizes[index]:g}" if boundary_index == 0 else None)
    axes[0].text(3, 3, "Empty core", ha="center", color="#334155")
    axes[1].text(3, 3, "Filled core", ha="center", color="#0f172a",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    axes[0].set_title(f"Focus mesh: {source.element_count} quads")
    axes[1].set_title(f"Full adaptive mesh: {result.mesh.element_count} quads")
    axes[1].legend(loc="upper right")
    figure.suptitle("Adaptive transitions around every focus boundary", fontsize=15)
    return figure


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--element-sizes", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--expand-ratio", type=float, default=1.5)
    parser.add_argument("--no-show", action="store_true", help="Skip the interactive plot window.")
    parser.add_argument("--plot", type=Path, help="Save the before/after plot.")
    parser.add_argument("--diagnostics", type=Path, help="Save construction diagnostics as JSON.")
    args = parser.parse_args()

    mesh, outer = build_fixture()
    source = Mesh2D(mesh.nodes, mesh.elements)
    result = generate_full_adaptive_mesh(mesh, args.element_sizes, outer_outline=outer,
                                         expand_ratio=args.expand_ratio)
    print(f"Focus mesh: {source.node_count} nodes, {source.element_count} quads")
    print(f"Full mesh: {mesh.node_count} nodes, {mesh.element_count} quads")
    print(f"{len(result.layer_outlines)} adaptive layers on each of "
          f"{len(result.diagnostics['boundaries'])} original boundaries")
    print(f"{len(result.diagnostics['global_regions'])} global regions, including the filled ring core")
    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(result.diagnostics, indent=2, default=_json_value) + "\n",
                                    encoding="utf-8")
        print(f"Saved {args.diagnostics.resolve()}")
    if args.plot is not None or not args.no_show:
        if args.no_show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = plot_result(source, outer, result, args.element_sizes)
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.plot, dpi=170)
            print(f"Saved {args.plot.resolve()}")
        if not args.no_show:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
