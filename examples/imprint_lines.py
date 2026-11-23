"""Render the four accepted line-imprint examples from the actual algorithm.

Run: PYTHONPATH=src venv/bin/python examples/imprint_lines.py --output /tmp/imprint-lines
Matplotlib is only needed by this example, not the mesher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesher.mesh2d.imprint_line import imprint_lines


def pair(axis, bounds, coordinates=None, expand=2):
    """Fixture shorthand; emit both explicit endpoint-distance fields."""
    x0, y0, x1, y1 = bounds
    if coordinates is None:
        coordinates = np.linspace(y0 if axis == 0 else x0, y1 if axis == 0 else x1, 6)
    e1, e2 = np.broadcast_to(expand, (2,)).tolist()
    return {"direction": "xy"[axis], "line1_expand_size": e1, "line2_expand_size": e2,
            "line1": [[x0, y0], [x0, y1]] if axis == 0 else [[x0, y0], [x1, y0]],
            "line2": [[x1, y0], [x1, y1]] if axis == 0 else [[x0, y1], [x1, y1]],
            "points": [[x0, v] for v in coordinates] if axis == 0 else [[v, y0] for v in coordinates]}


def cases():
    return [
        ("focus", "Single focus: four end transitions", 2,
         [pair(0, (4, 4, 6, 6)), pair(1, (4, 4, 6, 6))], [],
         [[-2, -2], [12, -2], [12, 12], [-2, 12]], .5),
        ("merged", "Offset focuses: merged horizontal band", 3,
         [pair(a, b) for b in ((4, 5, 6, 7), (10, 4, 12, 6)) for a in (0, 1)], [],
         [[-4, -4], [20, -4], [20, 15], [-4, 15]], .5),
        ("fillet", "Fillet: spacing 5-20, axial size 1000", 1000,
         [pair(0, (3000, 0, 7000, 500), np.r_[0, np.cumsum([5, 10, 15, 20] * 10)], 2000)], [],
         [[-1000, -1000], [11000, -1000], [11000, 1500], [-1000, 1500]], 200),
        ("patterns", "Thin control and patterns: outward relocation", 2,
         [pair(0, (3, 4, 5, 6), expand=1), pair(1, (6.2, 1, 6.5, 9), [6.2, 6.3, 6.4, 6.5], .5)],
         [[[5.5, 4.1], [8.1, 4.1]], [[10.1, 0], [10.1, 10]]],
         [[-2, 0], [12.1, 0], [12.1, 10], [-2, 10]], .5),
    ]


def plot_case(ax, case, mesh):
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.patches import Rectangle

    _, title, _, pairs, patterns, outer, _ = case
    polygons = [mesh.nodes[e[:3] if e[2] == e[3] else e, :2] for e in mesh.elements]
    centers = np.array([p.mean(axis=0) for p in polygons])
    colors = np.full(len(polygons), "#f4f5f7", dtype=object)

    def inside(box):
        return np.all(centers >= box[:2], axis=1) & np.all(centers <= box[2:], axis=1)

    bounds = np.r_[np.min(outer, axis=0), np.max(outer, axis=0)]
    # Public Mesh2D has geometry/topology, but no planner region metadata.
    # Highlight slanted quads without claiming these are complete A outlines.
    for i, polygon in enumerate(polygons):
        edges = np.roll(polygon, -1, axis=0) - polygon
        if len(polygon) == 4 and np.any(np.all(edges != 0, axis=1)):
            colors[i] = "#ffe4c5"
    for p in pairs:
        points = np.array([p["line1"], p["line2"]]).reshape(-1, 2)
        box = np.r_[points.min(axis=0), points.max(axis=0)]
        colors[inside(box)] = "#cce3ff"
        ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, edgecolor="#1765b5", lw=1.5, zorder=4))
    colors[mesh.element_types == 3] = "#b38bda"
    ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="#455463", linewidths=.45))
    if patterns:
        ax.add_collection(LineCollection(patterns, colors="#c53543", linewidths=2, zorder=5))
    ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal")
    ax.set_title(title + f"\n{mesh.element_count} cells, {np.count_nonzero(mesh.element_types == 3)} triangles", fontsize=11)
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    reports = []
    for ax, case in zip(axes.flat, cases()):
        name, _, size, pairs, patterns, outer, minimum = case
        mesh = imprint_lines(size, pairs, patterns, outer_outline=outer,
                             min_transition_size=minimum)
        plot_case(ax, case, mesh)
        single, single_ax = plt.subplots(figsize=(11, 8))
        plot_case(single_ax, case, mesh)
        single.tight_layout()
        single.savefig(args.output / f"{name}.png", dpi=150)
        plt.close(single)
        np.savez_compressed(args.output / f"{name}.npz", nodes=mesh.nodes, elements=mesh.elements)
        reports.append({"case": name, "nodes": mesh.node_count, "elements": mesh.element_count,
                        "triangles": int(np.count_nonzero(mesh.element_types == 3))})
    fig.legend(handles=[Patch(facecolor=c, label=label) for c, label in
                        [("#cce3ff", "Input control"),
                         ("#ffe4c5", "Slanted quad"), ("#b38bda", "Triangle")]],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(args.output / "overview.png", dpi=150)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
