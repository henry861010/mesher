"""Recreate the hand-drawn perimeter-box layout and render actual meshes.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/imprint_boxes_outline.py --output /tmp/box-outline

Dimensions are illustrative model units, not measurements of the sketch.
Matplotlib is needed only for these previews.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines


EXPAND_SIZE = 2.0
DISTANCE = 12.0
SPACING = 2.0
ELEMENT_SIZE = 8.0


def model():
    outer = [[0, 0], [100, 0], [100, 130], [0, 130]]
    boxes = [
        [4, 112, 12, 122], [36, 112, 44, 122],  # top row
        [70, 112, 78, 122], [87, 112, 95, 122],
        [19, 92, 27, 102], [78, 89, 86, 99],   # interior controls
        [8, 64, 16, 74], [85, 62, 93, 72],     # middle left/right
        [9, 37, 17, 47], [76, 35, 84, 45],     # lower left / interior right
        [10, 11, 18, 21], [86, 10, 94, 20],    # bottom corners
        [39, 3, 47, 11], [58, 2, 66, 10],     # bottom row
    ]
    return boxes, outer


def selected_extensions(pairs, outer):
    """Describe sides meeting the threshold, including unchanged numeric values."""
    bounds = np.r_[np.min(outer, axis=0), np.max(outer, axis=0)]
    selected = []
    for index, pair in enumerate(pairs):
        axis = 0 if pair["direction"] == "x" else 1
        for side in (0, 1):
            line = np.asarray(pair[f"line{side + 1}"], dtype=float)
            edge = bounds[axis + 2 * side]
            gap = abs(float(line[0, axis] - edge))
            if gap >= DISTANCE:
                continue
            expand = pair[f"line{side + 1}_expand_size"]
            if expand != gap:
                raise AssertionError("Selected expansion must equal the outline gap")
            target = line.copy()
            target[:, axis] = edge
            region = np.vstack((line, target))
            selected.append(dict(
                box=index // 2 + 1, pair=index, axis=axis, side=side,
                name=(("left", "right") if axis == 0 else ("bottom", "top"))[side],
                gap=gap, expand_size=expand, line=line.tolist(), target=target.tolist(),
                bounds=np.r_[region.min(axis=0), region.max(axis=0)].tolist(),
            ))
    return selected


def verify_extension_edges(mesh, pairs, extensions):
    """Check that every sampled extension is covered by actual mesh edges."""
    elements = mesh.elements
    edges = np.stack((elements, np.roll(elements, -1, axis=1)), axis=-1).reshape(-1, 2)
    xy = mesh.nodes[edges, :2]
    checked = 0
    for extension in extensions:
        axis = extension["axis"]
        start, end = sorted((extension["line"][0][axis], extension["target"][0][axis]))
        for point in pairs[extension["pair"]]["points"]:
            on = np.all(np.isclose(xy[:, :, 1 - axis], point[1 - axis], atol=1e-9, rtol=0), axis=1)
            spans = np.sort(xy[on, :, axis], axis=1)
            cursor = start
            for a, b in sorted(map(tuple, spans)):
                if b <= cursor or a >= end:
                    continue
                if a > cursor + 1e-9:
                    raise AssertionError(f"Missing extension edge: {extension}, point={point}")
                cursor = max(cursor, b)
            if cursor < end - 1e-9:
                raise AssertionError(f"Extension stops before outline: {extension}, point={point}")
            checked += 1
    return checked


def draw(ax, boxes, outer, extensions, *, mesh=None, arrows=False, labels=True):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    if mesh is not None:
        polygons = [mesh.nodes[e[:3] if e[2] == e[3] else e, :2] for e in mesh.elements]
        centers = np.array([p.mean(axis=0) for p in polygons])
        colors = np.full(len(polygons), "#f6f8fb", dtype=object)
        regions = [(e["bounds"], "#ffdfaa") for e in extensions]
        regions += [(box, "#bddcf2") for box in boxes]
        for region, color in regions:
            mask = np.all(centers >= region[:2], axis=1) & np.all(centers <= region[2:], axis=1)
            colors[mask] = color
        ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="#506071",
                                         linewidths=.38, zorder=2))
    else:
        for e in extensions:
            x1, y1, x2, y2 = e["bounds"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="#ffdfaa", zorder=1))

    for index, (x1, y1, x2, y2) in enumerate(boxes, 1):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               facecolor="#bddcf2" if mesh is None else "none",
                               edgecolor="#216390", linewidth=1.2, zorder=3))
        if labels:
            ax.text((x1 + x2) / 2, y1 + (y2 - y1) / 2 if mesh is None else y2 + 1,
                    f"{index:02d}", ha="center", va="center" if mesh is None else "bottom",
                    fontsize=8, color="#164361", zorder=5)
    for extension in extensions:
        line = np.asarray(extension["line"])
        ax.plot(line[:, 0], line[:, 1], color="#be650b", lw=2, zorder=4)
        if arrows:
            source = line.mean(axis=0)
            target = np.mean(extension["target"], axis=0)
            ax.annotate("", xy=target, xytext=source,
                        arrowprops=dict(arrowstyle="->", color="#be650b", lw=1.4), zorder=4)
    closed = np.vstack((outer, outer[0]))
    ax.plot(closed[:, 0], closed[:, 1], color="#172d45", lw=1.8, zorder=4)
    ax.set(xlim=(-3, 103), ylim=(-3, 133), aspect="equal", xlabel="X (model units)", ylabel="Y")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    boxes, outer = model()

    # Public helper -> public mesher. No hand-built pairs or private planner calls.
    pairs = get_pair_line(boxes, EXPAND_SIZE, outer_outline=outer,
                          distance=DISTANCE, spacing=SPACING)
    mesh = imprint_lines(ELEMENT_SIZE, pairs, outer_outline=outer,
                         min_transition_size=1, pattern_snap_distance=0)
    baseline_pairs = get_pair_line(boxes, EXPAND_SIZE, spacing=SPACING)
    baseline = imprint_lines(ELEMENT_SIZE, baseline_pairs, outer_outline=outer,
                             min_transition_size=1, pattern_snap_distance=0)
    extensions = selected_extensions(pairs, outer)
    verified = verify_extension_edges(mesh, pairs, extensions)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    legend = [Patch(facecolor="#bddcf2", edgecolor="#216390", label="Input box"),
              Patch(facecolor="#ffdfaa", label="Requested extension to outline"),
              plt.Line2D([], [], color="#506071", lw=.7, label="Actual mesh edges")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))
    draw(axes[0], boxes, outer, extensions, arrows=True)
    axes[0].set_title("Sketch layout | 14 boxes / 15 selected sides", loc="left", pad=14)
    draw(axes[1], boxes, outer, extensions, mesh=mesh)
    axes[1].set_title(f"Actual imprint mesh | {mesh.element_count:,} cells", loc="left", pad=14)
    fig.suptitle("Box controls extending to the outer outline", fontsize=18, x=.065, ha="left")
    fig.text(.065, .94, "expand_size = 2   |   distance = 12 (gap < 12)   |   spacing = 2   |   element_size = 8",
             fontsize=10, color="#506071")
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(.01, .10, .99, .91))
    fig.savefig(args.output / "overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, (name, window) in enumerate((
            ("Top-left corner: left = 4, top = 8", (-1, 23, 103, 132)),
            ("Top side: top = 8", (28, 52, 103, 132)))):
        for col, (current, selected, title) in enumerate((
                (baseline, [], "Default expansion = 2"),
                (mesh, extensions, "Outline-aware expansion"))):
            ax = axes[row, col]
            draw(ax, boxes, outer, selected, mesh=current, labels=False)
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
            ax.set_title(f"{title}\n{name}", fontsize=11, pad=10)
    fig.suptitle("Same boxes and mesher settings | only the helper options change", fontsize=15)
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)

    for name, current in (("mesh", mesh), ("baseline", baseline)):
        np.savez_compressed(args.output / f"{name}.npz", nodes=current.nodes, elements=current.elements)
    report = dict(
        boxes=len(boxes), pairs=len(pairs), selected_sides=len(extensions),
        verified_extension_lines=verified,
        parameters=dict(expand_size=EXPAND_SIZE, distance=DISTANCE, spacing=SPACING,
                        element_size=ELEMENT_SIZE, min_transition_size=1, pattern_snap_distance=0),
        mesh=dict(nodes=mesh.node_count, elements=mesh.element_count,
                  triangles=int(np.count_nonzero(mesh.element_types == 3))),
        baseline=dict(nodes=baseline.node_count, elements=baseline.element_count),
        extensions=extensions,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "controls.json").write_text(json.dumps(dict(boxes=boxes, outer_outline=outer,
                                                              pairs=pairs), indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "extensions"}, indent=2))


if __name__ == "__main__":
    main()
