"""Render and verify transitions on all four sides of a 50-by-2000 box.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/imprint_thin_transition.py --output /tmp/imprint-thin-transition

Optional: --baseline /path/to/before.npz compares an actual mesh from the old
implementation using the same model. No old behavior is simulated here.
Matplotlib is required only for rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines


BOX = [0, 0, 50, 2000]
OUTER = [[-5000, -5000], [5000, -5000], [5000, 7000], [-5000, 7000]]
EXPAND = 1000
SIZE = 1000
MINIMUM = 100
SPACING = 10
# Complete coarse cells claimed by each end, checked against the actual mesh.
CAPS = {
    "left": (0, 0, [-2000, 0, -1000, 2000]),
    "right": (0, 1, [2000, 0, 3000, 2000]),
    "bottom": (1, 0, [0, -2000, 1000, -1000]),
    "top": (1, 1, [0, 3000, 1000, 4000]),
}


def model():
    x1, y1, x2, y2 = BOX
    return get_pair_line([
        dict(box=BOX,
             x_points=[[x1, y] for y in np.r_[np.arange(y1, y2, SPACING), y2]],
             y_points=[[x, y1] for x in np.r_[np.arange(x1, x2, SPACING), x2]]),
    ], expand_size=EXPAND)


def polygons(mesh):
    return [mesh.nodes[e[:3] if e[2] == e[3] else e, :2] for e in mesh.elements]


def verify(mesh, pairs):
    """Check actual straight-edge coverage, slanted cells and coarse end caps."""
    e = mesh.elements
    edges = mesh.nodes[np.stack((e, np.roll(e, -1, axis=1)), axis=-1).reshape(-1, 2), :2]
    cells = polygons(mesh)
    report = {}
    for name, (axis, side, bounds) in CAPS.items():
        transverse = 1 - axis
        outer = bounds[axis + (0 if side == 0 else 2)]
        source = BOX[axis + 2 * side]
        inner = bounds[axis + (2 if side == 0 else 0)]
        assert abs(inner - source) >= EXPAND
        slanted = 0
        for p in cells:
            if np.all(p >= bounds[:2]) and np.all(p <= bounds[2:]):
                slanted += bool(np.any(np.all(np.roll(p, -1, axis=0) - p != 0, axis=1)))
        assert slanted > 0, f"{name}: no actual transition cells"
        for point in pairs[axis]["points"]:
            coordinate = point[transverse]
            on = np.all(edges[:, :, transverse] == coordinate, axis=1)
            spans = np.sort(edges[on, :, axis], axis=1)
            start, end = sorted((source, inner))
            cursor = start
            for a, b in sorted(map(tuple, spans)):
                if b <= cursor or a >= end:
                    continue
                assert a <= cursor, f"{name}: missing straight extension"
                cursor = max(cursor, b)
            assert cursor >= end, f"{name}: incomplete straight extension"
        levels = np.unique(mesh.nodes[(mesh.nodes[:, axis] == outer)
                           & (mesh.nodes[:, transverse] >= bounds[transverse])
                           & (mesh.nodes[:, transverse] <= bounds[transverse + 2]), transverse])
        expected = np.linspace(bounds[transverse], bounds[transverse + 2],
                               int(np.ceil((bounds[transverse + 2] - bounds[transverse]) / SIZE)) + 1)
        np.testing.assert_array_equal(levels, expected)
        report[name] = dict(straight_extension=abs(inner - source), transition_start=inner,
                            transition_end=outer, width=bounds[transverse + 2] - bounds[transverse],
                            depth=abs(outer - inner), slanted_cells=slanted,
                            outer_edge_nodes=levels.tolist())
    return report


def draw(ax, mesh, window, *, detail=False):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    cells = polygons(mesh)
    colors = []
    for p in cells:
        center = p.mean(axis=0)
        delta = np.roll(p, -1, axis=0) - p
        color = "#e7f5ef" if np.any(np.all(delta != 0, axis=1)) else "#f7f9fc"
        if np.all(center >= BOX[:2]) and np.all(center <= BOX[2:]):
            color = "#c8e1ff"
        colors.append(color)
    ax.add_collection(PolyCollection(cells, facecolors=colors, edgecolors="#4b6076",
                                     linewidths=.65 if detail else .3))
    ax.add_patch(Rectangle((0, 0), 50, 2000, fill=False, edgecolor="#1765b5", lw=1.4))
    ax.set(xlim=window[:2], ylim=window[2:], xlabel="X", ylabel="Y")
    ax.set_aspect("auto" if detail else "equal")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def render(mesh, baseline, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                         "axes.titlesize": 12, "axes.titleweight": "bold"})
    versions = [("Before", baseline), ("After", mesh)] if baseline is not None else [("After", mesh)]
    fig, axes = plt.subplots(1, len(versions), figsize=(7 * len(versions), 8), squeeze=False)
    for ax, (name, current) in zip(axes[0], versions):
        draw(ax, current, (-5000, 5000, -5000, 7000))
        ax.set_title(f"{name} | {len(current.elements):,} actual mesh cells", loc="left")
        for label, y in (("Top", 3500), ("Bottom", -1500)):
            ax.annotate(label, xy=(25, y), xytext=(1700, y + 600),
                        arrowprops=dict(arrowstyle="->", color="#1765b5"), color="#1765b5")
    fig.suptitle("Thin-line transition | 50 x 2000 control box", fontsize=18)
    fig.text(.5, .925, "Same inputs: expand = 1000 | element_size = 1000 | min_transition_size = 100 | spacing = 10",
             ha="center", fontsize=10)
    fig.legend(handles=[Patch(facecolor="#c8e1ff", label="Input box"),
                        Patch(facecolor="#e7f5ef", label="Actual slanted transition cells")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .045, 1, .91))
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(versions), figsize=(7 * len(versions), 12), squeeze=False)
    for row, (side, window) in enumerate((("top", (-50, 1050, 1900, 4500)),
                                         ("bottom", (-50, 1050, -2500, 100)))):
        for col, (name, current) in enumerate(versions):
            ax = axes[row, col]
            draw(ax, current, window, detail=True)
            _, end, bounds = CAPS[side]
            inner, outer = (bounds[1], bounds[3]) if end else (bounds[3], bounds[1])
            source = BOX[3] if end else BOX[1]
            for y in (source, inner, outer):
                ax.axhline(y, color="#a3b1bf", ls="--", lw=.7, zorder=0)
            ax.annotate("", xy=(62, source), xytext=(62, inner),
                        arrowprops=dict(arrowstyle="<->", color="#b36b00", lw=1.5))
            ax.text(65, (source + inner) / 2, "expand\n1000", va="center", color="#925700")
            if name == "After":
                ax.annotate("", xy=(62, inner), xytext=(62, outer),
                            arrowprops=dict(arrowstyle="<->", color="#13835d", lw=1.5))
                ax.text(65, (inner + outer) / 2, "transition\ndepth 1000", va="center", color="#13835d")
            ax.set_xticks([0, 250, 500, 750, 1000])
            ax.set_title(f"{name} | {side.capitalize()} edge, line length = 50", loc="left")
    fig.suptitle("Actual mesh close-ups | narrow sides now transition", fontsize=18)
    fig.text(.5, .948, "X and Y use different display scales to reveal the narrow mesh. Coordinates are unchanged.",
             ha="center", fontsize=10, color="#4b6076")
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=3, w_pad=3)
    fig.savefig(output / "closeups.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Actual old mesh NPZ for the identical model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = model()
    mesh = imprint_lines(SIZE, pairs, outer_outline=OUTER,
                         tag="mesh_snap", min_transition_size=MINIMUM)
    report = dict(box=BOX, expand=EXPAND, element_size=SIZE, min_transition_size=MINIMUM,
                  spacing=SPACING, elements=mesh.element_count, verified_sides=verify(mesh, pairs))
    baseline = None
    if args.baseline:
        with np.load(args.baseline) as data:
            baseline = SimpleNamespace(nodes=data["nodes"], elements=data["elements"])
    np.savez_compressed(args.output / "after.npz", nodes=mesh.nodes, elements=mesh.elements)
    render(mesh, baseline, args.output)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
