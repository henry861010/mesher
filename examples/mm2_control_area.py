"""Edit BOXES below to inspect local die-corner/gap control areas.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/mm2_control_area.py
    PYTHONPATH=src venv/bin/python examples/mm2_control_area.py --show

The preview shows actual get_mm2_control_area output, not a generated mesh.
Matplotlib is required. --show opens its interactive pan/zoom window;
without --show the example only saves the preview and numerical results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesher.utils import get_mm2_control_area


# 修改這裡的 die 座標：[x1, y1, x2, y2]。各 die 的內部不可重疊。
# 預設為四顆 die，水平與垂直 gap 都是 80，可檢查中央交會區。
# BOXES = [
#     [0, 0, 100, 100],
#     [180, 0, 280, 100],
#     [0, 180, 100, 280],
#     [180, 180, 280, 280],
# ]
BOXES = [
    [0,0,1000,1000],
    [0,1100,900,2000],
    [1100,0,2000,2000]
]

# Corner 沿 X/Y 正負方向各延伸的距離；也可用 --corner-extent 調整。
CORNER_EXTENT = 25.0

# 其他可直接替換 BOXES 的配置：
# T 字配置（右側是一整顆 die）：
# BOXES = [[0, 0, 100, 100], [0, 180, 100, 280], [180, 0, 280, 280]]
# 對角分離（不應產生遠處延長線交集）：
# BOXES = [[0, 0, 100, 100], [300, 300, 400, 400]]
# 部分遮擋（中間較矮的 die 擋住下方 gap）：
# BOXES = [[0, 0, 100, 100], [200, 0, 300, 100], [140, 0, 160, 60]]


def plot_result(boxes, areas, corner_extent):
    """Use identical bounds and equal axis scales for the before/after views."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    die_color, control_color = "#216390", "#dc811c"
    for index, (x1, y1, x2, y2) in enumerate(areas, 1):
        axes[1].add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor=control_color, alpha=.22, edgecolor="none", zorder=2,
        ))
        axes[1].add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor="none", edgecolor=control_color, linewidth=.9,
            linestyle="--", zorder=3,
        ))
        axes[1].annotate(f"C{index}", ((x1 + x2) / 2, (y1 + y2) / 2),
                         xytext=(0, 6), textcoords="offset points",
                         color="#713d06", fontsize=8, ha="center", va="bottom", zorder=5)

    all_bounds = np.concatenate([boxes, areas])
    if len(all_bounds):
        low, high = all_bounds[:, :2].min(axis=0), all_bounds[:, 2:].max(axis=0)
        padding = max(float(np.max(high - low)) * .08, corner_extent * .25)
        limits = (low - padding, high + padding)
    else:
        limits = (np.array([-1., -1.]), np.array([1., 1.]))

    for ax in axes:
        for index, (x1, y1, x2, y2) in enumerate(boxes, 1):
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                facecolor="#e8f0f7", edgecolor="none", zorder=1,
            ))
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                facecolor="none", edgecolor=die_color, linewidth=1.5, zorder=4,
            ))
            ax.scatter([x1, x1, x2, x2], [y1, y2, y1, y2], s=12,
                       color=die_color, zorder=5)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"D{index}",
                    color=die_color, ha="center", va="center", fontsize=10, zorder=6)
        ax.set(xlim=(limits[0][0], limits[1][0]),
               ylim=(limits[0][1], limits[1][1]),
               xlabel="X (model units)", ylabel="Y (model units)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        if not len(boxes):
            ax.text(.5, .5, "No dies / no control areas", transform=ax.transAxes,
                    ha="center", va="center")

    axes[0].set_title(f"Input | {len(boxes)} dies", loc="left")
    axes[1].set_title(f"Result | {len(areas)} control rectangles", loc="left")
    fig.suptitle(f"MM2 control areas | corner extent = {corner_extent:g}", fontsize=15)
    fig.legend(handles=[
        Patch(facecolor="#e8f0f7", edgecolor=die_color, label="Die (D#)"),
        Line2D([], [], color=die_color, marker="o", linestyle="none", label="Die corner"),
        Patch(facecolor=control_color, alpha=.35, label="Control area (C#)"),
        Line2D([], [], color=control_color, linestyle="--", label="Output rectangle boundary"),
    ], loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .07, 1, .96))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corner-extent", type=float, default=CORNER_EXTENT)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build" / "mm2-control-area",
                        help="Output directory (default: build/mm2-control-area)")
    parser.add_argument("--show", action="store_true",
                        help="Also open the Matplotlib window for interactive pan/zoom")
    args = parser.parse_args()

    try:
        areas = get_mm2_control_area(BOXES, corner_extent=args.corner_extent)
    except ValueError as error:
        parser.error(str(error))
    boxes = np.asarray(BOXES, dtype=float).reshape(-1, 4)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plot_result(boxes, areas, args.corner_extent)
    args.output.mkdir(parents=True, exist_ok=True)
    preview = args.output / "control_areas.png"
    fig.savefig(preview, dpi=180, facecolor="white")
    np.save(args.output / "control_areas.npy", areas)
    (args.output / "control_areas.json").write_text(json.dumps({
        "boxes": boxes.tolist(),
        "corner_extent": args.corner_extent,
        "control_areas": areas.tolist(),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(boxes)} dies -> {len(areas)} control rectangles")
    print("Control coordinates [x1, y1, x2, y2] (IDs match the plot):")
    for index, box in enumerate(areas, 1):
        print(f"  C{index}: {box.tolist()}")
    print(f"Preview: {preview.resolve()}")
    print(f"Data: {(args.output / 'control_areas.json').resolve()}")
    print(f"NumPy: {(args.output / 'control_areas.npy').resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
