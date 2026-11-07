"""Plot committed adapter results without planning or generating another mesh.

Matplotlib is imported only when ``plot_quad_layer`` is called. Meshing stays
in ``mesher.mesh2d.adapter.generate_layer_adaptive_mesh``; these helpers consume its result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .layer._types import QuadLayerResult


_SECTION_COLORS = {
    "LINE": "#90c7dc",
    "OUTER": "#f2bc75",
    "INNER": "#b4a1df",
    "INNER_LEFT": "#d99ac2",
    "INNER_RIGHT": "#8fc6aa",
    "INNER_BOTH": "#e3aa93",
}

_REPAIR_STYLES = {
    "slot": ("#efc8dc", "#9a426e", "//", "R"),
    "corner": ("#bce0d5", "#347b69", "\\\\", "C"),
}


def _draw_outline(axis, nodes, outline, **style):
    indices = list(outline)
    axis.plot(*nodes[indices + indices[:1], :2].T, **style)


def _polygon_center(points):
    origin = points[0]
    relative = points - origin
    following = np.roll(relative, -1, axis=0)
    cross = relative[:, 0] * following[:, 1] - following[:, 0] * relative[:, 1]
    return origin + ((relative + following) * cross[:, None]).sum(axis=0) / (3 * cross.sum())


def _source_counts(result):
    return (len(result.mesh.nodes) - result.diagnostics["added_nodes"],
            len(result.mesh.elements) - result.diagnostics["added_elements"])


def plot_quad_layer(result: QuadLayerResult, source_outline, *, pattern_lines=(),
                    name: str = "Quad4 layer", axes=None):
    """Show sections, selected units, and final quads from one committed result.

    ``source_outline`` is the outline supplied to ``generate_layer_adaptive_mesh``. Call
    before extending the same mesh again. Supply three axes to compose several
    results in one figure, or omit them to create a figure. Returns the figure.
    Neither this function nor its diagnostic helpers call a mesher or planner.
    """
    from ..model import _validate_zero_z

    _validate_zero_z(result.mesh.nodes)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.layout_engine import ConstrainedLayoutEngine
    from matplotlib.patches import Patch, Polygon

    if axes is None:
        figure, row = plt.subplots(1, 3, figsize=(16, 5.1), layout="constrained")
        figure.suptitle("Single-layer Quad4 extension: sections → units → mesh",
                       fontsize=16, fontweight="bold")
    else:
        row = tuple(axes)
        if len(row) != 3 or any(axis.figure is not row[0].figure for axis in row):
            raise ValueError("axes must contain three axes from the same figure")
        figure = row[0].figure

    mesh, diagnostics = result.mesh, result.diagnostics
    source_nodes, source_elements = _source_counts(result)
    old_nodes, old_elements = mesh.nodes[:source_nodes], mesh.elements[:source_elements]
    sections, units = diagnostics["section_details"], diagnostics["units"]
    repair_groups = {kind: diagnostics.get(f"{kind}_repairs", ())
                     for kind in _REPAIR_STYLES}
    repairs = [(kind, index, repair) for kind, items in repair_groups.items()
               for index, repair in enumerate(items)]
    front_ids = result.outline
    points = np.vstack((old_nodes[:, :2], mesh.nodes[front_ids, :2]))
    lower, upper = points.min(axis=0), points.max(axis=0)
    pad = 0.07 * max(upper - lower)
    for axis in row:
        axis.add_collection(PolyCollection(old_nodes[old_elements, :2],
                                           facecolors="#f0f1f3", edgecolors="#c5c9cf",
                                           linewidths=0.4, zorder=1))
        _draw_outline(axis, old_nodes, source_outline, color="#b67812",
                      linewidth=1.6, linestyle="--", zorder=5)
        _draw_outline(axis, mesh.nodes, result.outline, color="#168562",
                      linewidth=1.5, zorder=4)
        for pattern in pattern_lines:
            axis.plot(*np.asarray(pattern).T, color="#c73535", linewidth=2.0,
                      marker="o", markersize=3.4, zorder=6)
        axis.set_xlim(lower[0] - pad, upper[0] + pad)
        axis.set_ylim(lower[1] - pad, upper[1] + pad)
        axis.set_aspect("equal")
        axis.set_xlabel("x", fontsize=9)
        axis.set_ylabel("y", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)

    kinds = {section["section_id"]: section["kind"] for section in sections}
    for section in sections:
        polygon = np.asarray(section["polygon"])
        row[0].add_patch(Polygon(polygon, facecolor=_SECTION_COLORS[section["kind"]],
                                edgecolor="#4c5260", linewidth=1.0, zorder=2))
        row[0].text(*_polygon_center(polygon), f"S{section['section_id']}",
                    ha="center", va="center", fontsize=7.5, weight="bold", zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
        row[1].add_patch(Polygon(polygon, fill=False, edgecolor="#303743",
                                linewidth=1.9, zorder=3))
    for unit in units:
        row[1].add_patch(Polygon(np.asarray(unit["polygon"]),
                                facecolor=_SECTION_COLORS[kinds[unit["section_id"]]],
                                edgecolor="#38444e", linewidth=1.0, alpha=0.85, zorder=2))
    # Section/unit ranges begin after all staged repair elements. Repairs
    # are a separate construction phase, not members of the cyclic section plan.
    repair_elements = diagnostics.get("repair_added_elements", 0)
    added = mesh.elements[source_elements + repair_elements:]
    row[2].add_collection(PolyCollection(mesh.nodes[added, :2],
                                         facecolors="#cde5e8", edgecolors="#2b5158",
                                         linewidths=0.55, zorder=2))
    for kind, repair_index, repair in repairs:
        facecolor, edgecolor, hatch, prefix = _REPAIR_STYLES[kind]
        polygon = np.asarray(repair["polygon"])
        mouth = np.asarray(repair["mouth_points"])
        for axis in row[:2]:
            axis.add_patch(Polygon(polygon, facecolor=facecolor, edgecolor=edgecolor,
                                   hatch=hatch, linewidth=1.0, alpha=0.8, zorder=2))
            axis.text(*_polygon_center(polygon), f"{prefix}{repair_index}",
                      ha="center", va="center", fontsize=8, weight="bold", zorder=7)
        cells = mesh.elements[repair["element_start"]:repair["element_end"]]
        row[2].add_collection(PolyCollection(mesh.nodes[cells, :2],
                                             facecolors=facecolor, edgecolors=edgecolor,
                                             linewidths=0.6, zorder=2))
        for axis in row:
            axis.plot(*mouth[:, :2].T, color="#a62573", linewidth=2,
                      marker="o", markersize=3, zorder=6)
    repair_label = "".join(f"; {len(items)} {kind} repair{'s' if len(items) != 1 else ''}"
                           for kind, items in repair_groups.items() if items)
    row[0].set_title(f"{name}\n1. {len(sections)} sections{repair_label}", fontsize=11)
    row[1].set_title(f"2. {len(units)} units within fixed sections\n"
                     "Bold edges: sections / thin edges: units", fontsize=11)
    quality = diagnostics["quality"]
    row[2].set_title(f"3. Final mesh / {diagnostics['elapsed_seconds']:.3f} s\n"
                     f"+{diagnostics['added_elements']} Quad4 cells\n"
                     f"max edge ratio {quality['max_aspect_ratio']:.3g}; "
                     f"min scaled J {quality['min_scaled_jacobian']:.3g}", fontsize=10)

    # Keep one adapter legend when several result rows share a figure.
    used_kinds = getattr(figure, "_quad_layer_kinds", set()) | set(kinds.values())
    previous_legend = getattr(figure, "_quad_layer_legend", None)
    if previous_legend is not None:
        previous_legend.remove()
    handles = [Patch(facecolor=color, edgecolor="#4c5260", label=kind)
               for kind, color in _SECTION_COLORS.items() if kind in used_kinds]
    handles.extend([
        Line2D([], [], color="#b67812", linestyle="--", linewidth=2, label="Fixed source"),
        Line2D([], [], color="#168562", linewidth=2, label="New outline"),
        Line2D([], [], color="#c73535", marker="o", linewidth=2, label="Finite patterns"),
    ])
    repair_kinds = ({kind for kind, items in repair_groups.items() if items}
                    | getattr(figure, "_quad_layer_repair_kinds", set()))
    for kind, (facecolor, edgecolor, hatch, _) in _REPAIR_STYLES.items():
        if kind in repair_kinds:
            handles.append(Patch(facecolor=facecolor, edgecolor=edgecolor, hatch=hatch,
                                 label=f"{kind.title()} repair"))
    if repair_kinds:
        handles.append(Line2D([], [], color="#a62573", marker="o", linewidth=2,
                              label="Repaired boundary"))
    figure._quad_layer_kinds = used_kinds
    figure._quad_layer_has_repairs = bool(repair_kinds)
    figure._quad_layer_repair_kinds = repair_kinds
    figure._quad_layer_legend = figure.legend(
        handles=handles, loc="outside lower center", ncol=min(len(handles), 8), frameon=False)
    # Fixed-aspect panels can leave their x labels inside the outside legend's
    # reserved area. Reserve one label row, preserving larger caller margins.
    layout = figure.get_layout_engine()
    if isinstance(layout, ConstrainedLayoutEngine):
        left, bottom, width, height = layout.get()["rect"]
        label_space = 16. / (72. * figure.get_figheight())
        extra = max(0., label_space - bottom)
        if extra < height:
            layout.set(rect=(left, bottom + extra, width, height - extra))
    return figure


def diagnostic_report(result: QuadLayerResult, *, name: str,
                      element_size: float, expand_ratio: float) -> dict:
    """Package a completed result and example settings for JSON export."""
    source_nodes, source_elements = _source_counts(result)
    return {"name": name, "element_size": element_size, "expand_ratio": expand_ratio,
            "source_nodes": source_nodes, "source_elements": source_elements,
            "diagnostics": result.diagnostics, "outline": result.outline}


def save_diagnostics(reports: list[dict], path: str | Path):
    """Save result reports without changing or rebuilding a mesh."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reports, indent=2, default=convert, allow_nan=False) + "\n")


__all__ = ["plot_quad_layer", "diagnostic_report", "save_diagnostics"]
