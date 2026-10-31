"""Plot committed adapter results without planning or generating another mesh.

Matplotlib is imported only when ``plot_quad_layer`` is called. Meshing stays
in ``mesher.mesh2d.adapter.extend_quad_layer``; these helpers consume its result.
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


def _draw_outline(axis, nodes, outline, **style):
    indices = [node for chain in outline for node in chain[:-1]]
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

    ``source_outline`` is the outline supplied to ``extend_quad_layer``. Call
    before extending the same mesh again. Supply three axes to compose several
    results in one figure, or omit them to create a figure. Returns the figure.
    Neither this function nor its diagnostic helpers call a mesher or planner.
    """
    from ..model import _validate_zero_z

    _validate_zero_z(result.mesh.nodes)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
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
    front_ids = [node for chain in result.outline for node in chain[:-1]]
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
    added = mesh.elements[source_elements:]
    row[2].add_collection(PolyCollection(mesh.nodes[added, :2],
                                         facecolors="#cde5e8", edgecolors="#2b5158",
                                         linewidths=0.55, zorder=2))
    row[0].set_title(f"{name}\n1. {len(sections)} sections; corner first", fontsize=11)
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
    figure._quad_layer_kinds = used_kinds
    figure._quad_layer_legend = figure.legend(
        handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
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
