"""Inspect circular.imprint_v2 cases with the current parameter settings.

Run from the repository root with PYTHONPATH=src. Shapely is needed for
independent coverage measurements; Matplotlib is only needed with --plot.
Stage snapshots use the same local-patch engine as the public API.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
from unittest.mock import patch

import numpy as np
import shapely
from shapely import is_valid_reason, make_valid, union_all
from shapely.geometry import Polygon, box, mapping

from mesher import Mesh2D
from mesher.mesh2d.circular.imprint_v2 import imprint_circle
from mesher.mesh2d.circular.imprint_v2.local import _run_local_imprint
from mesher.mesh2d.circular.imprint_v2.utils import (
    _candidate_element_mask,
)
from mesher.mesh2d.generators import generate_rectilinear_mesh


HERE = Path(__file__).resolve().parent
STAGES = ("original", "projection", "cleanup", "split")


def load_cases(path=HERE / "cases.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def build_mesh(parameters):
    return generate_rectilinear_mesh(
        parameters["target_edge_size"],
        parameters["grid_x_list"],
        parameters["y_list"],
        coordinate_tolerance=parameters["coordinate_tolerance"],
    )


def imprint_arguments(parameters):
    return {
        "center": parameters["grid_circler_center_coordinate"],
        "radius": parameters["circle_radius"],
        **{name: parameters[name] for name in (
            "projection_tolerance", "guide_tolerance", "merge_tolerance",
            "minimum_area", "guide_segments",
        )},
    }


def copy_mesh(mesh):
    return Mesh2D(nodes=mesh.nodes, elements=mesh.elements)


def perimeter(element):
    return element[:3] if element[2] == element[3] else element


def polygons(mesh):
    return [Polygon(mesh.nodes[perimeter(row), :2]) for row in mesh.elements]


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in polygon_parts(child)]
    return []


def measure(mesh, domain, area_epsilon):
    """Measure coverage, keeping boundary notches separate from closed holes.

    make_valid is applied only to measurement copies of invalid polygons.
    It does not change the mesh or make invalid cells acceptable. This lets
    us measure the lobes of a self-crossing cell without asking GEOS to union
    an invalid polygon. Invalid element rows are reported separately.
    """
    cells = polygons(mesh)
    footprints = [cell if cell.is_valid else make_valid(cell) for cell in cells]
    covered = union_all(footprints)
    missing = domain.difference(covered)
    regions = [p for p in polygon_parts(missing) if p.area > area_epsilon]
    holes = [p for p in regions if not p.intersects(domain.boundary)]
    notches = [p for p in regions if p.intersects(domain.boundary)]
    invalid_rows = [i for i, cell in enumerate(cells) if not cell.is_valid]
    # Exact node IDs reveal nonconforming seams but are NOT a hole detector:
    # a hanging node can produce unmatched edges without missing any area.
    edge_counts = {}
    for row in mesh.elements:
        ids = perimeter(row)
        for a, b in zip(ids, np.roll(ids, -1), strict=True):
            if a != b:
                key = tuple(sorted((int(a), int(b))))
                edge_counts[key] = edge_counts.get(key, 0) + 1
    metrics = {
        "node_count": mesh.node_count,
        "element_count": mesh.element_count,
        "invalid_element_rows": invalid_rows,
        "covered_area": covered.area,
        "missing_area": missing.area,
        "outside_domain_area": covered.difference(domain).area,
        "closed_hole_count": len(holes),
        "closed_hole_area": sum(p.area for p in holes),
        "boundary_notch_count": len(notches),
        "boundary_notch_area": sum(p.area for p in notches),
        "overlap_area": max(0.0, sum(p.area for p in footprints) - covered.area),
        "nonmanifold_edge_count": sum(count > 2 for count in edge_counts.values()),
        "holes": [{
            "area": p.area,
            "bounds": list(p.bounds),
            "interior_point": list(p.representative_point().coords[0]),
        } for p in holes],
    }
    return metrics, missing


def cleanup_trace(projected, parameters, final_missing, metadata):
    """Attribute cleanup results to zero-based pre-cleanup element rows.

    Consume actual patch cleanup provenance, without replaying global cleanup.
    Node IDs refer to the projection snapshot, before global compaction.
    """
    nodes, elements = projected.nodes, projected.elements
    minimum_area = min(1.0, parameters["minimum_area"])
    rows = metadata["patch_rows"]
    if not len(rows):
        return []
    retained_mask = _candidate_element_mask(nodes, elements[rows], minimum_area)
    retained = dict(zip(rows.tolist(), retained_mask.tolist(), strict=True))
    representatives = np.arange(len(nodes), dtype=np.int64)
    representatives[metadata["node_ids"]] = metadata["representatives"]
    outputs = {}
    for source, row in zip(metadata["cleanup_sources"], metadata["cleanup_elements"], strict=True):
        outputs.setdefault(int(source), []).append(row)
    issues = []
    for index in rows.tolist():
        row = elements[index]
        remapped = representatives[row]
        repaired = np.asarray(outputs.get(index, []), dtype=np.int64).reshape(-1, 4)
        shape = Polygon(nodes[perimeter(row), :2])
        if shape.is_valid and np.array_equal(row, remapped) and np.array_equal(
            repaired, row[None, :],
        ):
            continue
        if not retained[index]:
            reason = "rejected_by_candidate_mask"
        elif len(repaired) == 0:
            reason = "discarded_by_repair"
        elif len(repaired) == 2:
            reason = "split_by_repair"
        else:
            reason = "remapped_or_collapsed"
        issues.append({
            "element_row": index,
            "reason": reason,
            "projected_node_ids": row.tolist(),
            "representative_node_ids": remapped.tolist(),
            "projected_xy": nodes[perimeter(row), :2].tolist(),
            "validity": is_valid_reason(shape),
            "signed_area": signed_area(nodes[perimeter(row), :2]),
            "repair_output_in_projected_node_ids": repaired.tolist(),
            "footprint_intersecting_final_missing_area": (
                make_valid(shape).intersection(final_missing).area
            ),
        })
    return issues


def signed_area(xy):
    local = xy - xy[:1]
    following = np.roll(local, -1, axis=0)
    return float(0.5 * np.sum(
        local[:, 0] * following[:, 1] - local[:, 1] * following[:, 0],
    ))


def mesh_profile(mesh, parameters):
    spacings = np.concatenate([
        np.diff(np.unique(mesh.nodes[:, axis])) for axis in (0, 1)
    ])
    return {
        "radius_to_target_edge_size": parameters["circle_radius"] / parameters["target_edge_size"],
        "minimum_grid_spacing": float(spacings.min()),
        "maximum_grid_spacing": float(spacings.max()),
        "original_element_count": mesh.element_count,
    }


def execute_stages(parameters, *, with_trace=False):
    mesh = build_mesh(parameters)
    snapshots = {"original": copy_mesh(mesh)}
    metadata = {}
    stats = {}

    def capture(stage, snapshot, provenance):
        snapshots[stage] = copy_mesh(snapshot)
        metadata.update(provenance)

    _run_local_imprint(mesh, **imprint_arguments(parameters), observer=capture, stats=stats)
    if with_trace:
        return snapshots, metadata, stats
    return snapshots


def control(parameters, domain, area_epsilon, *, disable_target_merge=False):
    """Run a counterfactual through the real public wrapper."""
    mesh = build_mesh(parameters)
    context = patch(
        "mesher.mesh2d.circular.imprint_v2.utils._merge_close_circle_targets",
        side_effect=lambda xy, *args: xy.copy(),
    ) if disable_target_merge else nullcontext()
    try:
        with context:
            imprint_circle(mesh, **imprint_arguments(parameters))
    except ValueError as error:
        return {"error": str(error)}
    metrics, _ = measure(mesh, domain, area_epsilon)
    return {key: metrics[key] for key in (
        "closed_hole_area", "missing_area", "overlap_area", "invalid_element_rows",
    )}


def diagnose(case):
    parameters = case["parameters"]
    snapshots, metadata, stats = execute_stages(parameters, with_trace=True)
    original = snapshots["original"]
    domain = box(*np.r_[original.nodes[:, :2].min(axis=0),
                        original.nodes[:, :2].max(axis=0)])
    area_epsilon = max(1.0, domain.area) * 1e-10
    metrics, missing = {}, {}
    for name, mesh in snapshots.items():
        metrics[name], missing[name] = measure(mesh, domain, area_epsilon)

    public_mesh = build_mesh(parameters)
    imprint_circle(public_mesh, **imprint_arguments(parameters))
    matches_public = bool(
        np.array_equal(public_mesh.nodes, snapshots["split"].nodes)
        and np.array_equal(public_mesh.elements, snapshots["split"].elements)
    )
    projected = snapshots["projection"]
    displacement = np.linalg.norm(projected.nodes[:, :2] - original.nodes[:, :2], axis=1)
    moved = np.flatnonzero(displacement > 1e-12)
    changed_rows = np.flatnonzero(np.any(projected.elements != original.elements, axis=1))
    controls = {
        "no_projection": control(dict(parameters, projection_tolerance=0.0), domain, area_epsilon),
        "no_guides": control(dict(parameters, guide_segments=[]), domain, area_epsilon),
        "no_target_merge": control(parameters, domain, area_epsilon, disable_target_merge=True),
        "no_area_cutoff": control(dict(parameters, minimum_area=0.0), domain, area_epsilon),
    }
    final = metrics["split"]
    failed = bool(final["invalid_element_rows"]) or any(
        final[key] > area_epsilon
        for key in ("missing_area", "overlap_area", "outside_domain_area")
    )
    result = {
        "id": case["id"],
        "status": "fail" if failed else "pass",
        "parameters": parameters,
        "mesh_profile": mesh_profile(original, parameters),
        "minimum_area_requested": parameters["minimum_area"],
        "minimum_area_effective": min(1.0, parameters["minimum_area"]),
        "inspection_bounds": case.get("inspection_bounds"),
        "area_epsilon": area_epsilon,
        "first_invalid_stage": next((name for name in STAGES if metrics[name]["invalid_element_rows"]), None),
        "first_hole_stage": next((name for name in STAGES if metrics[name]["closed_hole_area"] > area_epsilon), None),
        "public_call_matches_stages": matches_public,
        "stages": metrics,
        "projection_reoriented_element_rows": changed_rows.tolist(),
        "moved_nodes": [{
            "node_id": int(i), "original_xy": original.nodes[i, :2].tolist(),
            "projected_xy": projected.nodes[i, :2].tolist(),
            "distance": float(displacement[i]),
        } for i in moved],
        "cleanup_trace": cleanup_trace(projected, parameters, missing["split"], metadata),
        "local_patch": {key: stats[key] for key in ("global_elements", "core_elements", "patch_elements", "patch_nodes")},
        "controls": controls,
    }
    return result, snapshots, missing


def verify(case, result):
    """Validate replay consistency and input coverage for current results.

    Both passing and failing cases are valid diagnostic results. Mesh quality
    is reported separately in status; verification does not require a hole.
    """
    errors = []
    epsilon = result["area_epsilon"]
    if not result["public_call_matches_stages"]:
        errors.append("stage replay differs from public imprint_circle")
    original = result["stages"]["original"]
    if original["invalid_element_rows"] or original["missing_area"] > epsilon:
        errors.append("input grid is invalid or already has missing area")
    if result["stages"]["projection"]["missing_area"] > epsilon:
        errors.append("projection already has uncovered area")
    if result["stages"]["projection"]["outside_domain_area"] > epsilon:
        errors.append("projection moved mesh coverage outside the input domain")
    if "mesh_profile" in case:
        for key, expected in case["mesh_profile"].items():
            if not np.isclose(result["mesh_profile"][key], expected, rtol=1e-10, atol=1e-8):
                errors.append(f"mesh profile {key} differs from the recorded corpus")
        if result["mesh_profile"]["maximum_grid_spacing"] > case["parameters"]["target_edge_size"] * (1.0 + 1e-12):
            errors.append("input grid exceeds the target element size")
    if abs(result["stages"]["split"]["missing_area"] - result["stages"]["cleanup"]["missing_area"]) > epsilon:
        errors.append("split changed missing area")
    return errors


def draw_mesh(ax, mesh, missing, parameters, *, label_rows=(), bounds=None):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon as PlotPolygon

    segments = []
    for i, row in enumerate(mesh.elements):
        xy = mesh.nodes[perimeter(row), :2]
        segments.append(np.vstack((xy, xy[:1])))
        if i in label_rows:
            ax.text(*xy.mean(axis=0), str(i), color="#9333ea", fontsize=8, clip_on=True,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.add_collection(LineCollection(segments, colors="#475569", linewidths=0.7))
    for region in polygon_parts(missing):
        ax.add_patch(PlotPolygon(np.asarray(region.exterior.coords), facecolor="#ef4444",
                                 edgecolor="#b91c1c", alpha=0.75, zorder=3))
    ax.add_patch(Circle(parameters["grid_circler_center_coordinate"], parameters["circle_radius"],
                        fill=False, color="#0ea5e9", linestyle="--", linewidth=1.0))
    for line in parameters["guide_segments"]:
        xy = np.asarray(line)
        ax.plot(xy[:, 0], xy[:, 1], color="#d97706", linewidth=1.4)
    ax.set_aspect("equal")
    if bounds is None:
        bounds = (min(parameters["grid_x_list"]), min(parameters["y_list"]),
                  max(parameters["grid_x_list"]), max(parameters["y_list"]))
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.02
        bounds = (bounds[0] - margin, bounds[1] - margin,
                  bounds[2] + margin, bounds[3] + margin)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.tick_params(labelsize=7)


def write_plots(output, replays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 5, figsize=(18, 9.2), squeeze=False,
                                layout="constrained")
    for ax, (result, snapshots, missing) in zip(axes.flat, replays):
        draw_mesh(ax, snapshots["split"], missing["split"], result["parameters"],
                  bounds=result.get("inspection_bounds"))
        final = result["stages"]["split"]
        ax.set_title(f"{result['id']} | {result['status']} | hole area {final['closed_hole_area']:.6f}\n"
                     f"missing area {final['missing_area']:.6f}", fontsize=10)
    for ax in list(axes.flat)[len(replays):]:
        ax.set_visible(False)
    profiles = sorted({(r[0]["parameters"]["circle_radius"], r[0]["parameters"]["target_edge_size"])
                       for r in replays})
    scales = ", ".join(f"R = {radius:g}, h = {size:g}" for radius, size in profiles)
    figure.suptitle("imprint_v2: red = missing mesh, blue = circle, orange = guides\n"
                   f"{scales} | local inspection views",
                   fontsize=14)
    figure.savefig(output / "overview.png", dpi=160)
    plt.close(figure)
    for result, snapshots, missing in replays:
        figure, axes = plt.subplots(1, 4, figsize=(16, 4.6))
        for ax, name in zip(axes, STAGES, strict=True):
            bad_rows = result["stages"][name]["invalid_element_rows"]
            draw_mesh(ax, snapshots[name], missing[name], result["parameters"],
                      label_rows=bad_rows, bounds=result.get("inspection_bounds"))
            metric = result["stages"][name]
            ax.set_title(f"{name}: {metric['element_count']} elements\n"
                         f"invalid {len(bad_rows)}, missing {metric['missing_area']:.6f}", fontsize=10)
        figure.suptitle(f"{result['id']}: {result['status']} | purple labels = invalid element rows")
        figure.tight_layout()
        figure.savefig(output / result["id"] / "stages.png", dpi=150)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="case_ids", help="case_01, etc.; repeat to select multiple")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verify", action="store_true", help="check input coverage and replay consistency; pass/fail status is reported separately")
    args = parser.parse_args(argv)
    cases = load_cases()
    if args.case_ids:
        unknown = set(args.case_ids) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.case_ids]
    args.output.mkdir(parents=True, exist_ok=True)
    replays, errors = [], []
    for case in cases:
        result, snapshots, missing = diagnose(case)
        case_dir = args.output / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        for name, mesh in snapshots.items():
            np.savez_compressed(case_dir / f"{name}.npz", nodes=mesh.nodes, elements=mesh.elements)
        (case_dir / "missing.geojson").write_text(json.dumps(mapping(missing["split"]), indent=2) + "\n")
        (case_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.verify:
            errors.extend(f"{case['id']}: {error}" for error in verify(case, result))
        replays.append((result, snapshots, missing))
        final = result["stages"]["split"]
        print(f"{case['id']}: {result['status']}, first hole={result['first_hole_stage']}, "
              f"hole area={final['closed_hole_area']:.9f}, missing={final['missing_area']:.9f}")
    summary = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "shapely": shapely.__version__},
        "measurement": "union of polygon footprints; invalid polygons measured with make_valid; closed holes exclude boundary notches",
        "results": [replay[0] for replay in replays],
        "verification_errors": errors,
    }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot:
        write_plots(args.output, replays)
    if errors:
        for error in errors:
            print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
