"""Interactively compare a fail case before and after circular.imprint_v2.

Run directly from a checkout, for example::

    venv/bin/python examples/failures/imprint_v2_holes/view.py --case 10

Only the visualization extra is needed; this viewer does not load saved
diagnostics or import the Shapely-based diagnose.py script.
Small N/E labels show zero-based node/element indices in each mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "cases.json"
SOURCE_ROOT = HERE.parents[2] / "src"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="case_01", metavar="CASE",
        help="case ID or number, e.g. case_01 or 10 (default: case_01)",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_cases",
        help="list available cases without opening a window",
    )
    return parser


def load_cases(path=CASES_PATH):
    """Read the corpus using only the standard library, including for --list."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read cases file {path}: {error}") from error
    cases = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"cases file {path} must contain a nonempty 'cases' list")
    ids = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or not isinstance(case.get("parameters"), dict)
        ):
            raise ValueError(f"each case in {path} needs an ID and parameters object")
        if case["id"] in ids:
            raise ValueError(f"duplicate case ID in {path}: {case['id']}")
        ids.add(case["id"])
    return cases


def select_case(cases, selector):
    """Accept a canonical case ID or its numeric shorthand."""
    identifier = f"case_{int(selector):02d}" if selector.isdecimal() else selector
    for case in cases:
        if case["id"] == identifier:
            return case
    available = ", ".join(case["id"] for case in cases)
    raise ValueError(f"unknown case {selector!r}; available cases: {available}")


def _enable_checkout_imports():
    source_root = str(SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


def prepare_meshes(case):
    """Generate a fresh grid and imprint an independently owned copy."""
    _enable_checkout_imports()
    from mesher import Mesh2D
    from mesher.mesh2d.circular import imprint_v2
    from mesher.mesh2d.generators import generate_rectilinear_mesh

    parameters = case["parameters"]
    try:
        before = generate_rectilinear_mesh(
            parameters["target_edge_size"],
            parameters["grid_x_list"],
            parameters["y_list"],
            coordinate_tolerance=parameters["coordinate_tolerance"],
        )
        after = Mesh2D(nodes=before.nodes, elements=before.elements)
        arguments = {
            "center": parameters["grid_circler_center_coordinate"],
            "radius": parameters["circle_radius"],
            **{name: parameters[name] for name in (
                "projection_tolerance", "guide_tolerance", "merge_tolerance",
                "minimum_area", "guide_segments",
            )},
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{case['id']}: cannot generate input mesh: {error}") from error
    try:
        imprint_v2.imprint_circle(after, **arguments)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{case['id']}: imprint_v2 failed: {error}") from error
    return before, after


def create_plotter(case, before, after, *, off_screen=False):
    """Build two linked XY views; leave showing/closing to the caller."""
    _enable_checkout_imports()
    import numpy as np
    import pyvista as pv

    from mesher.mesh2d.visualization import build_faces

    parameters = case["parameters"]
    all_nodes = np.concatenate((before.nodes, after.nodes))
    span = max(float(np.ptp(all_nodes[:, :2], axis=0).max()), 1.0)
    # Put reference lines just above the faces to prevent coplanar flicker.
    reference_z = float(all_nodes[:, 2].max()) + span * 1e-5
    center = np.asarray(parameters["grid_circler_center_coordinate"])
    # Keep circle chord error below 1% of the smallest input grid spacing.
    # A fixed 512-segment circle is visibly inaccurate beside very fine cells
    # when the radius is much larger than the nominal element size.
    spacings = np.concatenate([
        np.diff(np.unique(before.nodes[:, axis])) for axis in (0, 1)
    ])
    chord_error = max(float(spacings.min()) * 0.01, parameters["circle_radius"] * 1e-12)
    half_angle = np.arccos(np.clip(1.0 - chord_error / parameters["circle_radius"], -1.0, 1.0))
    segment_count = min(1_000_000, max(512, int(np.ceil(np.pi / half_angle))))
    angles = np.linspace(0.0, 2.0 * np.pi, segment_count + 1)
    circle = np.column_stack((
        center[0] + parameters["circle_radius"] * np.cos(angles),
        center[1] + parameters["circle_radius"] * np.sin(angles),
        np.full(angles.shape, reference_z),
    ))
    guides_xy = np.asarray(parameters["guide_segments"], dtype=float).reshape(-1, 2)
    guides = np.column_stack((guides_xy, np.full(len(guides_xy), reference_z)))
    scene_points = np.concatenate((all_nodes, circle, guides))
    minimum, maximum = scene_points.min(axis=0), scene_points.max(axis=0)
    padding = max(float(np.ptp(scene_points[:, :2], axis=0).max()), 1.0) * 0.05
    bounds = (
        minimum[0] - padding, maximum[0] + padding,
        minimum[1] - padding, maximum[1] + padding,
        minimum[2], maximum[2],
    )
    inspection = case.get("inspection_bounds")
    inspection_bounds = (
        (inspection[0], inspection[2], inspection[1], inspection[3], minimum[2], maximum[2])
        if inspection else bounds
    )

    plotter = pv.Plotter(
        shape=(1, 2), window_size=(1440, 800),
        title=f"imprint_v2 | {case['id']}", off_screen=off_screen,
    )
    plotter.set_background("white", all_renderers=True)
    for column, (label, mesh) in enumerate((
        ("Before imprint", before), ("After imprint", after),
    )):
        plotter.subplot(0, column)
        label_options = dict(
            font_size=10, bold=False, shape=None, margin=0,
            show_points=False, always_visible=True, render=False,
        )
        if mesh.node_count:
            plotter.add_point_labels(
                mesh.nodes, [f"N{i}" for i in range(mesh.node_count)],
                text_color="#1d4ed8", name=f"node_ids_{column}", **label_options,
            )
        if mesh.element_count:
            surface = pv.PolyData(mesh.nodes, build_faces(mesh))
            plotter.add_mesh(
                surface,
                color="lightblue", show_edges=True, edge_color="#334155",
                line_width=1.2, lighting=False, name="mesh",
            )
            plotter.add_point_labels(
                surface.cell_centers().points,
                [f"E{i}" for i in range(mesh.element_count)],
                text_color="#9f1239", name=f"element_ids_{column}", **label_options,
            )
        plotter.add_lines(circle, connected=True, color="blue", width=2, name="circle")
        if guides.size:
            plotter.add_lines(
                guides, connected=False, color="orange", width=3, name="guides",
            )
        plotter.add_text(
            f"{case['id']} | {label}\n{mesh.node_count} nodes / {mesh.element_count} elements",
            position="upper_left", font_size=13, color="#0f172a", name="title",
        )
        plotter.add_text(
            "Wheel / right drag: zoom | Left drag: pan\n"
            "R: full mesh | F: fail region | Views linked\n"
            "N: node ID | E: element ID (zero-based)",
            position="lower_left", font_size=10, color="#334155", name="controls",
        )

    plotter.link_views()
    plotter.enable_custom_trackball_style(
        left="pan", shift_left="pan", control_left="pan",
        middle="pan", shift_middle="pan", control_middle="pan",
        right="dolly", shift_right="dolly", control_right="dolly",
    )

    def fit_view(view_bounds):
        # Always fit the same scene bounds, regardless of which side is active.
        plotter.subplot(0, 0)
        plotter.view_xy(bounds=view_bounds, render=False)
        plotter.enable_parallel_projection()
        plotter.render()

    for key in ("r", "R"):
        plotter.clear_events_for_key(key)
        plotter.add_key_event(key, lambda: fit_view(bounds))
    for key in ("f", "F"):
        plotter.clear_events_for_key(key)
        plotter.add_key_event(key, lambda: fit_view(inspection_bounds))
    # PyVista's default V shortcut switches to an isometric camera.
    for key in ("v", "V"):
        plotter.clear_events_for_key(key)
    fit_view(inspection_bounds)
    return plotter


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cases = load_cases()
        if args.list_cases:
            for case in cases:
                print(f"{case['id']}: {case.get('mechanism', 'imprint_v2')}")
            return 0
        case = select_case(cases, args.case)
        before, after = prepare_meshes(case)
        plotter = create_plotter(case, before, after)
    except ImportError as error:
        parser.error(
            f"visualization dependencies are unavailable ({error}); "
            "install them from the repository root with: python -m pip install '.[visualization]'"
        )
    except ValueError as error:
        parser.error(str(error))
    plotter.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
