"""Case selection, fresh public-API replay, and linked comparison views."""

import contextlib
import ast
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
VIEWER_PATH = ROOT / "examples/failures/imprint_v2_holes/view.py"
spec = importlib.util.spec_from_file_location("imprint_v2_viewer", VIEWER_PATH)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class CaseSelectionTests(unittest.TestCase):
    def test_selects_every_case_by_id_and_number(self):
        cases = viewer.load_cases()
        self.assertEqual(len(cases), 10)
        for index, case in enumerate(cases, 1):
            with self.subTest(case=case["id"]):
                self.assertIs(viewer.select_case(cases, str(index)), case)
                self.assertIs(viewer.select_case(cases, case["id"]), case)

    def test_unknown_case_reports_available_ids_before_creating_mesh(self):
        for identifier in ("0", "11", "case_99", "invalid"):
            with self.subTest(case=identifier):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with patch.object(viewer, "prepare_meshes") as prepare:
                        with self.assertRaises(SystemExit) as error:
                            viewer.main(["--case", identifier])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("unknown case", stderr.getvalue())
                self.assertIn("case_01", stderr.getvalue())
                self.assertIn("case_10", stderr.getvalue())
                prepare.assert_not_called()

    def test_default_case_is_case_01(self):
        self.assertEqual(viewer._parser().parse_args([]).case, "case_01")

    def test_list_and_help_work_without_third_party_imports_or_checkout_cwd(self):
        # -S disables site-packages: listing must not need NumPy/PyVista.
        for argument in ("--list", "--help"):
            with self.subTest(argument=argument), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run(
                    [sys.executable, "-S", str(VIEWER_PATH), argument],
                    cwd=cwd, capture_output=True, text=True, check=False,
                )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("case_01", result.stdout)
            if argument == "--list":
                self.assertIn("case_10", result.stdout)

    def test_missing_or_invalid_cases_file_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            with self.assertRaisesRegex(ValueError, "cannot read cases file"):
                viewer.load_cases(path)
            for content in ("{invalid", "{}", "[]", '{"cases": [{}]}'):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        viewer.load_cases(path)
            case = {"id": "case_01", "parameters": {}}
            path.write_text(json.dumps({"cases": [case, case]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case ID"):
                viewer.load_cases(path)


class MeshReplayTests(unittest.TestCase):
    def test_corpus_uses_large_circles_and_current_process_flow_tolerances(self):
        tree = ast.parse((ROOT / "src/mesher/process_flow/pipeline.py").read_text())
        ratios = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("_CIRCLE_")
        }
        for case in viewer.load_cases():
            with self.subTest(case=case["id"]):
                p = case["parameters"]
                self.assertEqual(p["circle_radius"], 300000)
                self.assertEqual(p["target_edge_size"], 5000)
                self.assertGreaterEqual(p["circle_radius"] / p["target_edge_size"], 60)
                for name, exponent in (("projection_tolerance", 1), ("guide_tolerance", 1),
                                       ("merge_tolerance", 1), ("minimum_area", 2)):
                    ratio = ratios[f"_CIRCLE_{name.upper()}_RATIO"]
                    self.assertEqual(p[name], ratio * p["target_edge_size"] ** exponent)
                cx, cy = p["grid_circler_center_coordinate"]
                self.assertLess(min(p["grid_x_list"]), cx - p["circle_radius"])
                self.assertGreater(max(p["grid_x_list"]), cx + p["circle_radius"])
                self.assertLess(min(p["y_list"]), cy - p["circle_radius"])
                self.assertGreater(max(p["y_list"]), cy + p["circle_radius"])

    def test_all_ten_cases_keep_the_input_unchanged_and_call_public_v2(self):
        from mesher.mesh2d.circular import imprint_v2
        from mesher.mesh2d.generators import generate_rectilinear_mesh

        for case in viewer.load_cases():
            with self.subTest(case=case["id"]):
                parameters = case["parameters"]
                original = generate_rectilinear_mesh(
                    parameters["target_edge_size"], parameters["grid_x_list"],
                    parameters["y_list"],
                    coordinate_tolerance=parameters["coordinate_tolerance"],
                )
                with patch.object(imprint_v2, "imprint_circle", wraps=imprint_v2.imprint_circle) as imprint:
                    before, after = viewer.prepare_meshes(case)
                imprint.assert_called_once()
                self.assertIs(imprint.call_args.args[0], after)
                self.assertEqual(imprint.call_args.kwargs["center"], parameters["grid_circler_center_coordinate"])
                self.assertEqual(imprint.call_args.kwargs["radius"], parameters["circle_radius"])
                for name in ("projection_tolerance", "guide_tolerance", "merge_tolerance",
                             "minimum_area", "guide_segments"):
                    self.assertEqual(imprint.call_args.kwargs[name], parameters[name])
                np.testing.assert_array_equal(before.nodes, original.nodes)
                np.testing.assert_array_equal(before.elements, original.elements)
                spacings = np.concatenate([
                    np.diff(np.unique(before.nodes[:, axis])) for axis in (0, 1)
                ])
                self.assertLessEqual(spacings.max(), parameters["target_edge_size"] * (1 + 1e-12))
                self.assertAlmostEqual(spacings.min(), case["mesh_profile"]["minimum_grid_spacing"])
                self.assertIsNot(before, after)
                self.assertFalse(np.shares_memory(before.nodes, after.nodes))
                self.assertFalse(np.shares_memory(before.elements, after.elements))

    def test_missing_case_parameter_and_imprint_error_include_case_id(self):
        case = viewer.load_cases()[0]
        with self.assertRaisesRegex(ValueError, "case_01: cannot generate input mesh"):
            viewer.prepare_meshes(dict(case, parameters={}))
        with patch("mesher.mesh2d.circular.imprint_v2.imprint_circle", side_effect=ValueError("bad geometry")):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with patch.object(viewer, "create_plotter") as create:
                    with self.assertRaises(SystemExit) as error:
                        viewer.main([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("case_01: imprint_v2 failed: bad geometry", stderr.getvalue())
        create.assert_not_called()

    def test_missing_visualization_dependency_has_install_instructions(self):
        with patch.object(viewer, "create_plotter", side_effect=ImportError("pyvista")):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    viewer.main([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn(".[visualization]", stderr.getvalue())


@unittest.skipIf(importlib.util.find_spec("pyvista") is None, "requires optional PyVista")
class LinkedViewTests(unittest.TestCase):
    def test_real_cameras_and_geometry_are_linked_and_reset_from_either_view(self):
        for identifier in ("case_01", "case_10"):
            with self.subTest(case=identifier):
                case = viewer.select_case(viewer.load_cases(), identifier)
                before, after = viewer.prepare_meshes(case)
                # Construct real VTK actors/cameras without starting a render
                # window. Actual drawing is checked separately on a display.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="\nThis system does not appear")
                    plotter = viewer.create_plotter(case, before, after, off_screen=True)
                try:
                    left, right = plotter.renderers
                    self.assertIs(left.camera, right.camera)
                    self.assertTrue(left.camera.parallel_projection)
                    np.testing.assert_allclose(left.camera.direction, [0, 0, -1])
                    np.testing.assert_allclose(left.camera.up, [0, 1, 0])
                    self.assertEqual(left.actors["mesh"].mapper.dataset.n_cells, before.element_count)
                    self.assertEqual(right.actors["mesh"].mapper.dataset.n_cells, after.element_count)
                    for renderer in (left, right):
                        circle = renderer.actors["circle"].mapper.dataset.points
                        center = case["parameters"]["grid_circler_center_coordinate"]
                        np.testing.assert_allclose(
                            np.linalg.norm(circle[:, :2] - center, axis=1),
                            case["parameters"]["circle_radius"],
                        )
                        self.assertEqual("guides" in renderer.actors, bool(case["parameters"]["guide_segments"]))
                    initial_position = np.array(left.camera.position)
                    initial_focus = np.array(left.camera.focal_point)
                    initial_scale = left.camera.parallel_scale
                    for column, key in ((0, "f"), (1, "F")):
                        plotter.subplot(0, column)
                        plotter.camera.zoom(2)
                        plotter.camera.position = initial_position + [1, 2, 0]
                        plotter.camera.focal_point = initial_focus + [1, 2, 0]
                        self.assertAlmostEqual(right.camera.parallel_scale, initial_scale / 2)
                        for callback in plotter.iren._key_press_event_callbacks[key]:
                            callback()
                        np.testing.assert_allclose(left.camera.position, initial_position)
                        np.testing.assert_allclose(right.camera.focal_point, initial_focus)
                        self.assertAlmostEqual(left.camera.parallel_scale, initial_scale)
                    for key in ("r", "R"):
                        for callback in plotter.iren._key_press_event_callbacks[key]:
                            callback()
                        self.assertGreater(left.camera.parallel_scale, initial_scale)
                        np.testing.assert_allclose(left.camera.focal_point[:2], [0, 0])
                    for callback in plotter.iren._key_press_event_callbacks["f"]:
                        callback()
                    np.testing.assert_allclose(left.camera.focal_point, initial_focus)
                finally:
                    plotter.close()


if __name__ == "__main__":
    unittest.main()
