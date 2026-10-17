"""Check the independent hole measurement used by the failure examples."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh


if importlib.util.find_spec("shapely") is not None:
    path = Path(__file__).resolve().parents[4] / "examples/failures/imprint_v2_holes/diagnose.py"
    spec = importlib.util.spec_from_file_location("imprint_v2_diagnostics", path)
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)
else:
    diagnostics = None


@unittest.skipIf(diagnostics is None, "diagnostics require optional Shapely")
class HoleMeasurementTests(unittest.TestCase):
    def test_resolved_case_is_a_valid_passing_diagnostic(self):
        case = {
            "id": "resolved",
            "parameters": {
                "target_edge_size": 1.0,
                "grid_x_list": [-2.0, 2.0],
                "y_list": [-2.0, 2.0],
                "coordinate_tolerance": 1e-6,
                "grid_circler_center_coordinate": [0.0, 0.0],
                "circle_radius": 1.0,
                "projection_tolerance": 0.0,
                "guide_tolerance": 1.0,
                "merge_tolerance": 0.0,
                "minimum_area": 0.0,
                "guide_segments": [],
            },
        }

        result, _, _ = diagnostics.diagnose(case)

        self.assertEqual(result["status"], "pass")
        self.assertIsNone(result["first_hole_stage"])
        self.assertEqual(diagnostics.verify(case, result), [])

    def test_enclosed_missing_cell_is_a_hole(self):
        mesh = generate_rectilinear_mesh(1, [0, 3], [0, 3])
        mesh = Mesh2D(nodes=mesh.nodes, elements=np.delete(mesh.elements, 4, axis=0))

        report, _ = diagnostics.measure(mesh, diagnostics.box(0, 0, 3, 3), 1e-8)

        self.assertEqual(report["closed_hole_count"], 1)
        self.assertEqual(report["closed_hole_area"], 1)
        self.assertEqual(report["boundary_notch_count"], 0)

    def test_boundary_notch_is_not_an_enclosed_hole(self):
        mesh = generate_rectilinear_mesh(1, [0, 3], [0, 3])
        mesh = Mesh2D(nodes=mesh.nodes, elements=np.delete(mesh.elements, 1, axis=0))

        report, _ = diagnostics.measure(mesh, diagnostics.box(0, 0, 3, 3), 1e-8)

        self.assertEqual(report["closed_hole_count"], 0)
        self.assertEqual(report["boundary_notch_area"], 1)

    def test_hanging_node_does_not_imply_missing_area(self):
        mesh = Mesh2D(
            nodes=[[0, 0], [1, 0], [1, 1], [0, 1], [2, 0],
                   [2, 0.5], [1, 0.5], [2, 1]],
            elements=[[0, 1, 2, 3], [1, 4, 5, 6], [6, 5, 7, 2]],
        )

        report, _ = diagnostics.measure(mesh, diagnostics.box(0, 0, 2, 1), 1e-8)

        self.assertEqual(report["missing_area"], 0)
        self.assertEqual(report["closed_hole_count"], 0)

    def test_overlap_cannot_hide_a_hole_in_total_element_area(self):
        mesh = generate_rectilinear_mesh(1, [0, 3], [0, 3])
        elements = mesh.elements.copy()
        elements[4] = elements[0]  # Removed area equals duplicated area.
        mesh = Mesh2D(nodes=mesh.nodes, elements=elements)

        report, _ = diagnostics.measure(mesh, diagnostics.box(0, 0, 3, 3), 1e-8)

        self.assertEqual(report["closed_hole_area"], 1)
        self.assertEqual(report["overlap_area"], 1)

    def test_invalid_measurement_preserves_the_original_mesh(self):
        mesh = Mesh2D(nodes=[[0, 0], [1, 1], [0, 1], [1, 0]],
                      elements=[[0, 1, 2, 3]])
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()

        report, _ = diagnostics.measure(mesh, diagnostics.box(0, 0, 1, 1), 1e-8)

        self.assertEqual(report["invalid_element_rows"], [0])
        self.assertAlmostEqual(report["covered_area"], 0.5)
        np.testing.assert_array_equal(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.elements, elements)
