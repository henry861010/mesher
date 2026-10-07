import copy
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh3d.extrusion import Dragger
from mesher.process_flow import build_mesh_from_structure


def _box(left=-1, bottom=-1, right=1, top=1):
    return {"type": "BoxGeometry", "bottom_left": [left, bottom, 0],
            "top_right": [right, top, 0], "thk": 1}


def _structure(geometry, *, field="vias", koz=0.2):
    return {"root": {
        "bodies": [{"geometry": _box(), "material": "Si"}],
        "vias": [], "bumps": [], "circuits": [], "children": [{
            "bodies": [], "vias": [], "bumps": [], "circuits": [], "children": [],
            field: [{"id": "feature-1", "geometry": geometry, "koz": koz,
                     "density": 100, "material": "Cu"}],
        }],
    }}


def _build(structure, *, size=0.4, symmetry="full", events=None):
    return build_mesh_from_structure(structure, {
        "schemaVersion": "1.0.0", "unitSystem": "um", "mesher": "process_flow_2_5d",
        "globalElementSize": size, "symmetry": symmetry, "controls": [],
    }, progress=None if events is None else events.append)


def _metal_corners(mesh):
    metal = mesh.elements[mesh.element_comps == mesh.comps["Cu"], :8]
    return mesh.nodes[metal, :2]


class KozMeshingIntegrationTests(unittest.TestCase):
    def test_rectangular_koz_boundary_and_material_for_all_features_and_symmetries(self):
        for field in ("bumps", "circuits", "vias"):
            for symmetry in ("full", "upper_half", "right_half", "upper_right_quarter"):
                with self.subTest(field=field, symmetry=symmetry):
                    structure = _structure(_box(-0.8, -0.8, 0.8, 0.8), field=field)
                    before = copy.deepcopy(structure)
                    mesh = _build(structure, symmetry=symmetry)
                    corners = _metal_corners(mesh)
                    self.assertGreater(len(corners), 0)
                    self.assertTrue(np.all(np.abs(corners) <= 0.6 + 1e-10))
                    self.assertTrue(np.any(np.isclose(mesh.nodes[:, 0], 0.6)))
                    self.assertTrue(np.any(np.isclose(mesh.nodes[:, 1], 0.6)))
                    if symmetry in ("right_half", "upper_right_quarter"):
                        self.assertTrue(np.all(mesh.nodes[:, 0] >= 0))
                    if symmetry in ("upper_half", "upper_right_quarter"):
                        self.assertTrue(np.all(mesh.nodes[:, 1] >= 0))
                    self.assertEqual(structure, before)

    def test_circle_keeps_both_rings_and_refines_xy_without_refining_z(self):
        circle = {"type": "CylinderGeometry", "center": [0, 0, 0],
                  "bottom_radius": 0.6, "thk": 1}
        for symmetry in ("full", "upper_half", "upper_right_quarter"):
            with self.subTest(symmetry=symmetry):
                structure = _structure(circle, koz=0.1)
                events, planar = [], {}
                original = Dragger.set_2D

                def capture(dragger, nodes, elements):
                    planar["nodes"], planar["elements"] = nodes.copy(), elements.copy()
                    return original(dragger, nodes, elements)

                with patch("mesher.process_flow.pipeline.Dragger.set_2D", capture):
                    mesh = _build(structure, size=0.5, symmetry=symmetry, events=events)
                nodes, elements = planar["nodes"], planar["elements"]
                radii = np.linalg.norm(nodes[:, :2], axis=1)
                for radius in (0.5, 0.6):
                    self.assertGreater(np.count_nonzero(np.isclose(radii, radius, atol=1e-10)), 4)
                points = nodes[elements, :2]
                signed_areas = 0.5 * np.sum(
                    points[:, :, 0] * np.roll(points[:, :, 1], -1, axis=1)
                    - points[:, :, 1] * np.roll(points[:, :, 0], -1, axis=1), axis=1)
                self.assertTrue(np.all(signed_areas < 0))
                self.assertTrue(np.all(elements >= 0))
                self.assertTrue(np.all(elements < len(nodes)))
                corners = _metal_corners(mesh)
                self.assertGreater(len(corners), 0)
                self.assertTrue(np.all(np.linalg.norm(corners, axis=2) <= 0.5 + 1e-8))
                np.testing.assert_array_equal(np.unique(mesh.nodes[:, 2]), [0, 0.5, 1])
                analysis = next(e["data"] for e in events if e["event"] == "stage.completed"
                                and e["stage"] == "analyzing_geometry")
                self.assertAlmostEqual(analysis["planarElementSize"], 0.025)
                circles = [e["data"] for e in events if e["event"] == "item.completed"
                           and e["stage"] == "building_2d_mesh"]
                self.assertEqual(len(circles), 2)
                self.assertTrue(all(e["sourceRefs"] == ["feature-1"] for e in circles))

    def test_polygon_hole_koz_uses_inner_hull_and_expanded_hole(self):
        polygon = {"type": "PolygonGeometry", "thk": 1, "polys": [
            [[-0.8, -0.8, 0], [0.8, -0.8, 0], [0.8, 0.8, 0], [-0.8, 0.8, 0]],
            [[-0.2, -0.2, 0], [0.2, -0.2, 0], [0.2, 0.2, 0], [-0.2, 0.2, 0]],
        ]}
        for symmetry in ("full", "upper_half", "upper_right_quarter"):
            with self.subTest(symmetry=symmetry):
                mesh = _build(_structure(polygon, koz=0.1), size=0.2, symmetry=symmetry)
                corners = _metal_corners(mesh)
                self.assertGreater(len(corners), 0)
                self.assertTrue(np.all(np.abs(corners) <= 0.7 + 1e-10))
                self.assertFalse(np.any(np.all(np.abs(corners) < 0.3 - 1e-10, axis=2)))
                centers = corners.mean(axis=1)
                self.assertFalse(np.any(np.all(np.abs(centers) < 0.3, axis=1)))

    def test_thin_box_koz_survives_grid_and_material_selection_tolerances(self):
        mesh = _build(_structure(_box(), koz=1e-4), size=0.5)
        corners = _metal_corners(mesh)
        self.assertGreater(len(corners), 0)
        self.assertTrue(np.all(np.abs(corners) <= 1 - 1e-4 + 1e-10))
        for coordinate in (-1, -1 + 1e-4, 1 - 1e-4, 1):
            self.assertTrue(np.any(np.abs(mesh.nodes[:, 0] - coordinate) < 1e-10))

    def test_circle_koz_smaller_than_legacy_selection_tolerance(self):
        circle = {"type": "CylinderGeometry", "center": [0, 0, 0],
                  "bottom_radius": 0.03, "thk": 1}
        structure = _structure(circle, koz=0.005)
        structure["root"]["bodies"][0]["geometry"] = _box(-0.05, -0.05, 0.05, 0.05)
        mesh = _build(structure)
        corners = _metal_corners(mesh)
        self.assertGreater(len(corners), 0)
        self.assertTrue(np.all(np.linalg.norm(corners, axis=2) <= 0.025 + 1e-10))

    def test_zero_and_missing_koz_produce_identical_meshes(self):
        structure = _structure(_box(-0.8, -0.8, 0.8, 0.8), koz=0)
        explicit = _build(structure)
        del structure["root"]["children"][0]["vias"][0]["koz"]
        implicit = _build(structure)
        np.testing.assert_array_equal(explicit.nodes, implicit.nodes)
        np.testing.assert_array_equal(explicit.elements, implicit.elements)
        np.testing.assert_array_equal(explicit.element_comps, implicit.element_comps)

    def test_circular_body_extends_outward_from_koz_ring(self):
        circle = {"type": "CylinderGeometry", "center": [0, 0, 0],
                  "bottom_radius": 0.6, "thk": 1}
        structure = _structure(circle, koz=0.1)
        structure["root"]["bodies"][0]["geometry"] = copy.deepcopy(circle)
        events = []
        mesh = _build(structure, events=events)
        analysis = next(e["data"] for e in events if e["event"] == "stage.completed"
                        and e["stage"] == "analyzing_geometry")
        self.assertEqual(analysis["extensionOperationCount"], 1)
        corners = _metal_corners(mesh)
        self.assertGreater(len(corners), 0)
        self.assertTrue(np.all(np.linalg.norm(corners, axis=2) <= 0.5 + 1e-10))

    def test_collapsed_koz_preserves_body_and_assigns_no_metal(self):
        geometries = [
            _box(),
            {"type": "CylinderGeometry", "center": [0, 0, 0], "bottom_radius": 1, "thk": 1},
            {"type": "PolygonGeometry", "polys": [
                [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]], "thk": 1},
        ]
        for geometry in geometries:
            with self.subTest(shape=geometry["type"]):
                mesh = _build(_structure(geometry, koz=1.1))
                self.assertGreater(mesh.element_count, 0)
                self.assertFalse(np.any(mesh.element_comps == mesh.comps["Cu"]))


if __name__ == "__main__":
    unittest.main()
