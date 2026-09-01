import copy
import unittest
from unittest.mock import patch

import numpy as np
from mesher import Mesh3D
from mesher.mesh2d.circular import extend_circular_mesh, imprint_circle

from mesher.process_flow import build_mesh_from_structure


def _box_structure():
    return {
        "root": {
            "bodies": [
                {
                    "geometry": {
                        "type": "BoxGeometry",
                        "bottom_left": [0.0, 0.0, 0.0],
                        "top_right": [2.0, 2.0, 0.0],
                        "thk": 1.0,
                    },
                    "material": "Si",
                }
            ],
            "vias": [],
            "circuits": [],
            "bumps": [],
            "children": [],
        }
    }


def _multi_circle_structure():
    return {
        "root": {
            "bodies": [
                {
                    "geometry": {
                        "type": "CylinderGeometry",
                        "center": [-3.0, 0.0, 0.0],
                        "bottom_radius": 2.0,
                        "thk": 1.0,
                    },
                    "material": "Si",
                },
                {
                    "geometry": {
                        "type": "CylinderGeometry",
                        "center": [3.0, 0.0, 0.0],
                        "bottom_radius": 2.0,
                        "thk": 1.0,
                    },
                    "material": "Si",
                },
            ],
            "vias": [],
            "circuits": [],
            "bumps": [],
            "children": [],
        }
    }


def _circle_structure(*circles):
    return {
        "root": {
            "bodies": [
                {
                    "geometry": {
                        "type": "CylinderGeometry",
                        "center": [center_x, center_y, z],
                        "bottom_radius": radius,
                        "thk": thickness,
                    },
                    "material": material,
                }
                for center_x, center_y, radius, z, thickness, material in circles
            ],
            "vias": [],
            "circuits": [],
            "bumps": [],
            "children": [],
        }
    }


def _append_box(structure, *, x1, y1, x2, y2):
    structure["root"]["bodies"].append(
        {
            "geometry": {
                "type": "BoxGeometry",
                "bottom_left": [x1, y1, 0.0],
                "top_right": [x2, y2, 0.0],
                "thk": 1.0,
            },
            "material": "Cu",
        }
    )
    return structure


def _offset_box_structure():
    structure = _box_structure()
    geometry = structure["root"]["bodies"][0]["geometry"]
    geometry["bottom_left"] = [10.0, 20.0, 0.0]
    geometry["top_right"] = [14.0, 24.0, 0.0]
    return structure


def _append_polygon(structure, *, points, material):
    structure["root"]["bodies"].append(
        {
            "geometry": {
                "type": "PolygonGeometry",
                "polys": [[[x, y, 0.0] for x, y in points]],
                "thk": 1.0,
            },
            "material": material,
        }
    )
    return structure


class BuilderIntegrationTests(unittest.TestCase):
    def test_rejects_unknown_semantic_keys(self):
        structure = _box_structure()
        structure["root"]["key"] = "mesh-root"

        with self.assertRaisesRegex(ValueError, "Unsupported container.key"):
            build_mesh_from_structure(structure, element_size=1.0)

    def test_preserves_a_box_pattern_crossing_a_circle_boundary(self):
        structure = {
            "root": {
                "key": "carrier.wafer",
                "bodies": [
                    {
                        "geometry": {
                            "type": "CylinderGeometry",
                            "center": [0.0, 0.0, 0.0],
                            "bottom_radius": 100.0,
                            "thk": 1.0,
                        },
                        "material": "wafer",
                    }
                ],
                "vias": [],
                "circuits": [],
                "bumps": [],
                "children": [
                    {
                        "key": "soc",
                        "bodies": [
                            {
                                "geometry": {
                                    "type": "BoxGeometry",
                                    "bottom_left": [85.0, -15.0, 1.0],
                                    "top_right": [115.0, 15.0, 1.0],
                                    "thk": 1.0,
                                },
                                "material": "soc",
                            }
                        ],
                        "vias": [],
                        "circuits": [],
                        "bumps": [],
                        "children": [],
                    }
                ],
            }
        }

        with patch(
            "mesher.process_flow.pipeline.imprint_circle",
            wraps=imprint_circle,
        ) as mocked_imprint:
            mesh = build_mesh_from_structure(structure, element_size=5.0)

        self.assertEqual(
            mocked_imprint.call_args.kwargs["guide_segments"],
            [
                ((85.0, -15.0), (115.0, -15.0)),
                ((115.0, -15.0), (115.0, 15.0)),
                ((115.0, 15.0), (85.0, 15.0)),
                ((85.0, 15.0), (85.0, -15.0)),
            ],
        )

        soc_elements = mesh.elements[mesh.element_comps == mesh.comps["soc"]]
        soc_xy = mesh.nodes[soc_elements[:, :4], :2]
        x = soc_xy[:, :, 0]
        y = soc_xy[:, :, 1]
        element_areas = 0.5 * np.abs(
            np.sum(
                x * np.roll(y, -1, axis=1)
                - y * np.roll(x, -1, axis=1),
                axis=1,
            )
        )
        self.assertAlmostEqual(float(np.sum(element_areas)), 900.0)

    def test_public_builder_matches_known_mesh_contract(self):
        mesh = build_mesh_from_structure(_box_structure(), element_size=1.0)

        self.assertIsInstance(mesh, Mesh3D)
        self.assertEqual(mesh.node_count, 18)
        self.assertEqual(mesh.element_count, 4)
        self.assertEqual(mesh.component_count, 2)
        self.assertEqual(mesh.comps, {"EMPTY": 0, "Si": 1})
        np.testing.assert_array_equal(mesh.element_comps, [1, 1, 1, 1])
        np.testing.assert_array_equal(
            mesh.nodes,
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [0.0, 2.0, 1.0],
                [1.0, 2.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
        )
        np.testing.assert_array_equal(
            mesh.elements,
            [
                [0, 3, 4, 1, 9, 12, 13, 10],
                [1, 4, 5, 2, 10, 13, 14, 11],
                [3, 6, 7, 4, 12, 15, 16, 13],
                [4, 7, 8, 5, 13, 16, 17, 14],
            ],
        )

    def test_progress_callback_reports_features_inside_building_2d_mesh(self):
        events = []

        mesh = build_mesh_from_structure(
            _multi_circle_structure(),
            element_size=1.0,
            progress=events.append,
        )

        self.assertGreater(mesh.element_count, 0)
        started_stages = [
            event["stage"]
            for event in events
            if event["event"] == "stage.started"
        ]
        self.assertEqual(
            started_stages,
            [
                "validating",
                "analyzing_geometry",
                "building_2d_mesh",
                "building_3d_mesh",
            ],
        )
        feature_items = [
            event
            for event in events
            if event["event"] == "item.completed"
            and event["data"].get("featureType") == "circle"
        ]
        self.assertGreater(len(feature_items), 0)
        self.assertTrue(
            all(event["stage"] == "building_2d_mesh" for event in feature_items)
        )
        self.assertNotIn("processing_features", started_stages)

    def test_progress_callback_matches_the_stable_box_event_contract(self):
        events = []

        build_mesh_from_structure(
            _box_structure(),
            element_size=1.0,
            progress=events.append,
        )

        volatile_fields = {"wallDurationMs", "cpuDurationMs", "peakRssBytes"}
        normalized = []
        for event in events:
            payload = copy.deepcopy(event)
            for field in volatile_fields:
                payload.get("data", {}).pop(field, None)
            normalized.append(payload)

        self.assertEqual(
            normalized,
            [
                {
                    "event": "stage.started",
                    "stage": "validating",
                    "current": None,
                    "total": None,
                    "unit": None,
                    "message": "Checking geometry input.",
                    "data": {},
                },
                {"event": "stage.completed", "stage": "validating", "data": {}},
                {
                    "event": "stage.started",
                    "stage": "analyzing_geometry",
                    "current": None,
                    "total": None,
                    "unit": None,
                    "message": "Analyzing geometry patterns.",
                    "data": {},
                },
                {
                    "event": "stage.completed",
                    "stage": "analyzing_geometry",
                    "data": {
                        "faceCount": 1,
                        "circlePatternCount": 0,
                        "imprintOperationCount": 0,
                        "extensionOperationCount": 0,
                        "xGridLineCount": 2,
                        "yGridLineCount": 2,
                        "layerBoundaryCount": 2,
                        "layerIntervalCount": 1,
                        "assignmentCount": 1,
                        "planarElementSize": 1.0,
                        "symmetry": "full",
                    },
                },
                {
                    "event": "stage.started",
                    "stage": "building_2d_mesh",
                    "current": None,
                    "total": None,
                    "unit": None,
                    "message": "Generating base grid.",
                    "data": {},
                },
                {
                    "event": "stage.completed",
                    "stage": "building_2d_mesh",
                    "data": {
                        "featureOperationCount": 0,
                        "node2DCount": 9,
                        "element2DCount": 4,
                    },
                },
                {
                    "event": "stage.started",
                    "stage": "building_3d_mesh",
                    "current": 0,
                    "total": 1,
                    "unit": "layers",
                    "message": "Building 3D mesh layers.",
                    "data": {},
                },
                {
                    "event": "progress",
                    "current": 0,
                    "total": 1,
                    "unit": "layers",
                    "message": "Assigning feature 1 of 1 in layer 1 of 1.",
                    "data": {},
                },
                {
                    "event": "item.completed",
                    "stage": "building_3d_mesh",
                    "data": {
                        "itemType": "assignment",
                        "layerIndex": 1,
                        "assignmentIndex": 1,
                        "sourceRef": "root.bodies[0]",
                        "containerRef": "root",
                        "featureType": "bodie",
                        "geometryType": "BoxGeometry",
                        "operation": "start",
                        "selectedElementCount": 4,
                    },
                },
                {
                    "event": "item.completed",
                    "stage": "building_3d_mesh",
                    "data": {
                        "itemType": "layer",
                        "layerIndex": 1,
                        "assignmentCount": 1,
                        "nodesAdded": 18,
                        "elementsAdded": 4,
                    },
                },
                {
                    "event": "progress",
                    "current": 1,
                    "total": 1,
                    "unit": "layers",
                    "message": "Built layer 1 of 1.",
                    "data": {},
                },
                {
                    "event": "stage.completed",
                    "stage": "building_3d_mesh",
                    "data": {
                        "nodeCount": 18,
                        "elementCount": 4,
                        "componentCount": 2,
                    },
                },
            ],
        )

    def test_symmetry_modes_use_the_full_boundary_box_center(self):
        structure = _offset_box_structure()
        cases = (
            ("full", 16, [10.0, 20.0], [14.0, 24.0]),
            ("upper_right_quarter", 4, [12.0, 22.0], [14.0, 24.0]),
            ("upper_half", 8, [10.0, 22.0], [14.0, 24.0]),
            ("right_half", 8, [12.0, 20.0], [14.0, 24.0]),
        )

        for symmetry, element_count, xy_min, xy_max in cases:
            with self.subTest(symmetry=symmetry):
                mesh = build_mesh_from_structure(
                    structure,
                    element_size=1.0,
                    symmetry=symmetry,
                )

                self.assertEqual(mesh.element_count, element_count)
                np.testing.assert_array_equal(mesh.nodes[:, :2].min(axis=0), xy_min)
                np.testing.assert_array_equal(mesh.nodes[:, :2].max(axis=0), xy_max)

    def test_quarter_model_filters_outside_patterns_and_keeps_crossing_patterns(self):
        structure = {
            "root": {
                "bodies": [
                    {
                        "geometry": {
                            "type": "BoxGeometry",
                            "bottom_left": [-4.0, -4.0, 0.0],
                            "top_right": [4.0, 4.0, 0.0],
                            "thk": 1.0,
                        },
                        "material": "base",
                    },
                    {
                        "geometry": {
                            "type": "BoxGeometry",
                            "bottom_left": [-3.0, -3.0, 0.0],
                            "top_right": [-2.0, -2.0, 0.0],
                            "thk": 1.0,
                        },
                        "material": "outside",
                    },
                    {
                        "geometry": {
                            "type": "BoxGeometry",
                            "bottom_left": [-1.0, 1.0, 0.0],
                            "top_right": [1.0, 2.0, 0.0],
                            "thk": 1.0,
                        },
                        "material": "crossing",
                    },
                ],
                "vias": [],
                "circuits": [],
                "bumps": [],
                "children": [],
            }
        }
        snapshot = copy.deepcopy(structure)

        mesh = build_mesh_from_structure(
            structure,
            element_size=1.0,
            symmetry="upper_right_quarter",
        )

        self.assertEqual(structure, snapshot)
        self.assertNotIn("outside", mesh.comps)
        self.assertIn("crossing", mesh.comps)
        self.assertTrue(np.all(mesh.nodes[:, 0] >= 0.0))
        self.assertTrue(np.all(mesh.nodes[:, 1] >= 0.0))

    def test_quarter_model_uses_actual_polygon_footprint_not_its_bounds(self):
        structure = {
            "root": {
                "bodies": [
                    {
                        "geometry": {
                            "type": "BoxGeometry",
                            "bottom_left": [-4.0, -4.0, 0.0],
                            "top_right": [4.0, 4.0, 0.0],
                            "thk": 1.0,
                        },
                        "material": "base",
                    }
                ],
                "vias": [],
                "circuits": [],
                "bumps": [],
                "children": [],
            }
        }
        _append_polygon(
            structure,
            points=[(-3.0, 1.0), (1.0, -3.0), (-3.0, -3.0)],
            material="diagonal-outside",
        )

        mesh = build_mesh_from_structure(
            structure,
            element_size=1.0,
            symmetry="upper_right_quarter",
        )

        self.assertNotIn("diagonal-outside", mesh.comps)

    def test_rejects_a_selected_quarter_without_positive_area_geometry(self):
        structure = {
            "root": {
                "bodies": [
                    {
                        "geometry": {
                            "type": "BoxGeometry",
                            "bottom_left": [-4.0, 2.0, 0.0],
                            "top_right": [-2.0, 4.0, 0.0],
                            "thk": 1.0,
                        },
                        "material": "upper-left",
                    },
                    {
                        "geometry": {
                            "type": "BoxGeometry",
                            "bottom_left": [2.0, -4.0, 0.0],
                            "top_right": [4.0, -2.0, 0.0],
                            "thk": 1.0,
                        },
                        "material": "lower-right",
                    },
                ],
                "vias": [],
                "circuits": [],
                "bumps": [],
                "children": [],
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            r"no geometry with positive XY area in upper_right_quarter",
        ):
            build_mesh_from_structure(
                structure,
                element_size=1.0,
                symmetry="upper_right_quarter",
            )

    def test_quarter_model_imprints_an_offset_open_circle(self):
        mesh = build_mesh_from_structure(
            _circle_structure((10.0, 20.0, 3.0, 0.0, 1.0, "Cu")),
            element_size=1.0,
            symmetry="upper_right_quarter",
        )

        self.assertGreater(mesh.element_count, 0)
        self.assertTrue(np.all(mesh.nodes[:, 0] >= 10.0 - 1.0e-12))
        self.assertTrue(np.all(mesh.nodes[:, 1] >= 20.0 - 1.0e-12))
        self.assertTrue(np.any(np.isclose(mesh.nodes[:, 0], 10.0)))
        self.assertTrue(np.any(np.isclose(mesh.nodes[:, 1], 20.0)))

    def test_rejects_a_non_radial_open_circle(self):
        structure = {
            "root": {
                "bodies": [
                    {
                        "geometry": {
                            "type": "BoxGeometry",
                            "bottom_left": [-5.0, -5.0, 0.0],
                            "top_right": [5.0, 5.0, 0.0],
                            "thk": 1.0,
                        },
                        "material": "base",
                    },
                    {
                        "geometry": {
                            "type": "CylinderGeometry",
                            "center": [1.0, 0.0, 0.0],
                            "bottom_radius": 2.0,
                            "thk": 1.0,
                        },
                        "material": "Cu",
                    },
                ],
                "vias": [],
                "circuits": [],
                "bumps": [],
                "children": [],
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            r"requires each intersecting symmetry boundary.*boundary x=0",
        ):
            build_mesh_from_structure(
                structure,
                element_size=1.0,
                symmetry="upper_right_quarter",
            )

    def test_rejects_an_unknown_symmetry(self):
        with self.assertRaisesRegex(
            ValueError,
            r"full, upper_half, right_half, upper_right_quarter",
        ):
            build_mesh_from_structure(
                _box_structure(),
                element_size=1.0,
                symmetry="Upper_Model",
            )

    def test_builds_disjoint_cylinders_in_one_mesh(self):
        mesh = build_mesh_from_structure(_multi_circle_structure(), element_size=1.0)

        self.assertEqual(mesh.element_count, 66)
        self.assertEqual(mesh.comps, {"EMPTY": 0, "Si": 1})
        self.assertTrue(np.all(mesh.element_comps == 1))

        element_xy = mesh.nodes[mesh.elements, :2]
        distance_squared = np.minimum(
            (element_xy[:, :, 0] + 3.0) ** 2 + element_xy[:, :, 1] ** 2,
            (element_xy[:, :, 0] - 3.0) ** 2 + element_xy[:, :, 1] ** 2,
        )
        self.assertTrue(np.all(distance_squared <= 2.0**2 + 1.0e-10))

        bottom_nodes = mesh.nodes[np.isclose(mesh.nodes[:, 2], 0.0)]
        for center_x in (-3.0, 3.0):
            radii = np.hypot(
                bottom_nodes[:, 0] - center_x,
                bottom_nodes[:, 1],
            )
            self.assertEqual(np.count_nonzero(np.isclose(radii, 2.0)), 19)

        padded_wedges = mesh.elements[:, 2] == mesh.elements[:, 3]
        self.assertGreater(np.count_nonzero(padded_wedges), 0)
        np.testing.assert_array_equal(
            mesh.elements[padded_wedges, 6],
            mesh.elements[padded_wedges, 7],
        )

    def test_refines_xy_mesh_for_a_circle_smaller_than_element_size(self):
        mesh = build_mesh_from_structure(
            _circle_structure((0.0, 0.0, 0.5, 0.0, 1.0, "Cu")),
            element_size=1.0,
        )

        bottom_nodes = mesh.nodes[np.isclose(mesh.nodes[:, 2], 0.0)]
        radii = np.hypot(bottom_nodes[:, 0], bottom_nodes[:, 1])
        self.assertEqual(np.count_nonzero(np.isclose(radii, 0.5)), 19)
        self.assertTrue(np.all(radii <= 0.5 + 1.0e-10))

    def test_imprints_a_repeated_xy_circle_only_once_across_z(self):
        structure = _circle_structure(
            (0.0, 0.0, 3.0, 0.0, 1.0, "Cu"),
            (0.0, 0.0, 3.0, 1.0, 1.0, "Cu"),
        )

        with patch(
            "mesher.process_flow.pipeline.imprint_circle",
            wraps=imprint_circle,
        ) as mocked_imprint:
            mesh = build_mesh_from_structure(structure, element_size=1.0)

        self.assertEqual(mocked_imprint.call_count, 1)
        self.assertGreater(mesh.element_count, 0)

    def test_extends_a_clean_outer_concentric_circle_chain(self):
        structure = _circle_structure(
            (0.0, 0.0, 10000.0, 0.0, 1.0, "Cu"),
            (0.0, 0.0, 12000.0, 0.0, 1.0, "Cu"),
            (0.0, 0.0, 13000.0, 0.0, 1.0, "Cu"),
        )

        with (
            patch(
                "mesher.process_flow.pipeline.imprint_circle",
                wraps=imprint_circle,
            ) as mocked_imprint,
            patch(
                "mesher.process_flow.pipeline.extend_circular_mesh",
                wraps=extend_circular_mesh,
            ) as mocked_extend,
        ):
            mesh = build_mesh_from_structure(structure, element_size=500.0)

        self.assertEqual(mocked_imprint.call_count, 1)
        self.assertEqual(mocked_imprint.call_args.kwargs["radius"], 10000.0)
        self.assertEqual(mocked_extend.call_count, 2)
        self.assertEqual(
            [
                (
                    call.kwargs["inner_radius"],
                    call.kwargs["outer_radius"],
                )
                for call in mocked_extend.call_args_list
            ],
            [(10000.0, 12000.0), (12000.0, 13000.0)],
        )

        bottom_nodes = mesh.nodes[np.isclose(mesh.nodes[:, 2], 0.0)]
        radii = np.hypot(bottom_nodes[:, 0], bottom_nodes[:, 1])
        for radius in (10000.0, 12000.0, 13000.0):
            self.assertGreater(np.count_nonzero(np.isclose(radii, radius)), 0)

    def test_line_pattern_moves_the_extension_source_outward(self):
        structure = _append_box(
            _circle_structure(
                (0.0, 0.0, 10000.0, 0.0, 1.0, "Cu"),
                (0.0, 0.0, 12000.0, 0.0, 1.0, "Cu"),
                (0.0, 0.0, 13000.0, 0.0, 1.0, "Cu"),
            ),
            x1=10500.0,
            y1=-250.0,
            x2=11000.0,
            y2=250.0,
        )

        with (
            patch(
                "mesher.process_flow.pipeline.imprint_circle",
                wraps=imprint_circle,
            ) as mocked_imprint,
            patch(
                "mesher.process_flow.pipeline.extend_circular_mesh",
                wraps=extend_circular_mesh,
            ) as mocked_extend,
        ):
            build_mesh_from_structure(structure, element_size=500.0)

        self.assertEqual(
            [call.kwargs["radius"] for call in mocked_imprint.call_args_list],
            [10000.0, 12000.0],
        )
        self.assertEqual(mocked_extend.call_count, 1)
        self.assertEqual(mocked_extend.call_args.kwargs["inner_radius"], 12000.0)
        self.assertEqual(mocked_extend.call_args.kwargs["outer_radius"], 13000.0)

    def test_line_pattern_in_the_outermost_annulus_disables_extension(self):
        structure = _append_box(
            _circle_structure(
                (0.0, 0.0, 10000.0, 0.0, 1.0, "Cu"),
                (0.0, 0.0, 12000.0, 0.0, 1.0, "Cu"),
                (0.0, 0.0, 13000.0, 0.0, 1.0, "Cu"),
            ),
            x1=12400.0,
            y1=-100.0,
            x2=12600.0,
            y2=100.0,
        )

        with (
            patch(
                "mesher.process_flow.pipeline.imprint_circle",
                wraps=imprint_circle,
            ) as mocked_imprint,
            patch(
                "mesher.process_flow.pipeline.extend_circular_mesh",
                wraps=extend_circular_mesh,
            ) as mocked_extend,
        ):
            build_mesh_from_structure(structure, element_size=400.0)

        self.assertEqual(mocked_imprint.call_count, 3)
        self.assertEqual(mocked_extend.call_count, 0)

    def test_uses_one_center_for_a_tolerance_matched_extension_chain(self):
        structure = _circle_structure(
            (0.0000005, -0.0000005, 10000.0, 0.0, 1.0, "Cu"),
            (0.0, 0.0, 12000.0, 0.0, 1.0, "Cu"),
            (0.0, 0.0, 13000.0, 0.0, 1.0, "Cu"),
        )

        with patch(
            "mesher.process_flow.pipeline.extend_circular_mesh",
            wraps=extend_circular_mesh,
        ) as mocked_extend:
            mesh = build_mesh_from_structure(structure, element_size=500.0)

        self.assertGreater(mesh.element_count, 0)
        self.assertEqual(mocked_extend.call_count, 2)
        for call in mocked_extend.call_args_list:
            self.assertEqual(call.kwargs["center_x"], 0.0000005)
            self.assertEqual(call.kwargs["center_y"], -0.0000005)

    def test_wraps_extension_failure_with_both_circle_identities(self):
        structure = _circle_structure(
            (0.0, 0.0, 10000.0, 0.0, 1.0, "Cu"),
            (0.0, 0.0, 12000.0, 0.0, 1.0, "Cu"),
            (0.0, 0.0, 13000.0, 0.0, 1.0, "Cu"),
        )

        with patch(
            "mesher.process_flow.pipeline.extend_circular_mesh",
            side_effect=ValueError("invalid retained topology"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"radius=10000.*radius=12000: invalid retained topology",
            ):
                build_mesh_from_structure(structure, element_size=500.0)

    def test_rejects_intersecting_tangent_and_overlapping_circle_bands(self):
        cases = (
            ("intersecting", 3.0),
            ("tangent", 4.0),
            ("overlapping bands", 5.0),
        )
        for name, second_center_x in cases:
            with self.subTest(name=name):
                structure = _circle_structure(
                    (0.0, 0.0, 2.0, 0.0, 1.0, "Cu"),
                    (second_center_x, 0.0, 2.0, 0.0, 1.0, "Cu"),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    r"center=\(0, 0\), radius=2.*center=\([345], 0\), radius=2",
                ):
                    build_mesh_from_structure(structure, element_size=1.0)

    def test_wraps_imprint_failure_with_circle_identity(self):
        structure = _circle_structure(
            (1.0, -2.0, 3.0, 0.0, 1.0, "Cu"),
        )

        with patch(
            "mesher.process_flow.pipeline.imprint_circle",
            side_effect=ValueError("invalid topology"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"center=\(1, -2\), radius=3: invalid topology",
            ):
                build_mesh_from_structure(structure, element_size=1.0)


if __name__ == "__main__":
    unittest.main()
