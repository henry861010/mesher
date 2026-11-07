"""Behavioral coverage for opt-in, local XY region meshing."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh25d.controls import ResolvedControls
from mesher.mesh25d.domain import ModelDomain, SymmetryMode
from mesher.mesh25d.geometry.patterns import Pattern2D
from mesher.mesh25d.plan2d import build_2d_mesh, plan_2d


def _box(bounds, size):
    return {"region": {"type": "BOX", "dim": list(bounds)}, "element_size": size}


def _plan(controls=(), *, size=2, bounds=(-12, -12, 12, 12),
          strategy="adapter", faces=(), tolerance=1e-6):
    pattern = Pattern2D(
        {"type": "BOX", "dim": list(bounds)}, list(faces), {}, {},
        tolerance, tolerance,
    )
    return plan_2d(
        pattern,
        ResolvedControls(list(controls), [], [f"controls[{i}] ('Region {i}')"
                                             for i in range(len(controls))]),
        element_size=size, domain=ModelDomain(SymmetryMode.FULL),
        region_meshing=strategy,
    )


class RegionAdapterTests(unittest.TestCase):
    def assert_complete(self, mesh, bounds):
        """Check area, positive cells, and shared topology independently."""
        self.assertTrue(np.all(mesh.element_types == 4))
        points = mesh.nodes[mesh.elements, :2]
        edges = np.roll(points, -1, axis=1) - points
        next_edges = np.roll(edges, -1, axis=1)
        turns = edges[..., 0] * next_edges[..., 1] - edges[..., 1] * next_edges[..., 0]
        self.assertTrue(np.all(turns > 0))
        area = 0.5 * np.sum(
            points[..., 0] * np.roll(points[..., 1], -1, axis=1)
            - points[..., 1] * np.roll(points[..., 0], -1, axis=1)
        )
        x0, y0, x1, y1 = bounds
        self.assertAlmostEqual(area, (x1 - x0) * (y1 - y0), places=8)
        directed = Counter((int(a), int(b)) for cell in mesh.elements
                           for a, b in zip(cell, np.roll(cell, -1)))
        incidence = Counter()
        for edge, count in directed.items():
            incidence[tuple(sorted(edge))] += count
        self.assertTrue(all(count in (1, 2) for count in incidence.values()))
        for (a, b), count in incidence.items():
            if count == 2:
                self.assertEqual((directed[a, b], directed[b, a]), (1, 1))
            else:
                edge = mesh.nodes[[a, b], :2]
                self.assertTrue(any(np.all(np.isclose(edge[:, axis], value))
                                    for axis, value in ((0, x0), (0, x1), (1, y0), (1, y1))),
                                "an internal boundary exposes a hole or unmatched interface")

    def test_box_seed_stays_fine_without_propagating_fine_cuts_to_outer_boundary(self):
        controls = [_box((-2, -2, 2, 2), 0.5)]
        plan = _plan(controls)
        mesh = build_2d_mesh(plan)
        self.assert_complete(mesh, (-12, -12, 12, 12))
        points = mesh.nodes[mesh.elements, :2]
        inside = np.all(np.abs(points) <= 2 + 1e-10, axis=(1, 2))
        self.assertEqual(np.count_nonzero(inside), 64)
        lengths = np.linalg.norm(np.roll(points[inside], -1, axis=1) - points[inside], axis=2)
        self.assertLessEqual(np.max(lengths), 0.5 + 1e-10)
        adaptive_boundary = mesh.nodes[np.isclose(mesh.nodes[:, 1], -12), 0]
        grid = build_2d_mesh(_plan(controls, strategy="grid"))
        grid_boundary = grid.nodes[np.isclose(grid.nodes[:, 1], -12), 0]
        self.assertLess(len(adaptive_boundary), len(grid_boundary))

    def test_each_region_has_its_own_fivefold_schedule_capped_at_planar_size(self):
        plan = _plan([
            _box((-90, -2, -86, 2), 0.1),
            _box((-2, -2, 2, 2), 0.4),
            _box((86, -2, 90, 2), 8),
        ], size=5, bounds=(-120, -120, 120, 120))
        self.assertIsNotNone(plan.adaptive)
        self.assertEqual([r.seed_size for r in plan.adaptive.regions], [0.1, 0.4, 5])
        self.assertEqual([r.transition_sizes for r in plan.adaptive.regions],
                         [(0.5, 2.5, 5), (2, 5), (5,)])
        self.assertIn("controls[1]", plan.adaptive.regions[1].source)

    def test_different_region_schedules_build_one_conforming_mesh(self):
        plan = _plan([
            _box((-12, -2, -8, 2), 0.2), _box((8, -2, 12, 2), 0.4),
        ], bounds=(-24, -12, 24, 12))
        events = []
        mesh = build_2d_mesh(plan, progress=events.append)
        self.assert_complete(mesh, (-24, -12, 24, 12))
        points = mesh.nodes[mesh.elements, :2]
        for (x0, y0, x1, y1), expected_size in (((-12, -2, -8, 2), 0.2), ((8, -2, 12, 2), 0.4)):
            inside = np.all((points >= [x0 - 1e-10, y0 - 1e-10])
                            & (points <= [x1 + 1e-10, y1 + 1e-10]), axis=(1, 2))
            self.assertGreater(np.count_nonzero(inside), 0)
            lengths = np.linalg.norm(np.roll(points[inside], -1, axis=1) - points[inside], axis=2)
            self.assertLessEqual(np.max(lengths), expected_size + 1e-10)
        layers = [event["data"] for event in events if event["event"] == "item.completed"
                  and event["data"].get("operation") == "adapter_layer"]
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0]["targetSize"], 1)
        self.assertIn("controls[0]", layers[0]["sourceRefs"][0])

    def test_circle_reduced_planar_size_caps_the_transition_schedule(self):
        plan = _plan([_box((-7, -1, -5, 1), 0.1)], size=10,
                     faces=[{"type": "CIRCLE", "dim": [5, 0, 3]}])
        self.assertEqual(plan.element_size, 1)
        self.assertEqual(plan.adaptive.regions[0].transition_sizes, (0.5, 1))

    def test_no_controls_and_wholly_outside_controls_preserve_grid_result(self):
        for controls in ([], [_box((30, -2, 34, 2), 0.1)]):
            with self.subTest(controls=controls):
                plan = _plan(controls)
                self.assertIsNone(plan.adaptive)
                adapter = build_2d_mesh(plan)
                grid = build_2d_mesh(_plan(controls, strategy="grid"))
                np.testing.assert_array_equal(adapter.nodes, grid.nodes)
                np.testing.assert_array_equal(adapter.elements, grid.elements)

    def test_orthogonal_polygon_hole_is_filled_and_loop_order_winding_are_irrelevant(self):
        outer = [[-6, -6], [6, -6], [6, 6], [-6, 6]]
        hole = [[-3, -3], [3, -3], [3, 3], [-3, 3]]
        for loops in ([outer, hole], [hole[::-1], outer[::-1]]):
            with self.subTest(loops=loops):
                control = {"region": {"type": "POLYGON", "dim": loops}, "element_size": 0.5}
                mesh = build_2d_mesh(_plan([control], size=1))
                self.assert_complete(mesh, (-12, -12, 12, 12))
                centers = mesh.nodes[mesh.elements, :2].mean(axis=1)
                self.assertTrue(np.any(np.all(np.abs(centers) < 1, axis=1)),
                                "a sizing-region hole is not a physical model void")

    def test_concave_orthogonal_polygon_builds_a_complete_mesh(self):
        control = {"region": {"type": "POLYGON", "dim": [[
            [-3, -3], [3, -3], [3, 0], [0, 0], [0, 3], [-3, 3],
        ]]}, "element_size": 0.5}
        mesh = build_2d_mesh(_plan([control], size=1))
        self.assert_complete(mesh, (-12, -12, 12, 12))

    def test_one_polygon_control_supports_disconnected_parts_and_nested_islands(self):
        cases = (
            [
                [[-8, -2], [-4, -2], [-4, 2], [-8, 2]],
                [[4, -2], [8, -2], [8, 2], [4, 2]],
            ],
            [
                [[-8, -8], [8, -8], [8, 8], [-8, 8]],
                [[-5, -5], [5, -5], [5, 5], [-5, 5]],
                [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            ],
        )
        for loops in cases:
            with self.subTest(loop_count=len(loops)):
                control = {"region": {"type": "POLYGON", "dim": loops}, "element_size": 0.5}
                mesh = build_2d_mesh(_plan([control], size=1, bounds=(-14, -14, 14, 14)))
                self.assert_complete(mesh, (-14, -14, 14, 14))
                points = mesh.nodes[mesh.elements, :2]
                # Each outer component remains a fine seed after completion.
                focus = (-8, -2, -4, 2) if len(loops) == 2 else (-1, -1, 1, 1)
                x0, y0, x1, y1 = focus
                inside = np.all((points >= [x0 - 1e-10, y0 - 1e-10])
                                & (points <= [x1 + 1e-10, y1 + 1e-10]), axis=(1, 2))
                self.assertGreater(np.count_nonzero(inside), 0)
                lengths = np.linalg.norm(np.roll(points[inside], -1, axis=1) - points[inside], axis=2)
                self.assertLessEqual(np.max(lengths), 0.5 + 1e-10)

    def test_unsupported_regions_report_control_source_without_falling_back(self):
        cases = (
            [_box((-12, -2, -8, 2), 0.5)],
            [_box((-14, -2, -10, 2), 0.5)],
            [_box((-2, -2, 2, 2), 0.5), _box((2, -2, 6, 2), 0.5)],
            [_box((-2, -2, 2, 2), 0.5), _box((1, -2, 5, 2), 0.5)],
            [{"region": {"type": "POLYGON", "dim": [[[-2, -2], [2, -2], [0, 2]]]},
              "element_size": 0.5}],
        )
        for controls in cases:
            with self.subTest(controls=controls):
                with self.assertRaisesRegex(ValueError, r"controls\["):
                    build_2d_mesh(_plan(controls))

    def test_adapter_growth_collision_is_an_error_instead_of_grid_retry(self):
        controls = [_box((-2, -2, 2, 2), 0.5), _box((3, -2, 7, 2), 0.5)]
        with self.assertRaises(ValueError):
            build_2d_mesh(_plan(controls))

    def test_final_failure_keeps_typed_error_and_maps_boundary_to_its_control(self):
        ring = {"region": {"type": "POLYGON", "dim": [
            [[-9, -3], [-3, -3], [-3, 3], [-9, 3]],
            [[-7, -1], [-5, -1], [-5, 1], [-7, 1]],
        ]}, "element_size": 0.5}
        plan = _plan([ring, _box((4, -2, 8, 2), 0.5)], size=1)
        failure = QuadLayerError("injected fixed-boundary failure", code="test_fixed_boundary",
                                 stage="final_layer_construction", detail="preserved")

        def reject_final(mesh, *args, **kwargs):
            # The ring contributes two boundaries, so control position cannot
            # be used directly as the adapter boundary index.
            outlines = EditorMesh2D.outlines(mesh)
            for index, outline in enumerate(outlines):
                node = outline[0]
                if mesh.nodes[node, 0] > 0:
                    failure.context["boundary_index"] = index
                    break
            else:
                self.fail("the second control has no exposed seed boundary")
            raise failure

        with patch("mesher.mesh25d.region_adapter.generate_full_adaptive_mesh", side_effect=reject_final) as full:
            with patch("mesher.mesh25d.plan2d.generate_rectilinear_mesh") as grid:
                with self.assertRaises(QuadLayerError) as caught:
                    build_2d_mesh(plan)
        self.assertIs(caught.exception, failure)
        self.assertEqual(failure.code, "test_fixed_boundary")
        self.assertEqual(failure.context["stage"], "final_layer_construction")
        self.assertEqual(failure.context["detail"], "preserved")
        self.assertEqual(failure.context["region_sources"], ["controls[1] ('Region 1')"])
        self.assertEqual(failure.context["region_layer_indices"], {"controls[1] ('Region 1')": 0})
        self.assertIn("injected fixed-boundary failure", str(failure))
        full.assert_called_once()
        grid.assert_not_called()

    def test_intermediate_collision_reports_region_layer_index_without_retry(self):
        plan = _plan([_box((-6, -2, -2, 2), 0.2), _box((-1, -2, 3, 2), 0.4)])
        with patch("mesher.mesh25d.region_adapter.generate_full_adaptive_mesh") as full:
            with patch("mesher.mesh25d.plan2d.generate_rectilinear_mesh") as grid:
                with self.assertRaises(QuadLayerError) as caught:
                    build_2d_mesh(plan)
        failure = caught.exception
        self.assertEqual(failure.context["stage"], "intermediate_validation")
        self.assertEqual(failure.context["region_layer_index"], 0)
        self.assertEqual(failure.context["region_sources"],
                         ["controls[0] ('Region 0')", "controls[1] ('Region 1')"])
        self.assertIn("region_layer_index", str(failure))
        self.assertEqual(str(failure).count("3D_REGION adapter"), 1)
        full.assert_not_called()
        grid.assert_not_called()

    def test_oblique_material_boundary_is_rejected_instead_of_silently_omitted(self):
        face = {"type": "POLYGON", "dim": [[[4, 4], [7, 4], [4, 7]]]}
        with self.assertRaisesRegex(ValueError, "geometry"):
            _plan([_box((-2, -2, 2, 2), 0.5)], faces=[face])


if __name__ == "__main__":
    unittest.main()
