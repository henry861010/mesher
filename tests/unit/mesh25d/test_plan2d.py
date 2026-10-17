import unittest

import numpy as np

from mesher.mesh25d.controls import ResolvedControls
from mesher.mesh25d.domain import ModelDomain, SymmetryMode
from mesher.mesh25d.geometry.patterns import Pattern2D
from mesher.mesh25d.plan2d import build_2d_mesh, plan_2d


def box_control(bounds, size):
    return {"region": {"type": "BOX", "dim": bounds}, "element_size": size}


def box_plan(controls, *, bounds=None, global_size=20):
    pattern = Pattern2D(
        base_face={"type": "BOX", "dim": [0, 0, 20, 20] if bounds is None else bounds},
        faces=[], circle_sources={}, koz_circle_sources={},
        coordinate_tolerance=1e-6, clearance_tolerance=1e-6,
    )
    return plan_2d(
        pattern, ResolvedControls(controls, [], []), element_size=global_size,
        domain=ModelDomain(SymmetryMode.FULL),
    )


class XYGridPlanningTests(unittest.TestCase):
    def test_box_uses_section_avg_and_keeps_short_interval_endpoints(self):
        for size, expected in (
            (5, [0, 5, 10, 15, 20]),
            (6, [0, 20 / 3, 40 / 3, 20]),
            (25, [0, 20]),
        ):
            with self.subTest(size=size):
                plan = box_plan([box_control([0, 0, 20, 20], size)])
                np.testing.assert_allclose(sorted(set(plan.x_lines)), expected)
                np.testing.assert_allclose(sorted(set(plan.y_lines)), expected)
                mesh = build_2d_mesh(plan)
                np.testing.assert_allclose(np.unique(mesh.nodes[:, 0]), expected)
                np.testing.assert_allclose(np.unique(mesh.nodes[:, 1]), expected)

    def test_polygon_subdivides_between_all_loop_vertices(self):
        control = {"region": {"type": "POLYGON", "dim": [
            [[0, 0], [12, 0], [0, 12], [0, 0], [0, 0]],
            [[2, 1], [5, 1], [2, 5], [2, 1]],
        ]}, "element_size": 3}
        plan = box_plan([control], bounds=[0, 0, 12, 12])
        mesh = build_2d_mesh(plan)
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 0]), [0, 2, 5, 8.5, 12])
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 1]), [0, 1, 5, 8.5, 12])

    def test_overlapping_controls_union_independent_cuts_in_either_order(self):
        controls = [box_control([0, 0, 12, 12], 4), box_control([3, 3, 15, 15], 4)]
        expected = [0, 3, 4, 7, 8, 11, 12, 15, 20]
        for entries in (controls, controls[::-1], controls + controls):
            with self.subTest(entries=entries):
                mesh = build_2d_mesh(box_plan(entries))
                np.testing.assert_allclose(np.unique(mesh.nodes[:, 0]), expected)
                np.testing.assert_allclose(np.unique(mesh.nodes[:, 1]), expected)

    def test_global_size_fills_remaining_gaps_after_control_cuts(self):
        mesh = build_2d_mesh(box_plan(
            [box_control([0, 0, 20, 20], 6)], global_size=5,
        ))
        expected = np.linspace(0, 20, 7)
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 0]), expected)
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 1]), expected)

    def test_control_does_not_refine_the_entire_model_to_its_size(self):
        mesh = build_2d_mesh(box_plan(
            [box_control([8, 8, 12, 12], 1)], global_size=5,
        ))
        expected = [0, 4, 8, 9, 10, 11, 12, 16, 20]
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 0]), expected)
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 1]), expected)

    def test_partial_region_is_subdivided_before_clipping_to_model_bounds(self):
        mesh = build_2d_mesh(box_plan(
            [box_control([-5, -5, 15, 15], 6)], bounds=[0, 0, 10, 10],
        ))
        expected = [0, 5 / 3, 25 / 3, 10]
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 0]), expected)
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 1]), expected)

    def test_disjoint_regions_do_not_refine_either_axis(self):
        for bounds in ([30, 0, 40, 20], [0, 30, 20, 40], [-20, 0, -10, 20]):
            with self.subTest(bounds=bounds):
                mesh = build_2d_mesh(box_plan([box_control(bounds, 1)]))
                np.testing.assert_array_equal(np.unique(mesh.nodes[:, 0]), [0, 20])
                np.testing.assert_array_equal(np.unique(mesh.nodes[:, 1]), [0, 20])

    def test_circle_support_halo_does_not_expand_control_clipping_bounds(self):
        pattern = Pattern2D(
            base_face={"type": "CIRCLE", "dim": [0, 0, 1]}, faces=[],
            circle_sources={}, koz_circle_sources={},
            coordinate_tolerance=1e-6, clearance_tolerance=1e-6,
        )
        plans = [plan_2d(
            pattern, ResolvedControls(controls, [], []), element_size=1,
            domain=ModelDomain(SymmetryMode.FULL),
        ) for controls in ([], [box_control([1.2, -0.5, 1.5, 0.5], 0.1)])]
        self.assertEqual(plans[0].x_lines, plans[1].x_lines)
        self.assertEqual(plans[0].y_lines, plans[1].y_lines)


if __name__ == "__main__":
    unittest.main()
