"""Mandatory full-domain cuts survive every planar meshing stage."""

from collections import defaultdict
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.model import Mesh2D
from mesher.mesh25d.controls import ResolvedControls
from mesher.mesh25d.domain import ModelDomain, SymmetryMode
from mesher.mesh25d.geometry.context import prepare_geometry
from mesher.mesh25d.geometry.patterns import Pattern2D
from mesher.mesh25d.plan2d import build_2d_mesh, plan_2d
from mesher.tolerance import GeometryToleranceError


def assert_chain(test, nodes, cells, axis, value, lower, upper):
    """Require a shared-node edge path, rather than isolated collinear nodes."""
    adjacent = defaultdict(set)
    for cell in cells:
        for a, b in zip(cell, np.roll(cell, -1)):
            if a != b and np.all(np.abs(nodes[[a, b], axis] - value) < 1e-9):
                adjacent[int(a)].add(int(b))
                adjacent[int(b)].add(int(a))
    starts = [i for i in adjacent if abs(nodes[i, 1 - axis] - lower) < 1e-9]
    test.assertTrue(starts, (axis, value, lower))
    reached, pending = set(starts), list(starts)
    while pending:
        for other in adjacent[pending.pop()] - reached:
            reached.add(other)
            pending.append(other)
    test.assertTrue(any(abs(nodes[i, 1 - axis] - upper) < 1e-9 for i in reached),
                    (axis, value, lower, upper))


def _plan(*, bounds=(-10, -10, 10, 10), faces=(), controls=(), strategy="grid",
          center=None, base=None, size=1):
    return plan_2d(
        Pattern2D(base or {"type": "BOX", "dim": list(bounds)}, list(faces), {}, {}, 0.001),
        ResolvedControls(list(controls), [], ["controls[0]"] if controls else []),
        element_size=size, region_meshing=strategy,
        domain=ModelDomain(SymmetryMode.FULL, *(center or (None, None))),
    )


def _body(bounds):
    x0, y0, x1, y1 = bounds
    return {"geometry": {"type": "BoxGeometry", "bottom_left": [x0, y0, 0],
                         "top_right": [x1, y1, 0], "thk": 1}, "material": "Si"}


class FullCenterlineTests(unittest.TestCase):
    def assert_box_lines(self, mesh, bounds):
        x0, y0, x1, y1 = bounds
        assert_chain(self, mesh.nodes, mesh.elements, 0, (x0 + x1) / 2, y0, y1)
        assert_chain(self, mesh.nodes, mesh.elements, 1, (y0 + y1) / 2, x0, x1)

    def test_shifted_box_with_odd_original_subdivision_count(self):
        bounds = (11, 21, 20, 30)
        mesh = build_2d_mesh(_plan(bounds=bounds, size=3))
        self.assert_box_lines(mesh, bounds)

    def test_root_body_bounds_take_precedence_without_filtering(self):
        root = {"bodies": [_body((10, 20, 20, 30))],
                "children": [{"bodies": [_body((40, 50, 50, 60))]}]}
        original = deepcopy(root)
        geometry = prepare_geometry(root, SymmetryMode.FULL)
        self.assertEqual((geometry.domain.center_x, geometry.domain.center_y), (15, 25))
        self.assertEqual(len(geometry.container["children"]), 1)
        self.assertEqual(root, original)

    def test_no_root_bodies_uses_all_pattern_bounds(self):
        geometry = prepare_geometry(
            {"children": [{"bodies": [_body((10, 20, 20, 30))]}]}, SymmetryMode.FULL,
        )
        self.assertEqual((geometry.domain.center_x, geometry.domain.center_y), (15, 25))

    def test_existing_center_cut_is_not_duplicated(self):
        control = {"region": {"type": "BOX", "dim": [-2, -2, 2, 2]}, "element_size": 1}
        mesh = build_2d_mesh(_plan(controls=[control]))
        self.assert_box_lines(mesh, (-10, -10, 10, 10))
        self.assertEqual(len(np.unique(mesh.nodes, axis=0)), len(mesh.nodes))

    def test_nearby_grid_cut_uses_exact_center_anchor(self):
        plan = _plan(faces=[{"type": "BOX", "dim": [-0.0005, -2, 2, 2]}])
        mesh = build_2d_mesh(plan)
        self.assert_box_lines(mesh, (-10, -10, 10, 10))

    def test_adapter_seed_and_transitions_preserve_both_lines(self):
        control = {"region": {"type": "BOX", "dim": [-2.5, -1.5, 2.5, 1.5]},
                   "element_size": 0.2}
        mesh = build_2d_mesh(_plan(controls=[control], strategy="adapter", size=2))
        self.assert_box_lines(mesh, (-10, -10, 10, 10))

    def test_centered_and_offset_imprints_keep_connected_edges(self):
        for center in ((0, 0), (1, 0), (0.3, 0.7)):
            with self.subTest(center=center):
                mesh = build_2d_mesh(_plan(faces=[{"type": "CIRCLE", "dim": [*center, 3]}]))
                self.assert_box_lines(mesh, (-10, -10, 10, 10))

    def test_centered_extension_keeps_radial_center_edges(self):
        plan = _plan(base={"type": "CIRCLE", "dim": [4, 7, 12]},
                     faces=[{"type": "CIRCLE", "dim": [4, 7, 3]}])
        self.assertEqual(len(plan.circle_plan.extensions), 1)
        mesh = build_2d_mesh(plan)
        assert_chain(self, mesh.nodes, mesh.elements, 0, 4, -5, 19)
        assert_chain(self, mesh.nodes, mesh.elements, 1, 7, -8, 16)

    def test_off_center_extension_replans_to_imprint_only(self):
        plan = _plan(base={"type": "CIRCLE", "dim": [0, 0, 12]},
                     faces=[{"type": "CIRCLE", "dim": [0, 0, 3]}], center=(1, 1))
        self.assertEqual(plan.circle_plan.extensions, ())
        mesh = build_2d_mesh(plan)
        points = mesh.nodes[mesh.elements, :2]
        self.assertFalse(np.any((points.min(axis=1) < 1 - 1e-9)
                                & (points.max(axis=1) > 1 + 1e-9)))
        for axis in (0, 1):
            on_line = mesh.nodes[np.abs(mesh.nodes[:, axis] - 1) < 1e-9, 1 - axis]
            assert_chain(self, mesh.nodes, mesh.elements, axis, 1, on_line.min(), on_line.max())

    def test_small_center_offset_does_not_grow_during_radial_extension(self):
        plan = _plan(base={"type": "CIRCLE", "dim": [0, 0, 12]},
                     faces=[{"type": "CIRCLE", "dim": [0, 0, 3]}], center=(0.0005, 0.0005))
        self.assertEqual(plan.circle_plan.extensions, ())
        mesh = build_2d_mesh(plan)
        for axis in (0, 1):
            on_line = mesh.nodes[np.abs(mesh.nodes[:, axis] - 0.0005) < 1e-9, 1 - axis]
            assert_chain(self, mesh.nodes, mesh.elements, axis, 0.0005,
                         on_line.min(), on_line.max())

    def test_lost_centerline_raises_instead_of_returning_invalid_mesh(self):
        broken = Mesh2D([[-10, -10], [10, -10], [10, 10], [-10, 10]], [[0, 1, 2, 3]])
        with patch("mesher.mesh25d.plan2d.generate_rectilinear_mesh", return_value=broken):
            with self.assertRaises(GeometryToleranceError) as caught:
                build_2d_mesh(_plan())
        self.assertEqual(caught.exception.code, "fixed_constraint_conflict")
        self.assertEqual(caught.exception.context["stage"], "centerline_validation")

    def test_reduced_domains_only_add_their_existing_symmetry_boundaries(self):
        pattern = Pattern2D({"type": "BOX", "dim": [-9, -9, 9, 9]}, [], {}, {}, 0.001)
        for mode, xs, ys in (
            (SymmetryMode.UPPER_HALF, [-9, 9], [0, 9]),
            (SymmetryMode.RIGHT_HALF, [0, 9], [-9, 9]),
            (SymmetryMode.UPPER_RIGHT_QUARTER, [0, 9], [0, 9]),
        ):
            with self.subTest(mode=mode):
                plan = plan_2d(pattern, ResolvedControls([], [], []), element_size=3,
                               domain=ModelDomain(mode, 0, 0))
                self.assertEqual(sorted(set(plan.x_lines)), xs)
                self.assertEqual(sorted(set(plan.y_lines)), ys)
                self.assertIsNone(plan.center)


if __name__ == "__main__":
    unittest.main()
