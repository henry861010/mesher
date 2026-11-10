"""Mandatory full- and half-domain cuts survive every planar meshing stage."""

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
          center=None, base=None, size=1, mode=SymmetryMode.FULL):
    return plan_2d(
        Pattern2D(base or {"type": "BOX", "dim": list(bounds)}, list(faces), {}, {}, 0.001),
        ResolvedControls(list(controls), [], ["controls[0]"] if controls else []),
        element_size=size, region_meshing=strategy,
        domain=ModelDomain(mode, *(center or (None, None))),
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

    def test_half_domains_add_the_missing_center_cut_and_quarter_keeps_boundaries(self):
        pattern = Pattern2D({"type": "BOX", "dim": [-9, -9, 9, 9]}, [], {}, {}, 0.001)
        for mode, xs, ys in (
            (SymmetryMode.UPPER_HALF, [-9, 0, 9], [0, 9]),
            (SymmetryMode.RIGHT_HALF, [0, 9], [-9, 0, 9]),
            (SymmetryMode.UPPER_RIGHT_QUARTER, [0, 9], [0, 9]),
        ):
            with self.subTest(mode=mode):
                plan = plan_2d(pattern, ResolvedControls([], [], []), element_size=3,
                               domain=ModelDomain(mode, 0, 0))
                self.assertEqual(sorted(set(plan.x_lines)), xs)
                self.assertEqual(sorted(set(plan.y_lines)), ys)
                if mode is SymmetryMode.UPPER_RIGHT_QUARTER:
                    self.assertIsNone(plan.center)
                else:
                    self.assertEqual(plan.center, (0, 0))
                    expected = (((0, 0), (0, 9)) if mode is SymmetryMode.UPPER_HALF
                                else ((0, 0), (9, 0)))
                    self.assertIn(expected, plan.pattern_segments)

    def test_half_centerlines_survive_grid_adapter_and_imprint(self):
        for mode, axis in ((SymmetryMode.UPPER_HALF, 0), (SymmetryMode.RIGHT_HALF, 1)):
            region = [-2.5, 8, 2.5, 10] if axis == 0 else [8, -2.5, 10, 2.5]
            circle = [0.7, 5, 3] if axis == 0 else [5, 0.7, 3]
            for kind in ("grid", "adapter", "imprint", "nearby_cut"):
                with self.subTest(mode=mode, kind=kind):
                    kwargs = dict(mode=mode, center=(0, 0), size=2,
                                  bounds=(-20, -20, 20, 20))
                    if kind == "adapter":
                        kwargs.update(strategy="adapter", controls=[
                            {"region": {"type": "BOX", "dim": region}, "element_size": 0.2}])
                    elif kind == "imprint":
                        kwargs["faces"] = [{"type": "CIRCLE", "dim": circle}]
                    elif kind == "nearby_cut":
                        dimensions = [-0.0005, 4, 2, 6] if axis == 0 else [4, -0.0005, 6, 2]
                        kwargs["faces"] = [{"type": "BOX", "dim": dimensions}]
                    mesh = build_2d_mesh(_plan(**kwargs))
                    assert_chain(self, mesh.nodes, mesh.elements, axis, 0, 0, 20)
                    self.assertEqual(len(np.unique(mesh.nodes, axis=0)), len(mesh.nodes))

    def test_half_extensions_keep_centerline_or_replan_for_offset_center(self):
        for mode, axis in ((SymmetryMode.UPPER_HALF, 0), (SymmetryMode.RIGHT_HALF, 1)):
            for offset in (0, 1, 0.0005):
                with self.subTest(mode=mode, offset=offset):
                    center = (offset, 0) if axis == 0 else (0, offset)
                    plan = _plan(mode=mode, center=center,
                                 base={"type": "CIRCLE", "dim": [0, 0, 12]},
                                 faces=[{"type": "CIRCLE", "dim": [0, 0, 3]}])
                    self.assertEqual(len(plan.circle_plan.extensions), int(offset == 0))
                    mesh = build_2d_mesh(plan)
                    on_line = mesh.nodes[np.abs(mesh.nodes[:, axis] - offset) < 1e-9, 1 - axis]
                    assert_chain(self, mesh.nodes, mesh.elements, axis, offset,
                                 on_line.min(), on_line.max())

    def test_half_centerline_uses_domain_center_and_is_validated(self):
        for mode, axis in ((SymmetryMode.UPPER_HALF, 0), (SymmetryMode.RIGHT_HALF, 1)):
            with self.subTest(mode=mode):
                center = (2, 0) if axis == 0 else (0, 2)
                plan = _plan(mode=mode, center=center)
                mesh = build_2d_mesh(plan)
                assert_chain(self, mesh.nodes, mesh.elements, axis, 2, 0, 10)
                nodes = [[-10, 0], [10, 0], [10, 10], [-10, 10]]
                if axis == 1:
                    nodes = np.asarray(nodes)[:, ::-1]
                broken = Mesh2D(nodes, [[0, 1, 2, 3]])
                with patch("mesher.mesh25d.plan2d.generate_rectilinear_mesh", return_value=broken):
                    with self.assertRaises(GeometryToleranceError) as caught:
                        build_2d_mesh(plan)
                self.assertEqual(caught.exception.context["stage"], "centerline_validation")


if __name__ == "__main__":
    unittest.main()
