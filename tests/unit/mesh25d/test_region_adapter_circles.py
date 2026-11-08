"""Circular extension must preserve active adapter refinement regions."""

import unittest

import numpy as np

from mesher.mesh25d.controls import ResolvedControls
from mesher.mesh25d.domain import ModelDomain, SymmetryMode
from mesher.mesh25d.geometry.patterns import Pattern2D
from mesher.mesh25d.plan2d import build_2d_mesh, plan_2d


def _plan(bounds=None):
    pattern = Pattern2D(
        {"type": "CIRCLE", "dim": [0, 0, 20]},
        [{"type": "CIRCLE", "dim": [0, 0, 5]}],
        {}, {}, 0.001,
    )
    xy = ([] if bounds is None else [
        {"region": {"type": "BOX", "dim": list(bounds)}, "element_size": 0.1},
    ])
    return plan_2d(
        pattern, ResolvedControls(xy, [], ["controls[0] ('Circular region')"] if xy else []),
        element_size=1, domain=ModelDomain(SymmetryMode.FULL), region_meshing="adapter",
    )


class RegionAdapterCircleTests(unittest.TestCase):
    def assert_fine_region_preserved(self, mesh, bounds, minimum_cells):
        x0, y0, x1, y1 = bounds
        points = mesh.nodes[mesh.elements, :2]
        inside = np.all((points >= [x0 - 1e-10, y0 - 1e-10])
                        & (points <= [x1 + 1e-10, y1 + 1e-10]), axis=(1, 2))
        # Mandatory support cuts may subdivide the seed beyond its size target.
        self.assertGreaterEqual(np.count_nonzero(inside), minimum_cells)
        cells = points[inside]
        lengths = np.linalg.norm(np.roll(cells, -1, axis=1) - cells, axis=2)
        self.assertLessEqual(float(np.max(lengths)), 0.1 + 1e-10)
        area = 0.5 * np.sum(
            cells[..., 0] * np.roll(cells[..., 1], -1, axis=1)
            - cells[..., 1] * np.roll(cells[..., 0], -1, axis=1)
        )
        self.assertAlmostEqual(float(area), (x1 - x0) * (y1 - y0), places=9)

    def test_annulus_refinement_is_not_replaced_by_circular_extension(self):
        bounds = (6, -1, 6.5, 1)
        plan = _plan(bounds)
        self.assertEqual(plan.circle_plan.extensions, ())
        self.assertEqual([circle.radius for circle in plan.circle_plan.imprint_patterns], [5, 20])
        self.assert_fine_region_preserved(build_2d_mesh(plan), bounds, minimum_cells=100)

    def test_annulus_refinement_outside_optimized_inner_grid_can_be_built(self):
        bounds = (10, -1, 12, 1)
        plan = _plan(bounds)
        self.assertEqual(plan.circle_plan.extensions, ())
        self.assertLess(min(plan.x_lines), bounds[0])
        self.assertGreater(max(plan.x_lines), bounds[2])
        self.assert_fine_region_preserved(build_2d_mesh(plan), bounds, minimum_cells=400)

    def test_no_active_region_keeps_circular_extension_optimization(self):
        for bounds in (None, (30, -1, 32, 1)):
            with self.subTest(bounds=bounds):
                plan = _plan(bounds)
                self.assertIsNone(plan.adaptive)
                self.assertEqual([circle.radius for circle in plan.circle_plan.imprint_patterns], [5])
                self.assertEqual([(item.inner.radius, item.outer.radius)
                                  for item in plan.circle_plan.extensions], [(5, 20)])

    def test_region_inside_inner_circle_keeps_extension_and_local_refinement(self):
        bounds = (-1, -1, 1, 1)
        plan = _plan(bounds)
        self.assertIsNotNone(plan.adaptive)
        self.assertEqual([(item.inner.radius, item.outer.radius)
                          for item in plan.circle_plan.extensions], [(5, 20)])
        self.assert_fine_region_preserved(build_2d_mesh(plan), bounds, minimum_cells=400)


if __name__ == "__main__":
    unittest.main()
