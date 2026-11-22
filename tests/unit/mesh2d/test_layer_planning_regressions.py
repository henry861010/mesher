"""Public layer geometry regressions for finite guides and target outlines."""

import unittest

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh
from tests.unit.mesh2d.test_extend_quad_layer import _rectangle, _target_ids
from tests.unit.mesh2d.test_slot_repair import _c_slot


class LayerPlanningRegressions(unittest.TestCase):
    def test_snap_reach_uses_the_exact_guide_midline(self):
        below, above = np.nextafter(4., 0.), np.nextafter(4., np.inf)
        # The unequal ULPs either side of 4 make the fourth midpoint strictly
        # greater than 4 even though the floating-point mean rounds to 4.
        cases = ((below, below, below), (4., 4., 4.), (above, above, 3.5),
                 (below, above, 3.5), (np.nextafter(below, 0.), above, 4.))
        for first, last, expected in cases:
            with self.subTest(first=first, last=last):
                mesh, outline = _rectangle()
                kwargs = dict(expand_ratio=2., pattern_lines=[[[1., first], [2., last]]])
                preview = plan_quad_sections(mesh, outline, .5, **kwargs)
                result = generate_layer_adaptive_mesh(mesh, outline, .5, **kwargs)
                self.assertEqual(preview.front[:, 1].max(), expected)
                self.assertEqual(mesh.nodes[result.outline, 1].max(), expected)
                self.assertEqual(0 in result.diagnostics["snapped_patterns"], expected != 3.5)

    def test_disconnected_external_guides_do_not_change_slot_repair(self):
        baseline, outline = _c_slot()
        generate_layer_adaptive_mesh(baseline, outline, .5)
        # Both segments lie entirely beyond the final domain. Their normal
        # separation is below T, but the finite axial intervals are disjoint.
        for height in (1.5, 1.4):
            with self.subTest(height=height):
                mesh, outline = _c_slot()
                guides = [[[5., height], [6., height]],
                          [[-5., height + .0005], [-4., height + .0005]]]
                result = generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=guides)
                repair, = result.diagnostics["slot_repairs"]
                self.assertEqual(repair["pattern_chord_count"], 0)
                self.assertEqual(repair["pattern_ids"], [])
                np.testing.assert_array_equal(mesh.nodes, baseline.nodes)
                np.testing.assert_array_equal(mesh.elements, baseline.elements)

    @staticmethod
    def target_with_slope(slope):
        mesh = generate_rectilinear_mesh(1., [0., 4.], [0., 4.])
        outline = EditorMesh2D.outlines(mesh)[0]
        points = np.array([[-1, y] for y in range(-1, 5)] + [[x, 5] for x in range(-1, 5)]
                          + [[5, y] for y in range(5, -1, -1)] + [[x, -1] for x in range(5, -1, -1)],
                          dtype=float)
        top = points[:, 1] == 5
        points[top, 1] += slope * points[top, 0]
        # The affected logical edge wraps around the first supplied node.
        return mesh, outline, _target_ids(mesh, np.roll(points, -9, axis=0))

    def test_target_checks_axis_drift_over_a_complete_logical_edge(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            for fixed in (False, True):
                with self.subTest(api=api.__name__, fixed=fixed):
                    mesh, outline, target = self.target_with_slope(.0005)
                    nodes, elements = mesh.nodes, mesh.elements
                    with self.assertRaises(QuadLayerError) as caught:
                        api(mesh, outline, 1., outline_o=target, fixed_outer_boundary=fixed)
                    self.assertEqual(caught.exception.code, "invalid_input")
                    self.assertEqual(caught.exception.context["parameter"], "outline_o")
                    self.assertIs(mesh.nodes, nodes)
                    self.assertIs(mesh.elements, elements)

    def test_resolved_near_axis_target_remains_a_reusable_source(self):
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                mesh, outline, target = self.target_with_slope(.0001)
                first = generate_layer_adaptive_mesh(mesh, outline, 1., outline_o=target,
                                                     fixed_outer_boundary=fixed)
                second = generate_layer_adaptive_mesh(mesh, first.outline, 1.)
                self.assertGreater(second.diagnostics["added_elements"], 0)
