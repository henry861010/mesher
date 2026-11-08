"""Finite guides retain their local geometry across supported numeric ranges."""

import unittest
import warnings

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections


def _rectangle():
    return (Mesh2D([[0., 0.], [4., 0.], [4., 3.], [0., 3.]], [[0, 1, 2, 3]]),
            [0, 3, 2, 1])


def _grid():
    nodes = [[x, y] for y in range(4) for x in range(5)]
    cells = [[5*y+x, 5*y+x+1, 5*(y+1)+x+1, 5*(y+1)+x]
             for y in range(3) for x in range(4)]
    outline = [0, 5, 10, 15, 16, 17, 18, 19, 14, 9, 4, 3, 2, 1]
    return Mesh2D(nodes, cells), outline


class QuadLayerPatternRegressionTests(unittest.TestCase):
    @staticmethod
    def _snapshot(mesh):
        return mesh.nodes, mesh.elements, mesh.nodes.copy(), mesh.elements.copy()

    def assert_unchanged(self, mesh, snapshot):
        nodes, elements, node_values, element_values = snapshot
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def assert_frozen_prefix(self, mesh, snapshot):
        _, _, nodes, elements = snapshot
        np.testing.assert_array_equal(mesh.nodes[:len(nodes)], nodes)
        np.testing.assert_array_equal(mesh.elements[:len(elements)], elements)

    def assert_axis_covered(self, mesh, start, end, *, xy=None):
        """Independent coverage oracle using local axis coordinates, not t ratios."""
        xy = mesh.nodes[:, :2] if xy is None else xy
        start, end = np.asarray(start), np.asarray(end)
        axis = int(np.argmax(np.abs(end - start)))
        support = start[1-axis]
        low, high = sorted((float(start[axis]), float(end[axis])))
        keys = {tuple(sorted((int(a), int(b)))) for cell in mesh.elements
                for a, b in zip(cell, np.roll(cell, -1))}
        intervals = []
        for a, b in keys:
            points = xy[[a, b]]
            if np.all(np.abs(points[:, 1-axis] - support) <= 1e-8):
                left, right = sorted(points[:, axis])
                if right >= low and left <= high:
                    intervals.append((max(low, left), min(high, right)))
        reached = low
        for left, right in sorted(intervals):
            if right <= reached:
                continue
            self.assertLessEqual(left, reached + 1e-8, "guide has a gap in mesh edges")
            reached = right
        self.assertAlmostEqual(reached, high, delta=1e-8)

    def assert_atomic_error(self, mesh, outline, patterns, *, code=None, **kwargs):
        for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(operation=operation.__name__):
                snapshot = self._snapshot(mesh)
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always", RuntimeWarning)
                    with self.assertRaises(QuadLayerError) as caught:
                        operation(mesh, outline, .5, pattern_lines=patterns, **kwargs)
                self.assertFalse([w for w in emitted if issubclass(w.category, RuntimeWarning)])
                if code is not None:
                    self.assertEqual(caught.exception.code, code)
                if code in ("numeric_range_exceeded", "empty_mesh"):
                    self.assertEqual(caught.exception.context["stage"], "input_validation")
                if code == "numeric_range_exceeded":
                    self.assertEqual(caught.exception.context["pattern_index"], 0)
                    self.assertEqual(caught.exception.context["limit"], 1e6)
                self.assert_unchanged(mesh, snapshot)

    def test_long_front_guides_build_without_parameter_rounding_gaps(self):
        for extent in (1000., 1e6, 2e6):
            with self.subTest(extent=extent):
                mesh, outline = _rectangle()
                snapshot = self._snapshot(mesh)
                patterns = [[[-extent, 3.4], [extent, 3.4]]]
                plan = plan_quad_sections(mesh, outline, .5, pattern_lines=patterns)
                self.assert_unchanged(mesh, snapshot)
                np.testing.assert_allclose(plan.edge_expansion, [.5, .4, .5, .5])
                result = generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=patterns)
                self.assert_frozen_prefix(mesh, snapshot)
                front = mesh.nodes[np.asarray(result.outline), :2]
                np.testing.assert_allclose(front.min(axis=0), [-.5, -.5])
                np.testing.assert_allclose(front.max(axis=0), [4.5, 3.4])
                self.assert_axis_covered(mesh, [-.5, 3.4], [4.5, 3.4])

    def test_long_source_coincident_guide_keeps_exact_frozen_corners(self):
        mesh, outline = _rectangle()
        snapshot = self._snapshot(mesh)
        patterns = [[[-1e6, 3.], [1e6, 3.]]]
        plan = plan_quad_sections(mesh, outline, .5, pattern_lines=patterns)
        self.assert_unchanged(mesh, snapshot)
        np.testing.assert_allclose(plan.edge_expansion, [.5] * 4)
        generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=patterns)
        self.assert_frozen_prefix(mesh, snapshot)
        self.assert_axis_covered(mesh, [-.5, 3.], [4.5, 3.])

    def test_long_vertical_shared_edges_form_complete_chords(self):
        for extent in (1000., 1e6, 2e6):
            with self.subTest(extent=extent):
                mesh, outline = _grid()
                snapshot = self._snapshot(mesh)
                patterns = [[[1., -extent], [1., extent]]]
                plan = plan_quad_sections(mesh, outline, .5, pattern_lines=patterns)
                self.assertEqual(plan.diagnostics["pattern_chord_count"], 2)
                self.assert_unchanged(mesh, snapshot)
                result = generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=patterns)
                self.assert_frozen_prefix(mesh, snapshot)
                self.assert_axis_covered(mesh, [1., -.5], [1., 3.5])
                locked = [seam for seam in result.diagnostics["seam_states"]
                          if seam["pattern_ids"]]
                self.assertEqual(len(locked), 2)
                self.assertTrue(all(seam["pattern_ids"] == [0] for seam in locked))

    def test_long_guides_are_covariant_under_scale_translation_and_rotation(self):
        quarter_turn = np.array([[0., -1.], [1., 0.]])
        cases = ((1e-5, [0., 0.], np.eye(2)),
                 (17., [10000., -20000.], quarter_turn),
                 (1., [1e7, -2e7], np.eye(2)))
        canonical_patterns = np.array([[[-1000., 3.4], [1000., 3.4]],
                                       [[1., -1000.], [1., 1000.]]])
        for scale, shift, rotation in cases:
            with self.subTest(scale=scale, shift=shift, rotation=rotation.tolist()):
                mesh, outline = _grid()
                shift = np.asarray(shift)
                mesh.nodes[:, :2] = mesh.nodes[:, :2] @ rotation.T * scale + shift
                patterns = canonical_patterns @ rotation.T * scale + shift
                snapshot = self._snapshot(mesh)
                plan = plan_quad_sections(mesh, outline, .5 * scale, pattern_lines=patterns, tolerance=.001 * scale)
                self.assert_unchanged(mesh, snapshot)
                front = (plan.front - shift) @ rotation / scale
                np.testing.assert_allclose(front.min(axis=0), [-.5, -.5], atol=1e-8)
                np.testing.assert_allclose(front.max(axis=0), [4.5, 3.4], atol=1e-8)
                generate_layer_adaptive_mesh(mesh, outline, .5 * scale, pattern_lines=patterns, tolerance=.001 * scale)
                self.assert_frozen_prefix(mesh, snapshot)
                xy = (mesh.nodes[:, :2] - shift) @ rotation / scale
                self.assert_axis_covered(mesh, [-.5, 3.4], [4.5, 3.4], xy=xy)
                self.assert_axis_covered(mesh, [1., -.5], [1., 3.4], xy=xy)

    def test_partial_reversed_duplicate_guides_preserve_ids_and_finite_anchors(self):
        mesh, outline = _grid()
        patterns = np.array([[[-1000., 3.4], [2., 3.4]],
                             [[1000., 3.4], [2., 3.4]],
                             [[2., 3.4], [1000., 3.4]]])
        original = patterns.copy()
        patterns.setflags(write=False)
        snapshot = self._snapshot(mesh)
        plan = plan_quad_sections(mesh, outline, .5, pattern_lines=patterns)
        self.assert_unchanged(mesh, snapshot)
        self.assertEqual({index for section in plan.sections for index in section.pattern_ids},
                         {0, 1, 2})
        result = generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=patterns)
        self.assert_frozen_prefix(mesh, snapshot)
        np.testing.assert_array_equal(patterns, original)
        self.assertEqual({index for section in result.diagnostics["section_details"]
                          for index in section["pattern_ids"]}, {0, 1, 2})
        front = mesh.nodes[np.unique(result.outline), :2]
        self.assertEqual(np.count_nonzero(np.all(np.isclose(front, [2., 3.4],
                                                           rtol=0, atol=1e-12), axis=1)), 1)
        self.assert_axis_covered(mesh, [-.5, 3.4], [4.5, 3.4])

    def test_finite_pattern_gaps_and_interior_endpoints_are_not_extended(self):
        cases = ([[[1., 3.], [1., 3.2]], [[1., 3.3], [1., 3.5]]],
                 [[[1., 3.25], [1., 1000.]]])
        for patterns in cases:
            with self.subTest(patterns=patterns):
                mesh, outline = _grid()
                self.assert_atomic_error(mesh, outline, patterns, code="unsupported_pattern_layout")

    def test_long_guides_reject_crossings_but_accept_within_tolerance_contacts(self):
        for tilted, pattern in ((False, [[-1000., 1.5], [1000., 1.5]]),
                                (True, [[4.0002, -1000.], [4.0002, 1000.]])):
            with self.subTest(tilted=tilted):
                mesh, outline = _rectangle()
                if tilted:
                    mesh.nodes[2, 0] += .0004
                if tilted:
                    snapshot = self._snapshot(mesh)
                    generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=[pattern])
                    self.assert_frozen_prefix(mesh, snapshot)
                else:
                    self.assert_atomic_error(mesh, outline, [pattern], code="fixed_boundary_conflict")

    def test_pattern_crossing_a_remote_old_cell_is_rejected(self):
        mesh, outline = _rectangle()
        mesh.replace_data(nodes=np.vstack((mesh.nodes, [[10., 10., 0.], [11., 10., 0.],
                                                       [11., 11., 0.], [10., 11., 0.]])),
                          elements=np.vstack((mesh.elements, [4, 5, 6, 7])))
        self.assert_atomic_error(mesh, outline, [[[9., 10.5], [12., 10.5]]],
                                 code="fixed_boundary_conflict")

    def test_pattern_numeric_length_limit_is_inclusive(self):
        # The selected boundary's maximum extent S is 4, so the limit is 4e6.
        for extent, accepted in ((4e6, True), (np.nextafter(4e6, np.inf), False)):
            with self.subTest(extent=extent):
                mesh, outline = _rectangle()
                patterns = [[[0., 3.4], [extent, 3.4]]]
                if not accepted:
                    self.assert_atomic_error(mesh, outline, patterns, code="numeric_range_exceeded")
                    continue
                snapshot = self._snapshot(mesh)
                plan_quad_sections(mesh, outline, .5, pattern_lines=patterns)
                self.assert_unchanged(mesh, snapshot)
                generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=patterns)
                self.assert_frozen_prefix(mesh, snapshot)
                self.assert_axis_covered(mesh, [0., 3.4], [4.5, 3.4])

    def test_pattern_endpoint_limit_uses_distance_to_source_bbox(self):
        cases = (([[4000004., .5], [4000004., 1.5]], True),
                 ([[-4000000., .5], [-4000000., 1.5]], True),
                 ([[4000004., 4000003.], [4000004., 4000002.]], True),
                 ([[np.nextafter(4000004., np.inf), .5],
                   [np.nextafter(4000004., np.inf), 1.5]], False))
        for pattern, accepted in cases:
            with self.subTest(pattern=pattern):
                mesh, outline = _rectangle()
                if not accepted:
                    self.assert_atomic_error(mesh, outline, [pattern], code="numeric_range_exceeded")
                    continue
                snapshot = self._snapshot(mesh)
                plan = plan_quad_sections(mesh, outline, .5, pattern_lines=[pattern])
                self.assert_unchanged(mesh, snapshot)
                np.testing.assert_allclose(plan.edge_expansion, [.5] * 4)
                generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=[pattern])
                self.assert_frozen_prefix(mesh, snapshot)

    def test_extreme_finite_patterns_fail_without_overflow_warnings_or_mutation(self):
        cases = ([[[-1e160, 1.], [1e160, 1.]]],
                 [[[1.7e308, 0.], [1.7e308, 1.]]],
                 [[[-1.7e308, 3.4], [1.7e308, 3.4]]])
        for patterns in cases:
            with self.subTest(patterns=patterns):
                mesh, outline = _rectangle()
                self.assert_atomic_error(mesh, outline, patterns, code="numeric_range_exceeded")

    def test_malformed_patterns_have_typed_atomic_errors(self):
        cases = (None, [[0., 0.], [1., 0.]], [[[0., 0.], [1.]]],
                 [[[0., 0.], [10**400, 0.]]], [[[0., 0.], [np.inf, 0.]]])
        for patterns in cases:
            with self.subTest(patterns=patterns):
                mesh, outline = _rectangle()
                self.assert_atomic_error(mesh, outline, patterns)

    def test_empty_mesh_is_not_an_existing_source_boundary_in_either_direction(self):
        for clockwise in (True, False):
            mesh, outline = _rectangle()
            mesh.elements = np.empty((0, 4), dtype=int)
            if not clockwise:
                outline = outline[::-1]
            with self.subTest(clockwise=clockwise):
                self.assert_atomic_error(mesh, outline, [], code="empty_mesh")


if __name__ == "__main__":
    unittest.main()
