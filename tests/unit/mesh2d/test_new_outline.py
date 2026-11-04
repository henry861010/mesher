"""User-defined target fronts preserve geometry and every supplied boundary point."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.adapter.layer import _mesher
from tests.unit.mesh2d import test_extend_quad_layer as layer_checks
from tests.unit.mesh2d.test_corner_repair import _step
from tests.unit.mesh2d.test_slot_repair import _c_slot


def _target():
    return np.array([[-.7, -.4], [-.7, 3.8], [4.9, 3.8], [4.9, -.4]])


class NewOutlineTests(unittest.TestCase):
    assert_layer = layer_checks.ExtendQuadLayerTests.assert_layer
    assert_pattern_covered = layer_checks.ExtendQuadLayerTests.assert_pattern_covered
    assert_unchanged = layer_checks.ExtendQuadLayerTests.assert_unchanged
    run_layer = layer_checks.ExtendQuadLayerTests.run_layer
    _snapshot = staticmethod(layer_checks.ExtendQuadLayerTests._snapshot)

    def assert_target(self, mesh, result, target):
        target = np.asarray(target, dtype=float)
        if np.array_equal(target[0], target[-1]):
            target = target[:-1]
        front = mesh.nodes[layer_checks._outline_nodes(result.outline), :2]
        scale = float(np.max(np.ptp(target, axis=0)))
        origin = target[0]
        normalized_target = (target - origin) / scale
        normalized_front = (front - origin) / scale
        for point in normalized_target:
            matches = np.linalg.norm(normalized_front - point, axis=1) <= 1e-10
            self.assertEqual(np.count_nonzero(matches), 1,
                             "every supplied target point must be one front node")
        # Check the entire returned boundary, not only its extrema or vertices.
        for point in normalized_front:
            on_target = False
            for first, second in zip(normalized_target, np.roll(normalized_target, -1, axis=0)):
                vector, delta = second - first, point - first
                projection = float(delta @ vector / (vector @ vector))
                if (abs(layer_checks._cross(vector, delta)) <= 1e-10
                        and -1e-10 <= projection <= 1 + 1e-10):
                    on_target = True
                    break
            self.assertTrue(on_target, "the generated front must follow the supplied polygon")

    def test_none_retains_automatic_extension(self):
        automatic, outline = layer_checks._rectangle()
        explicit, _ = layer_checks._rectangle()
        first = generate_layer_adaptive_mesh(automatic, outline, .5)
        second = generate_layer_adaptive_mesh(explicit, outline, .5, new_outline=None)
        np.testing.assert_array_equal(automatic.nodes, explicit.nodes)
        np.testing.assert_array_equal(automatic.elements, explicit.elements)
        for before, after in zip(first.outline, second.outline):
            np.testing.assert_array_equal(before, after)
        mesh, outline = layer_checks._rectangle()
        np.testing.assert_array_equal(plan_quad_sections(mesh, outline, .5).front,
                                      plan_quad_sections(mesh, outline, .5, new_outline=None).front)

    def test_target_geometry_is_independent_of_element_size(self):
        counts = []
        for size in (.4, .8):
            with self.subTest(size=size):
                mesh, outline = layer_checks._rectangle()
                target = _target()
                original = target.copy()
                result = self.run_layer(mesh, outline, h=size, new_outline=target)
                self.assert_target(mesh, result, target)
                np.testing.assert_array_equal(target, original)
                counts.append(len(layer_checks._outline_nodes(result.outline)))
        self.assertGreater(counts[0], counts[1])

    def test_nonuniform_anchors_on_lines_corner_arms_and_seam_contacts_survive(self):
        target = np.array([
            [-.5, -.5], [-.5, -.25], [-.5, 0.], [-.5, 1.3], [-.5, 3.],
            [-.5, 3.5], [-.2, 3.5], [0., 3.5], [1.25, 3.5], [1.4, 3.5],
            [4., 3.5], [4.5, 3.5], [4.5, 3.2], [4.5, 3.], [4.5, 1.7],
            [4.5, 0.], [4.5, -.5], [4., -.5], [2.3, -.5], [0., -.5],
        ])
        mesh, outline = layer_checks._rectangle()
        result = self.run_layer(mesh, outline, new_outline=target)
        self.assert_target(mesh, result, target)
        self.assertGreater(len(layer_checks._outline_nodes(result.outline)), len(target))

    def test_target_accepts_cyclic_starts_winding_and_explicit_closure(self):
        target = np.insert(_target(), 1, [-.7, 1.2], axis=0)
        baseline = None
        for reverse, start, closed in ((False, 0, False), (False, 1, True),
                                       (True, 2, False), (True, 4, True)):
            with self.subTest(reverse=reverse, start=start, closed=closed):
                supplied = np.roll(target[::-1] if reverse else target, start, axis=0)
                if closed:
                    supplied = np.vstack((supplied, supplied[0]))
                mesh, outline = layer_checks._rectangle()
                result = self.run_layer(mesh, outline, new_outline=supplied.tolist())
                self.assert_target(mesh, result, target)
                if baseline is None:
                    baseline = mesh.nodes.copy(), mesh.elements.copy()
                else:
                    np.testing.assert_array_equal(mesh.nodes, baseline[0])
                    np.testing.assert_array_equal(mesh.elements, baseline[1])

    def test_preview_contains_anchors_is_readonly_and_does_not_mutate(self):
        mesh, outline = layer_checks._rectangle()
        snapshot = self._snapshot(mesh)
        target = np.insert(_target(), 2, [1.3, 3.8], axis=0)
        plan = plan_quad_sections(mesh, outline, .5, new_outline=target)
        self.assert_unchanged(mesh, snapshot)
        np.testing.assert_allclose(plan.front, _target())
        np.testing.assert_allclose(plan.edge_expansion, [.7, .8, .9, .4])
        boundaries = np.concatenate([section.outer_boundary for section in plan.sections])
        for point in target:
            self.assertTrue(np.any(np.linalg.norm(boundaries - point, axis=1) < 1e-12))
        for points in (plan.front, *(section.outer_boundary for section in plan.sections),
                       *(seam.points for seam in plan.seams)):
            self.assertFalse(points.flags.writeable)
        result = self.run_layer(mesh, outline, new_outline=target)
        self.assert_target(mesh, result, target)

    def test_returned_outline_can_be_extended_again(self):
        mesh, outline = layer_checks._rectangle()
        first = self.run_layer(mesh, outline, new_outline=_target())
        target = np.array([[-1., -.7], [-1., 4.1], [5.2, 4.1], [5.2, -.7]])
        second = self.run_layer(mesh, first.outline, h=.3, new_outline=target)
        self.assert_target(mesh, second, target)

    def test_complete_pattern_chords_end_at_custom_front(self):
        mesh, outline, _ = layer_checks._polyomino({(x, y) for x in range(4) for y in range(3)})
        patterns = [[[1., 3.], [1., 3.8]], [[3., 3.], [3., 3.8]]]
        result = self.run_layer(mesh, outline, h=.5, new_outline=_target(), pattern_lines=patterns)
        self.assert_target(mesh, result, _target())
        for pattern in patterns:
            self.assert_pattern_covered(mesh, pattern)

    def test_parallel_pattern_beyond_target_does_not_snap_custom_front(self):
        mesh, outline = layer_checks._rectangle()
        pattern = [[1., 4.], [2., 4.]]
        result = self.run_layer(mesh, outline, h=.5, new_outline=_target(),
                                pattern_lines=[pattern], expand_ratio=2.)
        self.assert_target(mesh, result, _target())

    def test_hole_target_preserves_user_points(self):
        cells = {(x, y) for x in range(12) for y in range(12)
                 if x < 4 or x >= 8 or y < 4 or y >= 8}
        mesh, _, lookup = layer_checks._polyomino(cells)
        mesh.nodes[:, :2] /= 4
        outline = [[lookup[x, 4] for x in range(4, 9)],
                   [lookup[8, y] for y in range(4, 9)],
                   [lookup[x, 8] for x in range(8, 3, -1)],
                   [lookup[4, y] for y in range(8, 3, -1)]]
        target = np.array([[1.15, 1.2], [1.5, 1.2], [1.8, 1.2],
                           [1.8, 1.85], [1.15, 1.85]])
        result = self.run_layer(mesh, outline, h=.2, new_outline=target)
        self.assert_target(mesh, result, target)

    def test_concave_target_matches_the_source_corner_sequence(self):
        coarse = {(0, 0), (1, 0), (0, 1)}
        cells = {(3*x+dx, 3*y+dy) for x, y in coarse for dx in range(3) for dy in range(3)}
        mesh, outline, _ = layer_checks._polyomino(cells)
        mesh.nodes[:, :2] /= 3
        target = np.array([[-.2, -.2], [-.2, 2.2], [1.2, 2.2],
                           [1.2, 1.2], [2.2, 1.2], [2.2, -.2]])
        result = self.run_layer(mesh, outline, h=.2, new_outline=target)
        self.assert_target(mesh, result, target)

    def test_target_applies_after_slot_and_corner_repair(self):
        for fixture, size, key in ((_c_slot, .5, 'slot_repairs'), (_step, 1., 'corner_repairs')):
            with self.subTest(repair=key):
                mesh, outline = fixture()
                baseline = plan_quad_sections(mesh, outline, size)
                snapshot = self._snapshot(mesh)
                target = np.array([[-.6, -.6], [-.6, 3.6], [3.6, 3.6], [3.6, -.6]])
                preview = plan_quad_sections(mesh, outline, size, new_outline=target)
                self.assert_unchanged(mesh, snapshot)
                self.assertIsNotNone(preview.prepared_source)
                np.testing.assert_array_equal(preview.prepared_source.nodes, baseline.prepared_source.nodes)
                np.testing.assert_array_equal(preview.prepared_source.elements, baseline.prepared_source.elements)
                result = self.run_layer(mesh, outline, h=size, new_outline=target)
                self.assertEqual(len(result.diagnostics[key]), 1)
                self.assert_target(mesh, result, target)

    def test_custom_target_does_not_enable_otherwise_untriggered_repair(self):
        mesh, outline = _step()
        snapshot = self._snapshot(mesh)
        # At .75 the existing automatic expansion does not repair this step.
        # A rectangular target therefore has incompatible corner topology.
        target = np.array([[-1., -1.], [-1., 4.], [4., 4.], [4., -1.]])
        for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(QuadLayerError):
                    operation(mesh, outline, .75, new_outline=target)
                self.assert_unchanged(mesh, snapshot)

    def test_misaligned_slot_target_matches_six_prepared_edges(self):
        mesh, outline = _c_slot(2.25)
        baseline = plan_quad_sections(mesh, outline, .5)
        self.assertEqual(len(baseline.prepared_source.outline), 6)
        target = np.array([[-.6, -.4], [-.6, 3.6], [1.3, 3.6], [3.6, 3.6],
                           [3.6, 1.6], [2.85, 1.6], [2.85, -.4]])
        snapshot = self._snapshot(mesh)
        preview = plan_quad_sections(mesh, outline, .5, new_outline=target)
        self.assert_unchanged(mesh, snapshot)
        self.assertEqual(len(preview.front), 6)
        np.testing.assert_array_equal(preview.prepared_source.nodes, baseline.prepared_source.nodes)
        np.testing.assert_array_equal(preview.prepared_source.elements, baseline.prepared_source.elements)
        result = self.run_layer(mesh, outline, new_outline=target)
        self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
        self.assertEqual(len(result.outline), 6)
        self.assert_target(mesh, result, target)

    def test_target_is_scale_translation_and_rotation_invariant(self):
        for scale, shift, rotate in ((1e-7, np.array([0., 0.]), False),
                                      (1e6, np.array([0., 0.]), True),
                                      (1., np.array([1e6, -2e6]), False)):
            with self.subTest(scale=scale, shift=shift, rotate=rotate):
                mesh, outline = layer_checks._rectangle()
                target = np.insert(_target(), 2, [1.3, 3.8], axis=0)
                rotation = np.array([[0., -1.], [1., 0.]]) if rotate else np.eye(2)
                mesh.nodes[:, :2] = mesh.nodes[:, :2] @ rotation.T * scale + shift
                target = target @ rotation.T * scale + shift
                result = self.run_layer(mesh, outline, h=.5*scale, new_outline=target)
                self.assert_target(mesh, result, target)

    def test_invalid_target_inputs_and_topology_are_atomic(self):
        invalid = (
            [], [0., 1.], [[0., 1.], [2., 3.]], np.zeros((4, 3)),
            [[-.5, -.5], [-.5, np.nan], [4.5, 3.5], [4.5, -.5]],
            [[-.5, -.5], [-.5, 3.5], [np.inf, 3.5], [4.5, -.5]],
            [[-.5, -.5], [-.5, -.5], [-.5, 3.5], [4.5, 3.5], [4.5, -.5]],
            [[-.5, -.5], [-.4, 3.5], [4.5, 3.5], [4.5, -.5]],
            [[-.5, -.5], [-.5, 3.5], [4.5, 3.5], [-.5, -.5], [4.5, -.5]],
            [[0., -.5], [0., 3.5], [4.5, 3.5], [4.5, -.5]],
            [[.5, -.5], [.5, 3.5], [4.5, 3.5], [4.5, -.5]],
            [[1., 1.], [1., 2.], [3., 2.], [3., 1.]],
            [[-.5, -.5], [-.5, 3.5], [2., 3.5], [2., 4.], [4.5, 4.], [4.5, -.5]],
            [[-.5, -.5], [-.5, 3.5], [3., 3.5], [3., -1.], [5., -1.], [5., 4.],
             [4.5, 4.], [4.5, -.5]],
        )
        for index, target in enumerate(invalid):
            for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
                with self.subTest(target=index, operation=operation.__name__):
                    mesh, outline = layer_checks._rectangle()
                    snapshot = self._snapshot(mesh)
                    with self.assertRaises(QuadLayerError):
                        operation(mesh, outline, .5, new_outline=target)
                    self.assert_unchanged(mesh, snapshot)

    def test_incomplete_pattern_to_custom_target_is_atomic(self):
        mesh, outline, _ = layer_checks._polyomino({(x, y) for x in range(4) for y in range(3)})
        snapshot = self._snapshot(mesh)
        for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(QuadLayerError):
                    operation(mesh, outline, .5, new_outline=_target(),
                              pattern_lines=[[[1., 3.], [1., 3.5]]])
                self.assert_unchanged(mesh, snapshot)

    def test_custom_target_honors_output_budget_atomically(self):
        baseline, outline = layer_checks._rectangle()
        count = generate_layer_adaptive_mesh(baseline, outline, .5, new_outline=_target()).diagnostics['added_elements']
        mesh, outline = layer_checks._rectangle()
        snapshot = self._snapshot(mesh)
        with self.assertRaises(QuadLayerError):
            generate_layer_adaptive_mesh(mesh, outline, .5, new_outline=_target(), max_added_elements=count-1)
        self.assert_unchanged(mesh, snapshot)
        result = self.run_layer(mesh, outline, new_outline=_target(), max_added_elements=count)
        self.assertEqual(result.diagnostics['added_elements'], count)

    def test_late_custom_target_validation_failure_is_atomic(self):
        mesh, outline = layer_checks._rectangle()
        snapshot = self._snapshot(mesh)
        with patch.object(_mesher, '_validate_patch', side_effect=QuadLayerError('injected failure')):
            with self.assertRaisesRegex(QuadLayerError, 'injected failure'):
                generate_layer_adaptive_mesh(mesh, outline, .5, new_outline=_target())
        self.assert_unchanged(mesh, snapshot)


if __name__ == '__main__':
    unittest.main()
