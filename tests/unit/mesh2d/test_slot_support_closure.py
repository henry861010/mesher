"""Slot repair follows the cardinal fronts used by actual layer expansion."""

import unittest

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.adapter.layer._mesher import _expansion_distances, _prepare
from mesher.mesh2d.adapter.layer._repair.slots import _find_slots
from tests.unit.mesh2d import test_extend_quad_layer as layer_checks
from tests.unit.mesh2d.test_slot_repair import _c_slot


def _sheared_slot(top_only=False):
    mesh, outline = _c_slot()
    selected = mesh.nodes[:, 1] >= 2. if top_only else np.ones(mesh.node_count, dtype=bool)
    mesh.nodes[selected, 1] += .0002 * mesh.nodes[selected, 0]
    return mesh, outline


class SlotSupportClosureTests(unittest.TestCase):
    assert_layer = layer_checks.ExtendQuadLayerTests.assert_layer
    assert_pattern_covered = layer_checks.ExtendQuadLayerTests.assert_pattern_covered
    assert_unchanged = layer_checks.ExtendQuadLayerTests.assert_unchanged
    _snapshot = staticmethod(layer_checks.ExtendQuadLayerTests._snapshot)

    def assert_repaired(self, mesh, outline, size, **kwargs):
        before = self._snapshot(mesh)
        preview = plan_quad_sections(mesh, outline, size, **kwargs)
        self.assert_unchanged(mesh, before)
        result = generate_layer_adaptive_mesh(mesh, outline, size, **kwargs)
        self.assert_layer(mesh, outline, result, before[2], before[3])
        self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
        prepared = preview.prepared_source
        np.testing.assert_array_equal(mesh.nodes[:len(prepared.nodes)], prepared.nodes)
        np.testing.assert_array_equal(mesh.elements[:len(prepared.elements)], prepared.elements)
        # Equal-length walls must still be closed at the original mouth.
        repair = result.diagnostics['slot_repairs'][0]
        source_ids = [*repair['source_chains'][0],
                      *(node for chain in repair['source_chains'][1:] for node in chain[1:])]
        np.testing.assert_allclose(repair['polygon'], before[2][source_ids, :2])
        return result

    def test_parallel_sheared_walls_repair_before_the_front_reverses(self):
        mesh, outline = _sheared_slot()
        self.assert_repaired(mesh, outline, .4999)

    def test_varying_width_slot_uses_supports_instead_of_maximum_width(self):
        mesh, outline = _sheared_slot(top_only=True)
        self.assert_repaired(mesh, outline, .5002)

    def test_actual_front_gap_uses_the_geometric_tolerance(self):
        mesh, outline = _c_slot()
        reference = _prepare(mesh, outline, .5, (), 1., 1e-3)
        tolerance = reference['tol'] * reference['scale']
        # Positive gap is twice the reduction in each wall's expansion.
        for gap_in_tolerances, expected in ((2., 0), (.5, 1), (0., 1), (-2., 1)):
            with self.subTest(gap=gap_in_tolerances):
                size = .5 - .5 * gap_in_tolerances * tolerance
                data = _prepare(mesh, outline, size, (), 1., 1e-3)
                _expansion_distances(data)
                self.assertEqual(len(_find_slots(data)), expected)

    def test_support_closure_is_invariant_under_rotation_scale_and_translation(self):
        for rotation, scale, shift in (
            (1, 1., [0., 0.]), (2, 1e-3, [100., -200.]),
            (3, 1e3, [1e6, -2e6]),
        ):
            with self.subTest(rotation=rotation, scale=scale):
                mesh, outline = _sheared_slot()
                for _ in range(rotation):
                    mesh.nodes[:, :2] = mesh.nodes[:, [1, 0]] * [-1., 1.]
                mesh.nodes[:, :2] = mesh.nodes[:, :2] * scale + shift
                self.assert_repaired(mesh, outline, .4999 * scale)

    def test_asymmetric_pattern_snapping_closes_the_actual_support_gap(self):
        mesh, outline = _sheared_slot(top_only=True)
        pattern = [[1., 1.5], [3.5, 1.5]]
        data = _prepare(mesh, outline, .5, [pattern], 1.01, 1e-3)
        _expansion_distances(data)
        slot, = _find_slots(data)
        first, _, last = slot.edges
        self.assertNotEqual(data['expansion'][first], data['expansion'][last])
        self.assert_repaired(mesh, outline, .5, pattern_lines=[pattern], expand_ratio=1.01)
        self.assert_pattern_covered(mesh, pattern)

    def test_opt_out_retains_atomic_front_collision_failure(self):
        for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(operation=operation.__name__):
                mesh, outline = _sheared_slot()
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    operation(mesh, outline, .4999, repair_slots=False)
                self.assertEqual(caught.exception.code, 'topology_change')
                self.assertEqual(caught.exception.context['stage'], 'expansion')
                self.assert_unchanged(mesh, before)

    def test_custom_target_keeps_the_original_automatic_repair_decision(self):
        target = [[-.6, -.4], [-.6, 3.8], [3.6, 3.8], [3.6, -.4]]
        mesh, outline = _sheared_slot()
        self.assert_repaired(mesh, outline, .4999, outline_o=layer_checks._target_ids(mesh, target))
        # A rectangular target does not itself trigger repair of an open slot.
        for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(operation=operation.__name__):
                mesh, outline = _sheared_slot()
                ids = layer_checks._target_ids(mesh, target)
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    operation(mesh, outline, .4, outline_o=ids)
                self.assertEqual(caught.exception.code, 'unsupported_new_outline')
                self.assert_unchanged(mesh, before)


if __name__ == '__main__':
    unittest.main()
