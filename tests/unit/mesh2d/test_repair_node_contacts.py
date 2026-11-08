"""Slot mouths and corner caps share ownership rules for original nodes."""

import unittest

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from tests.unit.mesh2d import test_extend_quad_layer as layer_checks
from tests.unit.mesh2d.test_corner_repair import _step
from tests.unit.mesh2d.test_slot_repair import _c_slot


class RepairNodeContactTests(unittest.TestCase):
    _snapshot = staticmethod(layer_checks.ExtendQuadLayerTests._snapshot)
    assert_unchanged = layer_checks.ExtendQuadLayerTests.assert_unchanged

    def test_unrelated_nodes_on_free_boundaries_are_rejected_atomically(self):
        cases = (
            # Include unsampled points and duplicate endpoint coordinates.
            ('slot', _c_slot, .5, ((3., 1.5), (3., 1.3), (3., 1.), (3., 2.))),
            ('corner', _step, 1., ((0., 2.), (.5, 3.), (0., 3.), (0., 1.), (1., 3.))),
        )
        for kind, fixture, size, points in cases:
            for point in points:
                for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
                    with self.subTest(kind=kind, point=point, operation=operation.__name__):
                        mesh, outline = fixture()
                        node_id = mesh.node_count
                        mesh.nodes = np.vstack((mesh.nodes, [*point, 0.]))
                        before = self._snapshot(mesh)
                        with self.assertRaises(QuadLayerError) as caught:
                            operation(mesh, outline, size)
                        self.assertEqual(caught.exception.code, f'unmeshable_{kind}_repair')
                        causes = caught.exception.context['causes']
                        self.assertTrue(any(
                            cause.get('context', {}).get('reason') == 'mouth_hits_existing_mesh'
                            and cause['context'].get('node_id') == node_id
                            for cause in causes))
                        self.assert_unchanged(mesh, before)

    def test_rotated_translated_slot_mouth_rejects_an_unrelated_node(self):
        for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(operation=operation.__name__):
                mesh, outline = _c_slot()
                mesh.nodes = np.vstack((mesh.nodes, [3., 1.5, 0.]))
                mesh.nodes[:, :2] = mesh.nodes[:, [1, 0]] * [-1000., 1000.] + [1e6, -2e6]
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    operation(mesh, outline, 500.)
                self.assert_unchanged(mesh, before)

    def test_slanted_mouth_checks_all_existing_anchor_candidates(self):
        for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(operation=operation.__name__):
                mesh, outline = _c_slot(2.25)
                # Each of the two candidate slopes meets one isolated node.
                mesh.nodes = np.vstack((mesh.nodes, [[2.125, 1.5, 0.], [2.375, 1.5, 0.]]))
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    operation(mesh, outline, .5)
                causes = caught.exception.context['causes']
                self.assertEqual({cause['context'].get('node_id') for cause in causes},
                                 {len(before[2]) - 2, len(before[2]) - 1})
                self.assert_unchanged(mesh, before)

    def test_free_boundary_contacts_respect_geometric_tolerance(self):
        # Contact is defined by the configured absolute physical distance.
        tolerance = .001
        for fixture, size, point, outward in (
            (_c_slot, .5, np.array([3., 1.5]), np.array([1., 0.])),
            (_step, 1., np.array([0., 2.]), np.array([-1., 0.])),
        ):
            for factor in (.5, 4.):
                for operation in (plan_quad_sections, generate_layer_adaptive_mesh):
                    with self.subTest(fixture=fixture.__name__, factor=factor,
                                      operation=operation.__name__):
                        mesh, outline = fixture()
                        position = point + factor * tolerance * outward
                        mesh.nodes = np.vstack((mesh.nodes, [*position, 0.]))
                        before = self._snapshot(mesh)
                        if factor < 1:
                            with self.assertRaises(QuadLayerError):
                                operation(mesh, outline, size, tolerance=tolerance)
                            self.assert_unchanged(mesh, before)
                        else:
                            operation(mesh, outline, size, tolerance=tolerance)
                            np.testing.assert_array_equal(mesh.nodes[:len(before[2])], before[2])

    def test_legal_source_endpoints_and_unrelated_remote_nodes_are_preserved(self):
        for fixture, size, kind in ((_c_slot, .5, 'slot'), (_step, 1., 'corner')):
            with self.subTest(kind=kind):
                mesh, outline = fixture()
                remote_id = mesh.node_count
                mesh.nodes = np.vstack((mesh.nodes, [10., 10., 0.]))
                before = self._snapshot(mesh)
                preview = plan_quad_sections(mesh, outline, size)
                self.assert_unchanged(mesh, before)
                result = generate_layer_adaptive_mesh(mesh, outline, size)
                np.testing.assert_array_equal(mesh.nodes[:len(before[2])], before[2])
                np.testing.assert_array_equal(mesh.elements[:len(before[3])], before[3])
                self.assertFalse(np.any(mesh.elements == remote_id))
                for diagnostics in (preview.diagnostics, result.diagnostics):
                    repair = diagnostics[f'{kind}_repairs'][0]
                    mouth = repair['mouth_node_ids']
                    self.assertEqual(mouth[0], repair['source_chains'][0][0])
                    self.assertEqual(mouth[-1], repair['source_chains'][-1][-1])

    def test_slot_can_choose_an_alternative_mouth_without_merging_old_nodes(self):
        mesh, outline = _c_slot(2.25)
        node_id = mesh.node_count
        # Only the preferred mouth crosses this node. The other existing
        # anchor defines a valid mouth with a different slope.
        mesh.nodes = np.vstack((mesh.nodes, [2.125, 1.5, 0.]))
        before = self._snapshot(mesh)
        preview = plan_quad_sections(mesh, outline, .5)
        self.assert_unchanged(mesh, before)
        result = generate_layer_adaptive_mesh(mesh, outline, .5)
        np.testing.assert_array_equal(mesh.nodes[:len(before[2])], before[2])
        self.assertFalse(np.any(mesh.elements == node_id))
        for diagnostics in (preview.diagnostics, result.diagnostics):
            points = np.asarray(diagnostics['slot_repairs'][0]['mouth_points'])
            np.testing.assert_allclose(points[[0, -1]], [[2.5, 2.], [2.25, 1.]])


if __name__ == '__main__':
    unittest.main()
