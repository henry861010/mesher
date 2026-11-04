"""Sizing and transaction contracts for adaptive slot fills."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.adapter.layer import _mesher
from mesher.mesh2d.adapter.layer._repair import slots as _repair, adaptive as _repair_adaptive, patch as _repair_patch
from mesher.mesh2d.adapter.layer._limits import ElementLimitExceeded
from tests.unit.mesh2d import test_extend_quad_layer as layer_checks
from tests.unit.mesh2d.test_slot_repair import _blocks, _c_slot


_TEST2_PATTERNS = [[[-50., 30.], [50., 30.]],
                   [[-100., 63.], [-50., 63.]],
                   [[0., 50.], [0., 200.]]]


def _test2():
    """The four-block test2 geometry, assembled without the adapter or viewer."""
    step = lambda start, end: np.arange(start, end + 1., 5.)
    return _blocks([
        (step(0., 100.), step(0., 100.)),
        (step(100., 200.), step(0., 200.)),
        (step(100., 300.), step(200., 250.)),
        (step(200., 400.), step(0., 100.)),
    ])


def _quality(nodes, elements):
    points = nodes[elements, :2]
    edges = np.roll(points, -1, axis=1) - points
    lengths = np.linalg.norm(edges, axis=2)
    following = np.roll(edges, -1, axis=1)
    jacobian = ((edges[..., 0] * following[..., 1]
                 - edges[..., 1] * following[..., 0])
                / (lengths * np.roll(lengths, -1, axis=1)))
    return float(np.max(lengths.max(axis=1) / lengths.min(axis=1))), float(jacobian.min())


def _rectangular_source(length, width, spacing=5.):
    return (np.array([(x, width, 0.) for x in np.arange(length, -spacing / 2, -spacing)]),
            np.array([(0., y, 0.) for y in np.arange(width, -spacing / 2, -spacing)]),
            np.array([(x, 0., 0.) for x in np.arange(0., length + spacing / 2, spacing)]))


class AdaptiveSlotRepairTests(unittest.TestCase):
    assert_layer = layer_checks.ExtendQuadLayerTests.assert_layer
    assert_unchanged = layer_checks.ExtendQuadLayerTests.assert_unchanged
    _snapshot = staticmethod(layer_checks.ExtendQuadLayerTests._snapshot)

    def assert_mouth(self, nodes, repair, size):
        mouth = nodes[repair['mouth_node_ids'], :2]
        lengths = np.linalg.norm(np.diff(mouth, axis=0), axis=1)
        np.testing.assert_allclose(lengths, size, rtol=1e-8, atol=size * 1e-8)

    def test_test2_fine_mouth_and_adaptive_interior_improve_quality(self):
        mesh, outline = _test2()
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 50., expand_ratio=1.5,
                                   pattern_lines=_TEST2_PATTERNS)
        self.assert_layer(mesh, outline, result, old_nodes, old_elements)
        repairs = result.diagnostics['slot_repairs']
        self.assertEqual(len(repairs), 1)
        repair = repairs[0]
        self.assertEqual(len(repair['mouth_node_ids']) - 1, 20)
        self.assert_mouth(mesh.nodes, repair, 5.)
        self.assertAlmostEqual(repair['boundary_element_size'], 5.)
        self.assertAlmostEqual(repair['max_element_size'], 20.)
        count = repair['element_end'] - repair['element_start']
        self.assertLess(count, 400, 'adaptive fill must beat a uniform 20 by 20 grid')
        cells = mesh.elements[repair['element_start']:repair['element_end']]
        aspect, jacobian = _quality(mesh.nodes, cells)
        self.assertLess(aspect, 5.)
        self.assertGreater(jacobian, .2)
        self.assertGreater(sum(face['ring_count'] for face in repair['faces']), 0)
        for face in repair['faces']:
            self.assertIsInstance(face['stop_reason'], str)
            self.assertTrue(face['stop_reason'])
        np.testing.assert_allclose(result.diagnostics['edge_expansion'], 50.)

    def test_cap_overrides_preserve_fine_mouth_and_preview_matches_extension(self):
        for cap, effective in ((5., 5.), (10., 10.), (100., 50.)):
            with self.subTest(cap=cap):
                mesh, outline = _test2()
                before = self._snapshot(mesh)
                plan = plan_quad_sections(mesh, outline, 50., repair_max_element_size=cap)
                self.assert_unchanged(mesh, before)
                repair = plan.diagnostics['slot_repairs'][0]
                self.assertAlmostEqual(repair['max_element_size'], effective)
                self.assertAlmostEqual(repair['boundary_element_size'], 5.)
                self.assert_mouth(plan.prepared_source.nodes, repair, 5.)
                for face in repair['faces']:
                    targets = face['ring_target_sizes']
                    self.assertEqual(len(targets), face['ring_count'])
                    self.assertTrue(all(0. < size <= effective * (1. + 1e-10)
                                        for size in targets))
                self.assertFalse(plan.prepared_source.nodes.flags.writeable)
                self.assertFalse(plan.prepared_source.elements.flags.writeable)
                if cap == 10.:
                    result = generate_layer_adaptive_mesh(mesh, outline, 50., repair_max_element_size=cap)
                    source = plan.prepared_source
                    np.testing.assert_array_equal(mesh.nodes[:len(source.nodes)], source.nodes)
                    np.testing.assert_array_equal(mesh.elements[:len(source.elements)], source.elements)
                    self.assertEqual(result.diagnostics['repair_added_elements'],
                                     len(source.elements) - source.original_element_count)

    def test_invalid_cap_is_rejected_even_when_slot_repair_is_disabled(self):
        invalid = (0., -1., float('nan'), float('inf'), -float('inf'), True,
                   np.bool_(False), '1.0', np.array([1.]), np.array(1.), 1 + 1j)
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            for cap in invalid:
                with self.subTest(api=api.__name__, cap=repr(cap)):
                    mesh, outline = _c_slot()
                    before = self._snapshot(mesh)
                    with self.assertRaises(QuadLayerError) as caught:
                        api(mesh, outline, .5, repair_slots=False, repair_max_element_size=cap)
                    self.assertEqual(caught.exception.code, 'invalid_input')
                    self.assertEqual(caught.exception.context.get('parameter'),
                                     'repair_max_element_size')
                    self.assert_unchanged(mesh, before)

    def test_cap_below_local_boundary_size_is_atomic(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            with self.subTest(api=api.__name__):
                mesh, outline = _c_slot()
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    api(mesh, outline, .5, repair_max_element_size=.25)
                self.assert_unchanged(mesh, before)

    def test_repair_sizing_diagnostics_use_world_units(self):
        for scale, translation in ((1e-3, [100., -20.]), (1e3, [-4e5, 2e5])):
            with self.subTest(scale=scale):
                mesh, outline = _test2()
                mesh.nodes[:, :2] = mesh.nodes[:, :2] * scale + translation
                before = self._snapshot(mesh)
                plan = plan_quad_sections(mesh, outline, 50. * scale)
                self.assert_unchanged(mesh, before)
                repair = plan.diagnostics['slot_repairs'][0]
                self.assertAlmostEqual(repair['boundary_element_size'] / scale, 5.)
                self.assertAlmostEqual(repair['max_element_size'] / scale, 20.)
                self.assert_mouth(plan.prepared_source.nodes, repair, 5. * scale)
                targets = [size / scale for face in repair['faces']
                           for size in face['ring_target_sizes']]
                self.assertTrue(targets)
                self.assertTrue(all(5. <= size <= 20. * (1. + 1e-8) for size in targets))
                np.testing.assert_allclose(plan.edge_expansion, 50. * scale)

    def test_narrow_misaligned_and_unequal_walls_preserve_frozen_segments(self):
        narrow = lambda: _blocks([
            ([0., .5, 1.], [0., .5, 1., 1.25, 1.75, 2.25]),
            ([1., 1.5, 2., 2.5, 3.], [0., .5, 1.]),
            ([1., 1.5, 2., 2.5, 3.], [1.25, 1.75, 2.25]),
        ])
        fixtures = (narrow, lambda: _c_slot(2.25),
                    lambda: _c_slot(bottom_breaks=[1., 1.25, 2., 3.]))
        for index, fixture in enumerate(fixtures):
            with self.subTest(case=index):
                mesh, outline = fixture()
                old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
                result = generate_layer_adaptive_mesh(mesh, outline, .5)
                self.assert_layer(mesh, outline, result, old_nodes, old_elements)
                self.assertEqual(len(result.diagnostics['slot_repairs']), 1)

    def test_repair_runs_once_and_post_repair_topology_failure_is_atomic(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            for fail in (False, True):
                with self.subTest(api=api.__name__, fail=fail):
                    mesh, outline = _c_slot()
                    before = self._snapshot(mesh)
                    original_front = _mesher._front

                    def front(data):
                        self.assertIn('_original_data', data)
                        if fail:
                            raise QuadLayerError('post-repair topology conflict', code='topology_change')
                        return original_front(data)

                    with patch.object(_repair, 'repair_slots', wraps=_repair.repair_slots) as repairs:
                        with patch.object(_mesher, '_front', side_effect=front) as fronts:
                            if fail:
                                with self.assertRaises(QuadLayerError) as caught:
                                    api(mesh, outline, .5)
                                self.assertEqual(caught.exception.code, 'topology_change')
                                self.assertEqual(caught.exception.context.get('stage'), 'expansion')
                                self.assert_unchanged(mesh, before)
                            else:
                                api(mesh, outline, .5)
                            self.assertEqual(repairs.call_count, 1)
                            self.assertEqual(fronts.call_count, 1)

    def test_adaptive_repair_and_expansion_share_exact_element_budget(self):
        mesh, outline = _test2()
        result = generate_layer_adaptive_mesh(mesh, outline, 50.)
        total = result.diagnostics['added_elements']
        repair = result.diagnostics['repair_added_elements']
        mesh, outline = _test2()
        exact = generate_layer_adaptive_mesh(mesh, outline, 50., max_added_elements=total)
        self.assertEqual(exact.diagnostics['added_elements'], total)
        for limit in (total - 1, repair - 1):
            with self.subTest(limit=limit):
                mesh, outline = _test2()
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    generate_layer_adaptive_mesh(mesh, outline, 50., max_added_elements=limit)
                self.assertEqual(caught.exception.code, 'element_limit_exceeded')
                self.assertGreater(caught.exception.context['required_elements'], limit)
                self.assert_unchanged(mesh, before)

    def test_global_size_equal_to_boundary_keeps_a_healthy_long_core(self):
        source = _rectangular_source(1000., 10.)
        for chains in (source, tuple(side[::-1] for side in source[::-1])):
            with self.subTest(reversed=chains is not source):
                result = _repair_patch.build_repair_patch(chains, (), 5., 1e-10)
                frozen = np.vstack((chains[0], chains[1][1:], chains[2][1:]))
                np.testing.assert_array_equal(result.nodes[result.source_ids], frozen)
                face = result.diagnostics['faces'][0]
                self.assertEqual(face['ring_count'], 0)
                self.assertIn(face['core_method'], ('structured', 'strip'))
                aspect, jacobian = _quality(result.nodes, result.quads)
                self.assertLess(aspect, 1.01)
                self.assertGreater(jacobian, .99)
                points = result.nodes[result.quads, :2]
                areas = .5 * np.sum(points[..., 0] * np.roll(points[..., 1], -1, axis=1)
                                    - points[..., 1] * np.roll(points[..., 0], -1, axis=1), axis=1)
                self.assertAlmostEqual(float(areas.sum()), 10000.)

    def test_ring_budget_preflight_precedes_inner_sampling(self):
        # The outer disk bound is 39, but the first ring plus its core needs
        # at least 71 cells. A budget of 50 must refuse before allocating it.
        with patch.object(_repair_adaptive, '_sample_inner',
                          side_effect=AssertionError('inner boundary was allocated')) as sample:
            with self.assertRaises(QuadLayerError) as caught:
                _repair_patch.build_repair_patch(_rectangular_source(100., 100.), (),
                                                 50., 1e-10, max_added_elements=50)
            self.assertEqual(caught.exception.code, 'element_limit_exceeded')
            self.assertEqual(caught.exception.context['stage'], 'slot_repair_ring_count_preflight')
            self.assertGreater(caught.exception.context['required_elements'], 50)
            sample.assert_not_called()

    def test_adaptive_ring_limit_never_falls_back_to_legacy_geometry(self):
        def exceed(*args, **kwargs):
            raise ElementLimitExceeded(101, 100, stage='injected_ring_budget')

        with patch.object(_repair_adaptive, '_ring', side_effect=exceed) as ring:
            with patch.object(_repair_patch, '_four_sided_units',
                              side_effect=AssertionError('limit was treated as geometry failure')) as legacy:
                with self.assertRaises(QuadLayerError) as caught:
                    _repair_patch.build_repair_patch(_rectangular_source(100., 100.), (),
                                                     50., 1e-10, max_added_elements=100)
                self.assertEqual(caught.exception.code, 'element_limit_exceeded')
                self.assertEqual(caught.exception.context['stage'], 'injected_ring_budget')
                self.assertGreater(ring.call_count, 0)
                legacy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
