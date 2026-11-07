"""Dense runs have fixed, local interfaces through every public adapter path."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import (QuadLayerError, generate_full_adaptive_mesh,
                                   generate_layer_adaptive_mesh, plan_quad_sections)
from mesher.mesh2d.adapter import _full
from mesher.mesh2d.adapter.layer import _mesher
from mesher.mesh2d.editor import EditorMesh2D
from tests.unit.mesh2d.test_quad_sections import _cells
from tests.unit.mesh2d import test_section_integration as section_checks
from tests.unit.mesh2d.test_slot_repair import _c_slot


def rectangle(xs=None, height=2000., rotation=0, scale=1., shear=0.):
    if xs is None:
        xs = np.r_[0., 1000., 2000., 3000., np.arange(4000., 4301., 5.),
                   5000., 6000., 7000., 8000.]
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    nodes = np.array([[x + shear * y, y] for y in (0., height) for x in xs])
    for _ in range(rotation):
        nodes = nodes[:, [1, 0]] * [-1, 1]
    mesh = Mesh2D(nodes * scale, np.array([[i, i + 1, i + n + 1, i + n]
                                          for i in range(n - 1)]))
    return mesh, EditorMesh2D.outlines_by_edge(mesh)[0]


class DenseSectionTests(unittest.TestCase):
    assert_built_partition = section_checks.SectionIntegrationTests.assert_built_partition

    def assert_frozen(self, mesh, nodes, elements):
        np.testing.assert_array_equal(mesh.nodes[:len(nodes)], nodes)
        np.testing.assert_array_equal(mesh.elements[:len(elements)], elements)

    def assert_plan(self, mesh, outline, plan):
        prepared = plan.prepared_source
        chains = prepared.outline if prepared is not None else outline
        expected = Counter((int(a), int(b)) for chain in chains
                           for a, b in zip(chain[:-1], chain[1:]))
        actual = Counter((int(a), int(b)) for section in plan.sections
                         for chain in section.source_chains for a, b in zip(chain[:-1], chain[1:]))
        self.assertEqual(actual, expected)
        self.assertTrue(all(len(seam.owners) == 2 for seam in plan.seams))
        for section, following in zip(plan.sections, plan.sections[1:] + plan.sections[:1]):
            self.assertEqual(section.right_seam, following.left_seam)

    def test_dense_run_is_a_section_with_shared_fixed_endpoints_in_every_direction(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                mesh, outline = rectangle(rotation=rotation)
                nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
                plan = plan_quad_sections(mesh, outline, 1000., small_element_size=10.)
                self.assertEqual(len(plan.diagnostics['dense_sections']), 2)
                self.assertEqual(len(plan.diagnostics['dense_seams']), 4)
                self.assert_plan(mesh, outline, plan)
                result = generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.,
                                                      max_transition_count=1000)
                self.assert_built_partition(mesh, nodes, cells, result)
                self.assertEqual(plan.diagnostics['dense_sections'], result.diagnostics['dense_sections'])
                for record in plan.diagnostics['dense_sections']:
                    section = plan.sections[record['section_id']]
                    self.assertEqual(section.kind, 'LINE')
                    self.assertEqual(len(section.source_chains[0]) - 1, 60)
                    for seam_id in (section.left_seam, section.right_seam):
                        seam = plan.seams[seam_id]
                        self.assertEqual(seam.points[0, rotation % 2], seam.points[-1, rotation % 2])
                    # Every cell stays in its section's tangent interval.
                    for detail in result.diagnostics['section_details']:
                        if detail['kind'] != 'LINE' or detail['source_edge'] != section.source_edge:
                            continue
                        points = mesh.nodes[mesh.elements[detail['element_start']:detail['element_end']], :2]
                        polygon = np.asarray(detail['polygon'])
                        axis = rotation % 2
                        self.assertGreaterEqual(points[..., axis].min(), polygon[:, axis].min() - 1e-8)
                        self.assertLessEqual(points[..., axis].max(), polygon[:, axis].max() + 1e-8)

    def test_maximal_runs_single_edges_endpoints_and_strict_threshold(self):
        for xs, expected, seams in (
            ([0, 20, 25, 30, 50, 55, 75], 4, 8),
            ([0, 5, 25, 45, 50], 4, 4),
            ([0, 5, 10, 15], 2, 0),
            ([0, 10, 20, 30], 0, 0),
        ):
            with self.subTest(xs=xs):
                mesh, outline = rectangle(xs, height=100.)
                plan = plan_quad_sections(mesh, outline, 20., small_element_size=10.)
                self.assertEqual(len(plan.diagnostics['dense_sections']), expected)
                self.assertEqual(len(plan.diagnostics['dense_seams']), seams)
                self.assert_plan(mesh, outline, plan)

    def test_disabled_partition_and_explicit_old_cap_remain_available(self):
        mesh, outline = rectangle()
        plan = plan_quad_sections(mesh, outline, 1000.)
        self.assertEqual(plan.diagnostics['sections'], {'OUTER': 4, 'LINE': 4})
        self.assertEqual(plan.diagnostics['dense_sections'], [])
        result = generate_layer_adaptive_mesh(mesh, outline, 1000., max_transition_count=10)
        self.assertEqual(result.diagnostics['max_transition_count'], 10)
        mesh, outline = rectangle()
        result = generate_layer_adaptive_mesh(mesh, outline, 1000.)
        self.assertEqual(result.diagnostics['max_transition_count'], 20)
        self.assertTrue(any(d.get('max_actual_transition_count', 0) > 10
                            for d in result.diagnostics['section_details']))
        self.assertTrue(all(d.get('max_actual_transition_count', 0) <= 20
                            for d in result.diagnostics['section_details']))
        # This extreme dense section cannot satisfy both cap 20 and the size band.
        mesh, outline = rectangle()
        nodes, cells = mesh.nodes, mesh.elements
        with self.assertRaisesRegex(QuadLayerError, 'transition limit and size band'):
            generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, cells)

    def test_scale_and_near_axis_source_keep_endpoint_correspondence(self):
        reference = None
        for scale in (1e-5, 1., 1e5):
            mesh, outline = rectangle(scale=scale, shear=1e-5)
            plan = plan_quad_sections(mesh, outline, 1000. * scale,
                                     small_element_size=10. * scale,
                                     dense_region_tolerance=.001 * scale)
            contacts = [plan.seams[r['seam_id']].points / scale
                        for r in plan.diagnostics['dense_seams']]
            if reference is None:
                reference = contacts
            else:
                np.testing.assert_allclose(contacts, reference, rtol=1e-12, atol=1e-8)
            for points in contacts:
                self.assertAlmostEqual(points[0, 0], points[-1, 0])

    def test_explicit_outer_anchor_merge_is_strict_and_does_not_move_inputs(self):
        for distance, tolerance, reused in ((.0625, .125, True), (.125, .125, False),
                                            (.25, .125, False), (0., 0., True)):
            with self.subTest(distance=distance, tolerance=tolerance):
                mesh, outline = rectangle([0, 100, 105, 110, 200], height=100.)
                target = np.array([[-50., -50.], [-50., 150.], [100. + distance, 150.],
                                   [250., 150.], [250., -50.]])
                saved = target.copy()
                plan = plan_quad_sections(mesh, outline, 50., new_outline=target,
                                         small_element_size=10., dense_region_tolerance=tolerance)
                record = next(r for r in plan.diagnostics['dense_seams']
                              if np.array_equal(mesh.nodes[r['source_node'], :2], [100., 100.]))
                point = plan.seams[record['seam_id']].points[-1]
                self.assertAlmostEqual(point[0], 100. + distance if reused else 100.)
                np.testing.assert_array_equal(target, saved)
                # Explicit anchors must remain in the plan even if not reused.
                self.assertTrue(any(np.any(np.all(np.isclose(s.outer_boundary, target[2]), axis=1))
                                    for s in plan.sections))

    def test_equal_distance_anchor_choice_follows_boundary_order(self):
        mesh, outline = rectangle([0, 100, 105, 110, 200], height=100.)
        target = [[-50., -50.], [-50., 150.], [99.75, 150.], [100.25, 150.],
                  [250., 150.], [250., -50.]]
        plan = plan_quad_sections(mesh, outline, 50., new_outline=target,
                                 small_element_size=10., dense_region_tolerance=.5)
        record = next(r for r in plan.diagnostics['dense_seams']
                      if np.array_equal(mesh.nodes[r['source_node'], :2], [100., 100.]))
        self.assertEqual(plan.seams[record['seam_id']].points[-1, 0], 99.75)

    def test_pattern_splits_dense_run_and_keeps_its_original_shared_seam(self):
        mesh, outline = rectangle()
        patterns = [[[4150., 2000.], [4150., 3000.]]]
        plan = plan_quad_sections(mesh, outline, 1000., small_element_size=10., pattern_lines=patterns)
        self.assertEqual(len(plan.diagnostics['dense_sections']), 3)
        self.assertEqual(plan.diagnostics['pattern_chord_count'], 1)
        self.assertEqual(sum(bool(s.pattern_ids) for s in plan.seams), 1)
        self.assert_plan(mesh, outline, plan)
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.,
                                              pattern_lines=patterns, max_transition_count=1000)
        self.assert_built_partition(mesh, nodes, cells, result)
        section_checks.SectionIntegrationTests.assert_pattern_covered(self, mesh, patterns[0])

    def test_dense_source_after_repair_uses_prepared_snapshot_and_preserves_original(self):
        mesh, outline = _c_slot(bottom_breaks=[1., 1.5, 2., 2.1, 2.2, 2.5, 3.])
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        plan = plan_quad_sections(mesh, outline, .5, small_element_size=.2)
        self.assertIsNotNone(plan.prepared_source)
        self.assertTrue(plan.diagnostics['dense_sections'])
        self.assert_plan(mesh, outline, plan)
        result = generate_layer_adaptive_mesh(mesh, outline, .5, small_element_size=.2)
        self.assert_frozen(mesh, nodes, cells)
        self.assertEqual(plan.diagnostics['dense_sections'], result.diagnostics['dense_sections'])

    def test_inward_hole_can_be_partitioned_in_single_and_full_meshes(self):
        mesh, _ = _cells({(x, y) for x in range(8) for y in range(8)
                          if x in (0, 7) or y in (0, 7)})
        xs = np.array([0, 10, 20, 21, 22, 30, 40, 50, 60.])
        mesh.nodes[:, 0] = xs[mesh.nodes[:, 0].astype(int)]
        mesh.nodes[:, 1] *= 10.
        outlines = EditorMesh2D.outlines_by_edge(mesh)
        outline = min(outlines, key=lambda o: np.ptp(mesh.nodes[np.concatenate(o), 0]))
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        plan = plan_quad_sections(mesh, outline, 2., small_element_size=2.)
        self.assertEqual(len(plan.diagnostics['dense_sections']), 2)
        result = generate_layer_adaptive_mesh(mesh, outline, 2., small_element_size=2.)
        self.assert_built_partition(mesh, nodes, cells, result)
        # Building the exterior first must not change the hole's saved contacts.
        full_mesh = Mesh2D(nodes.copy(), cells.copy())
        full = generate_full_adaptive_mesh(
            full_mesh, [2.], small_element_size=2.,
            outer_outline=[[-10., -10.], [70., -10.], [70., 90.], [-10., 90.]])
        self.assert_frozen(full_mesh, nodes, cells)
        self.assertEqual([len(d['dense_sections']) for d in full.diagnostics['layers'][0]], [2, 2])
        incidence = Counter(tuple(sorted((int(a), int(b)))) for q in full_mesh.elements
                            for a, b in zip(q, np.roll(q, -1)))
        self.assertEqual(len(full.layer_outlines[-1]), 2)
        for boundary in full.layer_outlines[-1]:
            for chain in boundary:
                for a, b in zip(chain[:-1], chain[1:]):
                    self.assertEqual(incidence[tuple(sorted((a, b)))], 2)

    def test_invalid_parameters_are_atomic_for_all_public_entrypoints(self):
        for parameter, values in (
            ('small_element_size', [0, -1, True, np.nan, np.inf, '10', [10], 10**1000]),
            ('dense_region_tolerance', [-1, None, True, np.nan, np.inf, '.001', [.001]]),
        ):
            for value in values:
                for operation in ('preview', 'single', 'full'):
                    with self.subTest(parameter=parameter, value=repr(value), operation=operation):
                        mesh, outline = rectangle()
                        nodes, cells = mesh.nodes, mesh.elements
                        with self.assertRaises(QuadLayerError) as caught:
                            kwargs = {parameter: value}
                            if operation == 'full':
                                generate_full_adaptive_mesh(mesh, [1000.], outer_outline=[], **kwargs)
                            else:
                                fn = plan_quad_sections if operation == 'preview' else generate_layer_adaptive_mesh
                                fn(mesh, outline, 1000., **kwargs)
                        self.assertEqual(caught.exception.code, 'invalid_input')
                        self.assertEqual(caught.exception.context['parameter'], parameter)
                        self.assertIs(mesh.nodes, nodes)
                        self.assertIs(mesh.elements, cells)

    def test_collapsed_projected_interval_is_a_typed_atomic_failure(self):
        mesh, outline = rectangle([0, 100, 100.0001, 200], height=100.)
        nodes, cells = mesh.nodes, mesh.elements
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 50., small_element_size=10.)
        self.assertEqual(caught.exception.code, 'unmeshable_section_layout')
        self.assertIn('source_nodes', caught.exception.context)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, cells)

    def test_full_mesh_routes_parameters_and_preserves_fixed_interface_for_one_or_two_layers(self):
        for sizes in ([1000.], [1000., 1000.]):
            with self.subTest(sizes=sizes):
                mesh, _ = rectangle()
                nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
                target = [[-4000., -4000.], [12000., -4000.], [12000., 6000.], [-4000., 6000.]]
                with patch.object(_full, '_generate_layer_adaptive_mesh',
                                  wraps=_full._generate_layer_adaptive_mesh) as fixed:
                    result = generate_full_adaptive_mesh(mesh, sizes, outer_outline=target,
                                                        small_element_size=10., max_transition_count=1000)
                self.assert_frozen(mesh, nodes, cells)
                self.assertEqual(fixed.call_args.kwargs['small_element_size'], 10.)
                self.assertNotIn('max_transition_count', fixed.call_args.kwargs)
                self.assertTrue(result.diagnostics['layers'][0][0]['dense_sections'])
                incidence = Counter(tuple(sorted((int(a), int(b)))) for q in mesh.elements
                                    for a, b in zip(q, np.roll(q, -1)))
                final = result.layer_outlines[-1][0]
                for chain in final:
                    for a, b in zip(chain[:-1], chain[1:]):
                        self.assertEqual(incidence[tuple(sorted((a, b)))], 2)

    def test_fixed_construction_reuses_saved_contact_instead_of_a_closer_new_anchor(self):
        mesh, outline = rectangle([0, 100, 105, 110, 200], height=100.)
        target = np.array([[-50., -50.], [-50., 150.], [100.0625, 150.],
                           [250., 150.], [250., -50.]])
        plan = plan_quad_sections(mesh, outline, 50., new_outline=target,
                                 small_element_size=10., dense_region_tolerance=.125)
        contacts = {(r['source_edge'], r['source_index']): plan.seams[r['seam_id']].points[-1]
                    for r in plan.diagnostics['dense_seams']}
        # Simulate the global mesher adding closer contacts after planning.
        dense_front = _full.section_outline_points(plan)
        anchor = np.array([100.0625, 150.])
        pos = int(np.argmin(np.linalg.norm(dense_front - anchor, axis=1)))
        fixed = np.insert(dense_front, pos, [[100., 150.]], axis=0)
        data, rebuilt = _mesher._prepare_plan(mesh, outline, 50., (), 1., 1e-3,
                                              new_outline=fixed, small_element_size=10.,
                                              dense_region_tolerance=.125,
                                              _dense_seam_targets=contacts)
        record = next(r for r in rebuilt.diagnostics['dense_seams']
                      if np.array_equal(mesh.nodes[r['source_node'], :2], [100., 100.]))
        actual = rebuilt.seams[record['seam_id']].points[-1] * data['scale'] + data['origin']
        np.testing.assert_allclose(actual, anchor, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
