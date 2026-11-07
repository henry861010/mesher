"""Public fixed interfaces and directional, optional dense endpoint mappings."""

from collections import Counter
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import (QuadLayerError, generate_full_adaptive_mesh,
                                   generate_layer_adaptive_mesh, plan_quad_sections)
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh, generate_rectilinear_mesh_optimal
from tests.unit.mesh2d.test_dense_sections import rectangle
from tests.unit.mesh2d.test_extend_quad_layer import _target_ids
from tests.unit.mesh2d import test_fixed_layer_boundary as fixed_checks


def dense_fixture(top=(), bottom=None):
    mesh, outline = rectangle([0, 100, 105, 110, 200], height=100.)
    bottom = top if bottom is None else bottom
    points = [[-50, -50], [-50, 0], [-50, 100], [-50, 150], [0, 150]]
    points += [[x, 150] for x in sorted(top)]
    points += [[200, 150], [250, 150], [250, 100], [250, 0], [250, -50], [200, -50]]
    points += [[x, -50] for x in sorted(bottom, reverse=True)]
    points += [[0, -50]]
    return mesh, outline, _target_ids(mesh, points)


def top_mappings(mesh, plan):
    return {r['endpoint']: r for r in plan.diagnostics['dense_mappings']
            if mesh.nodes[r['source_node'], 1] == 100}


class BoundaryMappingTests(unittest.TestCase):
    assert_atomic_failure = fixed_checks.FixedLayerBoundaryTests.assert_atomic_failure
    assert_exact_interface = fixed_checks.FixedLayerBoundaryTests.assert_exact_interface

    def test_first_outward_node_and_exact_angle_boundary(self):
        boundary = 50 / np.sqrt(3)
        for nodes, angle, expected in (
            ([80, 95, 115, 130], 30., (95, 115)),
            ([100, 110], 0., (100, 110)),
            ([100 - boundary, 110 + boundary], 30., (100 - boundary, 110 + boundary)),
            ([100 - boundary - .01, 110 + boundary + .01], 30., (None, None)),
            ([102, 108], 30., (None, None)),
        ):
            with self.subTest(nodes=nodes, angle=angle):
                mesh, outline, target = dense_fixture(nodes)
                plan = plan_quad_sections(mesh, outline, 50, outline_o=target,
                                          fixed_outer_boundary=True, small_element_size=10,
                                          dense_mapping_angle=angle)
                records = top_mappings(mesh, plan)
                for role, wanted in zip(('begin', 'end'), expected):
                    record = records[role]
                    if wanted is None:
                        self.assertEqual(record['status'], 'skipped')
                    else:
                        self.assertEqual(record['status'], 'mapped')
                        self.assertAlmostEqual(mesh.nodes[record['target_node'], 0], wanted)
                self.assertEqual(len(plan.diagnostics['dense_seams']),
                                 sum(r['status'] == 'mapped' for r in plan.diagnostics['dense_mappings']))

    def test_each_endpoint_can_split_independently_and_generation_matches_preview(self):
        for top, bottom, mapped in (([95], [115], 'begin'), ([115], [95], 'end')):
            with self.subTest(mapped=mapped):
                mesh, outline, target = dense_fixture(top, bottom)
                kwargs = dict(outline_o=target, fixed_outer_boundary=True, small_element_size=10)
                plan = plan_quad_sections(mesh, outline, 50, **kwargs)
                records = top_mappings(mesh, plan)
                self.assertEqual(records[mapped]['status'], 'mapped')
                self.assertEqual(records['end' if mapped == 'begin' else 'begin']['status'], 'skipped')
                self.assertEqual(len(plan.diagnostics['dense_seams']), 2)
                result = generate_layer_adaptive_mesh(mesh, outline, 50, **kwargs)
                self.assert_exact_interface(mesh, result, target)
                self.assertEqual(result.diagnostics['dense_mappings'], plan.diagnostics['dense_mappings'])

    def test_ordinary_mode_projects_when_no_eligible_node_exists(self):
        mesh, outline, target = dense_fixture([102, 108])
        plan = plan_quad_sections(mesh, outline, 50, outline_o=target, small_element_size=10)
        for role, x in (('begin', 100), ('end', 110)):
            record = top_mappings(mesh, plan)[role]
            self.assertEqual(record['reason'], 'projected_contact')
            self.assertAlmostEqual(record['target'][0], x)
            self.assertIsNone(record['target_node'])

    def test_rotation_scale_start_and_target_winding_preserve_mapping_ids(self):
        reference = None
        for rotation in range(4):
            for scale in (1e-5, 1., 1e5):
                mesh, outline, target = dense_fixture([80, 95, 115, 130])
                for _ in range(rotation):
                    mesh.nodes[:, :2] = mesh.nodes[:, [1, 0]] * [-1, 1]
                mesh.nodes[:, :2] *= scale
                for reverse in (False, True):
                    supplied = np.roll(target[::-1] if reverse else target, 3)
                    source = np.roll(outline, 2)
                    plan = plan_quad_sections(mesh, source, 50 * scale, outline_o=supplied,
                                              fixed_outer_boundary=True, small_element_size=10 * scale)
                    actual = {r['source_node']: r['target_node'] for r in plan.diagnostics['dense_seams']}
                    if reference is None:
                        reference = actual
                    self.assertEqual(actual, reference)

    def test_corner_contacts_are_not_limited_by_dense_angle(self):
        for gap in (2., .5):
            mesh = generate_rectilinear_mesh(1, [0, 4], [0, 4])
            outline = EditorMesh2D.outlines(mesh)[0]
            target = _target_ids(mesh, [
                [-gap, -gap], [-gap, 1.3], [-gap, 2.7], [-gap, 4 + gap],
                [1.3, 4 + gap], [2.7, 4 + gap], [4 + gap, 4 + gap],
                [4 + gap, 2.7], [4 + gap, 1.3], [4 + gap, -gap], [2.7, -gap], [1.3, -gap]])
            nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
            plan = plan_quad_sections(mesh, outline, 2, outline_o=target,
                                      fixed_outer_boundary=True, dense_mapping_angle=0)
            result = generate_layer_adaptive_mesh(mesh, outline, 2, outline_o=target,
                                                  fixed_outer_boundary=True, dense_mapping_angle=0,
                                                  max_transition_count=1, min_size_ratio=1)
            self.assert_exact_interface(mesh, result, target)
            np.testing.assert_array_equal(mesh.nodes[:len(nodes)], nodes)
            np.testing.assert_array_equal(mesh.elements[:len(cells)], cells)
            self.assertFalse(result.diagnostics['line_policy_active'])
            self.assertEqual(plan.diagnostics['corner_mappings'], result.diagnostics['corner_mappings'])
            self.assertTrue(all(r['substituted'] for r in result.diagnostics['corner_mappings']))

    def test_conflicting_nearest_corner_nodes_try_other_candidates(self):
        mesh = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        outline = EditorMesh2D.outlines(mesh)[0]
        target = _target_ids(mesh, [[-2, -2], [-2, 0], [-2, 4], [-2, 6], [2, 6], [10, 6],
                                    [12, 6], [12, 4], [12, 0], [12, -2], [10, -2], [2, -2]])
        result = generate_layer_adaptive_mesh(mesh, outline, 2, outline_o=target, fixed_outer_boundary=True)
        self.assert_exact_interface(mesh, result, target)
        self.assertTrue(any(r['candidate_rank'] > 1 for r in result.diagnostics['corner_mappings']))

    def test_corner_distance_ties_are_stable_under_transformations(self):
        reference = None
        for rotation in range(4):
            for scale in (1e-5, 1., 1e5):
                mesh = generate_rectilinear_mesh(1, [0, 4], [0, 4])
                outline = EditorMesh2D.outlines(mesh)[0]
                target = _target_ids(mesh, [
                    [-2, -2], [-2, -.2], [-2, .2], [-2, 3.8], [-2, 4.2], [-2, 6],
                    [-.2, 6], [.2, 6], [3.8, 6], [4.2, 6], [6, 6], [6, 4.2],
                    [6, 3.8], [6, .2], [6, -.2], [6, -2], [4.2, -2], [3.8, -2],
                    [.2, -2], [-.2, -2]])
                for _ in range(rotation):
                    mesh.nodes[:, :2] = mesh.nodes[:, [1, 0]] * [-1, 1]
                mesh.nodes[:, :2] = mesh.nodes[:, :2] * scale + [.137, -1.13]
                for shift in (0, 3, 7):
                    plan = plan_quad_sections(mesh, np.roll(outline, shift), 2 * scale,
                                              outline_o=np.roll(target, shift), fixed_outer_boundary=True)
                    actual = {(r['source_node'], r['side']): r['target_node']
                              for r in plan.diagnostics['corner_mappings']}
                    if reference is None:
                        reference = actual
                    self.assertEqual(actual, reference)

    def test_missing_pattern_contact_is_an_atomic_error_in_both_apis(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            mesh = generate_rectilinear_mesh(1, [0, 4], [0, 4])
            outline = EditorMesh2D.outlines(mesh)[0]
            target, _ = fixed_checks._global_target(mesh, outline, 2)
            error = self.assert_atomic_failure(mesh, lambda: api(
                mesh, outline, 2, outline_o=target, fixed_outer_boundary=True,
                pattern_lines=[[[1, 4], [1, 6]]]))
            self.assertEqual(error.code, 'unsupported_pattern_layout')
            self.assertEqual(error.context['parameter'], 'outline_o')

    def test_two_existing_blocks_join_without_hanging_interface_nodes(self):
        inner = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        source = EditorMesh2D.outlines(inner)[0]
        outer = generate_rectilinear_mesh_optimal(
            2, [[-4, -4], [8, -4], [8, 8], [-4, 8]],
            holes=[[[-2, -2], [6, -2], [6, 6], [-2, 6]]])
        offset = inner.node_count
        mesh = Mesh2D(np.vstack((inner.nodes, outer.mesh.nodes)),
                      np.vstack((inner.elements, outer.mesh.elements + offset)))
        target = np.asarray(outer.hole_outlines[0]) + offset
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, source, 2, outline_o=target, fixed_outer_boundary=True)
        self.assert_exact_interface(mesh, result, target)
        np.testing.assert_array_equal(mesh.nodes[:len(nodes)], nodes)
        np.testing.assert_array_equal(mesh.elements[:len(elements)], elements)
        incidence = Counter(tuple(sorted((int(a), int(b)))) for cell in mesh.elements
                            for a, b in zip(cell, np.roll(cell, -1)))
        for loop in (source, target):
            self.assertTrue(all(incidence[tuple(sorted((int(a), int(b))))] == 2
                                for a, b in zip(loop, np.roll(loop, -1))))

    def test_invalid_fixed_flag_and_removed_keyword_do_not_mutate(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            for flag in (1, 'true', None):
                mesh, outline, target = dense_fixture()
                error = self.assert_atomic_failure(mesh, lambda: api(
                    mesh, outline, 50, outline_o=target, fixed_outer_boundary=flag))
                self.assertEqual(error.context['parameter'], 'fixed_outer_boundary')
            mesh, outline, target = dense_fixture()
            with self.assertRaises(TypeError):
                api(mesh, outline, 50, outline_o=target, dense_region_tolerance=.001)
        mesh, _, _ = dense_fixture()
        with self.assertRaises(TypeError):
            generate_full_adaptive_mesh(mesh, [50], outer_outline=[], dense_region_tolerance=.001)

    def test_disabled_fixed_line_policies_still_validate_their_values(self):
        for name, value in (('max_transition_count', 0), ('max_transition_count', None),
                            ('min_size_ratio', 0), ('min_size_ratio', 1.01)):
            mesh, outline, target = dense_fixture()
            error = self.assert_atomic_failure(mesh, lambda: generate_layer_adaptive_mesh(
                mesh, outline, 50, outline_o=target, fixed_outer_boundary=True, **{name: value}))
            self.assertEqual(error.context['parameter'], name)


if __name__ == '__main__':
    unittest.main()
