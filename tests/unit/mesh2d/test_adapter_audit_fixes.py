"""Public correctness and atomicity regressions for the adapter risk audit."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.tolerance import GeometryToleranceError
from mesher.mesh2d.adapter import (
    QuadLayerError, generate_layer_adaptive_mesh as layer, plan_quad_sections as plan,
)
from mesher.mesh2d.adapter.unit import build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z
from mesher.mesh2d.adapter.layer import _mesher
from mesher.mesh2d.adapter.layer._validation import _edge_conflicts
from mesher.mesh2d.adapter.layer._planning import fixed_corners
from mesher.mesh2d.editor import EditorMesh2D


def square(width=1., offset=0.):
    return Mesh2D(np.array([[0, 0], [width, 0], [width, width], [0, width]]) + offset,
                  [[0, 1, 2, 3]])


def target(mesh, xy):
    ids = np.arange(len(mesh.nodes), len(mesh.nodes) + len(xy))
    mesh.nodes = np.vstack((mesh.nodes, np.column_stack((xy, np.zeros(len(xy))))))
    return ids


def strip(xs, height):
    n = len(xs)
    mesh = Mesh2D([[x, y] for y in (0, height) for x in xs],
                  [[i, i + 1, n + i + 1, n + i] for i in range(n - 1)])
    return mesh, EditorMesh2D.outlines(mesh)[0]


def cells(indices, spacing):
    points = sorted({(x + dx, y + dy) for x, y in indices
                     for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))})
    ids = {p: i for i, p in enumerate(points)}
    mesh = Mesh2D(np.array(points) * spacing,
                  [[ids[x, y], ids[x + 1, y], ids[x + 1, y + 1], ids[x, y + 1]]
                   for x, y in indices])
    return mesh, EditorMesh2D.outlines(mesh)[0]


class AuditFixTests(unittest.TestCase):
    def atomic_error(self, mesh, operation):
        nodes, elements = mesh.nodes, mesh.elements
        saved_nodes, saved_elements = nodes.copy(), elements.copy()
        with self.assertRaises(QuadLayerError) as caught:
            operation()
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(nodes, saved_nodes)
        np.testing.assert_array_equal(elements, saved_elements)
        return caught.exception

    def test_final_contact_threshold_uses_encoded_world_coordinates(self):
        for gap, expected in ((np.nextafter(.125, 0), True), (.125, True),
                              (np.nextafter(.125, np.inf), False)):
            with self.subTest(gap=gap):
                world = np.array([[-1., 0], [0, 0], [gap, -.5], [gap, .5]])
                origin, scale = np.array([-3., .7]), 7.
                local = (world - origin) / scale
                result = _edge_conflicts(np.array([0, 1]), local[:2], np.array([[2, 3]]),
                                         local[None, 2:], .125 / scale,
                                         (world, origin, scale, .125))
                self.assertEqual(bool(result[0]), expected)

    def test_world_rounding_cannot_commit_short_edges(self):
        mesh = square(.006, 1e9)
        error = self.atomic_error(mesh, lambda: layer(mesh, [0, 3, 2, 1], .004,
            expand_size=.00200000005, tolerance=.001, repair_slots=False))
        self.assertEqual(error.code, 'feature_below_resolution')

    def test_source_attachment_requires_one_old_owner_per_edge(self):
        for owners in (0, 2):
            with self.subTest(owners=owners):
                mesh = square()
                outline = [0, 3, 2, 1]
                if owners == 0:
                    outline = target(mesh, mesh.nodes[outline, :2] + [10., 0.])
                else:
                    mesh.elements = np.vstack((mesh.elements, mesh.elements))
                error = self.atomic_error(mesh, lambda: layer(mesh, outline, .5))
                self.assertEqual(error.code, 'topology_change')
                self.assertEqual(error.context['parameter'], 'outline_i')
                self.assertTrue(set(error.context['edge']).issubset(outline))

    def test_generated_front_must_resolve_tolerance_in_world_coordinates(self):
        for operation in (plan, layer):
            with self.subTest(operation=operation.__name__):
                mesh = square(4.)
                mesh.nodes[:, 0] += 2.**41 - 4.25
                # Growing another .5 crosses a float64 exponent boundary.
                error = self.atomic_error(mesh, lambda: operation(mesh, [0, 3, 2, 1], .5))
                self.assertEqual(error.code, 'numeric_resolution_exceeded')
                self.assertEqual(error.context['stage'], 'world_coordinates')

    def test_snap_search_range_is_not_extended_by_geometry_tolerance(self):
        for gap, expected in ((1., 1.), (1.0005, .5)):
            with self.subTest(gap=gap):
                mesh = square(4.)
                outline = [0, 3, 2, 1]
                pattern = [[[1., 4. + gap], [2., 4. + gap]]]
                kwargs = dict(pattern_lines=pattern, expand_ratio=2., tolerance=.001)
                preview = plan(mesh, outline, .5, **kwargs)
                result = layer(mesh, outline, .5, **kwargs)
                self.assertEqual(preview.edge_expansion[1], expected)
                self.assertEqual(result.diagnostics['edge_expansion'][1], expected)
                self.assertEqual(mesh.nodes[result.outline, 1].max(), 4. + expected)

    def test_world_rounding_cannot_close_existing_mesh_clearance(self):
        mesh = square(.006, 1e9)
        right, tol = 1e9 + .01, .0010001
        expansion = right - mesh.nodes[1, 0] - tol - 3e-9
        other = [[right, 1e9 - .01], [right + .01, 1e9 - .01],
                 [right + .01, 1e9 + .02], [right, 1e9 + .02]]
        mesh = Mesh2D(np.vstack((mesh.nodes[:, :2], other)), [[0, 1, 2, 3], [4, 5, 6, 7]])
        error = self.atomic_error(mesh, lambda: layer(mesh, [0, 3, 2, 1], .004,
            expand_size=expansion, tolerance=tol, repair_slots=False))
        self.assertEqual(error.code, 'topology_change')

    def test_mapped_anchor_must_obey_line_size_band(self):
        mesh = square()
        ids = target(mesh, [[-1, -1], [-.0008, -1], [2, -1], [2, 2], [-1, 2]])
        error = self.atomic_error(mesh, lambda: layer(mesh, [0, 3, 2, 1], 1,
            outline_o=ids, min_size_ratio=1, max_size_ratio=1, tolerance=.001, repair_slots=False))
        self.assertEqual(error.code, 'line_constraints_infeasible')

    def test_mapped_anchor_diagnostics_use_committed_coordinates(self):
        mesh = square()
        ids = target(mesh, [[-1, -1], [-.0008, -1], [2, -1], [2, 2], [-1, 2]])
        result = layer(mesh, [0, 3, 2, 1], 1, outline_o=ids, tolerance=.001, repair_slots=False)
        self.assertTrue(set(ids).issubset(result.outline))
        for record in result.diagnostics['section_details']:
            if 'line_sampling' not in record:
                continue
            gaps = np.linalg.norm(np.diff(mesh.nodes[record['outer_ids'], :2], axis=0), axis=1)
            self.assertEqual(record['min_outer_spacing'], gaps.min())
            self.assertEqual(record['max_outer_spacing'], gaps.max())
            self.assertLessEqual(gaps.max(), 1.25 + 1e-13)
        for record in result.diagnostics['seam_states']:
            np.testing.assert_array_equal(record['points'], mesh.nodes[record['node_ids'], :2])

    def test_thin_bands_above_physical_resolution_are_accepted(self):
        for thickness in (.0015, .002, .0021):
            with self.subTest(thickness=thickness):
                mesh = square(.004)
                layer(mesh, [0, 3, 2, 1], .004, expand_size=thickness,
                      tolerance=.001, repair_slots=False)
                points = mesh.nodes[mesh.elements[1:], :2]
                self.assertGreater(np.linalg.norm(np.roll(points, -1, axis=1) - points, axis=2).min(), .001)

    def test_small_corner_and_slot_repairs_have_numerical_area_thresholds(self):
        for kind, indices in [('corner', [(0, 0), (1, 0), (1, 1)]),
                              ('slot', [(0, 0), (1, 0), (0, 1), (0, 2), (1, 2)])]:
            with self.subTest(kind=kind):
                mesh, ids = cells(indices, .003)
                result = layer(mesh, ids, .003, tolerance=.001)
                self.assertTrue(result.diagnostics[kind + '_repairs'])
                points = mesh.nodes[mesh.elements[len(indices):], :2]
                self.assertGreater(np.linalg.norm(np.roll(points, -1, axis=1) - points, axis=2).min(), .001)

    def test_dense_threshold_is_not_reduced_by_physical_tolerance(self):
        mesh, ids = strip([0, .2, .2995, .4995], .2)
        result = plan(mesh, ids, .2, small_element_size=.1, tolerance=.001, repair_slots=False)
        self.assertEqual(len(result.diagnostics['dense_sections']), 2)
        self.assertEqual(len(result.diagnostics['dense_seams']), 4)

    def test_zero_dense_angle_is_not_widened_by_physical_tolerance(self):
        mesh, ids = strip([0, .02, .026, .046], .02)
        outer = target(mesh, [[-.0022, -.0022], [.0482, -.0022], [.0482, .0222],
                             [.0269, .0222], [.0191, .0222], [-.0022, .0222]])
        result = plan(mesh, ids, .01, outline_o=outer, small_element_size=.01,
                      dense_mapping_angle=0, tolerance=.001, repair_slots=False)
        for record in result.diagnostics['dense_seams']:
            self.assertLessEqual(abs(record['angle']), 1e-10)
        self.assertTrue(any(r['status'] == 'skipped' for r in result.diagnostics['dense_mappings']))

    def test_ambiguous_fixed_contacts_raise_public_typed_error(self):
        for operation in (plan, layer):
            with self.subTest(operation=operation.__name__):
                mesh = square()
                ids = target(mesh, [[-1, -1], [-.00075, -1], [.00075, -1],
                                    [2, -1], [2, 2], [-1, 2]])
                error = self.atomic_error(mesh, lambda: operation(mesh, [0, 3, 2, 1], 1,
                    outline_o=ids, tolerance=.001, repair_slots=False))
                self.assertEqual(error.code, 'fixed_constraint_conflict')
                self.assertIsInstance(error.__cause__, GeometryToleranceError)
                self.assertTrue(error.context['owner_ids'])

    def test_large_finite_unit_coordinates_produce_ccw_without_overflow(self):
        xy = np.array([[0, 0], [1, 3], [.5, 2.75], [-.5, -.25]]) * 1e200
        edges = tuple(np.array(pair) for pair in ((0, 3), (3, 2), (2, 1), (1, 0)))
        for builder in (build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z):
            with self.subTest(builder=builder.__name__), warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                mesh = Mesh2D(xy, np.empty((0, 4), dtype=int))
                builder(mesh, *edges, tolerance=1e190)
                np.testing.assert_array_equal(mesh.nodes[:, :2], xy)
                points = mesh.nodes[mesh.elements, :2] / 1e200
                v = np.roll(points, -1, axis=1) - points
                following = np.roll(v, -1, axis=1)
                self.assertTrue(np.all(v[..., 0] * following[..., 1] - v[..., 1] * following[..., 0] > 0))

    def test_large_units_preserve_unrelated_data_and_remap_added_nodes(self):
        xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, 2], [0, 2],
                       [10, 10], [11, 10], [11, 11], [10, 11]], dtype=float) * 1e200
        edges = tuple(np.array(ids) for ids in ((0, 1, 2, 3), (3, 4), (4, 5), (5, 0)))
        for builder in (build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z):
            with self.subTest(builder=builder.__name__), warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                mesh = Mesh2D(xy, [[6, 7, 8, 9]])
                original = mesh.nodes.copy()
                builder(mesh, *edges, tolerance=1e190)
                np.testing.assert_array_equal(mesh.nodes[:10], original)
                np.testing.assert_array_equal(mesh.elements[0], [6, 7, 8, 9])
                self.assertGreater(len(mesh.nodes), 10)
                points = mesh.nodes[mesh.elements[1:], :2] / 1e200
                vectors = np.roll(points, -1, axis=1) - points
                following = np.roll(vectors, -1, axis=1)
                self.assertTrue(np.all(vectors[..., 0] * following[..., 1]
                                       - vectors[..., 1] * following[..., 0] > 0))

    def test_unit_resolution_failures_leave_mesh_untouched(self):
        for builder in (build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z):
            with self.subTest(builder=builder.__name__):
                mesh = square(1e200)
                nodes, elements = mesh.nodes, mesh.elements
                with self.assertRaises(GeometryToleranceError) as caught:
                    builder(mesh, *(np.array(ids) for ids in ((0, 1), (1, 2), (2, 3), (3, 0))),
                            tolerance=1e-3)
                self.assertEqual(caught.exception.code, 'numeric_resolution_exceeded')
                self.assertIs(mesh.nodes, nodes)
                self.assertIs(mesh.elements, elements)

    def test_impossible_late_edge_is_rejected_before_geometric_search(self):
        for n in (8, 12):
            with self.subTest(n=n):
                v = np.linspace(-1, 2, n + 2)
                outer = np.vstack(([-1, -1], np.column_stack((np.full(n + 1, -1), v[1:])),
                    np.column_stack((v[1:], np.full(n + 1, 2))),
                    np.column_stack((np.full(n + 1, 2), v[-2::-1]))))
                mesh = square()
                ids = target(mesh, outer)
                with patch.object(fixed_corners, 'valid_interface', side_effect=AssertionError('DFS entered')):
                    self.atomic_error(mesh, lambda: plan(mesh, [0, 3, 2, 1], 1,
                        outline_o=ids, fixed_outer_boundary=True, repair_slots=False))

    def test_fixed_contact_diagnostics_retain_exact_target_coordinates(self):
        mesh = square()
        values = np.linspace(-.3, 1.3, 5)
        outer = np.vstack((np.column_stack((np.full(4, -.3), values[:-1])),
                           np.column_stack((values[:-1], np.full(4, 1.3))),
                           np.column_stack((np.full(4, 1.3), values[:0:-1])),
                           np.column_stack((values[:0:-1], np.full(4, -.3)))))
        ids = target(mesh, outer)
        mesh.nodes[:, :2] += [11.7, -24.1]
        result = plan(mesh, [0, 3, 2, 1], 1, outline_o=ids,
                      fixed_outer_boundary=True, repair_slots=False)
        for record in result.diagnostics['corner_mappings']:
            np.testing.assert_array_equal(record['target'], mesh.nodes[record['target_node'], :2])

    def test_ordinary_success_uses_one_final_validation(self):
        mesh = square()
        with patch.object(_mesher, '_validate_patch', wraps=_mesher._validate_patch) as validate:
            layer(mesh, [0, 3, 2, 1], 1, repair_slots=False)
        self.assertEqual(validate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
