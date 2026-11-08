"""Cleanup warnings identify repair locations without changing mesh behavior."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from mesher import Mesh2D
from mesher.tolerance import GeometryToleranceError
from mesher.mesh2d.circular.imprint_v2 import imprint_circle
from mesher.mesh2d.circular.imprint_v2.local import _run_local_imprint
from mesher.mesh2d.circular.imprint_v2.utils import _remove_redundant_element
from mesher.mesh2d.generators import generate_rectilinear_mesh


def imprint(mesh, **overrides):
    arguments = dict(
        center=[0., 0.], radius=1., projection_distance=0.,
         tolerance=1e-12, minimum_area=0.,
    )
    arguments.update(overrides)
    return imprint_circle(mesh, **arguments)


def mesh_with_fold():
    # The untouched remote element ensures local and input element IDs differ.
    return Mesh2D(
        [[10., 10.], [11., 10.], [10., 11.],
         [0., 0.], [1., 1.], [0., 1.], [1., 0.]],
        [[0, 1, 2, 2], [3, 4, 5, 6]],
    )


class CleanupWarningTests(unittest.TestCase):
    def test_zero_and_positive_area_self_intersections_warn(self):
        for points in (
            [[0., 0.], [1., 1.], [0., 1.], [1., 0.]],
            [[0., 0.], [2., 1.], [0., 1.], [1., 0.]],
        ):
            with self.subTest(points=points):
                mesh = Mesh2D(points, [[0, 1, 2, 3]])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    _remove_redundant_element(mesh, 1e-12, 0.)
                self.assertEqual(mesh.element_count, 0)
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, RuntimeWarning)
                self.assertIn('element 0 (self-intersecting/folded)', str(caught[0].message))
                self.assertIn('holes', str(caught[0].message))
                self.assertIn('manual inspection and repair', str(caught[0].message))
                self.assertEqual(caught[0].filename, __file__)

    def test_merge_inversion_uses_geometry_at_deletion(self):
        # The first triangle's lower-index node pulls the second triangle's
        # apex below its base; all three nodes remain distinct.
        mesh = Mesh2D(
            [[0., -.05], [2., -2.], [2., -1.],
             [-1., 0.], [1., 0.], [0., .05]],
            [[0, 1, 2, 2], [3, 4, 5, 5]],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _remove_redundant_element(mesh, .11, 0.)
        self.assertEqual(mesh.element_count, 1)
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertIn('element 1 (inverted)', message)
        self.assertIn('XY=(0, -0.025', message)

    def test_degeneracy_and_area_cutoff_removals_are_silent(self):
        cases = (
            ([[0., 0.], [1., 0.], [2., 0.]], [[0, 1, 2, 2]], 0., 0.),
            ([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], [[0, 2, 1, 3]], 0., 0.),
            ([[0., 0.], [.05, 0.], [0., 100.]], [[0, 1, 2, 2]], .1, 0.),
            ([[0., 0.], [.1, 0.], [0., .1]], [[0, 1, 2, 2]], 0., .01),
            # A simple concave quad whose children fail the area cutoff is
            # not a self-intersection, despite its mixed corner orientations.
            ([[0., 0.], [2., 0.], [.5, .2], [0., 1.]], [[0, 1, 2, 3]], 0., .3),
        )
        for nodes, elements, merge, area in cases:
            with self.subTest(nodes=nodes):
                mesh = Mesh2D(nodes, elements)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    _remove_redundant_element(mesh, max(merge, 1e-12), area)
                self.assertEqual(mesh.element_count, 0)
                self.assertEqual(caught, [])

    def test_repaired_concave_and_collapsed_quads_are_silent(self):
        for points, merge, count in (
            ([[0., 0.], [2., 0.], [.5, .2], [0., 1.]], 0., 2),
            ([[0., 0.], [.05, 0.], [1., 1.], [0., 1.]], .1, 1),
        ):
            with self.subTest(points=points):
                mesh = Mesh2D(points, [[0, 1, 2, 3]])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    _remove_redundant_element(mesh, max(merge, 1e-12), 0.)
                self.assertEqual(mesh.element_count, count)
                self.assertEqual(caught, [])

    def test_direct_cleanup_warning_as_error_preserves_owned_arrays(self):
        mesh = mesh_with_fold()
        nodes, elements = mesh.nodes, mesh.elements
        before_nodes, before_elements = nodes.copy(), elements.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaises(RuntimeWarning):
                _remove_redundant_element(mesh, 1e-12, 0.)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)

    def test_classification_is_stable_across_scales_and_offsets(self):
        points = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
        for scale, offset in ((1.e-100, 0.), (1., 1.e12), (1.e100, 0.)):
            with self.subTest(scale=scale, offset=offset):
                mesh = Mesh2D(points * scale + offset, [[0, 1, 2, 3]])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    _remove_redundant_element(mesh, 1e-12, 0.)
                self.assertEqual(mesh.element_count, 0)
                self.assertEqual(len(caught), 1)
                self.assertIn('self-intersecting/folded', str(caught[0].message))


class LocalImprintValidationTests(unittest.TestCase):
    def test_reports_global_input_ids_before_any_commit(self):
        mesh = mesh_with_fold()
        nodes, elements = mesh.nodes, mesh.elements
        with warnings.catch_warnings(record=True) as caught:
            with self.assertRaises(GeometryToleranceError) as error:
                imprint(mesh)
        self.assertEqual(error.exception.code, 'invalid_input_geometry')
        self.assertEqual(error.exception.context['element_indices'], [1])
        self.assertEqual(caught, [])
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_invalid_source_is_rejected_before_splitting(self):
        mesh = mesh_with_fold()
        with patch('mesher.mesh2d.circular.imprint_v2.local._split_circle') as split:
            with self.assertRaises(GeometryToleranceError):
                imprint(mesh)
        split.assert_not_called()

    def test_multiple_invalid_sources_are_reported_in_input_order(self):
        mesh = mesh_with_fold()
        mesh.replace_data(
            nodes=np.concatenate((mesh.nodes, mesh.nodes[3:] + [.1, 0., 0.])),
            elements=np.concatenate((mesh.elements, [[7, 8, 9, 10]])),
        )
        with self.assertRaises(GeometryToleranceError) as error:
            imprint(mesh)
        self.assertEqual(error.exception.context['element_indices'], [1, 2])

    def test_remote_invalid_elements_are_preserved(self):
        mesh = Mesh2D(
            [[0., 0.], [0., 2.], [2., 0.],
             [10., 0.], [11., 1.], [10., 1.], [11., 0.]],
            [[0, 1, 2, 2], [3, 4, 5, 6]],
        )
        before = mesh.nodes[mesh.elements[1]].copy()
        imprint(mesh)
        np.testing.assert_array_equal(mesh.nodes[mesh.elements[1]], before)

    def test_later_split_failure_remains_atomic(self):
        mesh = Mesh2D([[0., 0.], [2., 0.], [0., 2.]], [[0, 1, 2, 2]])
        nodes, elements = mesh.nodes, mesh.elements
        with patch('mesher.mesh2d.circular.imprint_v2.local._split_circle',
                   side_effect=ValueError('forced split failure')):
            with self.assertRaisesRegex(ValueError, 'forced split failure'):
                imprint(mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_warning_filters_cannot_allow_invalid_source_deletion(self):
        mesh = mesh_with_fold()
        original = mesh.nodes.copy(), mesh.elements.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaises(GeometryToleranceError):
                imprint(mesh)
        np.testing.assert_array_equal(mesh.nodes, original[0])
        np.testing.assert_array_equal(mesh.elements, original[1])

    def test_case_13_withdraws_invalid_moves_instead_of_deleting_cells(self):
        from mesher.mesh2d.quality import MeshQualityChecker
        path = Path(__file__).resolve().parents[4] / 'examples/failures/imprint_v2_holes/cases.json'
        cases = json.loads(path.read_text())['cases']
        parameters = next(case['parameters'] for case in cases if case['id'] == 'case_13')
        mesh = generate_rectilinear_mesh(
            parameters['target_edge_size'], parameters['grid_x_list'], parameters['y_list'],
            tolerance=parameters['tolerance'],
        )
        count = mesh.element_count
        arguments = {name: parameters[name] for name in (
            'projection_distance', 'tolerance', 'minimum_area', 'guide_segments',
        )}
        with warnings.catch_warnings(record=True) as caught:
            imprint(mesh, center=parameters['grid_circler_center_coordinate'],
                    radius=parameters['circle_radius'], **arguments)
        self.assertEqual(caught, [])
        self.assertGreaterEqual(mesh.element_count, count)
        self.assertEqual(MeshQualityChecker(mesh).calculate_jacobian().summary.invalid_count, 0)


if __name__ == '__main__':
    unittest.main()
