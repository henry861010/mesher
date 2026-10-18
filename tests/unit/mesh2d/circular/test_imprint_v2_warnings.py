"""Cleanup warnings identify repair locations without changing mesh behavior."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.circular.imprint_v2 import imprint_circle
from mesher.mesh2d.circular.imprint_v2.local import _run_local_imprint
from mesher.mesh2d.circular.imprint_v2.utils import _remove_redundant_element
from mesher.mesh2d.generators import generate_rectilinear_mesh


def imprint(mesh, **overrides):
    arguments = dict(
        center=[0., 0.], radius=1., projection_tolerance=0.,
        guide_tolerance=0., merge_tolerance=0., minimum_area=0.,
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
                    _remove_redundant_element(mesh, 0., 0.)
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
                    _remove_redundant_element(mesh, merge, area)
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
                    _remove_redundant_element(mesh, merge, 0.)
                self.assertEqual(mesh.element_count, count)
                self.assertEqual(caught, [])

    def test_direct_cleanup_warning_as_error_preserves_owned_arrays(self):
        mesh = mesh_with_fold()
        nodes, elements = mesh.nodes, mesh.elements
        before_nodes, before_elements = nodes.copy(), elements.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaises(RuntimeWarning):
                _remove_redundant_element(mesh, 0., 0.)
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
                    _remove_redundant_element(mesh, 0., 0.)
                self.assertEqual(mesh.element_count, 0)
                self.assertEqual(len(caught), 1)
                self.assertIn('self-intersecting/folded', str(caught[0].message))


class LocalImprintWarningTests(unittest.TestCase):
    def test_reports_global_input_id_and_location_once_before_commit(self):
        mesh = mesh_with_fold()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imprint_circle(
                mesh, center=[0., 0.], radius=1., projection_tolerance=0.,
                guide_tolerance=0., merge_tolerance=0., minimum_area=0.,
            )
        self.assertEqual(mesh.element_count, 1)
        self.assertEqual(len(caught), 1)
        self.assertEqual(caught[0].filename, __file__)
        message = str(caught[0].message)
        self.assertIn('removed 1 invalid element(s)', message)
        self.assertIn('element 1 (self-intersecting/folded) at XY=(0.5, 0.5)', message)
        self.assertIn('zero-based', message)
        self.assertIn('not IDs in the renumbered output mesh', message)
        np.testing.assert_array_equal(mesh.elements, [[0, 1, 2, 2]])

    def test_only_last_closure_iteration_contributes_to_warning(self):
        # Moving the surviving local node brings its remote owner into the
        # patch on the next iteration, repeating cleanup of the folded cell.
        mesh = mesh_with_fold()
        mesh.replace_data(
            nodes=np.concatenate((mesh.nodes, [[2., 1., 0.], [2., 2., 0.],
                                              [3., 1., 0.], [3., 2., 0.]])),
            elements=np.concatenate((mesh.elements, [[4, 7, 8, 8], [7, 9, 10, 10]])),
        )
        from mesher.mesh2d.circular.imprint_v2.local import _split_circle
        calls = 0

        def split_and_move(local, *arguments):
            nonlocal calls
            calls += 1
            result = _split_circle(local, *arguments)
            if calls == 1:
                moved = np.all(local.nodes[:, :2] == [2., 1.], axis=1)
                local.nodes[moved] += [.01, 0., 0.]
            return result

        stats = {}
        with patch('mesher.mesh2d.circular.imprint_v2.local._split_circle', split_and_move):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _run_local_imprint(
                    mesh, center=[0., 0.], radius=1., projection_tolerance=0.,
                    guide_tolerance=0., merge_tolerance=0., minimum_area=0., stats=stats,
                )
        self.assertEqual(stats['closure_iterations'], 2)
        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0].message).count('element 1 ('), 1)

    def test_multiple_removed_elements_are_aggregated_in_input_order(self):
        mesh = mesh_with_fold()
        mesh.replace_data(
            nodes=np.concatenate((mesh.nodes, mesh.nodes[3:] + [.1, 0., 0.])),
            elements=np.concatenate((mesh.elements, [[7, 8, 9, 10]])),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imprint(mesh)
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertIn('removed 2 invalid element(s)', message)
        self.assertLess(message.index('element 1 ('), message.index('element 2 ('))

    def test_reoriented_and_remote_invalid_elements_are_silent(self):
        mesh = Mesh2D(
            [[0., 0.], [0., 2.], [2., 0.],
             [10., 0.], [11., 1.], [10., 1.], [11., 0.]],
            [[0, 1, 2, 2], [3, 4, 5, 6]],
        )
        before = mesh.nodes[mesh.elements[1]].copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imprint(mesh)
        self.assertEqual(caught, [])
        np.testing.assert_array_equal(mesh.nodes[mesh.elements[1]], before)

    def test_later_split_failure_emits_no_warning_and_is_atomic(self):
        mesh = mesh_with_fold()
        nodes, elements = mesh.nodes, mesh.elements
        with patch('mesher.mesh2d.circular.imprint_v2.local._split_circle',
                   side_effect=ValueError('forced split failure')):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with self.assertRaisesRegex(ValueError, 'forced split failure'):
                    imprint(mesh)
        self.assertEqual(caught, [])
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_warning_as_error_is_atomic(self):
        mesh = mesh_with_fold()
        nodes, elements = mesh.nodes, mesh.elements
        before_nodes, before_elements = nodes.copy(), elements.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaises(RuntimeWarning):
                imprint(mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)

    def test_case_13_reports_known_deleted_self_intersection(self):
        path = Path(__file__).resolve().parents[4] / 'examples/failures/imprint_v2_holes/cases.json'
        cases = json.loads(path.read_text())['cases']
        parameters = next(case['parameters'] for case in cases if case['id'] == 'case_13')
        mesh = generate_rectilinear_mesh(
            parameters['target_edge_size'], parameters['grid_x_list'], parameters['y_list'],
            coordinate_tolerance=parameters['coordinate_tolerance'],
        )
        arguments = {name: parameters[name] for name in (
            'projection_tolerance', 'guide_tolerance', 'merge_tolerance',
            'minimum_area', 'guide_segments',
        )}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imprint(
                mesh, center=parameters['grid_circler_center_coordinate'],
                radius=parameters['circle_radius'], **arguments,
            )
        self.assertEqual(len(caught), 1)
        self.assertIn('element 10411 (self-intersecting/folded)', str(caught[0].message))


if __name__ == '__main__':
    unittest.main()
