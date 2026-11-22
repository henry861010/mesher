import unittest

import numpy as np

from mesher.mesh2d import EditorMesh2D, Mesh2D
from .._fixtures import edited_mesh
from ._fixtures import empty_mesh, two_quads


class JacobianRatioTests(unittest.TestCase):
    def test_mixed_mesh_and_selection_order(self):
        mesh = Mesh2D(
            [[0, 0], [2, 0], [1, 1], [0, 1], [3, 1], [1000, 0]],
            [[0, 1, 2, 3], [0, 5, 3, 3], [0, 1, 4, 2]],
        )
        gauss = (3 - 1 / np.sqrt(3)) / (3 + 1 / np.sqrt(3))
        for sampling, trapezoid in (("corner", 0.5), ("gauss", gauss)):
            with self.subTest(sampling=sampling):
                all_ratios = EditorMesh2D.jacobian_ratios(mesh, sampling=sampling)
                self.assertEqual(all_ratios.dtype, np.float64)
                np.testing.assert_allclose(all_ratios, [trapezoid, 1, 1])
                selected = EditorMesh2D.jacobian_ratios(mesh, [2, 0], sampling=sampling)
                np.testing.assert_allclose(selected, [1, trapezoid])
                minimum = EditorMesh2D.min_jacobian_ratio(mesh, [2, 0], sampling=sampling)
                self.assertIs(type(minimum), float)
                self.assertAlmostEqual(minimum, trapezoid)
                self.assertEqual(EditorMesh2D.min_jacobian_ratio(mesh, [2], sampling=sampling), 1)
        np.testing.assert_array_equal(EditorMesh2D.jacobian_ratios(mesh, [0]), [0.5])

    def test_concave_quad_changes_sign_between_sampling_modes(self):
        mesh = Mesh2D([[0, 0], [1, 0], [0.4, 0.4], [0, 1]], [[0, 1, 2, 3]])
        self.assertAlmostEqual(EditorMesh2D.min_jacobian_ratio(mesh), -0.2)
        expected = (0.4 - 0.6 / np.sqrt(3)) / (0.4 + 0.6 / np.sqrt(3))
        self.assertAlmostEqual(EditorMesh2D.min_jacobian_ratio(mesh, sampling="gauss"), expected)

    def test_reversed_and_degenerate_elements_after_direct_edits(self):
        mesh = edited_mesh(
            [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [3, 0]],
            [[0, 3, 2, 1], [0, 3, 1, 1], [0, 1, 4, 5], [0, 1, 4, 4]],
        )
        for sampling in ("corner", "gauss"):
            np.testing.assert_array_equal(
                EditorMesh2D.jacobian_ratios(mesh, sampling=sampling), [-1, -1, 0, 0]
            )
        mesh.elements[0] = [0, 1, 2, 3]
        self.assertEqual(EditorMesh2D.min_jacobian_ratio(mesh, [0]), 1.0)

    def test_scale_invariance(self):
        points = np.array([[0, 0], [2, 0], [1, 1], [0, 1]])
        for scale in (1e-200, 1e200):
            mesh = Mesh2D(scale * points, [[0, 1, 2, 3], [0, 1, 2, 2]])
            np.testing.assert_allclose(EditorMesh2D.jacobian_ratios(mesh), [0.5, 1])

    def test_empty_and_invalid_selections(self):
        for mesh, indices in ((empty_mesh(), None), (two_quads(), [])):
            self.assertEqual(EditorMesh2D.jacobian_ratios(mesh, indices).shape, (0,))
            with self.assertRaisesRegex(ValueError, "at least one element"):
                EditorMesh2D.min_jacobian_ratio(mesh, indices)
        mesh = two_quads()
        for indices, error in (([1, 1], ValueError), ([-1], IndexError), ([2], IndexError),
                               ([[0]], ValueError), ([0.5], TypeError), ([True], TypeError)):
            with self.subTest(indices=indices), self.assertRaises(error):
                EditorMesh2D.jacobian_ratios(mesh, indices)
        with self.assertRaisesRegex(ValueError, "sampling"):
            EditorMesh2D.jacobian_ratios(mesh, [], sampling="center")

    def test_only_referenced_xy_must_be_finite(self):
        mesh = two_quads()
        mesh.nodes[0, 0] = np.nan
        np.testing.assert_array_equal(EditorMesh2D.jacobian_ratios(mesh, [1]), [1])
        with self.assertRaisesRegex(ValueError, "finite XY"):
            EditorMesh2D.min_jacobian_ratio(mesh)
