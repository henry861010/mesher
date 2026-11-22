"""A resolution failure in one trial must leave other Quad4 layouts available."""

import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from tests.unit.mesh2d.test_extend_quad_layer import _polyomino


class LayerCandidateResolutionTests(unittest.TestCase):
    def test_translated_hole_uses_resolvable_corner_units(self):
        cells = {(x, y) for x in range(8) for y in range(8)
                 if x < 2 or x >= 6 or y < 2 or y >= 6}
        for shift in (0., 11., -200.):
            with self.subTest(shift=shift):
                mesh, _, ids = _polyomino(cells)
                outline = ([ids[x, 2] for x in range(2, 6)] + [ids[6, y] for y in range(2, 6)]
                           + [ids[x, 6] for x in range(6, 2, -1)] + [ids[2, y] for y in range(6, 2, -1)])
                mesh.nodes[:, :2] = mesh.nodes[:, :2] * .001 + [0., shift]
                generate_layer_adaptive_mesh(mesh, outline, .0005, tolerance=1e-6)
                points = mesh.nodes[mesh.elements[len(cells):], :2]
                edges = np.roll(points, -1, axis=1) - points
                edges /= np.linalg.norm(edges, axis=2, keepdims=True)
                following = np.roll(edges, -1, axis=1)
                jacobians = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
                # At +11 the one-cell corner used to pass with Jacobian 1.8e-12.
                # At -200 it rounded flat, despite the available inner-ring fill.
                self.assertGreater(jacobians.min(), .3)

    def test_resolvable_ring_survives_an_unusable_sampling_alternative(self):
        for location, size, depth in (("corner", .0015, .0017), ("line", .00101, .00201)):
            with self.subTest(location=location):
                side, tol = .004, .001
                mesh = Mesh2D([[0, 0], [side, 0], [side, side], [0, side]], [[0, 1, 2, 3]])
                original = mesh.nodes.copy()
                result = generate_layer_adaptive_mesh(mesh, [0, 3, 2, 1], size,
                                                      expand_size=depth, tolerance=tol)
                np.testing.assert_array_equal(mesh.nodes[:4], original)
                np.testing.assert_array_equal(mesh.elements[0], [0, 1, 2, 3])
                points = mesh.nodes[mesh.elements[1:], :2]
                edges = np.roll(points, -1, axis=1) - points
                self.assertTrue(np.all(np.linalg.norm(edges, axis=2) > tol))
                self.assertGreater(result.diagnostics["quality"]["min_scaled_jacobian"], 0)
                x, y = points[..., 0], points[..., 1]
                area = (x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum() / 2
                self.assertAlmostEqual(area, (side + 2 * depth) ** 2 - side ** 2, delta=1e-16)
