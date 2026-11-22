"""Protect a known Quad4 layer and its response to uniform heat generation."""

import hashlib
from itertools import product
import unittest

import numpy as np

from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from tests.unit.mesh2d.test_dense_sections import rectangle


class AdapterFEMRegressionTests(unittest.TestCase):
    def test_equivalent_length_units_choose_the_same_projected_cuts(self):
        def generate(scale, size):
            mesh, outline = rectangle([0, 1, 2, 3.5, 4.5], height=3.)
            mesh.nodes[:, :2] *= scale
            layer = generate_layer_adaptive_mesh(mesh, outline, size * scale,
                                                 tolerance=.001 * scale)
            return mesh.nodes / scale, mesh.elements, layer.outline

        for size in (1.25, 2.5):
            nodes, elements, outline = generate(1., size)
            for scale in (.001, 1000.):
                with self.subTest(size=size, scale=scale):
                    converted, connectivity, boundary = generate(scale, size)
                    np.testing.assert_allclose(converted, nodes, atol=1e-14, rtol=1e-14)
                    np.testing.assert_array_equal(connectivity, elements)
                    self.assertEqual(boundary, outline)

    def test_nonuniform_transition_temperature(self):
        cases = (
            ({}, "ed3c7764065ceeea1208b5adfc49c4d030b56c38b516363ca903bdd9227244b6",
             129.09546941727265),
            ({"min_size_ratio": .5, "max_size_ratio": 1.25},
             "af5aa783f66fcd86def9ebe64ee30ed058abdab142ca904ddd34540a8d1c2762",
             128.05851863492813),
        )
        for (options, expected_topology, expected_temperature), scale in product(cases, (1., .01, 25.4)):
            with self.subTest(options=options, scale=scale):
                mesh, outline = rectangle([0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 34, 42],
                                          height=20)
                mesh.nodes[:, :2] *= scale
                result = generate_layer_adaptive_mesh(mesh, outline, 10 * scale,
                                                      tolerance=.001 * scale, **options)

                # Freeze connectivity and boundary IDs independently of platform integer width.
                topology = mesh.elements.astype("<i8").tobytes() + np.asarray(result.outline, "<i8").tobytes()
                self.assertEqual(hashlib.sha256(topology).hexdigest(), expected_topology)
                # In 2D the integrated unit source scales with area, whereas
                # the isotropic conduction stiffness is unchanged by scaling.
                temperature = self._probe_temperature(mesh, result.outline, scale) / scale**2
                self.assertAlmostEqual(temperature, expected_temperature, delta=1e-8)

    def _probe_temperature(self, mesh, outline, scale):
        # Steady isotropic conduction, k = 1, unit heat source, outer T = 0.
        # The non-affine solution is sensitive to movement of interior nodes.
        stiffness = np.zeros((mesh.node_count, mesh.node_count))
        load = np.zeros(mesh.node_count)
        xi_sign = np.array([-1, 1, 1, -1])
        eta_sign = np.array([-1, -1, 1, 1])
        gauss = np.array([-1, 1]) / np.sqrt(3.)
        for xi, eta in product(gauss, repeat=2):
            shape = (1 + xi_sign * xi) * (1 + eta_sign * eta) / 4
            derivatives = np.column_stack((xi_sign * (1 + eta_sign * eta),
                                           eta_sign * (1 + xi_sign * xi))) / 4
            for cell in mesh.elements:
                jacobian = mesh.nodes[cell, :2].T @ derivatives
                weight = np.linalg.det(jacobian)
                self.assertGreater(weight, 0)
                gradient = derivatives @ np.linalg.inv(jacobian)
                stiffness[np.ix_(cell, cell)] += weight * (gradient @ gradient.T)
                load[cell] += weight * shape

        free = np.setdiff1d(np.arange(mesh.node_count), outline)
        # The 62 by 40 envelope must receive the entire unit body load.
        self.assertAlmostEqual(load.sum() / scale**2, 62 * 40, delta=1e-9)
        # A linear temperature field has zero interior residual on a
        # conforming mesh, including across the original/added interface.
        linear = (mesh.nodes[:, 0] - 2 * mesh.nodes[:, 1]) / scale
        np.testing.assert_allclose(stiffness[free] @ linear, 0., atol=1e-10)
        temperature = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])
        # A fixed physical probe in the original mesh, away from the outer boundary.
        probe, = np.flatnonzero(np.all(mesh.nodes[free, :2] == np.array([24., 20.]) * scale, axis=1))
        return temperature[probe]
