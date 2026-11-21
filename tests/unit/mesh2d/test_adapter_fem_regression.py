"""Protect a known Quad4 layer and its response to uniform heat generation."""

import hashlib
from itertools import product
import unittest

import numpy as np

from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from tests.unit.mesh2d.test_dense_sections import rectangle


class AdapterFEMRegressionTests(unittest.TestCase):
    def test_nonuniform_transition_temperature(self):
        mesh, outline = rectangle([0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 34, 42], height=20)
        result = generate_layer_adaptive_mesh(mesh, outline, 10)

        # Freeze the solver's connectivity and boundary IDs independently of
        # floating-point representation and the platform's integer width.
        topology = mesh.elements.astype("<i8").tobytes() + np.asarray(result.outline, "<i8").tobytes()
        self.assertEqual(hashlib.sha256(topology).hexdigest(),
                         "af5aa783f66fcd86def9ebe64ee30ed058abdab142ca904ddd34540a8d1c2762")

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

        free = np.setdiff1d(np.arange(mesh.node_count), result.outline)
        temperature = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])
        # A fixed physical probe in the original mesh, away from the outer boundary.
        probe, = np.flatnonzero(np.all(mesh.nodes[free, :2] == [24., 20.], axis=1))
        self.assertAlmostEqual(temperature[probe], 128.05851863492813, delta=1e-8)
