"""Public mesh commits certify the represented XY geometry."""
import unittest

import numpy as np

from examples.extend_circular_mesh_open_sector import build_inner_sector
from mesher import GeometryToleranceError, Mesh2D
from mesher.mesh2d import EditorMesh2D
from mesher.mesh2d.adapter import generate_full_adaptive_mesh, generate_layer_adaptive_mesh
from mesher.mesh2d.circular.imprint_v2 import imprint_circle
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.quality import MeshQualityChecker, QualityReason

from ._fixtures import assert_ccw


class WindingTests(unittest.TestCase):
    def test_mixed_rows_preserve_ids_and_ownership_across_commits(self):
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        rows = [[1, 0, 3, 2], [3, 2, 1, 1], [2, 3, 0, 1]]
        expected = [[1, 2, 3, 0], [3, 1, 2, 2], rows[2]]
        for factory in (Mesh2D, Mesh2D.from_planar):
            mesh = factory(nodes, rows)
            np.testing.assert_array_equal(mesh.elements, expected)
            mesh.nodes = nodes * [-1, 1]
            assert_ccw(mesh)
            mesh.replace_data(nodes=nodes, elements=rows)
            mesh.elements = rows
            np.testing.assert_array_equal(mesh.elements, expected)
            assert_ccw(mesh)
            self.assertFalse(np.shares_memory(mesh.nodes, nodes))

    def test_invalid_assignments_are_atomic_and_identify_the_element(self):
        mesh = generate_rectilinear_mesh(1., [0., 2.], [0., 1.])
        nodes, elements = mesh.nodes, mesh.elements
        for bad in ([1, 5, 4, 2], [1, 2, 1, 4]):
            candidate = elements.copy()
            candidate[1] = bad
            with self.assertRaises(GeometryToleranceError) as caught:
                mesh.replace_data(nodes=nodes, elements=candidate)
            self.assertEqual(caught.exception.context['element_indices'], [1])
            self.assertIs(mesh.nodes, nodes)
            self.assertIs(mesh.elements, elements)
        candidate = nodes.copy()
        candidate[5, 0] = np.nan
        with self.assertRaises(GeometryToleranceError) as caught:
            mesh.nodes = candidate
        self.assertEqual(caught.exception.context['element_indices'], [1])
        self.assertIs(mesh.nodes, nodes)

    def test_mirrored_no_op_and_editor_outputs(self):
        for operation in (lambda m: EditorMesh2D.delete_element(m, []),
                          lambda m: EditorMesh2D.delete_element(m, [0]),
                          EditorMesh2D.equivalence, lambda m: EditorMesh2D.merge(m, m),
                          lambda m: imprint_circle(m, center=[20., 20.], radius=1.,
                                                   projection_distance=0., minimum_area=0.)):
            mesh = generate_rectilinear_mesh(1., [0., 2.], [0., 1.])
            mesh.nodes[:, 0] *= -1
            assert_ccw(operation(mesh))

    def test_adaptive_layers_normalize_writable_source(self):
        for full in (False, True):
            mesh = generate_rectilinear_mesh(.5, [0., 2.], [0., 2.])
            outline, = EditorMesh2D.outlines(mesh)
            original = mesh.elements.copy()
            mesh.elements[:] = original[:, [0, 3, 2, 1]]
            if full:
                result = generate_full_adaptive_mesh(mesh, [.5], outer_outline=[
                    [-2, -2], [-2, 4], [4, 4], [4, -2]])
            else:
                result = generate_layer_adaptive_mesh(mesh, outline, .5)
            np.testing.assert_array_equal(result.mesh.elements[:len(original)], original)
            assert_ccw(result.mesh)

    def test_sector_decreasing_angles_and_raw_diagnostics(self):
        mesh = build_inner_sector(np.zeros(2), 1., np.pi / 2, 0., 2)
        assert_ccw(mesh)
        raw = mesh.elements[:, [0, 2, 1, 1]]
        quality = MeshQualityChecker.from_arrays(mesh.nodes, raw).calculate_jacobian()
        self.assertTrue(all(reason & QualityReason.INVERTED for reason in quality.reason_flags))

    def test_exact_winding_at_extreme_scales(self):
        for scale in (1e-250, 1e250):
            mesh = Mesh2D([[0., 0.], [0., scale], [scale, 0.]], [[0, 1, 2, 2]])
            np.testing.assert_array_equal(mesh.elements, [[0, 2, 1, 1]])
