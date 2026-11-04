"""The shared XY contract applies before geometry selection or mesh mutation."""

from types import SimpleNamespace
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d import EditorMesh2D
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.adapter.unit import (
    build_unit_1_1_1_x, build_unit_1_1_x_y, build_unit_1_x_y_z,
)
from mesher.mesh2d.circular import extend_circular_mesh, imprint_circle
from mesher.mesh2d.circular.imprint_v2 import imprint_circle as imprint_v2
from mesher.mesh2d.quality import MeshQualityChecker


def rectangle():
    return Mesh2D([[0, 0], [4, 0], [4, 3], [0, 3]], [[0, 1, 2, 3]])


OUTLINE = [[0, 3], [3, 2], [2, 1], [1, 0]]
INVALID_Z = (1e-30, -1e-14, 17., 1e295, np.nan, np.inf, -np.inf)


class ZeroZModelTests(unittest.TestCase):
    def test_xy_xyz_empty_and_negative_zero_remain_owned_and_writable(self):
        for count in (0, 4):
            for columns in (2, 3):
                source = np.zeros((count, columns))
                if columns == 3:
                    source[:, 2] = -0.0
                mesh = Mesh2D(source, np.empty((0, 4), dtype=int))
                self.assertEqual(mesh.nodes.shape, (count, 3))
                self.assertEqual(mesh.nodes.dtype, np.float64)
                self.assertTrue(mesh.nodes.flags.c_contiguous)
                self.assertTrue(mesh.nodes.flags.writeable)
                self.assertFalse(np.shares_memory(mesh.nodes, source))
                np.testing.assert_array_equal(mesh.nodes[:, 2], 0.)

    def test_constructor_setter_and_replace_reject_z_atomically(self):
        for z in INVALID_Z:
            with self.subTest(z=z):
                mesh = rectangle()
                source = mesh.nodes.copy()
                source[2, 2] = z
                before = source.copy()
                nodes, elements = mesh.nodes, mesh.elements
                for write in (
                    lambda: Mesh2D(source, elements),
                    lambda: setattr(mesh, 'nodes', source),
                    lambda: mesh.replace_data(nodes=source, elements=elements),
                ):
                    with self.assertRaisesRegex(ValueError, r'Z=0; node 2 has Z='):
                        write()
                    self.assertIs(mesh.nodes, nodes)
                    self.assertIs(mesh.elements, elements)
                    np.testing.assert_array_equal(source, before)

    def test_direct_edits_can_be_inspected_and_repaired(self):
        mesh = rectangle()
        mesh.nodes[:, 0] += 10
        np.testing.assert_array_equal(EditorMesh2D.search_element(mesh, 10, 0, 14, 3), [0])
        for repair in ('in_place', 'setter', 'replace'):
            mesh.nodes[0, 2] = 1.
            self.assertEqual(mesh.node_count, 4)
            self.assertEqual(mesh.element_count, 1)
            self.assertEqual(mesh.nodes[0, 2], 1.)
            with self.assertRaisesRegex(ValueError, 'Z=0'):
                mesh.elements = mesh.elements
            if repair == 'in_place':
                mesh.nodes[:, 2] = 0.
            elif repair == 'setter':
                mesh.nodes = mesh.nodes[:, :2]
            else:
                mesh.replace_data(nodes=mesh.nodes[:, :2], elements=mesh.elements)
            self.assertEqual(len(EditorMesh2D.outline(mesh)), 1)


class ZeroZOperationTests(unittest.TestCase):
    def test_all_public_mesh_entrypoints_reject_unused_nonzero_z_before_work(self):
        from mesher.mesh2d.adapter.visualization import plot_quad_layer
        from mesher.mesh2d.visualization import build_faces, view_mesh
        from mesher.mesh25d.plan3d import LayerPlan, build_3d_mesh

        sides = tuple(np.array(side) for side in ([0, 1], [1, 2], [2, 3], [3, 0]))
        operations = {
            'delete_noop': lambda m: EditorMesh2D.delete_element(m, []),
            'merge_first': lambda m: EditorMesh2D.merge(m, rectangle()),
            'merge_second': lambda m: EditorMesh2D.merge(rectangle(), m),
            'search': lambda m: EditorMesh2D.search_element(m, 0, 0, 4, 3),
            'equivalence': lambda m: EditorMesh2D.equivalence(m),
            'outline': lambda m: EditorMesh2D.outline(m),
            'outlines_by_edge': lambda m: EditorMesh2D.outlines_by_edge(m),
            'plan_layer': lambda m: plan_quad_sections(m, OUTLINE, .5),
            'extend_layer': lambda m: generate_layer_adaptive_mesh(m, OUTLINE, .5),
            'unit111x': lambda m: build_unit_1_1_1_x(m, *sides),
            'unit11xy': lambda m: build_unit_1_1_x_y(m, *sides),
            'unit1xyz': lambda m: build_unit_1_x_y_z(m, *sides),
            'quality': lambda m: MeshQualityChecker(m),
            'faces': lambda m: build_faces(m),
            'viewer': lambda m: view_mesh(m),
            'layer_plot': lambda m: plot_quad_layer(SimpleNamespace(mesh=m), OUTLINE),
            'extrude': lambda m: build_3d_mesh(m, LayerPlan([])),
            'circular_extend': lambda m: extend_circular_mesh(
                m, element_size=.1, center_x=0, center_y=0, inner_radius=1, outer_radius=2),
            'imprint': lambda m: imprint_circle(m, center=[100, 100], radius=1, band_width=.1),
            'imprint_v2_noop': lambda m: imprint_v2(
                m, center=[100, 100], radius=1, projection_tolerance=0,
                guide_tolerance=0, merge_tolerance=0, minimum_area=0),
        }
        for name, operation in operations.items():
            for z in (1e-30, np.nan, np.inf):
                with self.subTest(operation=name, z=z):
                    mesh = rectangle()
                    mesh.nodes = np.vstack((mesh.nodes, [99., 99., 0.]))
                    mesh.nodes[-1, 2] = z  # No element references this node.
                    nodes, elements = mesh.nodes, mesh.elements
                    before_nodes, before_elements = nodes.copy(), elements.copy()
                    with self.assertRaisesRegex(ValueError, 'Z=0; node 4'):
                        operation(mesh)
                    self.assertIs(mesh.nodes, nodes)
                    self.assertIs(mesh.elements, elements)
                    np.testing.assert_array_equal(nodes, before_nodes)
                    np.testing.assert_array_equal(elements, before_elements)

    def test_layer_scaling_and_repeated_extension_stay_in_xy(self):
        results = []
        for scale in (1., 1000.):
            mesh = rectangle()
            mesh.nodes[:, :2] *= scale
            result = generate_layer_adaptive_mesh(mesh, OUTLINE, .5 * scale)
            generate_layer_adaptive_mesh(mesh, result.outline, .5 * scale)
            np.testing.assert_array_equal(mesh.nodes[:, 2], 0.)
            results.append(mesh)
        np.testing.assert_array_equal(results[0].elements, results[1].elements)
        np.testing.assert_allclose(results[0].nodes[:, :2], results[1].nodes[:, :2] / 1000., atol=1e-13)
        for scale in (1., 1000.):
            mesh = rectangle()
            mesh.nodes[:, :2] *= scale
            mesh.nodes[1, 2] = 1e-14 * scale
            with self.assertRaisesRegex(ValueError, 'Z=0; node 1'):
                generate_layer_adaptive_mesh(mesh, OUTLINE, .5 * scale)

    def test_extrusion_validates_z_but_generates_real_layer_heights(self):
        from mesher.mesh3d.extrusion import Dragger
        from mesher.mesh25d.plan3d import LayerPlan, build_3d_mesh

        mesh = rectangle()
        dragger = Dragger()
        for width in (2, 3, 4):
            nodes = np.zeros((mesh.node_count, width))
            nodes[:, :2] = mesh.nodes[:, :2]
            if width == 4:
                nodes[:, 3] = 123.  # Legacy extra columns are ignored.
            dragger.set_2D(nodes, mesh.elements)
        source, elements = dragger.node_2D, dragger.element_2D
        for z in INVALID_Z:
            invalid = mesh.nodes.copy()
            invalid[1, 2] = z
            with self.assertRaisesRegex(ValueError, 'Z=0; node 1'):
                dragger.set_2D(invalid, mesh.elements)
            self.assertIs(dragger.node_2D, source)
            self.assertIs(dragger.element_2D, elements)
        plan = LayerPlan([
            {'z': 5., 'element_size': 1., 'assignments': [
                {'type': 3, 'face': None, 'areas': [{'priority': 1., 'material': 'Cu'}]}]},
            {'z': 7., 'assignments': []},
        ])
        solid = build_3d_mesh(mesh, plan)
        np.testing.assert_array_equal(np.unique(solid.nodes[:, 2]), [5., 6., 7.])
        self.assertEqual(solid.element_count, 2)
        np.testing.assert_array_equal(mesh.nodes[:, 2], 0.)
