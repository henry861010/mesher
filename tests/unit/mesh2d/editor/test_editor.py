import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d import EditorMesh2D

from ._fixtures import empty_mesh, two_quads


class EditorMesh2DTests(unittest.TestCase):
    def test_static_api_has_no_mesh_or_cache_and_is_not_exported_at_root(self):
        import mesher
        from mesher.mesh2d.editor import EditorMesh2D as EditorFromPackage

        self.assertIs(EditorFromPackage, EditorMesh2D)
        self.assertNotIn("EditorMesh2D", mesher.__all__)
        for name in ("delete_element", "merge", "search_element", "equivalence", "outline", "outlines"):
            self.assertIsInstance(EditorMesh2D.__dict__[name], staticmethod)
        with self.assertRaises(AttributeError):
            EditorMesh2D().mesh = two_quads()

    def test_interleaved_calls_read_current_mesh_arrays(self):
        first, second = two_quads(), two_quads()
        np.testing.assert_array_equal(EditorMesh2D.search_element(first, 0, 0, 1, 1), [0])
        second.nodes[:, 0] += 10
        self.assertEqual(EditorMesh2D.search_element(second, 0, 0, 1, 1).size, 0)
        first.nodes[:, 0] += 10
        self.assertEqual(EditorMesh2D.search_element(first, 0, 0, 1, 1).size, 0)
        np.testing.assert_array_equal(EditorMesh2D.search_element(second, 10, 0, 11, 1), [0])

    def test_delete_duplicates_preserves_nodes_and_remaining_order(self):
        mesh = two_quads()
        nodes, elements = mesh.nodes, mesh.elements.copy()
        self.assertIs(EditorMesh2D.delete_element(mesh, [1, 1]), mesh)
        self.assertIs(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.elements, elements[:1])
        EditorMesh2D.delete_element(mesh, [0])
        self.assertEqual(mesh.elements.shape, (0, 4))
        self.assertEqual(mesh.node_count, 6)

    def test_delete_empty_is_no_op(self):
        mesh = two_quads()
        elements = mesh.elements
        EditorMesh2D.delete_element(mesh, [])
        self.assertIs(mesh.elements, elements)
        EditorMesh2D.delete_element(empty_mesh(), [])

    def test_delete_invalid_indices_are_atomic(self):
        for indices, error in (([2], IndexError), ([-1], IndexError),
                               ([0.0], TypeError), ([True], TypeError),
                               ([[0]], ValueError), (0, ValueError)):
            with self.subTest(indices=indices):
                mesh = two_quads()
                nodes, elements = mesh.nodes, mesh.elements
                with self.assertRaises(error):
                    EditorMesh2D.delete_element(mesh, indices)
                self.assertIs(mesh.nodes, nodes)
                self.assertIs(mesh.elements, elements)

    def test_merge_offsets_triangles_and_owns_independent_arrays(self):
        first = two_quads()
        second = Mesh2D([[0, 0], [1, 0], [0, 1]], [[0, 1, 2, 2]])
        result = EditorMesh2D.merge(first, second)
        np.testing.assert_array_equal(result.nodes, np.concatenate((first.nodes, second.nodes)))
        np.testing.assert_array_equal(result.elements, [[0, 1, 4, 3], [1, 2, 5, 4], [6, 7, 8, 8]])
        self.assertEqual(result.nodes.dtype, np.float64)
        self.assertEqual(result.elements.dtype, np.int32)
        for source in (first, second):
            self.assertFalse(np.shares_memory(result.nodes, source.nodes))
            self.assertFalse(np.shares_memory(result.elements, source.elements))
        result.nodes[:] = 100
        result.elements[:] = 0
        self.assertEqual(first.node_count, 6)
        self.assertEqual(second.element_count, 1)
        self.assertEqual(first.nodes[0, 0], 0)
        np.testing.assert_array_equal(second.elements, [[0, 1, 2, 2]])

    def test_merge_self_and_empty_inputs(self):
        mesh = two_quads()
        result = EditorMesh2D.merge(mesh, mesh)
        np.testing.assert_array_equal(result.elements[2:], mesh.elements + 6)
        self.assertEqual(mesh.element_count, 2)
        for first, second in ((mesh, empty_mesh()), (empty_mesh(), mesh)):
            result = EditorMesh2D.merge(first, second)
            np.testing.assert_array_equal(result.nodes, mesh.nodes)
            np.testing.assert_array_equal(result.elements, mesh.elements)
            self.assertIsNot(result, mesh)
        result = EditorMesh2D.merge(empty_mesh(), empty_mesh())
        self.assertEqual(result.nodes.shape, (0, 3))
        self.assertEqual(result.elements.shape, (0, 4))

    def test_search_inclusive_reversed_bounds_and_all_vertex_rule(self):
        mesh = two_quads()
        for bounds in ((0, 0, 1, 1), (1, 1, 0, 0), (1, 0, 0, 1)):
            result = EditorMesh2D.search_element(mesh, *bounds)
            np.testing.assert_array_equal(result, [0])
            self.assertEqual(result.dtype, np.int64)
        self.assertEqual(EditorMesh2D.search_element(mesh, 0.5, 0, 1.5, 1).size, 0)
        np.testing.assert_array_equal(EditorMesh2D.search_element(mesh, 0, 0, 2, 1), [0, 1])
        self.assertEqual(EditorMesh2D.search_element(mesh, 1, 0, 1, 1).size, 0)

    def test_search_supports_triangles_empty_mesh_and_invalid_bounds(self):
        triangle = Mesh2D([[0, 0], [1, 0], [0, 1]], [[0, 1, 2, 2]])
        np.testing.assert_array_equal(EditorMesh2D.search_element(triangle, 0, 0, 1, 1), [0])
        self.assertEqual(EditorMesh2D.search_element(empty_mesh(), 0, 0, 1, 1).shape, (0,))
        for value in (np.nan, np.inf, None, [0, 1], 1j):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EditorMesh2D.search_element(triangle, value, 0, 1, 1)

    def test_equivalence_welds_xy_preserves_lowest_id_xy_and_duplicate_cells(self):
        mesh = two_quads()
        other = two_quads()
        other.nodes[:, :2] += 0.0001
        merged = EditorMesh2D.merge(mesh, other)
        self.assertIs(EditorMesh2D.equivalence(merged), merged)
        np.testing.assert_array_equal(merged.nodes, mesh.nodes)
        np.testing.assert_array_equal(merged.elements, np.tile(mesh.elements, (2, 1)))

    def test_equivalence_is_transitive_and_boundary_inclusive(self):
        nodes = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        mesh = Mesh2D(
            np.concatenate((nodes, nodes + [0.125, 0], nodes + [0.25, 0])),
            [[0, 1, 2, 2], [3, 4, 5, 5], [6, 7, 8, 8]],
        )
        EditorMesh2D.equivalence(mesh, 0.125)
        self.assertEqual(mesh.node_count, 3)
        np.testing.assert_array_equal(mesh.elements, np.tile([0, 1, 2, 2], (3, 1)))

    def test_exact_equivalence_keeps_nearby_distinct_nodes(self):
        first = Mesh2D([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2, 2]])
        second = Mesh2D([[-0.0, 0, 0], [1.0001, 0, 0], [0, 1, 0]], [[0, 1, 2, 2]])
        mesh = EditorMesh2D.merge(first, second)
        EditorMesh2D.equivalence(mesh, 0)
        np.testing.assert_array_equal(mesh.nodes, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1.0001, 0, 0]])
        np.testing.assert_array_equal(mesh.elements, [[0, 1, 2, 2], [0, 3, 2, 2]])

    def test_quad_collapse_along_any_edge_becomes_tri3(self):
        for edge in range(4):
            with self.subTest(edge=edge):
                nodes = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
                nodes[(edge + 1) % 4] = nodes[edge] + [0.01, 0.01]
                mesh = Mesh2D(nodes, [[0, 1, 2, 3]])
                EditorMesh2D.equivalence(mesh, 0.1)
                self.assertEqual(mesh.node_count, 3)
                self.assertEqual(mesh.element_count, 1)
                self.assertEqual(mesh.elements[0, 2], mesh.elements[0, 3])
                self.assertEqual(len(np.unique(mesh.elements[0, :3])), 3)

    def test_removes_collapsed_triangles_nonadjacent_duplicates_and_zero_area(self):
        for nodes, elements, tolerance in (
            ([[0, 0], [0.01, 0], [0, 2]], [[0, 1, 2, 2]], 0.1),
            ([[0, 0], [1, 0], [0, 1]], [[0, 1, 0, 2]], 0),
            ([[0, 0], [1, 0], [2, 0]], [[0, 1, 2, 2]], 0),
        ):
            mesh = Mesh2D(nodes, elements)
            EditorMesh2D.equivalence(mesh, tolerance)
            self.assertEqual(mesh.nodes.shape, (0, 3))
            self.assertEqual(mesh.elements.shape, (0, 4))

    def test_keeps_clockwise_concave_and_very_small_elements(self):
        for nodes, element in (
            ([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 2, 3]),
            ([[0, 0], [2, 0], [0.5, 0.5], [0, 2]], [0, 1, 2, 3]),
            ([[0, 0], [1e-200, 0], [0, 1e-200]], [0, 1, 2, 2]),
            ([[-1e308, -1e308], [1e308, -1e308], [0, 1e308]], [0, 1, 2, 2]),
        ):
            mesh = Mesh2D(nodes, [element])
            before = mesh.elements.copy()
            EditorMesh2D.equivalence(mesh, 0)
            np.testing.assert_array_equal(mesh.elements, before)

    def test_unreferenced_nodes_do_not_supply_representatives_or_bridge_groups(self):
        nodes = [[0.09, 0], [0, 0], [0.18, 0], [0, 2], [2, 2]]
        mesh = Mesh2D(nodes, [[1, 3, 4, 4], [2, 3, 4, 4]])
        EditorMesh2D.equivalence(mesh, 0.1)
        expected_nodes = np.column_stack((np.asarray(nodes)[1:], np.zeros(4)))
        np.testing.assert_array_equal(mesh.nodes, expected_nodes)
        np.testing.assert_array_equal(mesh.elements, [[0, 2, 3, 3], [1, 2, 3, 3]])

    def test_empty_equivalence_removes_orphans_and_unchanged_mesh_is_no_op(self):
        mesh = Mesh2D([[np.nan, 0, 0]], np.empty((0, 4), dtype=np.int32))
        EditorMesh2D.equivalence(mesh)
        self.assertEqual(mesh.nodes.shape, (0, 3))
        mesh = two_quads()
        nodes, elements = mesh.nodes, mesh.elements
        EditorMesh2D.equivalence(mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_invalid_tolerances_and_referenced_coordinates_are_atomic(self):
        for tolerance in (-1, np.nan, np.inf, True, None, 1j):
            with self.subTest(tolerance=tolerance):
                mesh = two_quads()
                nodes, elements = mesh.nodes, mesh.elements
                with self.assertRaises(ValueError):
                    EditorMesh2D.equivalence(mesh, tolerance)
                self.assertIs(mesh.nodes, nodes)
                self.assertIs(mesh.elements, elements)
        mesh = two_quads()
        mesh.nodes[0, 0] = np.nan
        nodes, elements = mesh.nodes, mesh.elements
        with self.assertRaisesRegex(ValueError, "finite XY"):
            EditorMesh2D.equivalence(mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_wrong_mesh_types_are_rejected(self):
        operations = (
            lambda: EditorMesh2D.delete_element(None, []),
            lambda: EditorMesh2D.merge(two_quads(), None),
            lambda: EditorMesh2D.search_element(None, 0, 0, 1, 1),
            lambda: EditorMesh2D.equivalence(None),
            lambda: EditorMesh2D.outline(None),
            lambda: EditorMesh2D.outlines(None),
        )
        for operation in operations:
            with self.assertRaises(TypeError):
                operation()

