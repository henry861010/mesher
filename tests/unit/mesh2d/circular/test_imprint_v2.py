import inspect
import time
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.circular.imprint_v2.utils import (
    get_circle_intersect,
    get_inner_outer_areas,
    get_intersect_nodes,
    get_tri_quad,
)


class FunctionSignatureTests(unittest.TestCase):
    def test_all_functions_end_with_optional_indices_parameter(self):
        functions = (
            get_circle_intersect,
            get_inner_outer_areas,
            get_intersect_nodes,
            get_tri_quad,
        )

        for function in functions:
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters.values()
                parameter = tuple(parameters)[-1]
                self.assertEqual(parameter.name, "indices")
                self.assertIsNone(parameter.default)


class GetCircleIntersectTests(unittest.TestCase):
    @staticmethod
    def _mesh_from_polygons(polygons):
        nodes = []
        elements = []
        for polygon in polygons:
            start = len(nodes)
            polygon = np.asarray(polygon, dtype=np.float64)
            if polygon.shape[0] == 3:
                nodes.extend(np.column_stack((polygon, np.zeros(3))))
                elements.append([start, start + 1, start + 2, start + 2])
            else:
                nodes.extend(np.column_stack((polygon, np.zeros(4))))
                elements.append([start, start + 1, start + 2, start + 3])
        return Mesh2D(nodes=nodes, elements=elements)

    def test_orders_mixed_elements_counter_clockwise_from_positive_x(self):
        east = [[0.5, -0.2], [1.5, -0.2], [1.5, 0.2], [0.5, 0.2]]
        north = [[-0.2, 0.5], [0.2, 0.5], [0.2, 1.5], [-0.2, 1.5]]
        west_triangle = [[-0.5, -0.2], [-0.5, 0.2], [-1.5, 0.0]]
        south = [[-0.2, -1.5], [0.2, -1.5], [0.2, -0.5], [-0.2, -0.5]]
        mesh = self._mesh_from_polygons(
            [south, west_triangle, east, north]
        )

        result = get_circle_intersect(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(result, [2, 3, 1, 0])
        self.assertEqual(result.dtype, np.dtype(np.int64))

    def test_includes_edge_tangent_and_single_vertex_contact(self):
        tangent_quad = [[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]]
        vertex_triangle = [[0.0, 1.0], [2.0, 2.0], [-2.0, 2.0]]
        mesh = self._mesh_from_polygons([vertex_triangle, tangent_quad])

        result = get_circle_intersect(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(result, [1, 0])

    def test_uses_all_angles_for_lexicographic_ties(self):
        one_intersection = [[1.0, 0.0], [2.0, -0.3], [2.0, 0.3]]
        second_at_ninety = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        second_at_sixty = [
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
            [0.0, 0.0],
        ]
        duplicate_second_at_sixty = second_at_sixty
        mesh = self._mesh_from_polygons(
            [
                second_at_ninety,
                duplicate_second_at_sixty,
                one_intersection,
                second_at_sixty,
            ]
        )

        result = get_circle_intersect(mesh, 0.0, 0.0, 1.0)

        # The one-angle key is the shortest common prefix.  The two 60-degree
        # keys then tie by their original row indices, before the 90-degree key.
        np.testing.assert_array_equal(result, [2, 1, 3, 0])

    def test_deduplicates_contacts_at_shared_vertices_and_edges(self):
        mesh = Mesh2D(
            nodes=[
                [1.0, 0.0, 9.0],
                [0.0, 1.0, 8.0],
                [0.0, 0.0, 7.0],
                [2.0, 1.0, 6.0],
            ],
            elements=[
                [0, 1, 2, 2],
                [1, 0, 3, 3],
            ],
        )

        result = get_circle_intersect(mesh, 0.0, 0.0, 1.0)

        # Both rows share the complete chord from 0 to 90 degrees.  Each row
        # remains present and their identical keys are resolved by row index.
        np.testing.assert_array_equal(result, [0, 1])

    def test_excludes_nonintersecting_perimeters(self):
        inside = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
        outside = [[2.0, -0.2], [3.0, -0.2], [3.0, 0.2], [2.0, 0.2]]
        containing = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
        mesh = self._mesh_from_polygons([inside, outside, containing])

        result = get_circle_intersect(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(result, np.empty(0, dtype=np.int64))

    def test_only_processes_selected_elements_and_returns_global_indices(self):
        east = [[0.5, -0.2], [1.5, -0.2], [1.5, 0.2], [0.5, 0.2]]
        north = [[-0.2, 0.5], [0.2, 0.5], [0.2, 1.5], [-0.2, 1.5]]
        west = [[-1.5, -0.2], [-0.5, -0.2], [-0.5, 0.2], [-1.5, 0.2]]
        mesh = self._mesh_from_polygons([east, north, west])

        result = get_circle_intersect(
            mesh, 0.0, 0.0, 1.0, indices=[2, 0]
        )

        np.testing.assert_array_equal(result, [0, 2])

        empty = get_circle_intersect(
            mesh, 0.0, 0.0, 1.0, indices=[]
        )
        np.testing.assert_array_equal(empty, np.empty(0, dtype=np.int64))

    def test_handles_offset_center_seam_and_ignores_z(self):
        center = np.array([10.0, -4.0])
        near_positive_side = np.array(
            [[0.5, -0.1], [1.5, -0.1], [1.5, 0.1], [0.5, 0.1]]
        )
        upper = np.array(
            [[-0.1, 0.5], [0.1, 0.5], [0.1, 1.5], [-0.1, 1.5]]
        )
        mesh = self._mesh_from_polygons(
            [upper + center, near_positive_side + center]
        )
        mesh.nodes[:, 2] = np.linspace(-1.0e9, 1.0e9, mesh.node_count)

        result = get_circle_intersect(
            mesh, center[0], center[1], 1.0
        )

        np.testing.assert_array_equal(result, [1, 0])

    def test_validates_circle_and_mutated_mesh_data(self):
        mesh = self._mesh_from_polygons(
            [[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]
        )
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            get_circle_intersect(object(), 0.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            get_circle_intersect(mesh, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            get_circle_intersect(mesh, np.nan, 0.0, 1.0)

        mesh.nodes[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite XY"):
            get_circle_intersect(mesh, 0.0, 0.0, 1.0)

    def test_empty_mesh_returns_empty_int64_array(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3)),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        result = get_circle_intersect(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(result, np.empty(0, dtype=np.int64))


class GetCircleIntersectPerformanceTests(unittest.TestCase):
    maximum_seconds = 10.0

    def test_one_hundred_thousand_elements_with_effective_broad_phase(self):
        nx = 1000
        ny = 100
        x, y = np.meshgrid(
            np.arange(nx + 1, dtype=np.float64),
            np.arange(ny + 1, dtype=np.float64),
        )
        nodes = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        lower_left = np.arange(ny * (nx + 1), dtype=np.int32).reshape(
            ny, nx + 1
        )[:, :-1]
        elements = np.column_stack(
            (
                lower_left.ravel(),
                lower_left.ravel() + 1,
                lower_left.ravel() + nx + 2,
                lower_left.ravel() + nx + 1,
            )
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        result = get_circle_intersect(mesh, 500.0, 50.0, 30.0)
        elapsed = time.perf_counter() - started

        self.assertGreater(result.size, 0)
        self.assertLess(elapsed, self.maximum_seconds)

    def test_one_hundred_thousand_elements_all_reach_exact_sorting(self):
        count = 100_000
        square = np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ]
        )
        nodes = np.tile(square, (count, 1))
        elements = np.arange(count * 4, dtype=np.int32).reshape(count, 4)
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        result = get_circle_intersect(mesh, 0.0, 0.0, 2.2)
        elapsed = time.perf_counter() - started

        np.testing.assert_array_equal(
            result, np.arange(count, dtype=np.int64)
        )
        self.assertLess(elapsed, self.maximum_seconds)


class GetInnerOuterAreasTests(unittest.TestCase):
    @staticmethod
    def _mesh_from_polygons(polygons):
        nodes = []
        elements = []
        for polygon in polygons:
            start = len(nodes)
            polygon = np.asarray(polygon, dtype=np.float64)
            nodes.extend(
                np.column_stack(
                    (polygon, np.zeros(polygon.shape[0], dtype=np.float64))
                )
            )
            if polygon.shape[0] == 3:
                elements.append([start, start + 1, start + 2, start + 2])
            else:
                elements.append([start, start + 1, start + 2, start + 3])
        return Mesh2D(nodes=nodes, elements=elements)

    def test_returns_inner_and_outer_areas_in_requested_order(self):
        inside_triangle = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]
        outside_quad = [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]]
        partial_triangle = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
        containing_quad = [
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
        ]
        mesh = self._mesh_from_polygons(
            [inside_triangle, outside_quad, partial_triangle, containing_quad]
        )

        inner, outer = get_inner_outer_areas(mesh, 0.0, 0.0, 1.0, [2, 0, 3, 1])

        np.testing.assert_allclose(
            inner,
            [np.pi / 4.0, 0.125, np.pi, 0.0],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            outer,
            [2.0 - np.pi / 4.0, 0.0, 16.0 - np.pi, 1.0],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        self.assertEqual(inner.dtype, np.dtype(np.float64))
        self.assertEqual(outer.dtype, np.dtype(np.float64))

    def test_handles_tangency_offset_center_reversed_winding_and_ignores_z(self):
        center = np.array([10.0, -4.0])
        containing_clockwise = (
            np.array(
                [
                    [-2.0, -2.0],
                    [-2.0, 2.0],
                    [2.0, 2.0],
                    [2.0, -2.0],
                ]
            )
            + center
        )
        tangent_outside = (
            np.array(
                [[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]]
            )
            + center
        )
        mesh = self._mesh_from_polygons(
            [containing_clockwise, tangent_outside]
        )
        mesh.nodes[:, 2] = np.linspace(-1.0e12, 1.0e12, mesh.node_count)

        inner, outer = get_inner_outer_areas(
            mesh, center[0], center[1], 1.0, np.array([0, 1])
        )

        np.testing.assert_allclose(inner, [np.pi, 0.0], atol=1.0e-13)
        np.testing.assert_allclose(outer, [16.0 - np.pi, 1.0], atol=1.0e-13)

    def test_empty_indices_return_empty_float64_arrays(self):
        mesh = self._mesh_from_polygons(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]
        )

        inner, outer = get_inner_outer_areas(
            mesh, 0.0, 0.0, 1.0, np.empty(0, dtype=np.int32)
        )

        np.testing.assert_array_equal(inner, np.empty(0, dtype=np.float64))
        np.testing.assert_array_equal(outer, np.empty(0, dtype=np.float64))

    def test_none_processes_all_elements_in_mesh_order(self):
        mesh = self._mesh_from_polygons(
            [
                [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
                [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]],
            ]
        )

        default_inner, default_outer = get_inner_outer_areas(mesh, 0.0, 0.0, 1.0)
        explicit_inner, explicit_outer = get_inner_outer_areas(
            mesh, 0.0, 0.0, 1.0, indices=[0, 1]
        )

        np.testing.assert_array_equal(default_inner, explicit_inner)
        np.testing.assert_array_equal(default_outer, explicit_outer)

    def test_validates_mesh_circle_and_indices(self):
        mesh = self._mesh_from_polygons(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            get_inner_outer_areas(object(), 0.0, 0.0, 1.0, [0])
        with self.assertRaisesRegex(ValueError, "positive"):
            get_inner_outer_areas(mesh, 0.0, 0.0, 0.0, [0])
        with self.assertRaisesRegex(ValueError, "finite"):
            get_inner_outer_areas(mesh, np.nan, 0.0, 1.0, [0])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            get_inner_outer_areas(mesh, 0.0, 0.0, 1.0, [[0]])
        with self.assertRaisesRegex(TypeError, "integers"):
            get_inner_outer_areas(mesh, 0.0, 0.0, 1.0, [0.0])
        with self.assertRaisesRegex(TypeError, "integers"):
            get_inner_outer_areas(mesh, 0.0, 0.0, 1.0, np.array([True]))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            get_inner_outer_areas(mesh, 0.0, 0.0, 1.0, [0, 0])
        with self.assertRaisesRegex(IndexError, "out of range"):
            get_inner_outer_areas(mesh, 0.0, 0.0, 1.0, [1])


class GetInnerOuterAreasPerformanceTests(unittest.TestCase):
    maximum_seconds = 5.0

    def test_one_hundred_thousand_selected_elements(self):
        nx = 1000
        ny = 100
        x, y = np.meshgrid(
            np.arange(nx + 1, dtype=np.float64),
            np.arange(ny + 1, dtype=np.float64),
        )
        nodes = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        lower_left = np.arange(ny * (nx + 1), dtype=np.int32).reshape(
            ny, nx + 1
        )[:, :-1]
        elements = np.column_stack(
            (
                lower_left.ravel(),
                lower_left.ravel() + 1,
                lower_left.ravel() + nx + 2,
                lower_left.ravel() + nx + 1,
            )
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)
        indices = np.arange(mesh.element_count, dtype=np.int64)

        started = time.perf_counter()
        inner, outer = get_inner_outer_areas(mesh, 500.0, 50.0, 30.0, indices)
        elapsed = time.perf_counter() - started

        np.testing.assert_allclose(np.sum(inner), np.pi * 30.0**2)
        np.testing.assert_allclose(inner + outer, np.ones(mesh.element_count))
        self.assertLess(elapsed, self.maximum_seconds)


class GetIntersectNodesTests(unittest.TestCase):
    @staticmethod
    def _mesh_from_polygons(polygons):
        nodes = []
        elements = []
        for polygon in polygons:
            start = len(nodes)
            polygon = np.asarray(polygon, dtype=np.float64)
            nodes.extend(np.column_stack((polygon, np.zeros(len(polygon)))))
            if len(polygon) == 3:
                elements.append([start, start + 1, start + 2, start + 2])
            else:
                elements.append([start, start + 1, start + 2, start + 3])
        return Mesh2D(nodes=nodes, elements=elements)

    def test_returns_counts_and_counter_clockwise_padded_nodes(self):
        no_intersection = [
            [-0.2, -0.2],
            [0.2, -0.2],
            [0.2, 0.2],
            [-0.2, 0.2],
        ]
        two_intersections = [
            [0.0, -0.2],
            [2.0, -0.2],
            [2.0, 0.2],
            [0.0, 0.2],
        ]
        four_intersections = [
            [-2.0, -0.2],
            [2.0, -0.2],
            [2.0, 0.2],
            [-2.0, 0.2],
        ]
        mesh = self._mesh_from_polygons(
            [no_intersection, two_intersections, four_intersections]
        )

        counts, nodes = get_intersect_nodes(mesh, 0.0, 0.0, 1.0)

        x = np.sqrt(1.0 - 0.2**2)
        np.testing.assert_array_equal(counts, [0, 2, 4])
        np.testing.assert_allclose(
            nodes[1, :2], [[x, 0.2], [x, -0.2]], atol=1.0e-14
        )
        np.testing.assert_allclose(
            nodes[2, :4],
            [[x, 0.2], [-x, 0.2], [-x, -0.2], [x, -0.2]],
            atol=1.0e-14,
        )
        np.testing.assert_array_equal(nodes[0], np.zeros((8, 2)))
        np.testing.assert_array_equal(nodes[1, 2:], np.zeros((6, 2)))
        np.testing.assert_array_equal(nodes[2, 4:], np.zeros((4, 2)))
        self.assertEqual(counts.dtype, np.dtype(np.int64))
        self.assertEqual(nodes.dtype, np.dtype(np.float64))
        self.assertEqual(nodes.shape, (3, 8, 2))

    def test_supports_eight_quad_intersections_without_truncation(self):
        mesh = self._mesh_from_polygons(
            [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]]
        )

        counts, nodes = get_intersect_nodes(mesh, 0.0, 0.0, 1.1)

        offset = np.sqrt(1.1**2 - 1.0)
        expected = [
            [1.0, offset],
            [offset, 1.0],
            [-offset, 1.0],
            [-1.0, offset],
            [-1.0, -offset],
            [-offset, -1.0],
            [offset, -1.0],
            [1.0, -offset],
        ]
        np.testing.assert_array_equal(counts, [8])
        np.testing.assert_allclose(nodes[0], expected, atol=1.0e-14)

    def test_deduplicates_tangencies_and_shared_vertex_contacts(self):
        tangent_quad = [
            [1.0, -0.5],
            [2.0, -0.5],
            [2.0, 0.5],
            [1.0, 0.5],
        ]
        vertex_triangle = [[1.0, 0.0], [2.0, 1.0], [2.0, -1.0]]
        mesh = self._mesh_from_polygons([tangent_quad, vertex_triangle])

        counts, nodes = get_intersect_nodes(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(counts, [1, 1])
        np.testing.assert_allclose(nodes[0, 0], [1.0, 0.0], atol=1.0e-14)
        np.testing.assert_allclose(nodes[1, 0], [1.0, 0.0], atol=1.0e-14)
        np.testing.assert_array_equal(nodes[0, 1:], np.zeros((7, 2)))
        np.testing.assert_array_equal(nodes[1, 1:], np.zeros((7, 2)))

    def test_handles_offset_center_reversed_winding_and_ignores_z(self):
        center = np.array([10.0, -4.0])
        polygon = (
            np.array(
                [[0.0, 0.2], [2.0, 0.2], [2.0, -0.2], [0.0, -0.2]]
            )
            + center
        )
        mesh = self._mesh_from_polygons([polygon])
        mesh.nodes[:, 2] = np.linspace(-1.0e12, 1.0e12, mesh.node_count)

        counts, nodes = get_intersect_nodes(
            mesh, center[0], center[1], 1.0
        )

        x = np.sqrt(1.0 - 0.2**2)
        np.testing.assert_array_equal(counts, [2])
        np.testing.assert_allclose(
            nodes[0, :2], center + [[x, 0.2], [x, -0.2]], atol=1.0e-14
        )

    def test_empty_mesh_returns_correctly_shaped_arrays(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3)),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        counts, nodes = get_intersect_nodes(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(counts, np.empty(0, dtype=np.int64))
        np.testing.assert_array_equal(
            nodes, np.empty((0, 8, 2), dtype=np.float64)
        )

    def test_returns_only_selected_elements_in_requested_order(self):
        no_intersection = [
            [-0.2, -0.2],
            [0.2, -0.2],
            [0.2, 0.2],
            [-0.2, 0.2],
        ]
        two_intersections = [
            [0.0, -0.2],
            [2.0, -0.2],
            [2.0, 0.2],
            [0.0, 0.2],
        ]
        four_intersections = [
            [-2.0, -0.2],
            [2.0, -0.2],
            [2.0, 0.2],
            [-2.0, 0.2],
        ]
        mesh = self._mesh_from_polygons(
            [no_intersection, two_intersections, four_intersections]
        )

        counts, nodes = get_intersect_nodes(
            mesh, 0.0, 0.0, 1.0, indices=[2, 0]
        )

        np.testing.assert_array_equal(counts, [4, 0])
        self.assertEqual(nodes.shape, (2, 8, 2))
        np.testing.assert_array_equal(nodes[1], np.zeros((8, 2)))

        empty_counts, empty_nodes = get_intersect_nodes(
            mesh, 0.0, 0.0, 1.0, indices=[]
        )
        np.testing.assert_array_equal(
            empty_counts, np.empty(0, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            empty_nodes, np.empty((0, 8, 2), dtype=np.float64)
        )

    def test_validates_mesh_circle_and_mutated_mesh_data(self):
        mesh = self._mesh_from_polygons(
            [[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]
        )
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            get_intersect_nodes(object(), 0.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            get_intersect_nodes(mesh, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            get_intersect_nodes(mesh, np.nan, 0.0, 1.0)

        mesh.nodes[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite XY"):
            get_intersect_nodes(mesh, 0.0, 0.0, 1.0)


class GetTriQuadTests(unittest.TestCase):
    @staticmethod
    def _mesh(elements):
        elements = np.asarray(elements, dtype=np.int32).reshape(-1, 4)
        node_count = int(elements.max()) + 1 if elements.size else 0
        return Mesh2D(
            nodes=np.zeros((node_count, 3), dtype=np.float64),
            elements=elements,
        )

    def test_returns_mixed_element_indices_in_row_order(self):
        mesh = self._mesh(
            [
                [0, 1, 2, 2],
                [2, 3, 4, 5],
                [5, 6, 7, 7],
                [7, 8, 9, 10],
            ]
        )

        triangle_indices, quadrilateral_indices = get_tri_quad(mesh)

        np.testing.assert_array_equal(triangle_indices, [0, 2])
        np.testing.assert_array_equal(quadrilateral_indices, [1, 3])
        self.assertEqual(triangle_indices.dtype, np.dtype(np.int64))
        self.assertEqual(quadrilateral_indices.dtype, np.dtype(np.int64))

    def test_handles_single_topology_meshes(self):
        cases = (
            ([[0, 1, 2, 2], [2, 3, 4, 4]], [0, 1], []),
            ([[0, 1, 2, 3], [3, 4, 5, 6]], [], [0, 1]),
        )
        for elements, expected_triangles, expected_quadrilaterals in cases:
            with self.subTest(elements=elements):
                mesh = self._mesh(elements)

                triangle_indices, quadrilateral_indices = get_tri_quad(mesh)

                np.testing.assert_array_equal(
                    triangle_indices, expected_triangles
                )
                np.testing.assert_array_equal(
                    quadrilateral_indices, expected_quadrilaterals
                )

    def test_empty_mesh_returns_empty_int64_arrays(self):
        mesh = self._mesh([])

        triangle_indices, quadrilateral_indices = get_tri_quad(mesh)

        np.testing.assert_array_equal(
            triangle_indices, np.empty(0, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            quadrilateral_indices, np.empty(0, dtype=np.int64)
        )

    def test_only_uses_the_last_two_connectivity_entries(self):
        mesh = self._mesh(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 2],
            ]
        )

        triangle_indices, quadrilateral_indices = get_tri_quad(mesh)

        np.testing.assert_array_equal(triangle_indices, [0])
        np.testing.assert_array_equal(quadrilateral_indices, [1])

    def test_only_classifies_selected_elements_in_requested_order(self):
        mesh = self._mesh(
            [
                [0, 1, 2, 2],
                [2, 3, 4, 5],
                [5, 6, 7, 7],
                [7, 8, 9, 10],
            ]
        )

        triangle_indices, quadrilateral_indices = get_tri_quad(
            mesh, indices=[3, 2, 0]
        )

        np.testing.assert_array_equal(triangle_indices, [2, 0])
        np.testing.assert_array_equal(quadrilateral_indices, [3])

    def test_rejects_non_mesh_input(self):
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            get_tri_quad(object())


if __name__ == "__main__":
    unittest.main()
