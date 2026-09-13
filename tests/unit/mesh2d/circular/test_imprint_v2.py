import inspect
import random
import time
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.circular.imprint_v2.utils import (
    get_circle_intersect,
    get_inner_outer_areas,
    get_intersect_nodes,
    get_tri_quad,
    imprint_circle,
    to_circle,
)
from mesher.mesh2d.circular.utils.pattern_segments import _PatternGuideSet


class FunctionSignatureTests(unittest.TestCase):
    def test_all_functions_end_with_optional_indices_parameter(self):
        functions = (
            get_circle_intersect,
            get_inner_outer_areas,
            get_intersect_nodes,
            get_tri_quad,
            imprint_circle,
            to_circle,
        )

        for function in functions:
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters.values()
                parameter = tuple(parameters)[-1]
                self.assertEqual(parameter.name, "indices")
                self.assertIsNone(parameter.default)

    def test_imprint_circle_has_the_requested_public_signature(self):
        self.assertEqual(
            tuple(inspect.signature(imprint_circle).parameters),
            (
                "mesh",
                "center_x",
                "center_y",
                "radius",
                "tolerance",
                "indices",
            ),
        )


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


class ToCircleTests(unittest.TestCase):
    @staticmethod
    def _mesh(nodes, elements=None):
        nodes = np.asarray(nodes, dtype=np.float64)
        if elements is None:
            elements = np.empty((0, 4), dtype=np.int32)
        return Mesh2D(nodes=nodes, elements=elements)

    @staticmethod
    def _snapshot(mesh):
        return mesh.nodes.copy(), mesh.elements.copy()

    def test_moves_inside_and_outside_nodes_radially_and_preserves_z(self):
        mesh = self._mesh(
            [
                [0.9, 0.0, 7.0],
                [0.0, 1.1, 8.0],
                [-1.0, 0.0, 9.0],
                [2.0, 2.0, 10.0],
                [0.0, 0.0, 11.0],
            ],
            [[0, 1, 2, 3]],
        )
        original_elements = mesh.elements.copy()

        result = to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            None,
        )

        self.assertIs(result, mesh)
        np.testing.assert_allclose(
            mesh.nodes[:, :2],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [2.0, 2.0],
                [0.0, 0.0],
            ],
            atol=1.0e-15,
        )
        np.testing.assert_array_equal(mesh.nodes[:, 2], [7, 8, 9, 10, 11])
        np.testing.assert_array_equal(mesh.elements, original_elements)

    def test_element_indices_limit_unique_referenced_nodes(self):
        mesh = self._mesh(
            [
                [0.9, 0.0],
                [0.0, 0.9],
                [-0.9, 0.0],
                [0.0, -0.9],
                [1.1, 0.0],
                [0.0, 1.1],
                [-1.1, 0.0],
            ],
            [
                [0, 1, 2, 3],
                [3, 4, 5, 6],
            ],
        )

        to_circle(mesh, 0.0, 0.0, 1.0, 0.11, 0.0, None, indices=[0])

        np.testing.assert_allclose(
            mesh.nodes[:, :2],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.1, 0.0],
                [0.0, 1.1],
                [-1.1, 0.0],
            ],
        )

        snapshot = self._snapshot(mesh)
        to_circle(mesh, 0.0, 0.0, 1.0, 1.0, 1.0, None, indices=[])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])

    def test_vertical_horizontal_and_reversed_guides_choose_nearest_roots(self):
        root = np.sqrt(0.75)
        mesh = self._mesh(
            [
                [0.5, 0.8],
                [0.5, -0.8],
                [0.8, 0.5],
                [-0.8, 0.5],
            ]
        )
        guides = [
            [[0.5, 1.0], [0.5, -1.0]],
            [[-1.0, 0.5], [1.0, 0.5]],
        ]

        to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.0, guides)

        np.testing.assert_allclose(
            mesh.nodes[:, :2],
            [
                [0.5, root],
                [0.5, -root],
                [root, 0.5],
                [-root, 0.5],
            ],
        )

    def test_guide_tolerance_uses_true_distance_and_is_independent(self):
        source = np.array([[0.52, 0.79]], dtype=np.float64)
        segment = [[[0.5, 0.8], [0.5, 1.0]]]
        radial = self._mesh(source)
        guided = self._mesh(source)

        to_circle(radial, 0.0, 0.0, 1.0, 0.1, 0.021, segment)
        to_circle(guided, 0.0, 0.0, 1.0, 0.1, 0.023, segment)

        expected_radial = source[0] / np.linalg.norm(source[0])
        np.testing.assert_allclose(radial.nodes[0, :2], expected_radial)
        np.testing.assert_allclose(
            guided.nodes[0, :2],
            [0.5, np.sqrt(0.75)],
        )

    def test_finite_guide_that_does_not_reach_circle_blocks_the_node(self):
        mesh = self._mesh([[0.5, 0.8]])
        original = mesh.nodes.copy()

        to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            [[[0.5, 0.75], [0.5, 0.82]]],
        )

        np.testing.assert_array_equal(mesh.nodes, original)

    def test_compatible_duplicates_move_and_conflicting_guides_do_not(self):
        compatible = self._mesh([[0.5, 0.8]])
        conflict = self._mesh([[0.5, 0.8]])
        duplicate_guides = [
            [[0.5, -1.0], [0.5, 1.0]],
            [[0.5, 1.0], [0.5, -1.0]],
        ]

        to_circle(
            compatible,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            duplicate_guides,
        )
        to_circle(
            conflict,
            0.0,
            0.0,
            1.0,
            0.1,
            0.0,
            duplicate_guides + [[[-1.0, 0.8], [1.0, 0.8]]],
        )

        np.testing.assert_allclose(
            compatible.nodes[0, :2],
            [0.5, np.sqrt(0.75)],
        )
        np.testing.assert_array_equal(conflict.nodes[0, :2], [0.5, 0.8])

    def test_crossing_guides_can_agree_at_a_circle_anchor(self):
        anchor_y = np.sqrt(0.75)
        mesh = self._mesh([[0.49, anchor_y - 0.006]])

        to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.02,
            0.011,
            [
                [[0.5, 0.0], [0.5, 1.0]],
                [[0.0, anchor_y], [1.0, anchor_y]],
            ],
        )

        np.testing.assert_allclose(mesh.nodes[0, :2], [0.5, anchor_y])

    def test_ambiguous_roots_and_center_node_remain_unchanged(self):
        mesh = self._mesh([[0.5, 0.0], [0.0, 0.0]])
        original = mesh.nodes.copy()

        to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            [[[0.5, -1.0], [0.5, 1.0]]],
        )

        np.testing.assert_array_equal(mesh.nodes, original)

    def test_accepts_prepared_guides_and_is_idempotent(self):
        mesh = self._mesh([[0.5, 0.8]])
        guides = _PatternGuideSet.from_values(
            [[[0.5, -1.0], [0.5, 1.0]]],
            coordinate_scale=1.0,
        )

        to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.0, guides)
        once = mesh.nodes.copy()
        result = to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.0, guides)

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(mesh.nodes, once)

    def test_clockwise_merge_uses_current_representative_and_preserves_data(self):
        angles = np.deg2rad([0.0, -2.0, -4.0, 180.0])
        mesh = self._mesh(
            np.column_stack(
                (np.cos(angles), np.sin(angles), [1.0, 2.0, 3.0, 4.0])
            ),
            [[0, 1, 2, 3]],
        )
        original_nodes = mesh.nodes.copy()
        original_elements = mesh.elements.copy()

        result = to_circle(mesh, 0.0, 0.0, 1.0, 0.05, 0.0, None)

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(mesh.nodes[1, :2], mesh.nodes[0, :2])
        np.testing.assert_allclose(mesh.nodes[2, :2], original_nodes[2, :2])
        np.testing.assert_array_equal(mesh.nodes[:, 2], original_nodes[:, 2])
        np.testing.assert_array_equal(mesh.elements, original_elements)
        self.assertEqual(mesh.nodes.shape, original_nodes.shape)
        self.assertEqual(mesh.elements.shape, original_elements.shape)

        once = mesh.nodes.copy()
        to_circle(mesh, 0.0, 0.0, 1.0, 0.05, 0.0, None)
        np.testing.assert_array_equal(mesh.nodes, once)

    def test_clockwise_merge_reconciles_the_positive_x_seam(self):
        angles = np.deg2rad([-1.0, 180.0, 1.0])
        mesh = self._mesh(np.column_stack((np.cos(angles), np.sin(angles))))

        to_circle(mesh, 0.0, 0.0, 1.0, 0.04, 0.0, None)

        np.testing.assert_array_equal(mesh.nodes[2, :2], mesh.nodes[0, :2])
        np.testing.assert_allclose(
            mesh.nodes[1, :2],
            [-1.0, 0.0],
            atol=1.0e-15,
        )

    def test_merge_distance_is_strict_and_zero_disables_merging(self):
        exact_tolerance = float(np.hypot(1.0, 1.0))
        at_tolerance = self._mesh([[1.0, 0.0], [0.0, -1.0]])

        to_circle(
            at_tolerance,
            0.0,
            0.0,
            1.0,
            exact_tolerance,
            0.0,
            None,
        )

        self.assertFalse(
            np.array_equal(
                at_tolerance.nodes[0, :2],
                at_tolerance.nodes[1, :2],
            )
        )

        disabled = self._mesh([[1.0, 0.0], [0.0, 1.0]])
        original = disabled.nodes.copy()
        to_circle(disabled, 0.0, 0.0, 1.0, 0.0, 0.0, None)
        np.testing.assert_array_equal(disabled.nodes, original)

    def test_guided_targets_take_precedence_and_remain_distinct(self):
        first_angle = np.deg2rad(-2.0)
        first_anchor = np.array(
            [np.cos(first_angle), np.sin(first_angle)]
        )
        promoted = self._mesh([[1.0, 0.0], first_anchor])
        first_guide = [
            [[first_anchor[0], -0.1], [first_anchor[0], 0.0]]
        ]

        to_circle(
            promoted,
            0.0,
            0.0,
            1.0,
            0.05,
            1.0e-6,
            first_guide,
        )

        np.testing.assert_allclose(
            promoted.nodes[:, :2],
            np.tile(first_anchor, (2, 1)),
        )

        second_angle = np.deg2rad(-4.0)
        second_anchor = np.array(
            [np.cos(second_angle), np.sin(second_angle)]
        )
        distinct = self._mesh([first_anchor, second_anchor])
        guides = first_guide + [
            [[second_anchor[0], -0.1], [second_anchor[0], 0.0]]
        ]

        to_circle(
            distinct,
            0.0,
            0.0,
            1.0,
            0.05,
            1.0e-6,
            guides,
        )

        np.testing.assert_allclose(
            distinct.nodes[:, :2],
            [first_anchor, second_anchor],
            atol=1.0e-14,
        )

    def test_element_selection_limits_circle_target_merging(self):
        angles = np.deg2rad(
            [0.0, -2.0, 90.0, 180.0, 1.0, 45.0, 135.0, 225.0]
        )
        mesh = self._mesh(
            np.column_stack((np.cos(angles), np.sin(angles))),
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )
        unselected = mesh.nodes[4:].copy()

        to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.05,
            0.0,
            None,
            indices=[0],
        )

        np.testing.assert_array_equal(mesh.nodes[1, :2], mesh.nodes[0, :2])
        np.testing.assert_array_equal(mesh.nodes[4:], unselected)

    def test_seeded_rectilinear_example_merges_twenty_four_pairs(self):
        generator = random.Random(1)

        def coordinates(begin, end, minimum_step):
            values = [begin]
            current = begin
            while True:
                current += generator.uniform(minimum_step, 2.0 * minimum_step)
                if current > end:
                    return values
                values.append(current)

        mesh = generate_rectilinear_mesh(
            5.0,
            coordinates(-100.0, 100.0, 1.0),
            coordinates(-100.0, 100.0, 2.0),
        )
        original_elements = mesh.elements.copy()

        to_circle(mesh, 0.0, 0.0, 57.0, 1.0, 1.0, [])

        radii = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
        circle_nodes = mesh.nodes[
            np.isclose(radii, 57.0, rtol=0.0, atol=1.0e-10),
            :2,
        ]
        self.assertEqual(circle_nodes.shape[0], 150)
        self.assertEqual(np.unique(circle_nodes, axis=0).shape[0], 126)
        np.testing.assert_array_equal(mesh.elements, original_elements)

    def test_invalid_inputs_raise_before_mutation(self):
        invalid_calls = (
            ((0.0, 0.0, 0.0, 0.1, 0.1, None), ValueError),
            ((0.0, 0.0, 1.0, -0.1, 0.1, None), ValueError),
            ((0.0, 0.0, 1.0, 0.1, -0.1, None), ValueError),
            ((np.nan, 0.0, 1.0, 0.1, 0.1, None), ValueError),
            (
                (
                    0.0,
                    0.0,
                    1.0,
                    0.1,
                    0.1,
                    [[[0.0, 0.0], [1.0, 1.0]]],
                ),
                ValueError,
            ),
        )
        for arguments, error_type in invalid_calls:
            with self.subTest(arguments=arguments):
                mesh = self._mesh(
                    [[0.9, 0.0], [0.0, 0.9], [-0.9, 0.0], [0.0, -0.9]],
                    [[0, 1, 2, 3]],
                )
                snapshot = self._snapshot(mesh)

                with self.assertRaises(error_type):
                    to_circle(mesh, *arguments)

                np.testing.assert_array_equal(mesh.nodes, snapshot[0])
                np.testing.assert_array_equal(mesh.elements, snapshot[1])

        mesh = self._mesh(
            [[0.9, 0.0], [0.0, 0.9], [-0.9, 0.0], [0.0, -0.9]],
            [[0, 1, 2, 3]],
        )
        snapshot = self._snapshot(mesh)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.1, None, [0, 0])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])

        mesh.nodes[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite coordinates"):
            to_circle(mesh, 0.0, 0.0, 1.0, 0.1, 0.1, None)


class ToCirclePerformanceTests(unittest.TestCase):
    maximum_seconds = 5.0

    def test_one_hundred_thousand_nodes_and_two_thousand_guides(self):
        count = 100_000
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        radii = 1.0 + 1.0e-4 * np.sin(37.0 * angles)
        nodes = np.column_stack(
            (
                radii * np.cos(angles),
                radii * np.sin(angles),
                np.zeros(count),
            )
        )
        elements = np.tile(
            np.array([[0, 1, 2, 3]], dtype=np.int32),
            (count, 1),
        )
        fixed_values = np.linspace(-0.999, 0.999, 1000)
        vertical = np.stack(
            (
                np.column_stack((fixed_values, np.full(1000, -1.1))),
                np.column_stack((fixed_values, np.full(1000, 1.1))),
            ),
            axis=1,
        )
        horizontal = np.stack(
            (
                np.column_stack((np.full(1000, -1.1), fixed_values)),
                np.column_stack((np.full(1000, 1.1), fixed_values)),
            ),
            axis=1,
        )
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        result = to_circle(
            mesh,
            0.0,
            0.0,
            1.0,
            0.001,
            1.0e-5,
            np.concatenate((vertical, horizontal)),
        )
        elapsed = time.perf_counter() - started

        self.assertIs(result, mesh)
        self.assertLess(elapsed, self.maximum_seconds)


class ImprintCircleTests(unittest.TestCase):
    @staticmethod
    def _area(mesh, element):
        count = 3 if element[2] == element[3] else 4
        points = mesh.nodes[element[:count], :2]
        following = np.roll(points, -1, axis=0)
        return 0.5 * abs(
            np.sum(
                points[:, 0] * following[:, 1]
                - points[:, 1] * following[:, 0]
            )
        )

    @staticmethod
    def _snapshot(mesh):
        return (
            mesh.nodes.copy(),
            mesh.elements.copy(),
            id(mesh.nodes),
            id(mesh.elements),
        )

    def test_splits_triangle_into_triangle_and_quad_and_interpolates_z(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 4.0]],
            elements=[[0, 1, 2, 2]],
        )

        result = imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertIs(result, mesh)
        self.assertEqual(mesh.node_count, 5)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_allclose(
            mesh.nodes[3:],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
            atol=1.0e-14,
        )
        self.assertEqual(np.count_nonzero(mesh.element_types == 3), 1)
        self.assertEqual(np.count_nonzero(mesh.element_types == 4), 1)
        self.assertAlmostEqual(
            sum(self._area(mesh, element) for element in mesh.elements),
            2.0,
        )

    def test_preserves_clockwise_triangle_winding(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 2, 1, 1]],
        )

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        for element in mesh.elements:
            count = 3 if element[2] == element[3] else 4
            points = mesh.nodes[element[:count], :2]
            following = np.roll(points, -1, axis=0)
            signed_area = 0.5 * np.sum(
                points[:, 0] * following[:, 1]
                - points[:, 1] * following[:, 0]
            )
            self.assertLess(signed_area, 0.0)

    def test_handles_a_large_offset_circle_without_area_cancellation(self):
        center = np.array([1.0e12, -1.0e12])
        nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]) + center
        mesh = Mesh2D(nodes=nodes, elements=[[0, 1, 2, 2]])

        imprint_circle(mesh, center[0], center[1], 1.0, 0.0)

        self.assertEqual(mesh.node_count, 5)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_allclose(
            np.hypot(
                mesh.nodes[3:, 0] - center[0],
                mesh.nodes[3:, 1] - center[1],
            ),
            1.0,
        )

    def test_splits_opposite_edge_quad_into_two_quads(self):
        mesh = Mesh2D(
            nodes=[
                [0.8, -0.2, 0.0],
                [1.2, -0.2, 1.0],
                [1.2, 0.2, 2.0],
                [0.8, 0.2, 3.0],
            ],
            elements=[[0, 1, 2, 3]],
        )

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertEqual(mesh.node_count, 6)
        self.assertEqual(mesh.element_count, 2)
        np.testing.assert_array_equal(mesh.element_types, [4, 4])
        self.assertAlmostEqual(
            sum(self._area(mesh, element) for element in mesh.elements),
            0.16,
        )

    def test_splits_adjacent_edge_quad_into_two_triangles_and_quad(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 3]],
        )

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertEqual(mesh.node_count, 6)
        self.assertEqual(mesh.element_count, 3)
        self.assertEqual(np.count_nonzero(mesh.element_types == 3), 2)
        self.assertEqual(np.count_nonzero(mesh.element_types == 4), 1)
        self.assertAlmostEqual(
            sum(self._area(mesh, element) for element in mesh.elements),
            4.0,
        )

    def test_reuses_one_new_node_on_a_shared_selected_edge(self):
        mesh = Mesh2D(
            nodes=[
                [0.8, -0.2, 0.0],
                [1.2, -0.2, 1.0],
                [0.8, 0.0, 2.0],
                [1.2, 0.0, 3.0],
                [0.8, 0.2, 4.0],
                [1.2, 0.2, 5.0],
            ],
            elements=[[0, 1, 3, 2], [2, 3, 5, 4]],
        )

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        self.assertEqual(mesh.node_count, 9)
        self.assertEqual(mesh.element_count, 4)
        shared_hits = np.flatnonzero(
            np.all(np.isclose(mesh.nodes[:, :2], [1.0, 0.0]), axis=1)
        )
        np.testing.assert_array_equal(shared_hits, [7])
        self.assertEqual(np.count_nonzero(mesh.elements == 7), 4)

    def test_tolerance_snaps_endpoint_and_preserves_its_z(self):
        mesh = Mesh2D(
            nodes=[
                [0.95, 0.0, 7.0],
                [2.0, 1.0, 8.0],
                [0.0, 0.0, 9.0],
            ],
            elements=[[0, 1, 2, 2]],
        )

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.1)

        np.testing.assert_allclose(mesh.nodes[0], [1.0, 0.0, 7.0])
        self.assertEqual(mesh.node_count, 4)
        self.assertEqual(mesh.element_count, 2)

    def test_selection_leaves_unselected_connectivity_but_shared_snap_moves(self):
        mesh = Mesh2D(
            nodes=[
                [0.95, 0.0],
                [2.0, 1.0],
                [0.0, 0.0],
                [2.0, -1.0],
                [3.0, 0.0],
            ],
            elements=[[0, 1, 2, 2], [0, 3, 4, 4]],
        )
        unselected = mesh.elements[1].copy()

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.1, indices=[0])

        np.testing.assert_array_equal(mesh.elements[1], unselected)
        np.testing.assert_allclose(mesh.nodes[0, :2], [1.0, 0.0])
        self.assertEqual(mesh.element_count, 3)

    def test_other_intersection_topologies_are_no_ops(self):
        cases = (
            (
                [[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                [[0, 1, 2, 2]],
                "two intersections on one edge",
            ),
            (
                [
                    [-2.0, -0.2],
                    [2.0, -0.2],
                    [2.0, 0.2],
                    [-2.0, 0.2],
                ],
                [[0, 1, 2, 3]],
                "four intersections",
            ),
            (
                [[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]],
                [[0, 1, 2, 3]],
                "tangent",
            ),
            (
                [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
                [[0, 1, 2, 2]],
                "no intersection",
            ),
        )
        for nodes, elements, label in cases:
            with self.subTest(label=label):
                mesh = Mesh2D(nodes=nodes, elements=elements)
                snapshot = self._snapshot(mesh)

                imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

                np.testing.assert_array_equal(mesh.nodes, snapshot[0])
                np.testing.assert_array_equal(mesh.elements, snapshot[1])
                self.assertEqual(id(mesh.nodes), snapshot[2])
                self.assertEqual(id(mesh.elements), snapshot[3])

    def test_close_intersections_merge_to_one_contact_without_snapping(self):
        mesh = Mesh2D(
            nodes=[[0.95, 0.0], [2.0, 0.1], [2.0, -0.1]],
            elements=[[0, 1, 2, 2]],
        )
        snapshot = self._snapshot(mesh)

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.2)

        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])
        self.assertEqual(id(mesh.nodes), snapshot[2])
        self.assertEqual(id(mesh.elements), snapshot[3])

    def test_is_idempotent(self):
        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 2]],
        )
        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)
        snapshot = self._snapshot(mesh)

        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)

        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])
        self.assertEqual(id(mesh.nodes), snapshot[2])
        self.assertEqual(id(mesh.elements), snapshot[3])

    def test_invalid_inputs_raise_before_mutation(self):
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            imprint_circle(object(), 0.0, 0.0, 1.0, 0.0)

        invalid_arguments = (
            ((np.nan, 0.0, 1.0, 0.0), ValueError),
            ((0.0, 0.0, 0.0, 0.0), ValueError),
            ((0.0, 0.0, 1.0, -1.0), ValueError),
        )
        for arguments, error_type in invalid_arguments:
            mesh = Mesh2D(
                nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                elements=[[0, 1, 2, 2]],
            )
            snapshot = self._snapshot(mesh)
            with self.assertRaises(error_type):
                imprint_circle(mesh, *arguments)
            np.testing.assert_array_equal(mesh.nodes, snapshot[0])
            np.testing.assert_array_equal(mesh.elements, snapshot[1])

        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 2]],
        )
        snapshot = self._snapshot(mesh)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0, [0, 0])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])

    def test_empty_mesh_and_empty_selection_are_no_ops(self):
        empty = Mesh2D(
            nodes=np.empty((0, 3)),
            elements=np.empty((0, 4), dtype=np.int32),
        )
        self.assertIs(imprint_circle(empty, 0.0, 0.0, 1.0, 0.0), empty)

        mesh = Mesh2D(
            nodes=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            elements=[[0, 1, 2, 2]],
        )
        snapshot = self._snapshot(mesh)
        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0, indices=[])
        np.testing.assert_array_equal(mesh.nodes, snapshot[0])
        np.testing.assert_array_equal(mesh.elements, snapshot[1])


class ImprintCirclePerformanceTests(unittest.TestCase):
    maximum_seconds = 5.0

    def test_one_hundred_thousand_element_grid(self):
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
        imprint_circle(mesh, 500.0, 50.0, 30.0, 0.0)
        elapsed = time.perf_counter() - started

        self.assertGreater(mesh.element_count, 100_000)
        self.assertLess(elapsed, self.maximum_seconds)

    def test_one_hundred_thousand_elements_all_split(self):
        count = 100_000
        cell = np.array(
            [
                [0.8, -0.2, 0.0],
                [1.2, -0.2, 1.0],
                [1.2, 0.2, 2.0],
                [0.8, 0.2, 3.0],
            ]
        )
        nodes = np.tile(cell, (count, 1))
        elements = np.arange(count * 4, dtype=np.int32).reshape(count, 4)
        mesh = Mesh2D(nodes=nodes, elements=elements)

        started = time.perf_counter()
        imprint_circle(mesh, 0.0, 0.0, 1.0, 0.0)
        elapsed = time.perf_counter() - started

        self.assertEqual(mesh.node_count, 600_000)
        self.assertEqual(mesh.element_count, 200_000)
        self.assertLess(elapsed, self.maximum_seconds)


if __name__ == "__main__":
    unittest.main()
