import time
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.circular.imprint_v2.main import (
    _find_intersect_element_and_sort,
)


class FindIntersectElementAndSortTests(unittest.TestCase):
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

        result = _find_intersect_element_and_sort(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(result, [2, 3, 1, 0])
        self.assertEqual(result.dtype, np.dtype(np.int64))

    def test_includes_edge_tangent_and_single_vertex_contact(self):
        tangent_quad = [[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]]
        vertex_triangle = [[0.0, 1.0], [2.0, 2.0], [-2.0, 2.0]]
        mesh = self._mesh_from_polygons([vertex_triangle, tangent_quad])

        result = _find_intersect_element_and_sort(mesh, 0.0, 0.0, 1.0)

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

        result = _find_intersect_element_and_sort(mesh, 0.0, 0.0, 1.0)

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

        result = _find_intersect_element_and_sort(mesh, 0.0, 0.0, 1.0)

        # Both rows share the complete chord from 0 to 90 degrees.  Each row
        # remains present and their identical keys are resolved by row index.
        np.testing.assert_array_equal(result, [0, 1])

    def test_excludes_nonintersecting_perimeters(self):
        inside = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
        outside = [[2.0, -0.2], [3.0, -0.2], [3.0, 0.2], [2.0, 0.2]]
        containing = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
        mesh = self._mesh_from_polygons([inside, outside, containing])

        result = _find_intersect_element_and_sort(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(result, np.empty(0, dtype=np.int64))

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

        result = _find_intersect_element_and_sort(
            mesh, center[0], center[1], 1.0
        )

        np.testing.assert_array_equal(result, [1, 0])

    def test_validates_circle_and_mutated_mesh_data(self):
        mesh = self._mesh_from_polygons(
            [[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]
        )
        with self.assertRaisesRegex(TypeError, "Mesh2D"):
            _find_intersect_element_and_sort(object(), 0.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            _find_intersect_element_and_sort(mesh, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            _find_intersect_element_and_sort(mesh, np.nan, 0.0, 1.0)

        mesh.nodes[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite XY"):
            _find_intersect_element_and_sort(mesh, 0.0, 0.0, 1.0)

    def test_empty_mesh_returns_empty_int64_array(self):
        mesh = Mesh2D(
            nodes=np.empty((0, 3)),
            elements=np.empty((0, 4), dtype=np.int32),
        )

        result = _find_intersect_element_and_sort(mesh, 0.0, 0.0, 1.0)

        np.testing.assert_array_equal(result, np.empty(0, dtype=np.int64))


class FindIntersectElementAndSortPerformanceTests(unittest.TestCase):
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
        result = _find_intersect_element_and_sort(mesh, 500.0, 50.0, 30.0)
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
        result = _find_intersect_element_and_sort(mesh, 0.0, 0.0, 2.2)
        elapsed = time.perf_counter() - started

        np.testing.assert_array_equal(
            result, np.arange(count, dtype=np.int64)
        )
        self.assertLess(elapsed, self.maximum_seconds)


if __name__ == "__main__":
    unittest.main()
