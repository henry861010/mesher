"""Final quad triangulation must preserve coverage, seams, and provenance."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.circular.imprint_v2 import imprint_circle
from mesher.mesh2d.circular.imprint_v2.local import _run_local_imprint
from mesher.mesh2d.circular.imprint_v2.triangulation import _triangulate_flat_quads
from mesher.mesh2d.circular.imprint_v2.utils import _imprint_circle


def signed_areas(nodes, elements):
    points = nodes[elements, :2]
    points = points - points[:, :1]
    following = np.roll(points, -1, axis=1)
    return .5 * np.sum(
        points[..., 0] * following[..., 1] - points[..., 1] * following[..., 0],
        axis=1,
    )


def boundary_edges(elements):
    counts = {}
    for row in elements:
        perimeter = row[:3] if row[2] == row[3] else row
        for a, b in zip(perimeter, np.roll(perimeter, -1)):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    return {edge for edge, count in counts.items() if count == 1}


def imprint(mesh, **kwargs):
    return imprint_circle(
        mesh, center=[0., 0.], radius=1., projection_tolerance=0.,
        guide_tolerance=0., merge_tolerance=0., minimum_area=0., **kwargs,
    )


class FlatQuadTriangulationTests(unittest.TestCase):
    def test_every_corner_uses_the_diagonal_avoiding_a_sliver(self):
        nodes = np.array([[0., 0., 2.], [1., -.001, 4.],
                          [2., 0., 8.], [0., 1., 16.]])
        original_nodes = nodes.copy()
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                elements = np.roll(np.arange(4, dtype=np.int32), rotation)[None, :]
                before = elements.copy()
                output, sources = _triangulate_flat_quads(nodes, elements)

                self.assertEqual(output.shape, (2, 4))
                np.testing.assert_array_equal(output[:, 2], output[:, 3])
                np.testing.assert_array_equal(sources, [0, 0])
                self.assertEqual(set(output[0, :3]) & set(output[1, :3]), {1, 3})
                self.assertTrue(np.all(signed_areas(nodes, output) > 0.))
                self.assertAlmostEqual(signed_areas(nodes, output).sum(),
                                       signed_areas(nodes, elements).sum())
                self.assertEqual(boundary_edges(output), boundary_edges(elements))
                np.testing.assert_array_equal(elements, before)
                np.testing.assert_array_equal(nodes, original_nodes)

    def test_threshold_controls_whether_a_corner_is_split(self):
        for angle, threshold, count in ((169., 168., 2), (169., 170., 1),
                                         (171., 170., 2), (171., 175., 1)):
            with self.subTest(angle=angle, threshold=threshold):
                height = np.tan(np.deg2rad((180. - angle) / 2.))
                nodes = np.array([[0., 0.], [1., -height], [2., 0.], [0., 1.]])
                elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
                output, _ = _triangulate_flat_quads(nodes, elements, flat_quad_angle=threshold)
                self.assertEqual(len(output), count)

    def test_exactly_collinear_corner_keeps_both_children_positive(self):
        nodes = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 1.]])
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        output, _ = _triangulate_flat_quads(nodes, elements, flat_quad_angle=180.)
        self.assertEqual(len(output), 2)
        self.assertTrue(np.all(signed_areas(nodes, output) > 0.))
        self.assertEqual(boundary_edges(output), boundary_edges(elements))

    def test_mixed_rows_keep_order_and_parent_mapping(self):
        nodes = np.array([[0., 0.], [1., -.001], [2., 0.], [0., 1.],
                          [4., 0.], [5., 0.], [5., 1.], [4., 1.]])
        elements = np.array([[4, 5, 6, 6], [0, 1, 2, 3],
                             [4, 5, 6, 7], [1, 2, 3, 0]], dtype=np.int32)
        output, sources = _triangulate_flat_quads(nodes, elements)
        np.testing.assert_array_equal(sources, [0, 1, 1, 2, 3, 3])
        np.testing.assert_array_equal(output[[0, 3]], elements[[0, 2]])

    def test_angle_is_independent_of_scale_and_coordinate_offset(self):
        base = np.array([[0., 0.], [1., -.001], [2., 0.], [0., 1.]])
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        for scale, offset in ((1.e-9, 0.), (1., 0.), (1.e9, 0.), (1., 1.e12)):
            with self.subTest(scale=scale, offset=offset):
                nodes = base * scale + offset
                output, _ = _triangulate_flat_quads(nodes, elements)
                self.assertEqual(len(output), 2)
                self.assertTrue(np.all(signed_areas(nodes, output) > 0.))

    def test_no_op_keeps_arrays_for_empty_triangles_and_thin_rectangles(self):
        nodes = np.array([[0., 0.], [1000., 0.], [1000., 1.], [0., 1.]])
        for rows in ([], [[0, 1, 2, 2]], [[0, 1, 2, 3]]):
            elements = np.array(rows, dtype=np.int32).reshape(-1, 4)
            output, sources = _triangulate_flat_quads(nodes, elements)
            self.assertIs(output, elements)
            np.testing.assert_array_equal(sources, np.arange(len(elements)))


class ImprintFinalTriangulationTests(unittest.TestCase):
    @staticmethod
    def mesh():
        # Circle splitting produces a Quad4 with an approximately 171° corner.
        return Mesh2D([[-1.1, -.6, 2.], [-.44, -.14, 4.], [.52, .8, 8.]],
                      [[0, 1, 2, 2]])

    def test_public_call_triangulates_generated_quad_and_preserves_footprint(self):
        mesh = self.mesh()
        split_only = self.mesh()
        _imprint_circle(split_only, [0., 0.], 1., 0.)
        self.assertEqual(split_only.element_count, 2)
        self.assertEqual(np.count_nonzero(split_only.element_types == 4), 1)

        self.assertIs(imprint(mesh), mesh)

        self.assertEqual(mesh.element_count, 3)
        self.assertTrue(np.all(mesh.element_types == 3))
        np.testing.assert_array_equal(mesh.nodes, split_only.nodes)
        self.assertTrue(np.all(signed_areas(mesh.nodes, mesh.elements) > 0.))
        self.assertAlmostEqual(signed_areas(mesh.nodes, mesh.elements).sum(),
                               signed_areas(split_only.nodes, split_only.elements).sum())
        self.assertEqual(boundary_edges(mesh.elements), boundary_edges(split_only.elements))

        nodes, elements = mesh.nodes, mesh.elements
        imprint(mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)

    def test_public_angle_controls_generated_quad_triangulation(self):
        for threshold, count in ((160., 3), (170., 3), (175., 2), (180., 2)):
            with self.subTest(threshold=threshold):
                mesh = self.mesh()
                imprint(mesh, flat_quad_angle=threshold)
                self.assertEqual(mesh.element_count, count)
                self.assertEqual(np.count_nonzero(mesh.element_types == 4), 3 - count)

    def test_invalid_angles_raise_before_mutation_even_without_a_circle_patch(self):
        for offset in (0., 10.):
            for angle in (0., -1., 180.1, np.nan, np.inf, True, None, "invalid"):
                with self.subTest(offset=offset, angle=angle):
                    mesh = self.mesh()
                    mesh.nodes[:, :2] += offset
                    nodes, elements = mesh.nodes, mesh.elements
                    before_nodes, before_elements = nodes.copy(), elements.copy()
                    with self.assertRaisesRegex(ValueError, "flat_quad_angle"):
                        imprint(mesh, flat_quad_angle=angle)
                    self.assertIs(mesh.nodes, nodes)
                    self.assertIs(mesh.elements, elements)
                    np.testing.assert_array_equal(mesh.nodes, before_nodes)
                    np.testing.assert_array_equal(mesh.elements, before_elements)

    def test_final_failure_is_atomic(self):
        mesh = self.mesh()
        nodes, elements = mesh.nodes, mesh.elements
        original_nodes = nodes.copy()
        original_elements = elements.copy()

        def fail(local_nodes, local_elements, **kwargs):
            local_nodes[:] = 42.
            local_elements[:] = 0
            raise ValueError("forced triangulation failure")

        with patch("mesher.mesh2d.circular.imprint_v2.local._triangulate_flat_quads",
                   side_effect=fail):
            with self.assertRaisesRegex(ValueError, "forced triangulation failure"):
                imprint(mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, original_nodes)
        np.testing.assert_array_equal(mesh.elements, original_elements)

    def test_remote_flat_quad_is_preserved(self):
        near = self.mesh()
        remote = np.array([[20., 0., 1.], [21., -.001, 2.],
                           [22., 0., 3.], [20., 1., 4.]])
        mesh = Mesh2D(np.concatenate((near.nodes, remote)),
                      np.concatenate((near.elements, [[3, 4, 5, 6]])))
        imprint(mesh)
        quads = mesh.elements[mesh.element_types == 4]
        self.assertEqual(len(quads), 1)
        np.testing.assert_array_equal(mesh.nodes[quads[0]], remote)

    def test_final_source_metadata_tracks_both_triangles(self):
        mesh = self.mesh()
        stats, metadata = {}, {}

        def capture(stage, snapshot, provenance):
            metadata[stage] = provenance

        _run_local_imprint(
            mesh, center=[0., 0.], radius=1., projection_tolerance=0.,
            guide_tolerance=0., merge_tolerance=0., minimum_area=0.,
            observer=capture, stats=stats,
        )
        np.testing.assert_array_equal(metadata["split"]["split_sources"], [0, 0, 0])
        self.assertEqual(stats["triangulated_quads"], 1)
