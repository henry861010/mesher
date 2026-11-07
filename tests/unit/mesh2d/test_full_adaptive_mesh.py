"""End-to-end coverage of transactional, conforming full adaptive meshes."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import (
    FullAdaptiveMeshResult,
    QuadLayerError,
    generate_full_adaptive_mesh,
)
from mesher.mesh2d.editor import EditorMesh2D


def _rectangle(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _cell_mesh(cells, spacing=0.5):
    """Create a conforming Quad4 union, including disconnected or holed unions."""
    coordinates = sorted({(x + dx, y + dy) for x, y in cells
                          for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))})
    lookup = {point: index for index, point in enumerate(coordinates)}
    elements = [[lookup[x, y], lookup[x + 1, y], lookup[x + 1, y + 1], lookup[x, y + 1]]
                for x, y in sorted(cells)]
    return Mesh2D(np.asarray(coordinates) * spacing, np.asarray(elements, dtype=np.int32))


def _block(x0=0, y0=0, width=4, height=4):
    return {(x, y) for x in range(x0, x0 + width) for y in range(y0, y0 + height)}


def _ring(with_island=False):
    cells = _block(width=12, height=12) - _block(2, 2, 8, 8)
    if with_island:
        cells |= _block(5, 5, 2, 2)
    return _cell_mesh(cells)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _area(points):
    points = np.asarray(points)
    points = points - points[0]
    following = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]) / 2)


def _flat(outline):
    return np.asarray(outline, dtype=np.int64)


def _segments(loop):
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(loop, np.roll(loop, -1))}


def _overlap_area(first, second):
    """Independent convex clipping, including oblique adaptive quad edges."""
    polygon = list(first)
    for start, end in zip(second, np.roll(second, -1, axis=0)):
        if not polygon:
            return 0.0
        clipped = []
        previous = polygon[-1]
        previous_side = _cross(end - start, previous - start)
        for current in polygon:
            side = _cross(end - start, current - start)
            if (side >= 0) != (previous_side >= 0):
                clipped.append(previous + previous_side / (previous_side - side) * (current - previous))
            if side >= 0:
                clipped.append(current)
            previous, previous_side = current, side
        polygon = clipped
    return abs(_area(polygon)) if len(polygon) >= 3 else 0.0


class FullAdaptiveMeshTests(unittest.TestCase):
    @staticmethod
    def _snapshot(mesh):
        return mesh.nodes, mesh.elements, mesh.nodes.copy(), mesh.elements.copy()

    def assert_unchanged(self, mesh, snapshot):
        nodes, elements, node_values, element_values = snapshot
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def assert_complete(self, result, mesh, outer, sizes, source, original_outlines):
        self.assertIsInstance(result, FullAdaptiveMeshResult)
        self.assertIs(result.mesh, mesh)
        np.testing.assert_array_equal(mesh.nodes[:len(source[2])], source[2])
        np.testing.assert_array_equal(mesh.elements[:len(source[3])], source[3])
        self.assertTrue(np.all(mesh.element_types == 4))
        self.assertTrue(np.all(np.isfinite(mesh.nodes)))
        self.assertTrue(np.all(mesh.nodes[:, 2] == 0))
        self.assertGreater(mesh.element_count, len(source[3]))
        referenced = np.unique(mesh.elements)
        expected_used = np.r_[np.unique(source[3]), np.arange(len(source[2]), mesh.node_count)]
        np.testing.assert_array_equal(referenced, expected_used)

        # Normalize before testing geometry so the assertions do not depend on
        # a particular fixture's coordinate origin or scale.
        outer = np.asarray(outer, dtype=float)
        scale = float(np.max(np.ptp(outer, axis=0)))
        xy = (mesh.nodes[:, :2] - outer[0]) / scale
        domain = (outer - outer[0]) / scale
        points = xy[mesh.elements]
        vectors = np.roll(points, -1, axis=1) - points
        following = np.roll(vectors, -1, axis=1)
        turns = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
        self.assertTrue(np.all(turns > 0), "all quads must be strictly convex and CCW")
        self.assertAlmostEqual(sum(_area(cell) for cell in points), abs(_area(domain)), places=10)

        directed = Counter((int(a), int(b)) for cell in mesh.elements
                           for a, b in zip(cell, np.roll(cell, -1)))
        incidence = Counter()
        for edge, count in directed.items():
            incidence[tuple(sorted(edge))] += count
        self.assertTrue(all(count in (1, 2) for count in incidence.values()))
        for a, b in incidence:
            if incidence[a, b] == 2:
                self.assertEqual(directed[a, b], 1)
                self.assertEqual(directed[b, a], 1)
            delta = xy[b] - xy[a]
            offset = xy[referenced] - xy[a]
            length = np.linalg.norm(delta)
            cross = offset[:, 0] * delta[1] - offset[:, 1] * delta[0]
            fraction = offset @ delta / (length * length)
            inside = ((np.abs(cross) < 1e-11 * length)
                      & (fraction > 1e-10) & (fraction < 1 - 1e-10))
            self.assertFalse(np.any(inside), "unsplit edge contains a hanging node")

        boundary = np.asarray(result.outer_outline)
        self.assertEqual(boundary.ndim, 1)
        self.assertEqual(len(set(map(int, boundary))), len(boundary))
        self.assertEqual(len(boundary) % 2, 0)
        self.assertLess(_area(xy[boundary]), 0)
        self.assertEqual(_segments(boundary), {edge for edge, count in incidence.items() if count == 1})
        self.assertEqual(len(EditorMesh2D.outlines(mesh)), 1)
        for point in xy[boundary]:
            on_domain = False
            for a, b in zip(domain, np.roll(domain, -1, axis=0)):
                delta = b - a
                fraction = np.dot(point - a, delta) / np.dot(delta, delta)
                on_domain |= (abs(_cross(point - a, delta)) < 1e-11
                              and -1e-11 <= fraction <= 1 + 1e-11)
            self.assertTrue(on_domain, "final boundary must lie on the supplied outer polygon")

        self.assertEqual(len(result.layer_outlines), len(sizes))
        for layer in result.layer_outlines:
            self.assertEqual(len(layer), len(original_outlines))
            for outline, original in zip(layer, original_outlines):
                self.assertEqual(len(outline), len(set(outline)))
                self.assertTrue(all(isinstance(node, int) for node in outline))
                nodes = _flat(outline)
                self.assertEqual(len(set(map(int, nodes))), len(nodes))
                self.assertGreater(_area(xy[nodes]) * _area(xy[_flat(original)]), 0)
                for edge in _segments(nodes):
                    self.assertEqual(incidence[edge], 2, "every adaptive interface must be shared exactly")
        for outline in original_outlines:
            for edge in _segments(_flat(outline)):
                self.assertEqual(incidence[edge], 2, "source boundary must remain unsplit and fully shared")

        # Area and edge counts alone cannot exclude geometrically overlapping
        # cells, so independently clip cells with overlapping bounding boxes.
        lows, highs = points.min(axis=1), points.max(axis=1)
        for index, cell in enumerate(points):
            candidates = np.flatnonzero(np.all(
                np.minimum(highs[:index], highs[index]) - np.maximum(lows[:index], lows[index]) > 1e-11,
                axis=1))
            for other in candidates:
                self.assertLess(_overlap_area(cell, points[other]), 1e-10)

        diagnostics = result.diagnostics
        self.assertEqual(diagnostics["element_sizes"], sizes)
        self.assertEqual(len(diagnostics["boundaries"]), len(original_outlines))
        self.assertEqual(len(diagnostics["layers"]), len(sizes))
        self.assertTrue(all(len(layer) == len(original_outlines) for layer in diagnostics["layers"]))
        self.assertEqual(diagnostics["added_nodes"], mesh.node_count - len(source[2]))
        self.assertEqual(diagnostics["added_elements"], mesh.element_count - len(source[3]))
        self.assertGreaterEqual(diagnostics["elapsed_seconds"], 0)

    def run_full(self, mesh, sizes, outer, **kwargs):
        source = self._snapshot(mesh)
        outlines = EditorMesh2D.outlines(mesh)
        outer_before = np.asarray(outer).copy()
        result = generate_full_adaptive_mesh(mesh, sizes, outer_outline=outer, **kwargs)
        self.assert_complete(result, mesh, outer, sizes, source, outlines)
        np.testing.assert_array_equal(outer, outer_before)
        return result

    def test_rectangle_one_layer(self):
        result = self.run_full(_cell_mesh(_block()), [0.5], _rectangle(-2, -2, 4, 4))
        self.assertEqual(len(result.diagnostics["global_regions"]), 1)

    def test_small_coordinate_tolerance_allows_small_scale_geometry(self):
        outer = _rectangle(-2e-4, -2e-4, 4e-4, 4e-4)
        mesh = _cell_mesh(_block(), spacing=5e-5)
        snapshot = self._snapshot(mesh)
        with self.assertRaisesRegex(QuadLayerError, "coordinate_tolerance") as caught:
            generate_full_adaptive_mesh(mesh, [5e-5], outer_outline=outer)
        self.assertEqual(caught.exception.context["stage"], "input_validation")
        self.assert_unchanged(mesh, snapshot)

        for tolerance in (1e-6, 0.0):
            with self.subTest(coordinate_tolerance=tolerance):
                self.run_full(_cell_mesh(_block(), spacing=5e-5), [5e-5], outer,
                              coordinate_tolerance=tolerance)

    def test_coordinate_tolerance_checks_final_front_clearance_atomically(self):
        # The source is well separated, but its final front leaves a 1e-4 gap.
        # The same tolerance must govern source, front, and global geometry.
        outer = _rectangle(-0.5001, -0.5001, 2.5001, 2.5001)
        mesh = _cell_mesh(_block())
        snapshot = self._snapshot(mesh)
        with self.assertRaisesRegex(QuadLayerError, "coordinate_tolerance") as caught:
            generate_full_adaptive_mesh(mesh, [0.5], outer_outline=outer,
                                        coordinate_tolerance=2e-4)
        self.assertEqual(caught.exception.code, "topology_change")
        self.assertEqual(caught.exception.context["stage"], "final_front_validation")
        self.assert_unchanged(mesh, snapshot)
        self.run_full(mesh, [0.5], outer, coordinate_tolerance=1e-6)

    def test_invalid_coordinate_tolerance_fails_atomically(self):
        for tolerance in (True, np.bool_(False), -1, np.nan, np.inf, -np.inf, None, "bad"):
            with self.subTest(coordinate_tolerance=tolerance):
                mesh = _cell_mesh(_block())
                snapshot = self._snapshot(mesh)
                with self.assertRaisesRegex(QuadLayerError, "coordinate_tolerance") as caught:
                    generate_full_adaptive_mesh(
                        mesh, [0.5], outer_outline=_rectangle(-2, -2, 4, 4),
                        coordinate_tolerance=tolerance)
                self.assertEqual(caught.exception.code, "invalid_input")
                self.assertEqual(caught.exception.context["stage"], "input_validation")
                self.assert_unchanged(mesh, snapshot)

    def test_original_unused_nodes_are_preserved_without_equivalencing(self):
        mesh = _cell_mesh(_block())
        # The duplicate and exterior unused nodes deliberately prevent a
        # whole-mesh cleanup/renumbering from satisfying source preservation.
        mesh.nodes = np.vstack((mesh.nodes, [[100, 100, 0], [0, 0, 0]]))
        self.run_full(mesh, [0.5], _rectangle(-2, -2, 4, 4))

    def test_multiple_layers_and_nonincreasing_sizes(self):
        for sizes in ([0.5, 0.75], [0.5, 0.5], [0.75, 0.5]):
            with self.subTest(sizes=sizes):
                result = self.run_full(_cell_mesh(_block()), sizes, _rectangle(-3, -3, 5, 5))
                # Layer count and growth follow the supplied order, with no
                # sorting or implicit extra collar before the global region.
                distance = 0.0
                for size, layer in zip(sizes, result.layer_outlines):
                    distance += size
                    xy = result.mesh.nodes[_flat(layer[0]), :2]
                    np.testing.assert_allclose(xy.min(axis=0), [-distance, -distance])
                    np.testing.assert_allclose(xy.max(axis=0), [2 + distance, 2 + distance])

    def test_separate_focus_regions(self):
        mesh = _cell_mesh(_block() | _block(10, 0))
        result = self.run_full(mesh, [0.5], _rectangle(-2, -2, 9, 4))
        self.assertEqual(len(result.layer_outlines[0]), 2)
        self.assertTrue(all(item["parent_index"] is None for item in result.diagnostics["boundaries"]))
        self.assertEqual(len(result.diagnostics["global_regions"]), 1)

    def test_concave_focus_and_outer_boundary(self):
        cells = _block(width=6, height=2) | _block(width=2, height=6)
        outer = np.array([[-2, -2], [5, -2], [5, 2], [4, 2], [4, 5], [-2, 5]], dtype=float)
        self.run_full(_cell_mesh(cells), [0.5], outer)

    def test_ring_fills_its_core_and_expands_its_exterior(self):
        for sizes in ([0.5], [0.25, 0.5]):
            with self.subTest(sizes=sizes):
                result = self.run_full(_ring(), sizes, _rectangle(-2, -2, 8, 8))
                self.assertEqual(len(result.layer_outlines[0]), 2)
                self.assertEqual(len(result.diagnostics["global_regions"]), 2)
                self.assertEqual(sum(item["parent_index"] is not None
                                     for item in result.diagnostics["boundaries"]), 1)

    def test_focus_island_inside_ring_hole(self):
        result = self.run_full(_ring(with_island=True), [0.5], _rectangle(-2, -2, 8, 8))
        self.assertEqual(len(result.layer_outlines[0]), 3)
        self.assertEqual(len(result.diagnostics["global_regions"]), 2)
        parents = [item["parent_index"] for item in result.diagnostics["boundaries"]]
        self.assertEqual(sum(parent is None for parent in parents), 1)
        self.assertTrue(any(parent is not None and parents[parent] is not None for parent in parents))

    def test_global_pattern_cut_is_shared_by_final_transition(self):
        # Entire pattern lies outside the transition. Its global projection
        # terminates on the final interface at a point absent from the source.
        pattern = np.array([[[-1.8, 0.37], [-1.2, 0.37]]])
        before = pattern.copy()
        result = self.run_full(_cell_mesh(_block()), [0.5], _rectangle(-2, -2, 4, 4),
                               pattern_lines=pattern)
        np.testing.assert_array_equal(pattern, before)
        front = result.mesh.nodes[_flat(result.layer_outlines[-1][0]), :2]
        self.assertTrue(np.any(np.all(np.isclose(front, [-0.5, 0.37], atol=1e-12), axis=1)))
        global_diagnostics = result.diagnostics["global_regions"][0]["diagnostics"]
        self.assertTrue(global_diagnostics["parity_repairs"], "fixture must exercise global parity repair")

    def test_expand_ratio_controls_pattern_selected_front(self):
        pattern = [[[0.5, 2.7], [1.5, 2.7]]]
        for ratio, expected_top in ((1.0, 2.5), (1.5, 2.7)):
            with self.subTest(expand_ratio=ratio):
                result = self.run_full(_cell_mesh(_block()), [0.5], _rectangle(-2, -2, 4, 4),
                                       expand_ratio=ratio, pattern_lines=pattern)
                front = result.mesh.nodes[_flat(result.layer_outlines[0][0]), :2]
                self.assertAlmostEqual(front[:, 1].max(), expected_top)

    def test_invalid_sizes_fail_atomically(self):
        for sizes in ([], [0], [-1], [np.inf], [np.nan], [0.5, 0], [0.5, "bad"], None):
            with self.subTest(sizes=sizes):
                mesh = _cell_mesh(_block())
                snapshot = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    generate_full_adaptive_mesh(mesh, sizes, outer_outline=_rectangle(-2, -2, 4, 4))
                self.assert_unchanged(mesh, snapshot)

    def test_later_layer_collision_and_touch_fail_atomically(self):
        for final_size in (0.5, 0.75):
            with self.subTest(final_size=final_size):
                mesh = _cell_mesh(_block() | _block(8, 0))
                snapshot = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    generate_full_adaptive_mesh(mesh, [0.5, final_size],
                                                outer_outline=_rectangle(-4, -4, 10, 6))
                self.assert_unchanged(mesh, snapshot)

    def test_hole_collapse_fails_atomically(self):
        mesh = _cell_mesh(_block(width=8, height=8) - _block(3, 3, 2, 2))
        snapshot = self._snapshot(mesh)
        with self.assertRaises(QuadLayerError):
            generate_full_adaptive_mesh(mesh, [0.25, 0.5], outer_outline=_rectangle(-3, -3, 7, 7))
        self.assert_unchanged(mesh, snapshot)

    def test_outer_crossing_after_successful_first_layer_is_atomic(self):
        mesh = _cell_mesh(_block())
        snapshot = self._snapshot(mesh)
        with self.assertRaises(QuadLayerError):
            generate_full_adaptive_mesh(mesh, [0.5, 0.5], outer_outline=_rectangle(-0.75, -0.75, 2.75, 2.75))
        self.assert_unchanged(mesh, snapshot)

    def test_odd_source_boundaries_are_rejected_before_global_generation(self):
        outer = np.array([[0, 0], [6, 0], [6, 6], [0, 6], [0, 3]], dtype=float)
        inner = np.array([[1, 1], [5, 1], [5, 5], [1, 5], [1, 3]], dtype=float)
        cells = np.array([[i, (i + 1) % 5, 5 + (i + 1) % 5, 5 + i] for i in range(5)], dtype=np.int32)
        mesh = Mesh2D(np.vstack((outer, inner)), cells)
        snapshot = self._snapshot(mesh)
        with patch("mesher.mesh2d.adapter._full.generate_rectilinear_mesh_optimal") as generator:
            with self.assertRaises(QuadLayerError):
                generate_full_adaptive_mesh(mesh, [0.25], outer_outline=_rectangle(-2, -2, 8, 8))
        generator.assert_not_called()
        self.assert_unchanged(mesh, snapshot)

    def test_global_generator_failure_rolls_back_previous_layers(self):
        mesh = _cell_mesh(_block())
        snapshot = self._snapshot(mesh)
        failure = ValueError("injected global construction failure")
        with patch("mesher.mesh2d.adapter._full.generate_rectilinear_mesh_optimal", side_effect=failure) as generator:
            with self.assertRaises(QuadLayerError) as caught:
                generate_full_adaptive_mesh(mesh, [0.5, 0.75], outer_outline=_rectangle(-3, -3, 5, 5))
        generator.assert_called_once()
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(caught.exception.context["global_region_index"], 0)
        self.assertEqual(caught.exception.context["layer_index"], 1)
        self.assert_unchanged(mesh, snapshot)

    def test_fixed_boundary_failure_rolls_back_global_and_previous_layers(self):
        mesh = _cell_mesh(_block())
        snapshot = self._snapshot(mesh)
        with patch("mesher.mesh2d.adapter._full._generate_layer_adaptive_mesh",
                   side_effect=QuadLayerError("injected fixed boundary failure")) as fixed_layer:
            with self.assertRaises(QuadLayerError):
                generate_full_adaptive_mesh(mesh, [0.5, 0.75], outer_outline=_rectangle(-3, -3, 5, 5))
        fixed_layer.assert_called_once()
        self.assert_unchanged(mesh, snapshot)


if __name__ == "__main__":
    unittest.main()
