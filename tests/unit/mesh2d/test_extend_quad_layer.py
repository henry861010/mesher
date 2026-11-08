"""Direct Quad4 growth preserves fixed boundaries and finite constraints."""

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh


def _rectangle():
    mesh = Mesh2D(
        [[0, 0], [4, 0], [4, 3], [0, 3]],
        np.array([[0, 1, 2, 3]], dtype=np.int32),
    )
    return mesh, [0, 3, 2, 1]


def _target_ids(mesh, points):
    """Register target coordinates before taking transactional snapshots."""
    points = np.asarray(points, dtype=float)
    closed = len(points) > 1 and np.array_equal(points[0], points[-1])
    if closed:
        points = points[:-1]
    ids = np.arange(mesh.node_count, mesh.node_count + len(points), dtype=np.int64)
    mesh.nodes = np.vstack((mesh.nodes, np.column_stack((points, np.zeros(len(points))))))
    return np.r_[ids, ids[0]] if closed else ids


def _polyomino(cells):
    coordinates = sorted({(x + dx, y + dy) for x, y in cells
                          for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))})
    lookup = {point: index for index, point in enumerate(coordinates)}
    elements = np.array([
        [lookup[x, y], lookup[x + 1, y], lookup[x + 1, y + 1], lookup[x, y + 1]]
        for x, y in sorted(cells)
    ], dtype=np.int32)
    mesh = Mesh2D(coordinates, elements)
    directed = Counter((int(a), int(b)) for cell in elements
                       for a, b in zip(cell, np.roll(cell, -1)))
    clockwise = {b: a for (a, b) in directed if (b, a) not in directed}
    start = min(clockwise)
    loop = [start]
    while clockwise[loop[-1]] != start:
        loop.append(clockwise[loop[-1]])
    return mesh, loop, lookup


def _cross(first, second):
    return first[0] * second[1] - first[1] * second[0]


def _area(points):
    points = np.asarray(points)
    points = points - points[0]
    return 0.5 * sum(_cross(a, b) for a, b in zip(points, np.roll(points, -1, axis=0)))


def _outline_nodes(outline):
    return np.asarray(outline, dtype=np.int64)


def _segments(outline):
    return {frozenset((int(a), int(b)))
            for a, b in zip(outline, np.roll(outline, -1))}


def _corners(mesh, outline):
    """Count geometric turns independently of the adapter's cardinal grouping."""
    points = mesh.nodes[np.asarray(outline, dtype=int), :2]
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    directions = vectors / lengths[:, None]
    previous = np.roll(directions, 1, axis=0)
    turns = np.abs(directions[:, 0] * previous[:, 1]
                   - directions[:, 1] * previous[:, 0])
    return np.flatnonzero((turns > .01) | (np.sum(directions * previous, axis=1) <= 0))


def _overlap_area(first, second):
    """Independent convex clipping; no mesh implementation helpers are used."""
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
                t = previous_side / (previous_side - side)
                clipped.append(previous + t * (current - previous))
            if side >= 0:
                clipped.append(current)
            previous, previous_side = current, side
        polygon = clipped
    return abs(_area(polygon)) if len(polygon) >= 3 else 0.0


class ExtendQuadLayerTests(unittest.TestCase):
    def assert_layer(self, mesh, source, result, old_nodes, old_elements):
        self.assertIs(result.mesh, mesh)
        self.assertIsNotNone(result.diagnostics)
        np.testing.assert_array_equal(mesh.nodes[:len(old_nodes)], old_nodes)
        np.testing.assert_array_equal(mesh.elements[:len(old_elements)], old_elements)
        self.assertTrue(np.all(mesh.element_types == 4))
        self.assertTrue(np.all(np.isfinite(mesh.nodes)))

        front = result.outline
        self.assertGreaterEqual(len(front), 4)
        self.assertEqual(len(front), len(set(front)))
        self.assertTrue(np.issubdtype(np.asarray(front).dtype, np.integer))
        self.assertTrue(np.all(np.asarray(front) >= 0))
        self.assertTrue(np.all(np.asarray(front) < mesh.node_count))

        quads = mesh.elements[len(old_elements):]
        self.assertGreater(len(quads), 0)
        self.assertTrue(all(len(set(cell)) == 4 for cell in quads))
        self.assertTrue(set(range(len(old_nodes), mesh.node_count)) <= set(quads.ravel()))

        incidence = Counter((int(cell[i]), int(cell[(i + 1) % 4]))
                            for cell in quads for i in range(4))
        expected_boundary = _segments(source) | _segments(front)
        actual = {frozenset(pair) for pair in incidence}
        self.assertTrue(expected_boundary <= actual)
        for pair in actual:
            a, b = pair
            expected_count = 1 if pair in expected_boundary else 2
            self.assertEqual(incidence[a, b] + incidence[b, a], expected_count)
            if expected_count == 2:
                self.assertEqual(incidence[a, b], 1)
                self.assertEqual(incidence[b, a], 1)

        # Normalize independently so tiny and translated coordinates are checked
        # by the same convexity, area, and overlap assertions.
        front_ids, source_ids = _outline_nodes(front), _outline_nodes(source)
        scale = max(np.ptp(mesh.nodes[np.r_[front_ids, source_ids], :2], axis=0))
        xy = (mesh.nodes[:, :2] - mesh.nodes[source_ids[0], :2]) / scale
        cells = xy[quads]
        vectors = np.roll(cells, -1, axis=1) - cells
        following = np.roll(vectors, -1, axis=1)
        turns = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
        self.assertTrue(np.all(turns > 0), "every added quad must be strictly convex and CCW")
        source_area, front_area = _area(xy[source_ids]), _area(xy[front_ids])
        self.assertGreater(source_area * front_area, 0)
        expected_area = source_area - front_area
        self.assertGreater(expected_area, 0)
        self.assertAlmostEqual(sum(_area(cell) for cell in cells), expected_area, places=9)

        # Only clip cells whose bounding boxes overlap in both dimensions.
        all_cells = xy[mesh.elements]
        mins, maxs = all_cells.min(axis=1), all_cells.max(axis=1)
        for index in range(len(old_elements), len(all_cells)):
            candidates = np.flatnonzero(np.all(
                np.minimum(maxs[:index], maxs[index])
                - np.maximum(mins[:index], mins[index]) > 1e-12, axis=1))
            for other in candidates:
                self.assertLess(_overlap_area(all_cells[index], all_cells[other]), 1e-10)

    def assert_pattern_covered(self, mesh, pattern):
        start, end = np.asarray(pattern, dtype=float)
        direction = end - start
        length = np.linalg.norm(direction)
        direction /= length
        intervals = []
        for cell in mesh.elements:
            for first, second in zip(cell, np.roll(cell, -1)):
                a, b = mesh.nodes[[first, second], :2] - start
                if abs(_cross(a, direction)) < 1e-9 and abs(_cross(b, direction)) < 1e-9:
                    low, high = sorted((np.dot(a, direction), np.dot(b, direction)))
                    if high >= 0 and low <= length:
                        intervals.append((max(0, low), min(length, high)))
        covered_to = 0.0
        for low, high in sorted(intervals):
            if high <= covered_to:
                continue
            self.assertLessEqual(low, covered_to + 1e-9, "pattern has a gap in mesh-edge coverage")
            covered_to = high
        self.assertAlmostEqual(covered_to, length, places=9)

    @staticmethod
    def _snapshot(mesh):
        return mesh.nodes, mesh.elements, mesh.nodes.copy(), mesh.elements.copy()

    def assert_unchanged(self, mesh, snapshot):
        nodes, elements, node_values, element_values = snapshot
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, node_values)
        np.testing.assert_array_equal(mesh.elements, element_values)

    def run_layer(self, mesh, outline, h=0.5, **kwargs):
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        source = np.asarray(outline).copy()
        result = generate_layer_adaptive_mesh(mesh, outline, h, **kwargs)
        self.assert_layer(mesh, source, result, old_nodes, old_elements)
        np.testing.assert_array_equal(source, outline)
        return result

    def test_rectangle_is_quad_only_and_returns_reusable_outline(self):
        mesh, outline = _rectangle()
        result = self.run_layer(mesh, outline)
        xy = mesh.nodes[_outline_nodes(result.outline), :2]
        np.testing.assert_allclose(xy.min(axis=0), [-0.5, -0.5])
        np.testing.assert_allclose(xy.max(axis=0), [4.5, 3.5])
        second = self.run_layer(mesh, result.outline, h=0.35)
        xy = mesh.nodes[_outline_nodes(second.outline), :2]
        np.testing.assert_allclose(xy.min(axis=0), [-0.85, -0.85])
        np.testing.assert_allclose(xy.max(axis=0), [4.85, 3.85])

    def test_temporary_mesh_objects_also_contain_only_quads(self):
        mesh, outline, _ = _polyomino({(x, y) for x in range(4) for y in range(3)})
        original_init, original_replace = Mesh2D.__init__, Mesh2D.replace_data
        inspected = []

        def inspect(candidate):
            inspected.append(candidate.element_count)
            self.assertTrue(np.all(candidate.element_types == 4))
            self.assertTrue(all(len(set(cell)) == 4 for cell in candidate.elements))

        def checked_init(candidate, *args, **kwargs):
            original_init(candidate, *args, **kwargs)
            inspect(candidate)

        def checked_replace(candidate, *args, **kwargs):
            original_replace(candidate, *args, **kwargs)
            inspect(candidate)

        patterns = [[[1, 3], [1, 3.8]], [[3, 3], [3, 3.8]]]
        with patch.object(Mesh2D, "__init__", checked_init), \
                patch.object(Mesh2D, "replace_data", checked_replace):
            self.run_layer(mesh, outline, h=0.8, pattern_lines=patterns)
        self.assertTrue(inspected)

    def test_concave_l_and_cross_keep_all_fixed_collinear_segments(self):
        for cells in (
            {(0, 0), (1, 0), (0, 1)},
            {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        ):
            with self.subTest(cells=cells):
                refined = {(3*x+dx, 3*y+dy) for x,y in cells
                           for dx in range(3) for dy in range(3)}
                mesh, outline, _ = _polyomino(refined)
                mesh.nodes[:, :2] /= 3
                self.run_layer(mesh, outline, h=0.2)

    def test_counterclockwise_hole_grows_toward_hole_interior(self):
        cells = {(x, y) for x in range(12) for y in range(12)
                 if x < 4 or x >= 8 or y < 4 or y >= 8}
        mesh, _, lookup = _polyomino(cells)
        mesh.nodes[:, :2] /= 4
        outline = ([lookup[x, 4] for x in range(4, 8)]
                   + [lookup[8, y] for y in range(4, 8)]
                   + [lookup[x, 8] for x in range(8, 4, -1)]
                   + [lookup[4, y] for y in range(8, 4, -1)])
        result = self.run_layer(mesh, outline, h=0.2)
        xy = mesh.nodes[_outline_nodes(result.outline), :2]
        np.testing.assert_allclose(xy.min(axis=0), [1.2, 1.2])
        np.testing.assert_allclose(xy.max(axis=0), [1.8, 1.8])

    def test_partial_parallel_pattern_snaps_whole_edge_within_ratio(self):
        for distance, ratio, expected in ((0.3, 1.0, 3.3), (0.75, 1.5, 3.75),
                                           (0.75, 1.4, 3.5)):
            with self.subTest(distance=distance, ratio=ratio):
                mesh, outline = _rectangle()
                pattern = [[1, 3 + distance], [2, 3 + distance]]
                result = self.run_layer(mesh, outline, pattern_lines=[pattern], expand_ratio=ratio)
                front = mesh.nodes[result.outline, :2]
                self.assertAlmostEqual(front[:, 1].max(), expected)
                top = front[np.isclose(front[:, 1], expected)]
                np.testing.assert_allclose(top[:, 0].min(), -.5)
                np.testing.assert_allclose(top[:, 0].max(), 4.5)
                if distance <= 0.5 * ratio:
                    self.assert_pattern_covered(mesh, pattern)

    def test_nearest_pattern_wins_and_coincident_pattern_does_not_stall(self):
        mesh, outline = _rectangle()
        patterns = [[[1, 3], [2, 3]], [[1, 3.25], [2, 3.25]], [[1, 3.4], [2, 3.4]]]
        result = self.run_layer(mesh, outline, pattern_lines=patterns)
        self.assertAlmostEqual(mesh.nodes[_outline_nodes(result.outline), 1].max(), 3.25)
        self.assert_pattern_covered(mesh, patterns[0])
        self.assert_pattern_covered(mesh, patterns[1])

    def test_complete_transverse_patterns_share_existing_source_nodes(self):
        mesh, outline, _ = _polyomino({(x, y) for x in range(4) for y in range(3)})
        patterns = [[[1, 3], [1, 3.5]], [[3, 3], [3, 3.5]]]
        result = self.run_layer(mesh, outline, pattern_lines=patterns)
        self.assertEqual(result.diagnostics["sections"]["LINE"], 6)
        for pattern in patterns:
            self.assert_pattern_covered(mesh, pattern)

    def test_unsupported_finite_patterns_are_typed_atomic_rejections(self):
        cases = {
            "endpoint": [[[1, 3.1], [1, 3.4]]],
            "T": [[[4.2, 3.4], [4.6, 3.4]], [[4.4, 3.2], [4.4, 3.4]]],
            "cross": [[[4.2, 3.4], [4.6, 3.4]], [[4.4, 3.2], [4.4, 3.6]]],
            "overlap": [[[4.2, 3.4], [4.5, 3.4]], [[4.6, 3.4], [4.3, 3.4]]],
            "closed": [[[4.2, 3.2], [4.6, 3.2]], [[4.6, 3.2], [4.6, 3.6]],
                       [[4.6, 3.6], [4.2, 3.6]], [[4.2, 3.6], [4.2, 3.2]]],
        }
        for name, patterns in cases.items():
            with self.subTest(name=name):
                mesh, outline = _rectangle()
                snapshot = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    generate_layer_adaptive_mesh(mesh, outline, 0.8, pattern_lines=patterns)
                self.assertEqual(caught.exception.code, "unsupported_pattern_layout")
                self.assert_unchanged(mesh, snapshot)

    def test_approximately_axis_aligned_nodes_are_never_moved(self):
        mesh, outline = _rectangle()
        mesh.nodes[2:, 0] += 1e-4
        result = self.run_layer(mesh, outline)
        xy = mesh.nodes[result.outline, :2]
        vectors = np.roll(xy, -1, axis=0) - xy
        self.assertTrue(np.all(np.min(np.abs(vectors), axis=1) < 1e-12))

    def test_source_spacing_can_coarsen_on_the_new_front(self):
        mesh, outline, _ = _polyomino({(x, y) for x in range(12) for y in range(8)})
        result = self.run_layer(mesh, outline, h=3.0)
        self.assertLess(len(_outline_nodes(result.outline)), len(_outline_nodes(outline)))

    def test_odd_hole_boundary_count_is_balanced_by_new_front_parity(self):
        inner = np.array([[1, 1], [1.5, 1], [2, 1], [2.5, 1], [3, 1],
                          [3, 1.75], [3, 2.4], [3, 3], [2.4, 3], [1.7, 3],
                          [1, 3], [1, 2.4], [1, 1.7]], dtype=float)
        count = len(inner)
        outer = (inner - [2, 2]) * 2 + [2, 2]
        elements = np.array([[i, count+i, count+(i+1)%count, (i+1)%count]
                             for i in range(count)], dtype=np.int32)
        mesh = Mesh2D(np.vstack((inner, outer)), elements)
        outline = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        result = self.run_layer(mesh, outline, h=0.25)
        self.assertEqual(len(_outline_nodes(outline)), 13)
        self.assertEqual(len(_outline_nodes(result.outline)) % 2, 1)

    def test_near_tangent_pattern_is_matched_within_absolute_tolerance(self):
        mesh, outline = _rectangle()
        mesh.nodes[2:, 0] += 5e-4
        pattern = [[4.0002, 0.5], [4.0002, 1.0]]
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 0.5, pattern_lines=[pattern])
        np.testing.assert_array_equal(result.mesh.nodes[:len(before_nodes)], before_nodes)
        np.testing.assert_array_equal(result.mesh.elements[:len(before_elements)], before_elements)
        self.assertGreater(result.mesh.element_count, len(before_elements))
        strict, outline = _rectangle()
        strict.nodes[2:, 0] += 5e-4
        with self.assertRaises(QuadLayerError):
            generate_layer_adaptive_mesh(strict, outline, 0.5, pattern_lines=[pattern], tolerance=1e-6)

    def test_translation_scaling_rotation_and_loop_start(self):
        for scale, shift, turns, start in (
            (1e-4, [0, 0], 0, 0), (1.0, [1e7, -2e7], 0, 0),
            (1.0, [0, 0], 1, 2),
        ):
            with self.subTest(scale=scale, shift=shift, turns=turns, start=start):
                mesh, outline = _rectangle()
                rotation = np.array([[0, -1], [1, 0]]) if turns else np.eye(2)
                mesh.nodes[:, :2] = (mesh.nodes[:, :2] @ rotation.T) * scale + shift
                outline = np.roll(outline, -start)
                self.run_layer(mesh, outline, h=0.5 * scale, tolerance=.001 * scale)

    def test_invalid_scalars_and_patterns_are_atomic(self):
        invalid = (
            {"element_size": 0}, {"element_size": -1}, {"element_size": np.nan},
            {"element_size": np.inf}, {"expand_ratio": 0.9}, {"expand_ratio": np.inf},
            {"tolerance": -1}, {"tolerance": np.nan},
            {"pattern_lines": [[[1, 4], [1, 4]]]},
            {"pattern_lines": [[[1, 4], [2, 5]]]},
            {"pattern_lines": [[[1, np.nan], [2, np.nan]]]},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                mesh, outline = _rectangle()
                snapshot = self._snapshot(mesh)
                with self.assertRaises((ValueError, TypeError)):
                    generate_layer_adaptive_mesh(mesh, outline, **{"element_size": 0.5, **kwargs})
                self.assert_unchanged(mesh, snapshot)

    def test_invalid_outline_and_nonquad_input_are_atomic(self):
        bad_outlines = (
            [0, 3, 2],
            [0, 3, 2, 3, 1],
            [0, 3, 2, 99],
            [0, 3, 2, -1],
            [0, 2, 3, 1],
            [0, 3, 2, 1, 0, 0],
            [[0, 3], [3, 2], [2, 1], [1, 0]],
        )
        for outline in bad_outlines:
            with self.subTest(outline=outline):
                mesh, _ = _rectangle()
                snapshot = self._snapshot(mesh)
                with self.assertRaises(ValueError):
                    generate_layer_adaptive_mesh(mesh, outline, 0.5)
                self.assert_unchanged(mesh, snapshot)
        mesh, outline = _rectangle()
        mesh.elements = np.array([[0, 1, 2, 2], [0, 2, 3, 3]], dtype=np.int32)
        snapshot = self._snapshot(mesh)
        with self.assertRaises(ValueError):
            generate_layer_adaptive_mesh(mesh, outline, 0.5)
        self.assert_unchanged(mesh, snapshot)

    def test_pattern_cannot_split_frozen_source_or_cross_existing_element(self):
        for pattern in ([[1.5, 3], [1.5, 3.4]], [[1.5, 2.8], [1.5, 3.4]]):
            with self.subTest(pattern=pattern):
                mesh, outline = _rectangle()
                snapshot = self._snapshot(mesh)
                with self.assertRaises(ValueError):
                    generate_layer_adaptive_mesh(mesh, outline, 0.5, pattern_lines=[pattern])
                self.assert_unchanged(mesh, snapshot)

    def test_hole_collapse_is_atomic(self):
        cells = {(x, y) for x in range(3) for y in range(3)} - {(1, 1)}
        mesh, _, lookup = _polyomino(cells)
        loop = [lookup[1, 1], lookup[2, 1], lookup[2, 2], lookup[1, 2]]
        outline = [[loop[i], loop[(i + 1) % 4]] for i in range(4)]
        snapshot = self._snapshot(mesh)
        with self.assertRaises(ValueError):
            generate_layer_adaptive_mesh(mesh, outline, 0.5)
        self.assert_unchanged(mesh, snapshot)

    def test_narrow_u_slot_closure_without_repair_is_an_atomic_topology_error(self):
        cells = {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)}
        mesh, outline, _ = _polyomino(cells)
        snapshot = self._snapshot(mesh)
        with self.assertRaises(ValueError):
            generate_layer_adaptive_mesh(mesh, outline, 0.5, repair_slots=False)
        self.assert_unchanged(mesh, snapshot)

    def test_narrow_u_slot_closure_is_repaired_by_default(self):
        cells = {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)}
        mesh, outline, _ = _polyomino(cells)
        result = self.run_layer(mesh, outline, h=0.5)
        self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
        self.assertEqual(len(_corners(mesh, result.outline)), 4)

    def test_collision_with_other_mesh_is_atomic(self):
        mesh, outline = _rectangle()
        remote = [[4.2, 0, 0], [5.2, 0, 0], [5.2, 1, 0], [4.2, 1, 0]]
        mesh.replace_data(nodes=np.vstack((mesh.nodes, remote)),
                          elements=np.vstack((mesh.elements, [4, 5, 6, 7])))
        snapshot = self._snapshot(mesh)
        with self.assertRaises(ValueError):
            generate_layer_adaptive_mesh(mesh, outline, 0.5)
        self.assert_unchanged(mesh, snapshot)

    def test_late_generated_patch_validation_failure_is_atomic(self):
        from mesher.mesh2d.adapter.layer import _mesher

        mesh, outline = _rectangle()
        snapshot = self._snapshot(mesh)
        with patch.object(_mesher, "_validate_patch",
                          side_effect=ValueError("injected final patch rejection")) as validate:
            with self.assertRaisesRegex(ValueError, "injected final patch rejection"):
                generate_layer_adaptive_mesh(mesh, outline, 0.5)
        self.assertTrue(validate.called)
        self.assert_unchanged(mesh, snapshot)

    def test_section_failure_does_not_retry_with_a_thin_collar(self):
        from mesher.mesh2d.adapter.layer import _mesher

        mesh, outline = _rectangle()
        snapshot = self._snapshot(mesh)
        fill = _mesher.fill_sections

        def fail_after_fill(nodes, *args, **kwargs):
            staged = fill(nodes, *args, **kwargs)
            self.assertGreater(len(staged[0]), len(nodes))
            self.assertGreater(len(staged[1]), 0)
            raise QuadLayerError("injected completed section rejection",
                                 code="unmeshable_section_layout", stage="section_builder")

        with patch.object(_mesher, "fill_sections", side_effect=fail_after_fill) as builder:
            with self.assertRaisesRegex(QuadLayerError, "injected completed section rejection"):
                generate_layer_adaptive_mesh(mesh, outline, 0.5)
        self.assertEqual(builder.call_count, 1)
        self.assert_unchanged(mesh, snapshot)

    def test_shallow_unequal_length_segments_cross_in_either_order(self):
        from mesher.mesh2d.adapter.layer._geometry import segment_contacts

        a, b = np.array([0.4999995, 0.0]), np.array([0.5000005, 0.0])
        c, d = np.array([0.0, -1e-8]), np.array([1.0, 1e-8])
        tolerance = 128 * np.finfo(float).eps
        for segments in ((a, b, c, d), (c, d, a, b)):
            with self.subTest(segments=segments):
                contacts = segment_contacts(*segments, tolerance)
                self.assertEqual(len(contacts), 1)
                np.testing.assert_allclose(contacts[0], [0.5, 0.0], rtol=0, atol=tolerance)

    def test_thousand_source_nodes_reject_local_interior_pattern_endpoints(self):
        mesh, outline, _ = _polyomino({(x, y) for x in range(500) for y in range(2)})
        patterns = [
            [[2.5 + 4 * i, 2.1 + 0.001 * i],
             [2.5 + 4 * i, 2.21 + 0.001 * i]]
            for i in range(100)
        ]
        snapshot = self._snapshot(mesh)
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 0.5, pattern_lines=patterns)
        self.assertEqual(caught.exception.code, "unsupported_pattern_layout")
        self.assert_unchanged(mesh, snapshot)


if __name__ == "__main__":
    unittest.main()
