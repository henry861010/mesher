"""Geometry and conformity checks for the grid generator with opaque holes."""

from collections import Counter
from contextlib import nullcontext
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.generators import (
    generate_rectilinear_mesh,
    generate_rectilinear_mesh_optimal,
)
from mesher.mesh2d.generators import holed_rectilinear as holed_generator


def _rectangle(x0, y0, x1, y1):
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _area(points):
    points = np.asarray(points, dtype=float)
    points = points - points[0]
    following = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * following[:, 1]
                        - points[:, 1] * following[:, 0]) / 2)


def _inside(point, polygon):
    """Independent half-open ray crossing, only used away from boundaries."""
    x, y = point
    crossings = 0
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (a[1] > y) != (b[1] > y):
            intersection = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            crossings += intersection > x
    return crossings % 2 == 1


class GenerateRectilinearMeshWithHolesTests(unittest.TestCase):
    def assert_valid(self, result, outer, holes):
        mesh = result.mesh
        nodes = mesh.nodes[:, :2]
        cells = mesh.elements
        scale = float(np.max(np.ptp(np.asarray(outer), axis=0)))
        origin = np.asarray(outer)[0]
        xy = (nodes - origin) / scale
        polygon = (np.asarray(outer) - origin) / scale
        voids = [(np.asarray(hole) - origin) / scale for hole in holes]
        eps = 2e-11

        self.assertEqual(cells.ndim, 2)
        self.assertEqual(cells.shape[1], 4)
        self.assertGreater(len(cells), 0)
        self.assertTrue(np.all(np.isfinite(mesh.nodes)))
        self.assertTrue(np.all(mesh.nodes[:, 2] == 0))
        self.assertTrue(np.all((cells >= 0) & (cells < len(nodes))))
        self.assertTrue(np.all(np.diff(np.sort(cells, axis=1), axis=1) > 0))
        np.testing.assert_array_equal(np.unique(cells), np.arange(len(nodes)))

        points = xy[cells]
        vectors = np.roll(points, -1, axis=1) - points
        following = np.roll(vectors, -1, axis=1)
        cross = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
        self.assertTrue(np.all(cross > 0), "every cell must be strictly CCW")
        self.assertTrue(np.all(np.logical_xor(vectors[..., 0] == 0, vectors[..., 1] == 0)),
                        "every cell edge must be a nonzero axis segment")
        low, high = points.min(axis=1), points.max(axis=1)
        centers = (low + high) / 2
        for center in centers:
            self.assertTrue(_inside(center, polygon))
            self.assertFalse(any(_inside(center, hole) for hole in voids))
        expected_area = abs(_area(polygon)) - sum(abs(_area(hole)) for hole in voids)
        actual_area = np.prod(high - low, axis=1).sum()
        self.assertAlmostEqual(float(actual_area), expected_area, delta=eps)
        for i in range(len(cells)):
            overlap = np.minimum(high[i], high[i + 1:]) - np.maximum(low[i], low[i + 1:])
            self.assertFalse(np.any(np.all(overlap > eps, axis=1)), "cell interiors overlap")
        for geometry in [polygon, *voids]:
            for a, b in zip(geometry, np.roll(geometry, -1, axis=0)):
                axis = int(np.argmax(np.abs(b - a)))
                other = 1 - axis
                overlaps = (np.minimum(high[:, axis], max(a[axis], b[axis]))
                            - np.maximum(low[:, axis], min(a[axis], b[axis])))
                penetrates = ((low[:, other] + eps < a[other])
                              & (a[other] < high[:, other] - eps) & (overlaps > eps))
                self.assertFalse(np.any(penetrates), "a geometry edge crosses a cell interior")

        directed = Counter()
        for cell in cells:
            directed.update((int(a), int(b)) for a, b in zip(cell, np.roll(cell, -1)))
        incidence = Counter()
        for edge, count in directed.items():
            incidence[tuple(sorted(edge))] += count
        self.assertTrue(all(count in (1, 2) for count in incidence.values()))
        for a, b in incidence:
            if incidence[a, b] == 2:
                self.assertEqual(directed[a, b], 1)
                self.assertEqual(directed[b, a], 1)
            # A node in an unsplit edge interior is a hanging node, even if
            # the area and edge-owner counts happen to look correct.
            delta = xy[b] - xy[a]
            axis = int(np.argmax(np.abs(delta)))
            aligned = np.abs(xy[:, 1 - axis] - xy[a, 1 - axis]) <= eps
            lower, upper = sorted((xy[a, axis], xy[b, axis]))
            interior = (xy[:, axis] > lower + eps) & (xy[:, axis] < upper - eps)
            self.assertFalse(np.any(aligned & interior), "unsplit edge contains a mesh node")

        self.assertEqual(len(result.hole_outlines), len(holes))
        loops = [result.outer_outline, *result.hole_outlines]
        expected_boundary = set()
        for index, loop in enumerate(loops):
            self.assertEqual(loop.ndim, 1)
            self.assertEqual(loop.dtype, np.dtype(np.int64))
            self.assertEqual(len(loop), len(set(map(int, loop))))
            self.assertEqual(len(loop) % 2, 0, "each complete ring must have even parity")
            self.assertGreaterEqual(len(loop), 4)
            signed = _area(xy[loop])
            self.assertLess(signed, 0) if index == 0 else self.assertGreater(signed, 0)
            source = polygon if index == 0 else voids[index - 1]
            for point in xy[loop]:
                on_source = False
                for a, b in zip(source, np.roll(source, -1, axis=0)):
                    delta = b - a
                    axis = int(np.argmax(np.abs(delta)))
                    lower, upper = sorted((a[axis], b[axis]))
                    on_source |= (abs(point[1 - axis] - a[1 - axis]) <= eps
                                  and lower - eps <= point[axis] <= upper + eps)
                self.assertTrue(on_source, "returned ring leaves its source polygon")
            expected_boundary.update(tuple(sorted((int(a), int(b))))
                                     for a, b in zip(loop, np.roll(loop, -1)))
        self.assertEqual(expected_boundary, {edge for edge, count in incidence.items() if count == 1})

    def assert_node(self, result, point):
        self.assertTrue(np.any(np.all(result.mesh.nodes[:, :2] == point, axis=1)),
                        f"mandatory point {point} was not retained exactly")

    def assert_horizontal_covered(self, result, y, lower, upper):
        points = result.mesh.nodes[:, :2]
        intervals = []
        for cell in result.mesh.elements:
            for a, b in zip(points[cell], np.roll(points[cell], -1, axis=0)):
                if a[1] == y and b[1] == y:
                    intervals.append(sorted((a[0], b[0])))
        reached = lower
        for left, right in sorted(intervals):
            if left <= reached + 1e-12:
                reached = max(reached, right)
        self.assertGreaterEqual(reached + 1e-12, upper)

    def assert_no_horizontal_edges(self, result, y, lower, upper):
        points = result.mesh.nodes[:, :2]
        for cell in result.mesh.elements:
            for a, b in zip(points[cell], np.roll(points[cell], -1, axis=0)):
                if a[1] == y and b[1] == y:
                    left, right = sorted((a[0], b[0]))
                    self.assertLessEqual(min(right, upper) - max(left, lower), 1e-12)

    def assert_exact_conformity(self, result):
        """Check adjacent-float geometry without treating nearby lines as equal."""
        points = result.mesh.nodes[:, :2]
        cells = result.mesh.elements
        incidence = Counter()
        for cell in cells:
            coordinates = points[cell]
            self.assertEqual(len(set(map(tuple, coordinates))), 4)
            self.assertGreater(_area(coordinates), 0)
            for a, b in zip(cell, np.roll(cell, -1)):
                incidence[tuple(sorted((int(a), int(b))))] += 1
                start, end = points[a], points[b]
                self.assertEqual(np.count_nonzero(start == end), 1)
                fixed = 0 if start[0] == end[0] else 1
                varying = 1 - fixed
                lo, hi = sorted((start[varying], end[varying]))
                interior = ((points[:, fixed] == start[fixed])
                            & (lo < points[:, varying])
                            & (points[:, varying] < hi))
                self.assertFalse(np.any(interior), "unsplit edge contains a node")
        self.assertTrue(all(count in (1, 2) for count in incidence.values()))
        expected_boundary = set()
        for owner, chain in enumerate([result.outer_outline, *result.hole_outlines]):
            self.assertEqual(len(chain) % 2, 0)
            self.assertEqual(len(chain), len(set(map(int, chain))))
            area = _area(points[chain])
            self.assertLess(area, 0) if owner == 0 else self.assertGreater(area, 0)
            expected_boundary.update(tuple(sorted((int(a), int(b))))
                                     for a, b in zip(chain, np.roll(chain, -1)))
        self.assertEqual(expected_boundary,
                         {edge for edge, count in incidence.items() if count == 1})

    def assert_fallback_diagnostics(self, result):
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics["strategy"], "global_grid_fallback")
        self.assertEqual(diagnostics["fallback_reason"], "parity_repair_unavailable")
        actual = [len(result.outer_outline), *map(len, result.hole_outlines)]
        self.assertEqual(diagnostics["boundary_node_counts_before"], actual)
        self.assertEqual(diagnostics["boundary_node_counts_after"], actual)
        self.assertEqual(diagnostics["parity_repairs"], [])
        self.assertEqual(diagnostics["cuts"]["parity"], 0)
        self.assertEqual(diagnostics["element_count"], result.mesh.element_count)
        self.assertEqual(diagnostics["node_count"], result.mesh.node_count)

    def test_plain_rectangle_and_empty_holes(self):
        outer = _rectangle(0, 0, 5, 3)
        result = generate_rectilinear_mesh_optimal(1.3, outer)
        self.assert_valid(result, outer, [])
        self.assertEqual(result.hole_outlines, [])

    def test_left_pattern_stops_at_hole_and_keeps_both_finite_endpoints(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        pattern = np.array([[1.17, 3.37], [2.43, 3.37]])
        result = generate_rectilinear_mesh_optimal(4, outer, holes=[hole], pattern_lines=[pattern])
        self.assert_valid(result, outer, [hole])
        for point in pattern:
            self.assert_node(result, point)
        self.assert_horizontal_covered(result, 3.37, 0, 4)
        self.assert_no_horizontal_edges(result, 3.37, 8, 12)

    def test_corner_rays_stop_at_neighboring_hole(self):
        outer = _rectangle(0, 0, 20, 12)
        holes = [_rectangle(4, 4, 7, 8), _rectangle(10, 2, 14, 10)]
        result = generate_rectilinear_mesh_optimal(100, outer, holes=holes)
        self.assert_valid(result, outer, holes)
        for y in (4, 8):
            self.assert_horizontal_covered(result, y, 7, 10)
            self.assert_no_horizontal_edges(result, y, 14, 20)

    def test_parity_is_repaired_after_cut_between_two_holes(self):
        outer = _rectangle(0, 0, 18, 12)
        holes = [_rectangle(3, 3, 6, 9), _rectangle(10, 3, 13, 9)]
        result = generate_rectilinear_mesh_optimal(
            100, outer, holes=holes, pattern_lines=[[[7, 5.27], [9, 5.27]]])
        self.assert_valid(result, outer, holes)
        for point in ([6, 5.27], [10, 5.27]):
            self.assert_node(result, point)
        self.assert_horizontal_covered(result, 5.27, 6, 10)
        self.assert_no_horizontal_edges(result, 5.27, 0, 3)
        self.assert_no_horizontal_edges(result, 5.27, 13, 18)
        diagnostics = result.diagnostics
        before = diagnostics["boundary_node_counts_before"]
        after = diagnostics["boundary_node_counts_after"]
        self.assertEqual([count % 2 for count in before], [0, 1, 1])
        self.assertEqual(after, [len(result.outer_outline), *map(len, result.hole_outlines)])
        self.assertEqual(sum(after), sum(before) + 2 * len(diagnostics["parity_repairs"]))
        self.assertGreater(len(diagnostics["parity_repairs"]), 0)

    def test_concave_outer_and_hole_return_input_order_rings(self):
        outer = np.array([[0, 0], [14, 0], [14, 10], [10, 10], [10, 14], [0, 14]])
        holes = [np.array([[3, 3], [8, 3], [8, 5], [6, 5], [6, 8], [3, 8]]),
                 _rectangle(1, 10, 4, 12)]
        result = generate_rectilinear_mesh_optimal(2.7, outer, holes=holes)
        self.assert_valid(result, outer, holes)

    def test_vertical_patterns_and_crossings_are_conforming(self):
        outer, hole = _rectangle(0, 0, 10, 12), _rectangle(3, 5, 7, 9)
        patterns = [[[4.31, 1.13], [4.31, 3.71]], [[1.29, 2.41], [8.17, 2.41]]]
        result = generate_rectilinear_mesh_optimal(4, outer, holes=[hole], pattern_lines=patterns)
        self.assert_valid(result, outer, [hole])
        for point in np.asarray(patterns).reshape(-1, 2):
            self.assert_node(result, point)
        self.assert_node(result, [4.31, 2.41])

    def test_patterns_are_clipped_across_holes_and_outside_domain(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        result = generate_rectilinear_mesh_optimal(
            5, outer, holes=[hole], pattern_lines=[[[-2, 3.37], [14, 3.37]]])
        self.assert_valid(result, outer, [hole])
        self.assert_horizontal_covered(result, 3.37, 0, 4)
        self.assert_horizontal_covered(result, 3.37, 8, 12)
        self.assert_no_horizontal_edges(result, 3.37, 4, 8)

    def test_outside_and_hole_interior_patterns_do_not_change_mesh(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        baseline = generate_rectilinear_mesh_optimal(3, outer, holes=[hole])
        result = generate_rectilinear_mesh_optimal(
            3, outer, holes=[hole],
            pattern_lines=[[[5.13, 4.27], [6.79, 4.27]], [[-5, 1.13], [-2, 1.13]]])
        np.testing.assert_array_equal(result.mesh.nodes, baseline.mesh.nodes)
        np.testing.assert_array_equal(result.mesh.elements, baseline.mesh.elements)

    def test_boundary_pattern_retains_endpoints_and_corner_tangency_is_safe(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        patterns = [[[4, 3.31], [4, 6.73]], [[0, 2], [4, 2]]]
        result = generate_rectilinear_mesh_optimal(4, outer, holes=[hole], pattern_lines=patterns)
        self.assert_valid(result, outer, [hole])
        for point in np.asarray(patterns).reshape(-1, 2):
            self.assert_node(result, point)

    def test_duplicate_patterns_and_reversed_endpoints_are_idempotent(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        pattern = np.array([[1.17, 3.37], [2.43, 3.37]])
        baseline = generate_rectilinear_mesh_optimal(3, outer, holes=[hole], pattern_lines=[pattern])
        repeated = generate_rectilinear_mesh_optimal(
            3, outer, holes=[hole], pattern_lines=[pattern[::-1], pattern, pattern])
        np.testing.assert_array_equal(repeated.mesh.nodes, baseline.mesh.nodes)
        np.testing.assert_array_equal(repeated.mesh.elements, baseline.mesh.elements)

    def test_tolerance_does_not_merge_distinct_mandatory_coordinates(self):
        outer = _rectangle(0, 0, 6, 6)
        patterns = np.array([[[1.17, 2.37], [2.43, 2.37]],
                             [[1.17, 2.3701], [2.43, 2.3701]]])
        result = generate_rectilinear_mesh_optimal(
            3, outer, pattern_lines=patterns, coordinate_tolerance=1e-3)
        self.assert_valid(result, outer, [])
        for point in patterns.reshape(-1, 2):
            self.assert_node(result, point)

    def test_consecutive_float_pattern_coordinates_do_not_need_a_midpoint(self):
        outer = _rectangle(0, 0, 3, 3)
        next_x = np.nextafter(1.0, 2.0)
        patterns = [[[1.0, 0], [1.0, 3]], [[next_x, 0], [next_x, 3]]]
        result = generate_rectilinear_mesh_optimal(10, outer, pattern_lines=patterns)
        self.assert_valid(result, outer, [])
        self.assertEqual(result.mesh.element_count, 3)
        np.testing.assert_array_equal(np.unique(result.mesh.nodes[:, 0]), [0, 1.0, next_x, 3])
        for pattern in patterns:
            for point in pattern:
                self.assert_node(result, point)

    def test_inputs_are_unchanged_and_optional_closure_and_winding_work(self):
        outer = np.vstack((_rectangle(0, 0, 12, 10), [0, 0]))[::-1].copy()
        hole = np.vstack((_rectangle(4, 2, 8, 8), [4, 2]))[::-1].copy()
        patterns = np.array([[[1.17, 3.37], [2.43, 3.37]]])
        originals = [value.copy() for value in (outer, hole, patterns)]
        result = generate_rectilinear_mesh_optimal(4, outer, holes=[hole], pattern_lines=patterns)
        self.assert_valid(result, outer[:-1], [hole[:-1]])
        for value, original in zip((outer, hole, patterns), originals):
            np.testing.assert_array_equal(value, original)

    def test_scale_invariance_with_zero_coordinate_tolerance(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        pattern = np.array([[[1.17, 3.37], [2.43, 3.37]]])
        for scale in (1e-7, 1, 1e6):
            with self.subTest(scale=scale):
                result = generate_rectilinear_mesh_optimal(
                    4 * scale, outer * scale, holes=[hole * scale],
                    pattern_lines=pattern * scale, coordinate_tolerance=0)
                self.assert_valid(result, outer * scale, [hole * scale])
                for point in pattern.reshape(-1, 2) * scale:
                    self.assert_node(result, point)

    def test_unrepresentable_cell_area_and_aspect_ratio_are_rejected_without_input_changes(self):
        patterns = {
            "area_underflow": [[[1e-200, 1e-200], [.5, 1e-200]]],
            "aspect_overflow": [[[1e-310, .2], [1e-310, .8]]],
        }
        for case, supplied in patterns.items():
            with self.subTest(case=case):
                outer = _rectangle(0, 0, 1, 1)
                pattern = np.asarray(supplied, dtype=float)
                original_outer, original_pattern = outer.copy(), pattern.copy()
                with self.assertRaises(ValueError):
                    generate_rectilinear_mesh_optimal(
                        2, outer, pattern_lines=pattern, coordinate_tolerance=0)
                np.testing.assert_array_equal(outer, original_outer)
                np.testing.assert_array_equal(pattern, original_pattern)

    def test_adjacent_float_geometry_coordinates_work_in_both_axes(self):
        next_coordinate = np.nextafter(1.0, 2.0)
        original_outer = _rectangle(0, 0, 3, 3)
        original_hole = _rectangle(1, 1, next_coordinate, 2)
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                outer = original_outer[:, ::-1] if transpose else original_outer
                hole = original_hole[:, ::-1] if transpose else original_hole
                result = generate_rectilinear_mesh_optimal(
                    10, outer, holes=[hole], coordinate_tolerance=0)
                self.assert_exact_conformity(result)
                for point in np.vstack((outer, hole)):
                    self.assert_node(result, point)
                coordinates = result.mesh.nodes[result.mesh.elements, :2]
                areas = np.prod(np.ptp(coordinates, axis=1), axis=1)
                self.assertTrue(np.all(areas > 0))
                self.assertAlmostEqual(float(areas.sum()), 9.0, places=13)

    def test_parity_falls_back_when_left_edge_has_no_representable_midpoint(self):
        base = float(2**52)
        outer = _rectangle(0, base, 10, base + 6)
        hole = _rectangle(3, base + 2, 7, base + 3)
        patterns = np.array([[[5, base], [5, base + 2]]])
        result = generate_rectilinear_mesh_optimal(
            100, outer, holes=[hole], pattern_lines=patterns)
        self.assert_valid(result, outer, [hole])
        self.assert_exact_conformity(result)
        self.assert_fallback_diagnostics(result)
        for point in patterns.reshape(-1, 2):
            self.assert_node(result, point)

    def test_parity_falls_back_when_unrelated_levels_exhaust_midpoint_candidates(self):
        base = float(2**52)
        outer = _rectangle(0, base, 20, base + 16)
        holes = [_rectangle(4, base + 6, 6, base + 10),
                 _rectangle(10, base + 2, 12, base + 14)]
        patterns = [[[0, base + 8], [4, base + 8]]]
        patterns.extend([[[14, base + i], [15, base + i]] for i in (7, 8, 9)])
        result = generate_rectilinear_mesh_optimal(
            100, outer, holes=holes, pattern_lines=patterns)
        self.assert_valid(result, outer, holes)
        self.assert_exact_conformity(result)
        self.assert_fallback_diagnostics(result)
        for point in np.asarray(patterns).reshape(-1, 2):
            self.assert_node(result, point)

    def test_fallback_discards_repairs_completed_before_another_hole_fails(self):
        base = float(2**52)
        outer = _rectangle(0, base, 20, base + 16)
        holes = [_rectangle(3, base + 2, 7, base + 3),
                 _rectangle(11, base + 6, 15, base + 12)]
        patterns = [[[5, base], [5, base + 2]], [[15, base + 9], [20, base + 9]]]
        repair_parity = holed_generator._Builder.repair_parity
        local_repairs = []

        def capture_repairs(builder):
            try:
                return repair_parity(builder)
            finally:
                local_repairs.extend(cut.coordinate for cut in builder.cuts.values()
                                     if cut.kind == "parity")

        with patch.object(holed_generator._Builder, "repair_parity", capture_repairs):
            result = generate_rectilinear_mesh_optimal(
                100, outer, holes=holes, pattern_lines=patterns)
        self.assertEqual(local_repairs, [base + 8])
        self.assertFalse(np.any(result.mesh.nodes[:, 1] == base + 8))
        self.assert_fallback_diagnostics(result)
        self.assertEqual(result.diagnostics["cuts"],
                         {"geometry": 16, "pattern": 4, "sizing": 0, "parity": 0})
        self.assert_valid(result, outer, holes)
        self.assert_exact_conformity(result)

    def test_forced_fallback_preserves_concave_geometry_and_clipped_patterns(self):
        outer = np.array([[0, 0], [14, 0], [14, 10], [10, 10], [10, 14], [0, 14]])
        holes = [np.array([[3, 3], [8, 3], [8, 5], [6, 5], [6, 8], [3, 8]]),
                 _rectangle(1, 10, 4, 12)]
        patterns = np.array([[[-2, 4], [16, 4]], [[1.25, 1.5], [11.75, 1.5]],
                             [[9.25, 2.25], [9.25, 11.75]], [[3, 3.5], [3, 4.5]],
                             [[4, 4], [5, 4]]])
        with patch.object(holed_generator._Builder, "repair_parity",
                          side_effect=holed_generator._ParityRepairUnavailable):
            result = generate_rectilinear_mesh_optimal(
                4, outer, holes=holes, pattern_lines=patterns)
            repeated = generate_rectilinear_mesh_optimal(
                4, outer[::-1], holes=[hole[::-1] for hole in holes],
                pattern_lines=patterns[::-1, ::-1])
        self.assert_valid(result, outer, holes)
        self.assert_fallback_diagnostics(result)
        np.testing.assert_array_equal(result.mesh.nodes, repeated.mesh.nodes)
        np.testing.assert_array_equal(result.mesh.elements, repeated.mesh.elements)
        for point in np.asarray(patterns[1:4]).reshape(-1, 2):
            self.assert_node(result, point)
        self.assert_horizontal_covered(result, 4, 0, 3)
        self.assert_horizontal_covered(result, 4, 8, 14)
        self.assert_no_horizontal_edges(result, 4, 3, 8)

    def test_unrelated_parity_errors_are_not_hidden_by_fallback(self):
        outer = _rectangle(0, 0, 4, 4)
        for error in (ValueError("unrelated invalid geometry"),
                      RuntimeError("unexpected internal failure")):
            with self.subTest(error=type(error).__name__):
                with patch.object(holed_generator._Builder, "repair_parity", side_effect=error):
                    with self.assertRaisesRegex(type(error), str(error)):
                        generate_rectilinear_mesh_optimal(10, outer)

    def test_forced_fallback_ignores_patterns_outside_the_material(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        active = [[[0, 3], [4, 3]]]
        inactive = [[[5.13, 4.27], [6.79, 4.27]], [[-5, 1.13], [-2, 1.13]]]
        with patch.object(holed_generator._Builder, "repair_parity",
                          side_effect=holed_generator._ParityRepairUnavailable):
            baseline = generate_rectilinear_mesh_optimal(
                4, outer, holes=[hole], pattern_lines=active)
            result = generate_rectilinear_mesh_optimal(
                4, outer, holes=[hole], pattern_lines=active + inactive)
        np.testing.assert_array_equal(result.mesh.nodes, baseline.mesh.nodes)
        np.testing.assert_array_equal(result.mesh.elements, baseline.mesh.elements)
        self.assert_fallback_diagnostics(result)

    def test_max_elements_requires_a_positive_integer(self):
        outer = _rectangle(0, 0, 2, 2)
        invalid = (0, -1, True, np.bool_(False), 1.0, 2.5, "4", None, np.nan, np.inf)
        for limit in invalid:
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "max_elements must be a positive integer"):
                    generate_rectilinear_mesh_optimal(1, outer, max_elements=limit)
        result = generate_rectilinear_mesh_optimal(1, outer, max_elements=np.int64(4))
        self.assertEqual(result.mesh.element_count, 4)

    def test_max_elements_exact_limit_succeeds_and_one_below_fails(self):
        outer = _rectangle(0, 0, 2, 2)
        result = generate_rectilinear_mesh_optimal(1, outer, max_elements=4)
        self.assert_valid(result, outer, [])
        self.assertEqual(result.mesh.element_count, 4)
        with self.assertRaisesRegex(ValueError, "max_elements exceeded: limit=3,.*stage="):
            generate_rectilinear_mesh_optimal(1, outer, max_elements=3)

    def test_impossible_default_budget_is_rejected_before_index_allocation(self):
        outer = _rectangle(0, 0, 1, 1)
        with patch.object(holed_generator, "_RectangleIndex",
                          side_effect=AssertionError("index allocation must not start")):
            with self.assertRaisesRegex(ValueError, "max_elements exceeded: limit=1000000,"):
                generate_rectilinear_mesh_optimal(1e-5, outer)

    def test_seed_partition_respects_element_budget(self):
        outer = _rectangle(0, 0, 4, 4)
        patterns = [[[1, 0], [1, 4]], [[0, 1], [4, 1]]]
        result = generate_rectilinear_mesh_optimal(
            100, outer, pattern_lines=patterns, max_elements=4)
        self.assertEqual(result.mesh.element_count, 4)
        with self.assertRaisesRegex(ValueError, "max_elements exceeded: limit=3,.*stage="):
            generate_rectilinear_mesh_optimal(
                100, outer, pattern_lines=patterns, max_elements=3)

    def test_sizing_subdivisions_respect_element_budget(self):
        outer = _rectangle(0, 0, 3, 3)
        result = generate_rectilinear_mesh_optimal(2, outer, max_elements=4)
        self.assertEqual(result.mesh.element_count, 4)
        with self.assertRaisesRegex(ValueError, "max_elements exceeded: limit=3,.*stage="):
            generate_rectilinear_mesh_optimal(2, outer, max_elements=3)

    def test_parity_subdivisions_respect_element_budget_without_fallback(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        patterns = [[[0, 3], [4, 3]]]
        result = generate_rectilinear_mesh_optimal(
            100, outer, holes=[hole], pattern_lines=patterns, max_elements=10)
        self.assertEqual(result.mesh.element_count, 10)
        self.assertGreater(result.diagnostics["cuts"]["parity"], 0)
        with self.assertRaisesRegex(ValueError, "max_elements exceeded: limit=9,.*stage="):
            generate_rectilinear_mesh_optimal(
                100, outer, holes=[hole], pattern_lines=patterns, max_elements=9)

    def test_fallback_respects_exact_and_exceeded_element_budgets(self):
        outer, hole = _rectangle(0, 0, 12, 10), _rectangle(4, 2, 8, 8)
        patterns = [[[0, 3], [4, 3]]]
        with patch.object(holed_generator._Builder, "repair_parity",
                          side_effect=holed_generator._ParityRepairUnavailable):
            baseline = generate_rectilinear_mesh_optimal(
                100, outer, holes=[hole], pattern_lines=patterns)
            count = baseline.mesh.element_count
            result = generate_rectilinear_mesh_optimal(
                100, outer, holes=[hole], pattern_lines=patterns, max_elements=count)
            self.assert_valid(result, outer, [hole])
            self.assert_fallback_diagnostics(result)
            self.assertEqual(result.mesh.element_count, count)
            with self.assertRaisesRegex(ValueError, "max_elements exceeded:.*stage="):
                generate_rectilinear_mesh_optimal(
                    100, outer, holes=[hole], pattern_lines=patterns, max_elements=count - 1)

    def test_budget_preflight_uses_the_same_edge_roundoff_tolerance_as_meshing(self):
        outer = _rectangle(0, 0, 1, 1)
        target = np.nextafter(1.0, 0.0)
        for force_fallback in (False, True):
            with self.subTest(force_fallback=force_fallback):
                context = (patch.object(holed_generator._Builder, "repair_parity",
                                        side_effect=holed_generator._ParityRepairUnavailable)
                           if force_fallback else nullcontext())
                with context:
                    result = generate_rectilinear_mesh_optimal(
                        target, outer, max_elements=1)
                self.assertEqual(result.mesh.element_count, 1)
                self.assert_valid(result, outer, [])

    def test_largest_finite_target_does_not_overflow_budget_preflight(self):
        outer = _rectangle(0, 0, 1, 1)
        for force_fallback in (False, True):
            with self.subTest(force_fallback=force_fallback):
                context = (patch.object(holed_generator._Builder, "repair_parity",
                                        side_effect=holed_generator._ParityRepairUnavailable)
                           if force_fallback else nullcontext())
                with context, np.errstate(all="raise"):
                    result = generate_rectilinear_mesh_optimal(
                        np.finfo(float).max, outer, max_elements=1)
                self.assertEqual(result.mesh.element_count, 1)
                self.assert_valid(result, outer, [])

    def test_rounding_induced_extra_subdivisions_obey_target_and_final_budget(self):
        outer = _rectangle(1e12, 0, 1e12 + 1, .5)
        target = .1
        for force_fallback in (False, True):
            with self.subTest(force_fallback=force_fallback):
                context = (patch.object(holed_generator._Builder, "repair_parity",
                                        side_effect=holed_generator._ParityRepairUnavailable)
                           if force_fallback else nullcontext())
                with context:
                    result = generate_rectilinear_mesh_optimal(target, outer, max_elements=60)
                    self.assertEqual(result.mesh.element_count, 60)
                    self.assert_valid(result, outer, [])
                    maximum = result.diagnostics["quality"]["maximum_edge_length"]
                    self.assertLessEqual(maximum, target * (1 + 32 * np.finfo(float).eps))
                    with self.assertRaisesRegex(ValueError, "max_elements exceeded: limit=59,"):
                        generate_rectilinear_mesh_optimal(target, outer, max_elements=59)

    def test_final_validation_rejects_missing_interior_pattern_endpoints(self):
        outer = _rectangle(0, 0, 1, 1)
        domain = holed_generator.OrthogonalDomain(outer)
        # One cell covers the geometry and has a valid boundary, but cannot
        # satisfy a required pattern endpoint in the cell's interior.
        with self.assertRaisesRegex(ValueError, "retained pattern endpoint"):
            holed_generator._assemble(
                domain, [(0, 0, 1, 1)], [set(map(tuple, outer))], 1,
                holed_generator._ElementBudget(1),
                patterns=[np.array([[.25, .5], [.75, .5]])],
            )

    def test_solid_and_material_thickness_ratios_across_scales_and_origins(self):
        cases = [(1, 100, []), (10, 25, [_rectangle(10, 10, 90, 90)]),
                 (100, 25, [_rectangle(1, 1, 99, 99)]),
                 (10, 25, [_rectangle(10, 10, 45, 90), _rectangle(55, 10, 90, 90)]),
                 (100, 25, [_rectangle(1, 1, 49.5, 99), _rectangle(50.5, 1, 99, 99)])]
        for ratio, target, source_holes in cases:
            for scale, origin in ((1e-7, 0), (1, 0), (1e6, 0), (1, 1e12)):
                with self.subTest(ratio=ratio, holes=len(source_holes), scale=scale, origin=origin):
                    outer = _rectangle(0, 0, 100, 100) * scale + origin
                    holes = [hole * scale + origin for hole in source_holes]
                    result = generate_rectilinear_mesh_optimal(
                        target * scale, outer, holes=holes, coordinate_tolerance=0)
                    self.assert_valid(result, outer, holes)
                    quality = result.diagnostics["quality"]
                    self.assertTrue(all(np.isfinite(value) for value in quality.values()))
                    self.assertLessEqual(quality["maximum_edge_length"], target * scale * (1 + 1e-12))

    def test_fixed_seed_concave_holes_and_patterns_preserve_mesh_invariants(self):
        random = np.random.default_rng(51943)
        outer = np.array([[0, 0], [18, 0], [18, 18], [10, 18], [10, 16], [0, 16]])
        for case in range(8):
            holes = []
            for ox, oy in ((2, 2), (10, 2), (2, 10)):
                heights = random.integers(2, 5, size=3)
                points = [[ox, oy], [ox + 3, oy], [ox + 3, oy + heights[-1]]]
                for index in range(2, -1, -1):
                    points.append([ox + index, oy + heights[index]])
                    if index and heights[index] != heights[index - 1]:
                        points.append([ox + index, oy + heights[index - 1]])
                holes.append(np.asarray(points, dtype=float))
            patterns = []
            for index in range(3):
                point = random.uniform(0.5, 15.5, size=2)
                endpoint = point.copy()
                endpoint[index % 2] = random.uniform(0.5, 15.5)
                patterns.append([point, endpoint])
            with self.subTest(case=case):
                result = generate_rectilinear_mesh_optimal(
                    5, outer, holes=holes, pattern_lines=patterns)
                self.assert_valid(result, outer, holes)
                self.assert_exact_conformity(result)

    def test_dense_local_patterns_use_fewer_cells_than_full_grid(self):
        outer, hole = _rectangle(0, 0, 20, 12), _rectangle(4, 2, 8, 10)
        levels = np.linspace(3.13, 8.87, 17)
        patterns = np.asarray([[[0.77, y], [2.31, y]] for y in levels])
        result = generate_rectilinear_mesh_optimal(3, outer, holes=[hole], pattern_lines=patterns)
        self.assert_valid(result, outer, [hole])
        legacy = generate_rectilinear_mesh(
            3, [0, .77, 2.31, 4, 8, 20], [0, 2, *levels, 10, 12], coordinate_tolerance=0)
        centers = legacy.nodes[legacy.elements, :2].mean(axis=1)
        retained = np.asarray([not _inside(center, hole) for center in centers])
        self.assertLess(len(result.mesh.elements), int(retained.sum()))
        for y in levels:
            self.assert_horizontal_covered(result, y, .77, 2.31)
            self.assert_no_horizontal_edges(result, y, 8, 20)

    def test_invalid_sizes_tolerances_and_pattern_shapes(self):
        outer = _rectangle(0, 0, 10, 10)
        for size in (0, -1, np.nan, np.inf):
            with self.subTest(size=size), self.assertRaises(ValueError):
                generate_rectilinear_mesh_optimal(size, outer)
        for tolerance in (-1, np.nan, np.inf, None):
            with self.subTest(tolerance=tolerance), self.assertRaises((ValueError, TypeError)):
                generate_rectilinear_mesh_optimal(2, outer, coordinate_tolerance=tolerance)
        for pattern in ([[[1, 1], [2, 2]]], [[[1, 1], [1, 1]]],
                        [[[1, 1], [np.nan, 1]]], [[1, 2], [3, 4]]):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                generate_rectilinear_mesh_optimal(2, outer, pattern_lines=pattern)

    def test_invalid_loops_contacts_overlaps_and_nesting_are_rejected(self):
        outer = _rectangle(0, 0, 10, 10)
        bad_outers = [[], [[0, 0], [1, 0], [1, 1]],
                      [[0, 0], [10, 0], [9, 10], [0, 10]],
                      [[0, 0], [10, 0], [10, np.nan], [0, 10]],
                      [[0, 0], [10, 0], [10, 0], [10, 10], [0, 10]],
                      [[0, 0], [4, 0], [4, 4], [2, 4], [2, -2], [0, -2]]]
        for polygon in bad_outers:
            with self.subTest(outer=polygon), self.assertRaises(ValueError):
                generate_rectilinear_mesh_optimal(2, polygon)
        hole_cases = [
            [_rectangle(0, 2, 4, 6)],
            [_rectangle(8, 2, 12, 6)],
            [_rectangle(2, 2, 5, 5), _rectangle(5, 3, 7, 7)],
            [_rectangle(2, 2, 5, 5), _rectangle(5, 5, 7, 7)],
            [_rectangle(2, 2, 6, 6), _rectangle(4, 4, 8, 8)],
            [_rectangle(2, 2, 8, 8), _rectangle(3, 3, 4, 4)],
        ]
        for holes in hole_cases:
            with self.subTest(holes=holes), self.assertRaises(ValueError):
                generate_rectilinear_mesh_optimal(2, outer, holes=holes)

    def test_generator_import_and_execution_need_no_optional_packages(self):
        script = """
import sys

class OptionalImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'shapely', 'scipy', 'matplotlib', 'pyvista', 'PySide6', 'process_flow_kernel'
        }:
            raise AssertionError('unexpected optional dependency: ' + fullname)

sys.meta_path.insert(0, OptionalImportGuard())
from mesher.mesh2d.generators import generate_rectilinear_mesh_optimal
result = generate_rectilinear_mesh_optimal(
    2, [[0, 0], [8, 0], [8, 8], [0, 8]],
    holes=[[[2, 2], [4, 2], [4, 4], [2, 4]]],
    pattern_lines=[[[.5, 3.1], [1.5, 3.1]]])
assert result.mesh.elements.shape[1] == 4
assert len(result.hole_outlines[0]) % 2 == 0
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
