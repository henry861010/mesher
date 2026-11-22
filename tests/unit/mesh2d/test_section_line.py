"""LINE sections plan shared seams and preserve the whole source chain."""

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter.layer._construction.line import plan_line
from mesher.mesh2d.adapter.layer._limits import ElementLimitExceeded
from mesher.mesh2d.adapter.layer._construction.units import build_unit_patch, quality_score
from mesher import Mesh2D
from mesher.mesh2d.adapter.unit import build_unit_1_x_y_z
from mesher.mesh2d.adapter.unit.build_unit_1_1_1_x import (
    _segments_intersect,
    _validate_cyclic_boundary,
)
from mesher.mesh2d.adapter.unit.build_unit_1_1_x_y import _validate_patch
import mesher.mesh2d.adapter.layer._construction.strip as strip


def _fixture(n=20, width=100.0, height=50.0, left_nodes=2, right_nodes=2):
    source = np.column_stack((np.linspace(0, width, n + 1), np.zeros(n + 1), np.zeros(n + 1)))
    outer = np.array([[0.0, height, 0.0], [width, height, 0.0]])
    left = np.linspace(source[0], outer[0], left_nodes)
    right = np.linspace(source[-1], outer[-1], right_nodes)
    return source, outer, left, right


class SectionLineTests(unittest.TestCase):
    def assert_conforming(self, patch):
        boundary = (patch.source_ids, patch.right_ids,
                    patch.outer_ids[::-1], patch.left_ids[::-1])
        expected = {tuple(sorted((int(a), int(b)))) for side in boundary
                    for a, b in zip(side[:-1], side[1:])}
        incidence = Counter(tuple(sorted((int(a), int(b)))) for quad in patch.quads
                            for a, b in zip(quad, np.roll(quad, -1)))
        self.assertEqual({edge for edge, count in incidence.items() if count == 1}, expected)
        self.assertTrue(all(count == (1 if edge in expected else 2)
                            for edge, count in incidence.items()))
        self.assertEqual(set(patch.quads.ravel()), set(range(len(patch.nodes))))
        self.assertTrue(np.isfinite(quality_score(patch.nodes, patch.quads)))
        _validate_patch(patch.nodes, patch.quads, boundary)

    def test_large_coarsening_preserves_source_and_reports_actual_outer_spacing(self):
        args = _fixture()
        originals = tuple(array.copy() for array in args)
        candidates = plan_line(*args, 50.0, 1e-9, best_only=True)
        self.assertEqual(len(candidates), 1)
        patch = candidates[0]
        self.assertEqual(len(patch.outer_ids), 3)
        np.testing.assert_array_equal(patch.nodes[patch.source_ids], args[0])
        gaps = np.diff(patch.nodes[patch.outer_ids, 0])
        self.assertTrue(np.all((gaps >= 25.) & (gaps <= 62.5)))
        self.assertAlmostEqual(patch.size_error, np.mean((gaps / 50. - 1) ** 2))
        self.assertAlmostEqual(patch.diagnostics["max_alignment_angle"], 0.)
        # Large coarsening remains valid without a quality rejection threshold.
        self.assertGreater(patch.quality, 4)
        self.assert_conforming(patch)
        for original, current in zip(originals, args):
            np.testing.assert_array_equal(original, current)

    def test_all_ordinary_seam_pairs_close_conformingly(self):
        for left in (2, 3):
            for right in (2, 3):
                with self.subTest(left=left, right=right):
                    args = _fixture(6, 12, 4, left, right)
                    patches = plan_line(*args, 4, 1e-10)
                    self.assertTrue(patches)
                    for patch in patches:
                        self.assertEqual(len(patch.left_ids), left)
                        self.assertEqual(len(patch.right_ids), right)
                        self.assert_conforming(patch)

    def test_dense_sublayers_retry_the_middle_and_keep_the_final_anchor(self):
        source, outer, left, right = _fixture(37, 185., 400., 3, 3)
        outer = np.insert(outer, 1, [19.5, 400., 0.], axis=0)
        settings = dict(best_only=True, max_transition_count=10, size_band_active=False)
        first, = plan_line(source, [left[1], right[1]], left[:2], right[:2], 800., .001, **settings)
        preferred = first.nodes[first.outer_ids]
        result, = plan_line(source, outer, left, right, 800., .001, dense_sublayers=True, **settings)
        np.testing.assert_array_equal(result.nodes[result.diagnostics["midline_ids"]], preferred)
        self.assertTrue(np.any(np.all(result.nodes[result.outer_ids] == outer[1], axis=1)))

        def reject_first_middle(bottom, *args, **kwargs):
            if np.array_equal(bottom, preferred):
                return None
            return strip.build_strip(bottom, *args, **kwargs)

        with patch("mesher.mesh2d.adapter.layer._construction.line.build_strip", side_effect=reject_first_middle):
            result, = plan_line(source, outer, left, right, 800., .001, dense_sublayers=True, **settings)
        self.assertFalse(np.array_equal(result.nodes[result.diagnostics["midline_ids"]], preferred))
        self.assertTrue(np.any(np.all(result.nodes[result.outer_ids] == outer[1], axis=1)))
        self.assert_conforming(result)

    def test_both_dense_sublayers_share_one_element_budget(self):
        args = _fixture(1, 5., 400., 3, 3)
        settings = dict(best_only=True, dense_sublayers=True, size_band_active=False, max_transition_count=1)
        result, = plan_line(*args, 800., .001, max_added_elements=2, **settings)
        self.assertEqual(len(result.quads), 2)
        with self.assertRaises(ElementLimitExceeded) as caught:
            plan_line(*args, 800., .001, max_added_elements=1, **settings)
        self.assertEqual(caught.exception.context["required_elements"], 2)
        self.assertEqual(caught.exception.context["limit"], 1)

    def test_split_endpoints_leave_single_full_depth_quads_in_the_middle(self):
        for left in (2, 3):
            for right in (2, 3):
                with self.subTest(left=left, right=right):
                    args = _fixture(24, 24, 1, left, right)
                    patch, = plan_line(*args, 1., 1e-10, best_only=True)
                    np.testing.assert_array_equal(patch.nodes[patch.left_ids], args[2])
                    np.testing.assert_array_equal(patch.nodes[patch.right_ids], args[3])
                    self.assertTrue(all(u["boundary_counts"][1] == 2 for u in patch.units[:-1]))
                    self.assertTrue(all(u["boundary_counts"][3] == 2 for u in patch.units[1:]))
                    # Check actual connectivity in the regular center as well as unit records.
                    cells = patch.nodes[patch.quads, :2]
                    if left == right == 3:
                        self.assertEqual(len(cells), 28)
                        self.assertEqual(len(patch.units), 24)
                        self.assertEqual(patch.units[0]["boundary_counts"], [2, 2, 3, 3])
                        self.assertEqual(patch.units[-1]["boundary_counts"], [2, 3, 3, 2])
                    for x in range(1, 23):
                        idx = np.flatnonzero(np.logical_and(cells[:, :, 0].min(axis=1) == x,
                                                            cells[:, :, 0].max(axis=1) == x + 1))
                        self.assertEqual(len(idx), 1)
                        self.assertEqual(set(map(tuple, cells[idx[0]])),
                                         {(x, 0), (x + 1, 0), (x, 1), (x + 1, 1)})
                    self.assert_conforming(patch)

    def test_fixed_three_point_ends_keep_exact_nodes_and_unsplit_internal_cuts(self):
        source, _, left, right = _fixture(24, 24., 1., 3, 3)
        outer = source.copy()
        outer[:, 1] = 1.
        patch, = plan_line(source, outer, left, right, 1., 1e-10,
                           fixed_outer_boundary=True, best_only=True)
        np.testing.assert_array_equal(patch.nodes[patch.outer_ids], outer)
        self.assertGreater(len(patch.units), 2)
        self.assertAlmostEqual(patch.diagnostics["max_alignment_angle"], 45.)
        for unit in patch.units[:-1]:
            count, cut_count, _, _ = unit["boundary_counts"]
            self.assertEqual(cut_count, 2)
            cut = np.asarray(unit["polygon"][count - 1:count + 1])[:, :2]
            np.testing.assert_allclose(np.abs(cut[1] - cut[0]), [1., 1.])

        self.assert_conforming(patch)

    def test_refinement_keeps_ordered_source_assignments(self):
        patches = plan_line(*_fixture(3, 12, 4), 1.7, 1e-10)
        self.assertTrue(patches)
        for patch in patches:
            self.assertGreater(len(patch.outer_ids) - 1, 3)
            loads = [count for unit in patch.units for count in unit["source_segments_per_outer"]]
            self.assertEqual(sum(loads), 3)
            self.assertEqual(len(loads), len(patch.outer_ids) - 1)
            self.assert_conforming(patch)

    def test_outer_anchors_survive_sampling_and_parity(self):
        source, _, left, right = _fixture(8, 16, 4)
        outer = np.array([[0., 4, 0], [3, 4, 0], [16, 4, 0]])
        patches = plan_line(source, outer, left, right, 8, 1e-10)
        self.assertTrue(patches)
        for patch in patches:
            for point in outer:
                matches = np.linalg.norm(patch.nodes[patch.outer_ids] - point, axis=1) < 1e-10
                self.assertEqual(np.count_nonzero(matches), 1)
            self.assert_conforming(patch)

    def test_kinked_seam_and_four_node_seam_are_preserved(self):
        source, outer, _, right = _fixture(6, 12, 4, right_nodes=4)
        left = np.array([[0., 0, 0], [-1, 2, 0], [0, 4, 0]])
        patches = plan_line(source, outer, left, right, 4, 1e-10)
        self.assertTrue(patches)
        for patch in patches:
            np.testing.assert_array_equal(patch.nodes[patch.left_ids], left)
            np.testing.assert_array_equal(patch.nodes[patch.right_ids], right)
            self.assert_conforming(patch)

    def test_unit_diagnostics_and_input_winding_are_independent(self):
        source, outer, left, right = _fixture(3, 3, 1)
        sides = (source, right, outer[::-1], left[::-1])
        unit = build_unit_patch(sides, 1e-10)
        reverse = build_unit_patch(tuple(side[::-1] for side in sides[::-1]), 1e-10)
        self.assertEqual(unit.method, "template")
        self.assertAlmostEqual(unit.quality, reverse.quality)
        for side, ids in zip(sides, unit.sides):
            np.testing.assert_array_equal(unit.nodes[ids], side)

    def test_invalid_endpoint_does_not_modify_input(self):
        source, outer, left, right = _fixture()
        right[-1, 0] += 1
        originals = [array.copy() for array in (source, outer, left, right)]
        with self.assertRaises(ValueError):
            plan_line(source, outer, left, right, 50, 1e-9)
        for original, current in zip(originals, (source, outer, left, right)):
            np.testing.assert_array_equal(original, current)

    def test_direct_dispatch_matches_general_builder(self):
        for count, left_nodes, right_nodes in ((5, 2, 2), (6, 3, 2), (5, 3, 3)):
            with self.subTest(count=count, left=left_nodes, right=right_nodes):
                source, outer, left, right = _fixture(count, 10, 4, left_nodes, right_nodes)
                unit = build_unit_patch((source, right, outer[::-1], left[::-1]), 1e-10)
                boundary_count = sum(len(side) - 1 for side in unit.sides)
                boundary = unit.nodes[:boundary_count]
                origin, scale = boundary[0], np.ptp(boundary[:, :2], axis=0).max()
                general = Mesh2D((boundary - origin) / scale, np.empty((0, 4), dtype=int))
                build_unit_1_x_y_z(general, *unit.sides)
                np.testing.assert_array_equal(unit.quads, general.elements)
                np.testing.assert_allclose(unit.nodes, origin + scale * general.nodes, atol=1e-13)

    def test_exact_normalized_cache_restores_translated_units_without_rounding(self):
        source, outer, left, right = _fixture(3, 3, 1)
        sides = (source, right, outer[::-1], left[::-1])
        cache = {}
        original = build_unit_patch(sides, 1e-10, _cache=cache)
        translated = tuple(side + [4, 2, 0] for side in sides)
        cached = build_unit_patch(translated, 1e-10, _cache=cache)
        direct = build_unit_patch(translated, 1e-10)
        self.assertEqual(len(cache), 1)
        np.testing.assert_array_equal(cached.nodes, direct.nodes)
        np.testing.assert_array_equal(cached.quads, direct.quads)
        self.assertEqual(cached.quality, direct.quality)
        self.assertEqual(cached.method, original.method)
        changed = tuple(side.copy() for side in translated)
        changed[0][1, 0] += 1e-12
        build_unit_patch(changed, 1e-10, _cache=cache)
        self.assertEqual(len(cache), 2, "nearby inputs must not become byte-exact cache hits")

    def test_filtered_boundary_checks_match_exhaustive_pairs(self):
        random = np.random.default_rng(481)
        for scale in (1e-5, 1.0, 1e8):
            for shuffled in (False, True):
                for _ in range(8):
                    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                    radius = random.uniform(.8, 1.2, size=12)
                    xy = scale * np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
                    if shuffled:
                        xy = xy[random.permutation(len(xy))]
                    extent = np.ptp(xy, axis=0).max()
                    length_tol = 64 * np.finfo(float).eps * extent
                    area_tol = length_tol * extent
                    expected_crossing = any(
                        _segments_intersect(xy[i], xy[(i + 1) % 12],
                                            xy[j], xy[(j + 1) % 12],
                                            length_tol, area_tol)
                        for i in range(12) for j in range(i + 1, 12)
                        if j != i + 1 and (i, j) != (0, 11)
                    )
                    sides = (np.arange(4), np.arange(3, 7), np.arange(6, 10),
                             np.array([9, 10, 11, 0]))
                    nodes = np.column_stack((xy, np.zeros(12)))
                    if expected_crossing:
                        with self.assertRaises(ValueError):
                            _validate_cyclic_boundary(nodes, sides, require_convex=False)
                    else:
                        _validate_cyclic_boundary(nodes, sides, require_convex=False)

    def test_small_strips_are_valid_without_quality_search(self):
        for n, left_count, right_count in ((7, 2, 2), (8, 2, 3), (7, 3, 3)):
            with self.subTest(n=n, left=left_count, right=right_count):
                with patch.object(strip, "build_unit_patch", wraps=build_unit_patch) as calls:
                    selected = plan_line(*_fixture(n, n, 2, left_count, right_count),
                                         n / 3, 1e-10, best_only=True)
                self.assertTrue(selected)
                self.assertEqual(calls.call_count, selected[0].diagnostics["unit_trial_count"])
                self.assert_conforming(selected[0])

    def test_geometry_failure_repairs_only_the_shared_cut_with_its_neighbor(self):
        source, _, left, right = _fixture(13, 13., 2.)
        outer = np.array([[0., 2., 0.], [4., 2., 0.], [8., 2., 0.], [13., 2., 0.]])
        inputs = (source, outer, left, right)
        originals = [a.copy() for a in inputs]
        for blocked, preserved in (((0., 3.), (8, 2)), ((8., 13.), (3, 1))):
            with self.subTest(blocked=blocked):
                cuts = [(0, 0), (3, 1), (8, 2), (13, 3)]
                def reject_failed_span(sides, *args, **kwargs):
                    if tuple(sides[0][[0, -1], 0]) == blocked:
                        raise ValueError("invalid span")
                    return build_unit_patch(sides, *args, **kwargs)

                stats = {}
                with patch.object(strip, "build_unit_patch", side_effect=reject_failed_span):
                    result = strip.build_strip(*inputs, cuts, 0., 1e-10,
                                               cache={}, stats=stats, cap=7)
                self.assertIsNotNone(result)
                self.assertIn(list(preserved), result.diagnostics["internal_cuts"])
                self.assertGreaterEqual(stats["local_repair_count"], 1)
                self.assertTrue(all(u["boundary_counts"][1] == 2 for u in result.units[:-1]))
                self.assertTrue(all(max(u["source_segments_per_outer"]) <= 7 for u in result.units))
                self.assertEqual(cuts, [(0, 0), (3, 1), (8, 2), (13, 3)])
                for value, original in zip(inputs, originals):
                    np.testing.assert_array_equal(value, original)
                self.assert_conforming(result)

    def test_strict_upper_size_can_add_outer_points_without_slant(self):
        source, outer, left, right = _fixture(20, 20., .5)
        source[:, 0] = np.r_[0., np.cumsum([1.5] * 10 + [.5] * 10)]
        patch, = plan_line(source, outer, left, right, 1., 1e-10,
                           min_size_ratio=.5, max_size_ratio=1., best_only=True)
        xs = patch.nodes[patch.outer_ids, 0]
        self.assertGreater(len(xs), len(source))
        self.assertTrue(np.all((np.diff(xs) >= .5 - 1e-10) & (np.diff(xs) <= 1. + 1e-10)))
        self.assertAlmostEqual(patch.diagnostics["max_alignment_angle"], 0.)
        for unit in patch.units[:-1]:
            count, cut_count, _, _ = unit["boundary_counts"]
            self.assertEqual(cut_count, 2)
            cut = np.asarray(unit["polygon"][count - 1:count + 1])[:, :2]
            np.testing.assert_allclose(cut[1] - cut[0], [0., .5], atol=1e-10)
        self.assert_conforming(patch)

    def test_strict_size_band_excludes_oversized_parity_neighbor(self):
        chosen = plan_line(*_fixture(7, 3, 2), 2, 1e-10, max_size_ratio=1, best_only=True)
        self.assertEqual(len(chosen[0].outer_ids) - 1, 3)
        self.assert_conforming(chosen[0])

    def test_nonuniform_source_matches_geometry_and_keeps_all_nodes(self):
        source, outer, left, right = _fixture(8, 12, 4, 2, 3)
        source[:, 0] = [0, .2, .4, 1, 3, 7, 9, 11, 12]
        chosen = plan_line(source, outer, left, right, 4, 1e-10, best_only=True)[0]
        ends = [unit["polygon"][unit["boundary_counts"][0] - 1][0]
                for unit in chosen.units[:-1]]
        self.assertTrue(set(ends).issubset(set(source[:, 0])))
        np.testing.assert_array_equal(chosen.nodes[chosen.source_ids], source)
        self.assertGreaterEqual(chosen.diagnostics["unit_trial_count"]
                                    + chosen.diagnostics["direct_quad_count"], len(chosen.units))
        self.assert_conforming(chosen)

    def test_equal_counts_and_nonuniform_refinement_keep_source(self):
        for n, h in ((6, 2), (3, 1)):
            with self.subTest(n=n):
                source, outer, left, right = _fixture(n, 12, 4)
                if n == 3:
                    source[:, 0] = [0, 1, 8, 12]
                chosen = plan_line(source, outer, left, right, h, 1e-10, best_only=True)[0]
                self.assertGreaterEqual(chosen.diagnostics["unit_trial_count"]
                                    + chosen.diagnostics["direct_quad_count"], len(chosen.units))
                np.testing.assert_array_equal(chosen.nodes[chosen.source_ids], source)
                self.assert_conforming(chosen)

    def test_repeated_outer_contacts_are_deduplicated_without_moving_anchors(self):
        source, _, left, right = _fixture(8, 16, 4)
        outer = np.array([[0., 4, 0], [3, 4, 0], [3, 4, 0], [16, 4, 0]])
        chosen = plan_line(source, outer, left, right, 8, 1e-10, best_only=True)[0]
        self.assertEqual(np.count_nonzero(chosen.nodes[chosen.outer_ids, 0] == 3), 1)
        self.assert_conforming(chosen)

if __name__ == "__main__":
    unittest.main()
