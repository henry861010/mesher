"""LINE sections plan shared seams and preserve the whole source chain."""

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter.layer._line import (
    outer_count_candidates,
    plan_line,
    sample_outer,
)
from mesher.mesh2d.adapter.layer._unit_plan import build_unit_patch, quality_score
from mesher import Mesh2D
from mesher.mesh2d.adapter.unit import build_unit_1_x_y_z
from mesher.mesh2d.adapter.unit.build_unit_1_1_1_x import (
    _segments_intersect,
    _validate_cyclic_boundary,
)
from mesher.mesh2d.adapter.unit.build_unit_1_1_x_y import _validate_patch
import mesher.mesh2d.adapter.layer._line as line_module


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

    def test_full_source_chain_is_one_section_with_two_units(self):
        args = _fixture()
        originals = tuple(array.copy() for array in args)
        candidates = plan_line(*args, 50.0, 1e-9)
        self.assertEqual(len(candidates), 1)
        patch = candidates[0]
        self.assertEqual(len(patch.units), 2)
        self.assertEqual(len(patch.outer_ids), 3)
        np.testing.assert_array_equal(patch.nodes[patch.source_ids], args[0])
        self.assertEqual(patch.size_error, 0)
        # Large coarsening remains valid without a quality rejection threshold.
        self.assertGreater(patch.quality, 4)
        self.assert_conforming(patch)
        for original, current in zip(originals, args):
            np.testing.assert_array_equal(original, current)

    def test_all_ordinary_seam_pairs_close_and_share_midpoint_ids(self):
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

    def test_refinement_uses_one_source_segment_per_unit(self):
        patches = plan_line(*_fixture(3, 12, 4), 1.7, 1e-10)
        self.assertTrue(patches)
        for patch in patches:
            self.assertGreater(len(patch.outer_ids) - 1, 3)
            self.assertEqual(len(patch.units), 3)
            self.assertTrue(all(unit["boundary_counts"][0] == 2 for unit in patch.units))
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

    def test_count_candidates_are_nearest_legal_neighbors(self):
        self.assertEqual(outer_count_candidates(20, 2, 2, 100, 50, 1), [2])
        self.assertEqual(set(outer_count_candidates(19, 2, 2, 100, 50, 1)), {1, 3})
        self.assertEqual(outer_count_candidates(20, 2, 2, 100, 50, 5), [6])
        self.assertEqual(outer_count_candidates(1, 2, 2, 1, 50, 1), [1])

    def test_anchor_allocation_retains_endpoints_exactly(self):
        anchors = np.array([[0., 0, 0], [1, 0, 0], [5, 0, 0]])
        sampled, error = sample_outer(anchors, 5, 1, 1e-10)
        np.testing.assert_array_equal(sampled[:, 0], np.arange(6))
        self.assertEqual(error, 0)

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
        translated = tuple(side + [4, 2, 7] for side in sides)
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

    def test_best_only_skips_dominated_size_count_after_success(self):
        args = _fixture(8, 16, 4, 2, 3)
        expected = plan_line(*args, 8, 1e-10)[0]
        with patch.object(line_module, "_plan_count", wraps=line_module._plan_count) as calls:
            chosen = plan_line(*args, 8, 1e-10, best_only=True)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].size_error, expected.size_error)
        self.assertEqual(chosen[0].quality, expected.quality)

    def test_best_only_still_tries_other_count_after_geometry_failure(self):
        args = _fixture(8, 16, 4, 2, 3)
        original = line_module._plan_count
        attempts = []

        def fail_first(*values):
            attempts.append(len(values[1]))
            return None if len(attempts) == 1 else original(*values)

        with patch.object(line_module, "_plan_count", side_effect=fail_first):
            chosen = plan_line(*args, 8, 1e-10, best_only=True)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(chosen)

    def test_long_line_builds_each_unit_once_without_changing_section(self):
        args = _fixture(40, 40, 10)
        with patch.object(line_module, "build_unit_patch", wraps=build_unit_patch) as trials:
            candidates = plan_line(*args, 10, 1e-10, best_only=True)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].units), 4)
        self.assertEqual(len(candidates[0].source_ids), 41)
        self.assertAlmostEqual(candidates[0].quality, 10)
        self.assertEqual(trials.call_count, 4)
        self.assertEqual(candidates[0].diagnostics, {
            "line_planner": "outer_first", "unit_trial_count": 4,
            "local_repair_count": 0, "outer_count_attempts": 1,
        })
        self.assert_conforming(candidates[0])

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
                with patch.object(line_module, "build_unit_patch", wraps=build_unit_patch) as calls:
                    selected = plan_line(*_fixture(n, n, 2, left_count, right_count),
                                         n / 3, 1e-10, best_only=True)
                self.assertTrue(selected)
                self.assertEqual(calls.call_count, len(selected[0].units))
                self.assert_conforming(selected[0])

    def test_equal_size_error_prefers_fewer_segments_and_stops(self):
        with patch.object(line_module, "_plan_count", wraps=line_module._plan_count) as calls:
            chosen = plan_line(*_fixture(7, 3, 2), 2, 1e-10, best_only=True)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(len(chosen[0].outer_ids) - 1, 1)
        self.assertAlmostEqual(chosen[0].size_error, .25)
        self.assert_conforming(chosen[0])

    def test_nonuniform_source_matches_geometry_and_keeps_all_nodes(self):
        source, outer, left, right = _fixture(8, 12, 4, 2, 3)
        source[:, 0] = [0, .2, .4, 1, 3, 7, 9, 11, 12]
        chosen = plan_line(source, outer, left, right, 4, 1e-10, best_only=True)[0]
        ends = [unit["polygon"][unit["boundary_counts"][0] - 1][0]
                for unit in chosen.units[:-1]]
        self.assertEqual(ends, [3, 7])
        np.testing.assert_array_equal(chosen.nodes[chosen.source_ids], source)
        self.assertEqual(chosen.diagnostics["unit_trial_count"], 3)
        self.assert_conforming(chosen)

    def test_equal_counts_and_nonuniform_refinement_build_each_unit_once(self):
        for n, h in ((6, 2), (3, 1)):
            with self.subTest(n=n):
                source, outer, left, right = _fixture(n, 12, 4)
                if n == 3:
                    source[:, 0] = [0, 1, 8, 12]
                chosen = plan_line(source, outer, left, right, h, 1e-10, best_only=True)[0]
                self.assertEqual(chosen.diagnostics["unit_trial_count"], n)
                self.assertEqual(chosen.diagnostics["local_repair_count"], 0)
                np.testing.assert_array_equal(chosen.nodes[chosen.source_ids], source)
                self.assert_conforming(chosen)

    def test_repeated_outer_contacts_are_deduplicated_without_moving_anchors(self):
        source, _, left, right = _fixture(8, 16, 4)
        outer = np.array([[0., 4, 0], [3, 4, 0], [3, 4, 0], [16, 4, 0]])
        chosen = plan_line(source, outer, left, right, 8, 1e-10, best_only=True)[0]
        self.assertEqual(np.count_nonzero(chosen.nodes[chosen.outer_ids, 0] == 3), 1)
        self.assert_conforming(chosen)

    def test_failed_nearest_cut_tries_a_neighbor_then_stops(self):
        for blocked, expected in (
            ({(0, 4)}, [(0, 4), (0, 3), (3, 8)]),
            ({(0, 4), (0, 3)}, [(0, 4), (0, 3), (0, 5), (5, 8)]),
        ):
            with self.subTest(blocked=blocked):
                attempted = []

                def reject_first_cut(sides, *args, **kwargs):
                    interval = (sides[0][0, 0], sides[0][-1, 0])
                    attempted.append(interval)
                    if interval in blocked:
                        raise ValueError("injected invalid cut")
                    return build_unit_patch(sides, *args, **kwargs)

                with patch.object(line_module, "build_unit_patch", side_effect=reject_first_cut):
                    chosen = plan_line(*_fixture(8, 8, 4), 4, 1e-10, best_only=True)[0]
                self.assertEqual(attempted, expected)
                self.assertEqual(chosen.diagnostics["local_repair_count"], len(expected) - 2)
                self.assertEqual(chosen.diagnostics["unit_trial_count"], len(expected))
                self.assert_conforming(chosen)

    def test_closing_repair_rebuilds_only_the_last_cut_owners(self):
        for n, h, blocked, expected in (
            (8, 4, (4, 8), [(0, 4), (4, 8), (0, 3), (3, 8)]),
            (9, 3, (3, 6), [(0, 3), (6, 9), (3, 6), (5, 9), (3, 5)]),
        ):
            with self.subTest(n=n):
                attempted = []

                def reject_closure(sides, *args, **kwargs):
                    interval = (sides[0][0, 0], sides[0][-1, 0])
                    attempted.append(interval)
                    if interval == blocked:
                        raise ValueError("injected invalid closing unit")
                    return build_unit_patch(sides, *args, **kwargs)

                with patch.object(line_module, "build_unit_patch", side_effect=reject_closure):
                    chosen = plan_line(*_fixture(n, n, 4), h, 1e-10, best_only=True)[0]
                self.assertEqual(attempted, expected)
                self.assertEqual(chosen.diagnostics["local_repair_count"], 1)
                self.assertEqual(chosen.diagnostics["unit_trial_count"], len(expected))
                self.assert_conforming(chosen)

    def test_closing_repair_also_preserves_outer_anchors_when_refining(self):
        source, _, left, right = _fixture(2, 8, 4)
        outer = np.array([[0., 4, 0], [3., 4, 0], [8., 4, 0]])
        attempted = []

        def reject_closure(sides, *args, **kwargs):
            forward = sides[2][::-1]
            interval = (forward[0, 0], forward[-1, 0])
            attempted.append(interval)
            if interval == (4, 8):
                raise ValueError("invalid refining closure")
            return build_unit_patch(sides, *args, **kwargs)

        with patch.object(line_module, "build_unit_patch", side_effect=reject_closure):
            chosen = plan_line(source, outer, left, right, 1, 1e-10, best_only=True)[0]
        self.assertEqual(attempted, [(0, 4), (4, 8), (0, 3), (3, 8)])
        self.assertEqual(chosen.diagnostics["local_repair_count"], 1)
        for point in outer:
            self.assertEqual(np.count_nonzero(np.all(chosen.nodes[chosen.outer_ids] == point, axis=1)), 1)
        self.assert_conforming(chosen)

    def test_exhausted_repairs_are_bounded_and_leave_inputs_unchanged(self):
        args = _fixture(40, 40, 10)
        originals = [array.copy() for array in args]
        stats = {}
        with patch.object(line_module, "build_unit_patch", side_effect=ValueError("invalid")) as calls:
            chosen = plan_line(*args, 10, 1e-10, best_only=True, _diagnostics=stats)
        self.assertFalse(chosen)
        self.assertEqual(calls.call_count, 3)
        self.assertEqual(stats["unit_trial_count"], 3)
        self.assertEqual(stats["local_repair_count"], 2)
        for before, after in zip(originals, args):
            np.testing.assert_array_equal(before, after)

    def test_failed_closure_never_backtracks_past_one_neighbor(self):
        attempted = []

        def reject_center(sides, *args, **kwargs):
            interval = (sides[0][0, 0], sides[0][-1, 0])
            attempted.append(interval)
            if 5 <= interval[0] <= 7 and interval[1] == 9:
                raise ValueError("unmeshable center")
            return build_unit_patch(sides, *args, **kwargs)

        source, outer, left, right = _fixture(12, 12, 4)
        stats = {"unit_trial_count": 0, "local_repair_count": 0}
        with patch.object(line_module, "build_unit_patch", side_effect=reject_center):
            chosen = line_module._plan_count(source, np.linspace(outer[0], outer[-1], 5),
                                             left, right, 0., 1e-10, None, stats)
        self.assertIsNone(chosen)
        self.assertEqual(attempted, [(0, 3), (9, 12), (3, 6), (6, 9),
                                     (3, 5), (5, 9), (3, 7), (7, 9)])
        self.assertLessEqual(stats["unit_trial_count"], 8 * 4)
        self.assertEqual(stats["local_repair_count"], 2)

    def test_unit_work_scales_with_the_shorter_chain(self):
        for segments in (32, 128, 512):
            with self.subTest(segments=segments):
                chosen = plan_line(*_fixture(segments, segments, 8),
                                   8, 1e-10, best_only=True)[0]
                self.assertEqual(chosen.diagnostics["unit_trial_count"], segments // 8)
                self.assertEqual(chosen.diagnostics["local_repair_count"], 0)
                self.assert_conforming(chosen)


if __name__ == "__main__":
    unittest.main()
