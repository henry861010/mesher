"""Behavioral acceptance tests for the approved line-imprint examples."""
import copy
import unittest

import numpy as np

from .._fixtures import assert_ccw
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines, ImprintLineError
from mesher.mesh2d.imprint_line.mesher import _prepare_plan, _assemble
from mesher.mesh2d.imprint_line._unit import build_unit


def pair(axis, bounds, coordinates=None, expand=2, count=6):
    x0, y0, x1, y1 = bounds
    values = np.linspace(y0 if axis == 0 else x0, y1 if axis == 0 else x1, count) if coordinates is None else coordinates
    e1, e2 = np.broadcast_to(expand, (2,)).tolist()
    return dict(direction="xy"[axis], line1_expand_size=e1, line2_expand_size=e2,
                line1=([[x0, y0], [x0, y1]] if axis == 0 else [[x0, y0], [x1, y0]]),
                line2=([[x1, y0], [x1, y1]] if axis == 0 else [[x0, y1], [x1, y1]]),
                points=([[x0, v] for v in values] if axis == 0 else [[v, y0] for v in values]))


class ImprintTests(unittest.TestCase):
    def run_case(self, pairs, patterns=(), outer=((-2, -2), (12, -2), (12, 12), (-2, 12)),
                 size=2, minimum=.5, **kwargs):
        args = _prepare_plan(size, pairs, patterns, outer_outline=outer,
                             min_transition_size=minimum, **kwargs)
        mesh = _assemble(*args)
        assert_ccw(mesh)
        plan = args[1]
        self.assertTrue(np.all(mesh.nodes[:, 2] == 0))
        exterior, = EditorMesh2D.outlines(mesh)
        self.assertEqual(len(exterior) % 2, 0)
        boundary_edges = {frozenset(edge) for edge in zip(exterior, np.roll(exterior, -1))}
        triangles = mesh.elements[mesh.element_types == 3, :3]
        for triangle in triangles:
            self.assertFalse(any(frozenset(edge) in boundary_edges
                                 for edge in zip(triangle, np.roll(triangle, -1))))
        vertices = mesh.nodes[mesh.elements, :2]
        self.assertFalse(np.any((vertices.min(axis=1) < 0) & (vertices.max(axis=1) > 0)))
        for transition in plan.transitions:
            a = transition.axis
            b = transition.bounds
            self.assertGreaterEqual(b[a + 2] - b[a] + 1e-9 * minimum, minimum)
            self.assertLessEqual(b[a + 2] - b[a], size * (1 + 1e-9))
            for source_index in plan.bands[transition.band].sources:
                source = pairs[source_index]
                ends = [source["line1"][0][a], source["line2"][0][a]]
                distances = [source["line1_expand_size"], source["line2_expand_size"]]
                index = int(np.argmin(ends) if transition.side == 0 else np.argmax(ends))
                actual = (ends[index] - transition.inner if transition.side == 0 else transition.inner - ends[index])
                self.assertGreaterEqual(actual + 1e-9 * max(1, abs(ends[index])), distances[index])
            self.assertTrue(transition.cells)
            for cell in transition.cells:
                self.assertEqual((cell[a], cell[a + 2]), (b[a], b[a + 2]))
                self.assertGreater(min(cell[3 - a], b[3 - a]), max(cell[1 - a], b[1 - a]))
        for source in pairs:
            a = 0 if source["direction"] == "x" else 1
            t = 1 - a
            ends = sorted([source["line1"][0][a], source["line2"][0][a]])
            for point in source["points"]:
                self.assert_segment(mesh, a, point[t], *ends)
        return mesh, plan

    def assert_segment(self, mesh, axis, coordinate, start, end):
        e = mesh.elements
        edges = np.stack((e, np.roll(e, -1, axis=1)), axis=-1).reshape(-1, 2)
        pts = mesh.nodes[edges, :2]
        on = np.all(pts[:, :, 1 - axis] == coordinate, axis=1)
        spans = np.sort(pts[on, :, axis], axis=1)
        spans = spans[spans[:, 1] > spans[:, 0]]
        cursor = start
        for a, b in sorted(set(map(tuple, spans))):
            if b <= cursor or a >= end:
                continue
            self.assertLessEqual(a, cursor)
            cursor = max(cursor, b)
        self.assertGreaterEqual(cursor, end)

    def test_single_focus_matches_preview_topology(self):
        mesh, plan = self.run_case([pair(0, (4, 4, 6, 6)), pair(1, (4, 4, 6, 6))])
        self.assertEqual(plan.diagnostics["transition_count"], 4)
        self.assertEqual(mesh.element_count, 113)
        self.assertEqual(np.count_nonzero(mesh.element_types == 3), 0)

    def test_shifted_focus_ends_can_claim_the_same_cell(self):
        pairs = [pair(a, b) for b in ((4, 5, 6, 7), (12, 4, 14, 6)) for a in (0, 1)]
        _, plan = self.run_case(pairs, size=2, outer=((-4, -4), (20, -4), (20, 16), (-4, 16)))
        horizontal = [t for t in plan.transitions if t.axis == 0]
        right = next(t for t in horizontal if t.side == 1 and t.inner == 8)
        left = next(t for t in horizontal if t.side == 0 and t.inner == 10)
        self.assertTrue(set(right.cells) & set(left.cells))
        self.assertEqual(len(plan.transitions), 8)

    def test_fillet_nonuniform_5_to_20_intervals(self):
        ys = np.r_[0, np.cumsum([5, 10, 15, 20] * 10)]
        mesh, plan = self.run_case([pair(0, (3000, 0, 7000, 500), ys, 2000)], size=1000, minimum=200,
                                   outer=((-1000, -1000), (11000, -1000), (11000, 1500), (-1000, 1500)))
        self.assertEqual(plan.diagnostics["transition_count"], 2)
        self.assertTrue(all(len(t.cells) >= 1 for t in plan.transitions))
        self.assertEqual({t.inner for t in plan.transitions}, {1000, 9000})
        # 40 intervals remain straight across the original length.
        self.assert_segment(mesh, 0, 5, 3000, 7000)

    def test_finite_patterns_and_dense_lines_survive_outward_alignment(self):
        patterns = [[[5.5, 4.1], [8.1, 4.1]], [[10.1, 0], [10.1, 10]]]
        mesh, plan = self.run_case([pair(0, (3, 4, 5, 6), expand=1),
                                    pair(1, (6.2, 1, 6.5, 9), [6.2, 6.3, 6.4, 6.5], .5)],
                                   patterns, outer=((-2, 0), (12.1, 0), (12.1, 10), (-2, 10)))
        right = next(t for t in plan.transitions if t.axis == 0 and t.side == 1)
        self.assertEqual(right.inner, 6.8)
        self.assert_segment(mesh, 0, 4., 3., right.inner)
        self.assertFalse(any(t.axis == 1 for t in plan.transitions))
        for x in (6.2, 6.3, 6.4, 6.5):
            self.assert_segment(mesh, 1, x, 0, 10)
        self.assert_segment(mesh, 0, 4.1, 5.5, 8.1)
        self.assert_segment(mesh, 1, 10.1, 0, 10)

    def test_thin_pair_rebuilds_wider_cells_in_both_directions(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                bounds = (3, 4, 5, 4.2) if axis == 0 else (4, 3, 4.2, 5)
                _, plan = self.run_case([pair(axis, bounds, count=4)], minimum=.5)
                self.assertEqual(len(plan.transitions), 2)
                self.assertEqual({t.inner for t in plan.transitions}, {0, 8})
                self.assertEqual((plan.bands[0].low, plan.bands[0].high), (4, 4.2))
                for end in plan.transitions:
                    self.assertTrue(any(b[1-axis] < 4 or b[3-axis] > 4.2 for b in end.cells))

    def test_short_box_sides_transition_after_requested_expansion(self):
        for box in ([0, 0, 50, 2000], [0, 0, 2000, 50]):
            with self.subTest(box=box):
                x1, y1, x2, y2 = box
                pairs = get_pair_line([
                    dict(box=box,
                         x_points=[[x1, y] for y in np.arange(y1, y2 + 1, 10)],
                         y_points=[[x, y1] for x in np.arange(x1, x2 + 1, 10)]),
                ], expand_size=1000)
                _, plan = self.run_case(pairs, size=1000, minimum=100,
                                       outer=((-5000, -5000), (7000, -5000),
                                              (7000, 7000), (-5000, 7000)))
                self.assertEqual(len(plan.transitions), 4)
                self.assertFalse(plan.diagnostics['boundary_ends'])
                for t in plan.transitions:
                    expected = -1000 if t.side == 0 else 1000 * (1 + np.ceil(box[t.axis + 2] / 1000))
                    self.assertEqual(t.inner, expected)

    def test_thin_pair_still_requires_minimum_expansion_depth(self):
        mesh, plan = self.run_case([pair(0, (3, 4, 5, 4.2), count=4)], minimum=.5,
                                   outer=((-2, -2), (7.25, -2), (7.25, 12), (-2, 12)))
        self.assertEqual([t.side for t in plan.transitions], [0])
        self.assertEqual(plan.diagnostics['boundary_ends'], [((0,), 1)])
        for coordinate in plan.bands[0].coordinates:
            self.assert_segment(mesh, 0, coordinate, 5, 7.25)

    def test_no_room_extends_to_domain_without_transition(self):
        mesh, plan = self.run_case([pair(0, (3, 4, 5, 6), expand=100)])
        self.assertFalse(plan.transitions)
        self.assert_segment(mesh, 0, 4.4, -2, 12)

    def test_transition_may_end_on_outer_boundary(self):
        _, plan = self.run_case([pair(0, (4, 4, 6, 6))], outer=((0, 0), (10, 0), (10, 10), (0, 10)))
        self.assertEqual(len(plan.transitions), 2)
        self.assertEqual({t.bounds[t.axis + (0 if t.side == 0 else 2)] for t in plan.transitions}, {0, 10})

    def test_distinct_endpoint_distances_survive_reversed_lines(self):
        for axis in (0, 1):
            for reversed_lines in (False, True):
                with self.subTest(axis=axis, reversed_lines=reversed_lines):
                    p = pair(axis, (4, 4, 6, 6), expand=(1, 3))
                    if reversed_lines:
                        p["line1"], p["line2"] = p["line2"], p["line1"]
                    _, plan = self.run_case([p])
                    self.assertEqual({t.inner for t in plan.transitions},
                                     {0, 8} if reversed_lines else {2, 10})

    def test_endpoint_distance_fields_are_required(self):
        for name in ("line1_expand_size", "line2_expand_size"):
            p = pair(0, (4, 4, 6, 6))
            del p[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.run_case([p])
        for legacy in (2, (1, 3)):
            p = pair(0, (4, 4, 6, 6))
            del p["line1_expand_size"], p["line2_expand_size"]
            p["expand_size"] = legacy
            with self.subTest(legacy=legacy), self.assertRaisesRegex(ValueError, "missing"):
                self.run_case([p])

    def test_endpoint_distances_reject_invalid_values_independently(self):
        for name in ("line1_expand_size", "line2_expand_size"):
            for invalid in (-1, np.nan, np.inf, -np.inf, True, np.array(True),
                            [1], (1, 2), np.array([1.]), None):
                p = pair(0, (4, 4, 6, 6))
                p[name] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaisesRegex(ValueError, name):
                    self.run_case([p])

    def test_zero_endpoint_distances_are_valid(self):
        _, plan = self.run_case([pair(0, (4, 4, 6, 6), expand=(0, 0))])
        self.assertEqual({t.inner for t in plan.transitions}, {4, 6})

    def test_large_transverse_span_splits_into_single_grid_cells(self):
        _, plan = self.run_case([pair(0, (4, 2, 6, 8), count=20)])
        for t in plan.transitions:
            self.assertGreaterEqual(len(t.cells), 3)
            self.assertTrue(all(b[3] - b[1] <= 2 for b in t.cells))

    def test_parallel_pattern_can_become_unit_divider(self):
        mesh, plan = self.run_case([pair(0, (4, 4, 6, 6))], [[[0, 5], [10, 5]]])
        for t in plan.transitions:
            self.assertTrue(all(b[3] <= 5 or b[1] >= 5 for b in t.cells))
        self.assert_segment(mesh, 0, 5, 0, 10)

    def test_parallel_pattern_can_split_off_width_below_minimum(self):
        mesh, plan = self.run_case([pair(0, (4, 4, 6, 6))], [[[0, 4.1], [10, 4.1]]])
        self.assertEqual({t.inner for t in plan.transitions}, {2, 8})
        for t in plan.transitions:
            self.assertTrue(all(b[3] <= 4.1 or b[1] >= 4.1 for b in t.cells))
        self.assert_segment(mesh, 0, 4.1, 0, 10)

    def test_perpendicular_pattern_can_bound_a_smaller_valid_grid_cell(self):
        mesh, plan = self.run_case([pair(0, (4, 4, 6, 6))], [[[9, 3], [9, 7]]])
        self.assert_segment(mesh, 1, 9, 3, 7)
        right = next(t for t in plan.transitions if t.side == 1)
        self.assertEqual(right.bounds, (9., 4., 10.5, 6.))

    def test_repeated_perpendicular_patterns_do_not_force_boundary_fallback(self):
        patterns = [[[x, 3], [x, 7]] for x in (8.5, 9.5, 10.5, 11.5)]
        _, plan = self.run_case([pair(0, (4, 4, 6, 6))], patterns)
        right = next(t for t in plan.transitions if t.side == 1)
        self.assertEqual(right.bounds, (8.5, 4., 9.5, 6.))

    def test_nearest_mesh_edge_wins_regardless_of_pattern_origin(self):
        control = pair(0, (900, 200, 1000, 400), expand=1000)
        outer = ((200, 0), (3200, 0), (3200, 600), (200, 600))
        for pattern_x in ((2450,), (2200, 2450)):
            with self.subTest(pattern_x=pattern_x):
                patterns = [[[x, 0], [x, 600]] for x in pattern_x]
                mesh, plan = self.run_case([control], patterns, outer, size=250, minimum=50)
                right, = plan.transitions
                self.assertEqual(right.bounds, (2200, 200, 2450, 400))
                self.assert_segment(mesh, 0, 240, 1000, 2200)
                self.assertFalse(np.any(mesh.nodes[:, 0] == 2000))
                self.assert_segment(mesh, 1, 2450, 0, 600)

    def test_shallow_outward_layers_are_skipped_whole(self):
        control = pair(0, (4, 4, 6, 6), expand=2)
        patterns = [[[x, 0], [x, 12]] for x in (8, 8.2, 8.4, 10)]
        mesh, plan = self.run_case([control], patterns, minimum=.5)
        right = next(t for t in plan.transitions if t.side == 1)
        self.assertEqual(right.bounds, (8.4, 4, 10, 6))
        self.assert_segment(mesh, 0, 4.4, 6, 8.4)

    def test_edge_in_a_disconnected_arm_is_not_a_candidate(self):
        outer = [[0, 0], [18, 0], [18, 4], [3, 4], [3, 8], [18, 8], [18, 12], [0, 12]]
        control = pair(0, (5, 1, 7, 3), expand=3)
        mesh, plan = self.run_case([control], [[[10, 9], [10, 11]]], outer, size=4)
        right, = plan.transitions
        self.assertEqual(right.cells, ((10.5, 0, 14.25, 4),))
        self.assert_segment(mesh, 1, 10, 9, 11)

    def test_remote_pattern_endpoint_splits_narrow_width_without_moving_transition(self):
        mesh, plan = self.run_case([pair(0, (4, 4, 6, 6))], [[[900, 4.1], [900, 4.2]]],
                                   outer=((0, 0), (1000, 0), (1000, 10), (0, 10)))
        right = next(t for t in plan.transitions if t.side == 1)
        self.assertEqual(right.inner, 8)
        self.assertTrue(any(b[1] == 4.1 and b[3] == 4.2 for b in right.cells))
        self.assert_segment(mesh, 1, 900, 4.1, 4.2)

    def test_concave_exterior_and_outside_pattern_clipping(self):
        outer = [[0, 0], [12, 0], [12, 5], [8, 5], [8, 12], [0, 12]]
        mesh, _ = self.run_case([pair(0, (2, 2, 4, 4), expand=.5)],
                                [[[-10, 1], [20, 1]]], outer=outer)
        self.assert_segment(mesh, 0, 1, 0, 12)

    def test_zero_pairs_produce_rectangular_grid(self):
        mesh, plan = self.run_case([], [[[1, 0], [1, 10]]])
        self.assertFalse(plan.transitions)
        self.assertTrue(np.all(mesh.element_types == 4))
        self.assert_segment(mesh, 1, 1, 0, 10)

    def test_odd_exterior_repaired_on_the_contacted_coarse_edge(self):
        p = pair(0, (1, 1, 3, 3), expand=(10, .75), count=3)
        outer = ((-3, -3), (7, -3), (7, 7), (-3, 7))
        mesh, plan = self.run_case([p], outer=outer)
        exterior, = EditorMesh2D.outlines(mesh)
        self.assertEqual(len(exterior) % 2, 0)
        transition, = plan.transitions
        self.assert_segment(mesh, 0, 2, -3, transition.inner)
        self.assertTrue(all(np.any(mesh.nodes[mesh.nodes[:, 0] == -3, 1] == y) for y in (1, 2, 3)))

    def test_narrow_dense_side_keeps_a_local_transition(self):
        p = pair(0, (1, 1, 3, 1.003), expand=(10, .75), count=3)
        mesh, plan = self.run_case([p], outer=((-3, -3), (7, -3), (7, 7), (-3, 7)))
        end, = plan.transitions
        self.assert_segment(mesh, 0, p["points"][1][1], -3, end.inner)

    def test_exterior_walls_have_only_quad_owners(self):
        for low, high in ((0, 5), (-4, 2), (0, 2)):
            with self.subTest(low=low, high=high):
                p = pair(0, (4, 0, 6, 2), expand=.5, count=3)
                mesh, plan = self.run_case([p], outer=((0, low), (10, low), (10, high), (0, high)), size=3)
                self.assertTrue(plan.transitions)
                self.assert_segment(mesh, 0, 1, 3.5, 6.5)

    def test_touching_transitions_preserve_their_fronts_and_boundary_anchors(self):
        p = [pair(0, (1, 1, 3, 2), expand=(10, 1), count=3),
             pair(0, (1, 2, 3, 3), expand=(10, 2), count=3)]
        outer = ((-3, -3), (80, -3), (80, 7), (-3, 7))
        mesh, plan = self.run_case(p, [[[-3, 5.25], [-3, 9]]], outer=outer)
        self.assertEqual([t.inner for t in plan.transitions], [4, 6])
        self.assertTrue(np.any(np.all(mesh.nodes[:, :2] == (-3, 5.25), axis=1)))

    def test_axes_through_dense_lines_transitions_and_background(self):
        outer = ((-6, -6), (7, -6), (7, 7), (-6, 7))
        p = pair(0, (1, -1, 3, 1), expand=.5, count=4)
        patterns = [[[-4, -1.75], [4, -1.75]]]
        mesh, plan = self.run_case([p], patterns, outer=outer)
        self.assertTrue(plan.transitions)
        self.assertEqual(np.count_nonzero(np.all(mesh.nodes[:, :2] == 0, axis=1)), 1)
        for axis in (0, 1):
            self.assert_segment(mesh, axis, 0, -6, 7)
        self.assert_segment(mesh, 0, -1.75, -4, 4)
        axes = [[[-6, 0], [7, 0]], [[0, 7], [0, -6]]]
        explicit, _ = self.run_case([p], patterns + axes + axes, outer=outer)
        np.testing.assert_array_equal(mesh.nodes, explicit.nodes)
        np.testing.assert_array_equal(mesh.elements, explicit.elements)

    def test_clipped_axes_on_disjoint_spans_and_single_sided_domains(self):
        outer = np.array([[-6, -5], [6, -5], [6, -2], [-2, -2],
                          [-2, 2], [6, 2], [6, 5], [-6, 5]])
        for axis in (0, 1):
            mesh, _ = self.run_case([], outer=outer if axis == 1 else outer[:, ::-1])
            self.assert_segment(mesh, axis, 0, -5, -2)
            self.assert_segment(mesh, axis, 0, 2, 5)
            self.assert_segment(mesh, 1 - axis, 0, -6, -2)
            self.assertFalse(np.any(np.all(mesh.nodes[:, :2] == 0, axis=1)))
        mesh, _ = self.run_case([], outer=((.0003, 0), (5, 0), (5, 4), (.0003, 4)))
        self.assertEqual(mesh.nodes[:, 0].min(), .0003)
        self.assert_segment(mesh, 0, 0, .0003, 5)

    def test_input_order_and_reversal_are_deterministic(self):
        pairs = [pair(0, (4, 4, 6, 6)), pair(1, (4, 4, 6, 6))]
        first, _ = self.run_case(pairs)
        reverse = copy.deepcopy(pairs[::-1])
        for p in reverse:
            p["points"].reverse()
            p["line1"].reverse()
            p["line2"].reverse()
        second, _ = self.run_case(reverse)
        np.testing.assert_array_equal(first.nodes, second.nodes)
        np.testing.assert_array_equal(first.elements, second.elements)

    def test_inputs_unchanged_and_new_mesh_owned(self):
        pairs = [pair(0, (4, 4, 6, 6))]
        before = copy.deepcopy(pairs)
        kwargs = dict(outer_outline=[[-2, -2], [12, -2], [12, 12], [-2, 12]], min_transition_size=.5)
        a = imprint_lines(2, pairs, **kwargs)
        b = imprint_lines(2, pairs, **kwargs)
        self.assertEqual(pairs, before)
        a.nodes[0, 0] += 1
        self.assertNotEqual(a.nodes[0, 0], b.nodes[0, 0])

    def test_budget_includes_transition_elements(self):
        pairs = [pair(0, (4, 4, 6, 6)), pair(1, (4, 4, 6, 6))]
        self.run_case(pairs, max_elements=113)
        with self.assertRaisesRegex(ValueError, "max_elements"):
            self.run_case(pairs, max_elements=112)

    def test_invalid_inputs(self):
        for override in ({"direction": "z"}, {"line1_expand_size": -1}, {"line2_expand_size": [1, 2, 3]},
                         {"points": [[4, 7]]}, {"points": [[np.nan, 5]]},
                         {"line1": [[4, 4], [5, 6]]}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.run_case([{**pair(0, (4, 4, 6, 6)), **override}])
        with self.assertRaises(ImprintLineError):
            self.run_case([pair(0, (-3, 4, 5, 6))])
        with self.assertRaises(ValueError):
            self.run_case([], [[[0, 0], [2, 1]]])

    def test_hundred_dense_nodes_and_scaled_translated_geometry(self):
        for scale, origin in [(1, 0), (1000, 1e6), (.01, -100)]:
            p = pair(0, (4, 4, 6, 6), count=100)
            for field in ("line1", "line2", "points"):
                p[field] = (np.asarray(p[field]) * scale + origin).tolist()
            p["line1_expand_size"] *= scale
            p["line2_expand_size"] *= scale
            outer = np.asarray([[-2, -2], [12, -2], [12, 12], [-2, 12]]) * scale + origin
            mesh, plan = self.run_case([p], outer=outer, size=2 * scale, minimum=.5 * scale,
                                       tolerance=1e-5 * scale)
            self.assertEqual(plan.diagnostics["transition_count"], 2)
            self.assertGreater(np.count_nonzero(mesh.element_types == 4),
                               np.count_nonzero(mesh.element_types == 3))

    def test_seeded_crossings_and_roundtrip_support_regression(self):
        # Seed 29 exposed two computed copies of a support differing by one ULP.
        for seed in (0, 3, 29, 63, 99):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                pairs, patterns = [], []
                for i in range(8):
                    x, y = rng.uniform(4, 22, 2)
                    w, h = rng.uniform(.7, 3, 2)
                    pairs.append(pair(i % 2, (x, y, x + w, y + h), expand=float(rng.uniform(.5, 2))))
                for i in range(12):
                    x, y = rng.uniform(1, 28, 2)
                    length = float(rng.uniform(.5, 3))
                    patterns.append([[x, y], [x + length, y]] if i % 2 else [[x, y], [x, y + length]])
                mesh, _ = self.run_case(pairs, patterns, outer=((0, 0), (30, 0), (30, 30), (0, 30)),
                                        size=3, minimum=.3, tolerance=1e-6)
                for p in patterns:
                    axis = 0 if p[0][1] == p[1][1] else 1
                    self.assert_segment(mesh, axis, p[0][1 - axis], min(p[0][axis], p[1][axis]),
                                        min(30, max(p[0][axis], p[1][axis])))

    def test_huge_subdivision_request_fails_before_allocating_coordinates(self):
        with self.assertRaisesRegex(ValueError, "max_elements"):
            self.run_case([], outer=((0, 0), (1e12, 0), (1e12, 1), (0, 1)), max_elements=100)


class UnitTests(unittest.TestCase):
    def test_parity_and_convexity_in_all_four_directions(self):
        for n in (2, 3, 6, 7, 100, 101):
            for axis in (0, 1):
                for side in (0, 1):
                    with self.subTest(n=n, axis=axis, side=side):
                        xy, cells = build_unit((0, 0, 1000, 1000), axis, side, np.linspace(0, 1000, n), 1e-5)
                        self.assertEqual(np.count_nonzero(cells[:, 2] == cells[:, 3]), n % 2)
                        edges = np.stack((cells, np.roll(cells, -1, axis=1)), axis=-1).reshape(-1, 2)
                        edges = edges[edges[:, 0] != edges[:, 1]]
                        self.assertGreater(np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1).min(), .1)


if __name__ == "__main__":
    unittest.main()
