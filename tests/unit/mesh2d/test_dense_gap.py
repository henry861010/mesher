"""Adjacent dense runs protect only the ordinary LINE between their facing ends."""

import unittest

import numpy as np

from mesher.mesh2d.adapter import generate_full_adaptive_mesh, generate_layer_adaptive_mesh, plan_quad_sections
from tests.unit.mesh2d.test_dense_sections import rectangle
from tests.unit.mesh2d.test_extend_quad_layer import _target_ids


def two_runs(gap=20.2):
    return rectangle([0, 50, 100, 103, 103 + gap, 106 + gap, 200, 250], height=100.)


def top_mappings(mesh, diagnostics):
    return {mesh.nodes[r["source_node"], 0]: r for r in diagnostics["dense_mappings"]
            if mesh.nodes[r["source_node"], 1] == 100.}


class DenseGapTests(unittest.TestCase):
    def test_translating_normal_to_the_line_preserves_dense_topology(self):
        # Both source and threshold use exact binary values. Translating Y
        # leaves every X gap (including the below-threshold difference) exact.
        for case in ("short_edge", "gap_between_runs"):
            reference = None
            for offset in (0., 2.**40):
                with self.subTest(case=case, offset=offset):
                    if case == "short_edge":
                        mesh, outline = rectangle([0, 20, 25, 45], height=24.)
                        settings = dict(small_element_size=5. + 2.**-12,
                                        dense_mapping_angle=0., expand_size=4.)
                        sections, protected = 2, 0
                    else:
                        mesh, outline = two_runs(gap=40. - 2.**-12)
                        settings = dict(small_element_size=5., dense_mapping_angle=45.,
                                        expand_size=10.)
                        sections, protected = 4, 4
                    mesh.nodes[:, 1] += offset
                    plan = plan_quad_sections(mesh, outline, 10., **settings)
                    result = generate_layer_adaptive_mesh(mesh, outline, 10., **settings)
                    for info in (plan.diagnostics, result.diagnostics):
                        self.assertEqual(len(info["dense_sections"]), sections)
                        self.assertEqual(sum(r["gap_limited"] for r in info["dense_mappings"]),
                                         protected)
                    if reference is None:
                        reference = mesh.elements, result.outline
                    else:
                        np.testing.assert_array_equal(mesh.elements, reference[0])
                        self.assertEqual(result.outline, reference[1])

    def test_close_runs_preserve_gap_quad_quality_and_free_ends(self):
        qualities = []
        for settings, protected, width in [({"dense_min_gap_ratio": 0}, False, .2), ({}, True, 20.2)]:
            mesh, outline = two_runs()
            kwargs = dict(expand_size=10., small_element_size=5., dense_mapping_angle=45., **settings)
            plan = plan_quad_sections(mesh, outline, 20., **kwargs)
            result = generate_layer_adaptive_mesh(mesh, outline, 20., **kwargs)
            preview = top_mappings(mesh, plan.diagnostics)
            actual = top_mappings(mesh, result.diagnostics)
            for x, r in actual.items():
                self.assertEqual(r["gap_limited"], protected and x in (103., 123.2))
                self.assertEqual(r["gap_limited"], preview[x]["gap_limited"])
                np.testing.assert_allclose(r["target"], preview[x]["target"])
            self.assertAlmostEqual(actual[100.]["angle"], -45.)
            self.assertAlmostEqual(actual[126.2]["angle"], 45.)
            if protected:
                self.assertAlmostEqual(actual[103.]["angle"], 0.)
                self.assertAlmostEqual(actual[123.2]["angle"], 0.)
            middle, = [s for s in result.diagnostics["section_details"] if s["kind"] == "LINE"
                       and s["source_chains"][0][0] == actual[103.]["source_node"]]
            self.assertAlmostEqual(np.ptp(mesh.nodes[middle["outer_ids"], 0]), width)
            quads = mesh.elements[middle["element_start"]:middle["element_end"]]
            self.assertGreater(len(quads), 0)
            self.assertTrue(all(len(set(q)) == 4 for q in quads))
            qualities.append(middle["quality"])
        self.assertLess(qualities[1], qualities[0])

    def test_threshold_equality_scales_and_ratio_extremes(self):
        # A 10-unit layer at 45 degrees consumes 20 units of a free gap.
        for gap, ratio, limited in ((39.99, .5, True), (40., .5, False), (40.01, .5, False),
                                    (40., .75, True), (60., 1., True), (20., .01, True)):
            for scale in (1., 1000.):
                with self.subTest(gap=gap, ratio=ratio, scale=scale):
                    mesh, outline = two_runs(gap)
                    mesh.nodes[:, :2] *= scale
                    plan = plan_quad_sections(mesh, outline, 10. * scale, small_element_size=5. * scale,
                                              dense_mapping_angle=45., dense_min_gap_ratio=ratio,
                                              tolerance=.001 * scale)
                    records = plan.diagnostics["dense_mappings"]
                    for r in records:  # Top and bottom traverse the source in opposite directions.
                        x = mesh.nodes[r["source_node"], 0] / scale
                        facing = np.isclose(x, 103.) or np.isclose(x, 103. + gap)
                        self.assertEqual(r["gap_limited"], limited and facing)
                        self.assertAlmostEqual(abs(r["angle"]), 0. if limited and facing else 45.)

    def test_three_runs_only_lock_the_close_pair(self):
        mesh, outline = rectangle([0, 50, 100, 103, 123, 126, 186, 189, 240], height=100.)
        plan = plan_quad_sections(mesh, outline, 10., small_element_size=5., dense_mapping_angle=45.)
        records = top_mappings(mesh, plan.diagnostics)
        self.assertEqual({x for x, r in records.items() if r["gap_limited"]}, {103., 123.})
        np.testing.assert_allclose([records[x]["angle"] for x in sorted(records)], [-45, 0, 0, 45, -45, 45])

    def test_pattern_seam_delimits_the_neighbor_search(self):
        mesh, outline = rectangle([0, 50, 100, 103, 113.1, 123.2, 126.2, 200, 250], height=100.)
        plan = plan_quad_sections(mesh, outline, 10., small_element_size=5., dense_mapping_angle=45.,
                                  pattern_lines=[[[113.1, 100.], [113.1, 110.]]])
        # The upper runs lie in separate LINEs. Only the lower gap is protected.
        limited = [mesh.nodes[r["source_node"], :2] for r in plan.diagnostics["dense_mappings"] if r["gap_limited"]]
        np.testing.assert_allclose(sorted(map(tuple, limited)), [(103., 0.), (123.2, 0.)])
        self.assertTrue(all(r["reason"] == "expanded_contact" for r in top_mappings(mesh, plan.diagnostics).values()))

    def test_fixed_and_resampled_anchors_use_the_actual_candidate_gap(self):
        for fixed in (False, True):
            for anchors, ratio, limited in (([112, 114], .5, True), ([107, 119], .5, False),
                                            ([103, 114], .75, True)):
                with self.subTest(fixed=fixed, anchors=anchors):
                    mesh, outline = two_runs()
                    xs = [0, 90, *anchors, 136.2, 250]
                    target = _target_ids(mesh, [[-10, -10], [-10, 0], [-10, 100], [-10, 110],
                                               *[[x, 110] for x in xs], [260, 110], [260, 100],
                                               [260, 0], [260, -10], *[[x, -10] for x in reversed(xs)]])
                    result = generate_layer_adaptive_mesh(mesh, outline, 20., outline_o=target,
                                                          fixed_outer_boundary=fixed, small_element_size=5.,
                                                          dense_mapping_angle=45., dense_min_gap_ratio=ratio)
                    records = top_mappings(mesh, result.diagnostics)
                    for x, anchor in zip((103., 123.2), anchors):
                        r = records[x]
                        self.assertEqual(r["gap_limited"], limited)
                        if fixed and limited and x != anchor:
                            self.assertEqual(r["status"], "skipped")
                        else:
                            self.assertAlmostEqual(r["target"][0], x if limited else anchor)
                            self.assertEqual(r["reason"], "existing_contact" if x == anchor or not limited
                                             else "projected_contact")
                    if fixed:
                        self.assertEqual(set(result.outline), set(target))

    def test_full_mesh_replays_custom_gap_decisions_from_preview(self):
        mesh, outline = two_runs(40.)
        settings = dict(small_element_size=5., dense_mapping_angle=45., dense_min_gap_ratio=.75)
        plan = plan_quad_sections(mesh, outline, 10., **settings)
        result = generate_full_adaptive_mesh(mesh, [10.], **settings, outer_outline=[
            [-50., -50.], [300., -50.], [300., 150.], [-50., 150.]])
        layer = result.diagnostics["layers"][0][0]
        self.assertEqual(result.diagnostics["dense_min_gap_ratio"], .75)
        self.assertEqual(layer["dense_min_gap_ratio"], .75)
        actual = {r["source_node"]: r for r in layer["dense_mappings"]}
        self.assertEqual(sum(r["gap_limited"] for r in actual.values()), 4)
        for expected in plan.diagnostics["dense_mappings"]:
            r = actual[expected["source_node"]]
            self.assertEqual(r["reason"], "canonical_contact")
            self.assertEqual(r["gap_limited"], expected["gap_limited"])
            np.testing.assert_allclose(r["target"], expected["target"])
