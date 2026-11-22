"""Dense runs have fixed, local interfaces through every public adapter path."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import (QuadLayerError, generate_full_adaptive_mesh,
                                   generate_layer_adaptive_mesh, plan_quad_sections)
from mesher.mesh2d.adapter import _full
from mesher.mesh2d.adapter.layer import _mesher
from mesher.mesh2d.editor import EditorMesh2D
from tests.unit.mesh2d.test_quad_sections import _cells
from tests.unit.mesh2d import test_section_integration as section_checks
from tests.unit.mesh2d.test_slot_repair import _c_slot
from tests.unit.mesh2d.test_extend_quad_layer import _target_ids


def rectangle(xs=None, height=2000., rotation=0, scale=1., shear=0.):
    if xs is None:
        xs = np.r_[0., 1000., 2000., 3000., np.arange(4000., 4301., 5.),
                   5000., 6000., 7000., 8000.]
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    nodes = np.array([[x + shear * y, y] for y in (0., height) for x in xs])
    for _ in range(rotation):
        nodes = nodes[:, [1, 0]] * [-1, 1]
    mesh = Mesh2D(nodes * scale, np.array([[i, i + 1, i + n + 1, i + n]
                                          for i in range(n - 1)]))
    return mesh, EditorMesh2D.outlines(mesh)[0]


class DenseSectionTests(unittest.TestCase):
    assert_built_partition = section_checks.SectionIntegrationTests.assert_built_partition

    def assert_frozen(self, mesh, nodes, elements):
        np.testing.assert_array_equal(mesh.nodes[:len(nodes)], nodes)
        np.testing.assert_array_equal(mesh.elements[:len(elements)], elements)

    def assert_plan(self, mesh, outline, plan):
        prepared = plan.prepared_source
        loop = prepared.outline if prepared is not None else outline
        expected = Counter((int(a), int(b)) for a, b in zip(loop, np.roll(loop, -1)))
        actual = Counter((int(a), int(b)) for section in plan.sections
                         for chain in section.source_chains for a, b in zip(chain[:-1], chain[1:]))
        self.assertEqual(actual, expected)
        self.assertTrue(all(len(seam.owners) == 2 for seam in plan.seams))
        for section, following in zip(plan.sections, plan.sections[1:] + plan.sections[:1]):
            self.assertEqual(section.right_seam, following.left_seam)

    def test_dense_run_is_a_section_with_shared_fixed_endpoints_in_every_direction(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                mesh, outline = rectangle(rotation=rotation)
                nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
                plan = plan_quad_sections(mesh, outline, 1000., small_element_size=10.)
                self.assertEqual(len(plan.diagnostics['dense_sections']), 2)
                self.assertEqual(len(plan.diagnostics['dense_seams']), 4)
                self.assert_plan(mesh, outline, plan)
                result = generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.,
                                                      max_transition_count=1000)
                self.assert_built_partition(mesh, nodes, cells, result)
                new_points = mesh.nodes[mesh.elements[len(cells):], :2]
                self.assertGreater(np.linalg.norm(np.roll(new_points, -1, axis=1) - new_points,
                                                 axis=2).min(), .001)
                self.assertEqual(plan.diagnostics['dense_sections'], result.diagnostics['dense_sections'])
                for record in plan.diagnostics['dense_sections']:
                    section = plan.sections[record['section_id']]
                    self.assertEqual(section.kind, 'LINE')
                    self.assertEqual(len(section.source_chains[0]) - 1, 60)
                    axis = rotation % 2
                    begin, end = (plan.seams[i].points for i in (section.left_seam, section.right_seam))
                    direction = np.sign(end[0, axis] - begin[0, axis])
                    offset = 1000. * np.tan(np.radians(30.))
                    self.assertAlmostEqual(direction * (begin[-1, axis] - begin[0, axis]), -offset)
                    self.assertAlmostEqual(direction * (end[-1, axis] - end[0, axis]), offset)
                    # Every cell stays in its section's tangent interval.
                    for detail in result.diagnostics['section_details']:
                        if detail['kind'] != 'LINE' or detail['source_edge'] != section.source_edge:
                            continue
                        points = mesh.nodes[mesh.elements[detail['element_start']:detail['element_end']], :2]
                        polygon = np.asarray(detail['polygon'])
                        self.assertGreaterEqual(points[..., axis].min(), polygon[:, axis].min() - 1e-8)
                        self.assertLessEqual(points[..., axis].max(), polygon[:, axis].max() + 1e-8)

    def test_maximal_runs_single_edges_endpoints_and_strict_threshold(self):
        for xs, expected, seams in (
            ([0, 20, 25, 30, 50, 55, 75], 4, 8),
            ([0, 5, 25, 45, 50], 4, 4),
            ([0, 5, 10, 15], 2, 0),
            ([0, 10, 20, 30], 0, 0),
        ):
            with self.subTest(xs=xs):
                mesh, outline = rectangle(xs, height=100.)
                plan = plan_quad_sections(mesh, outline, 20., small_element_size=10.)
                self.assertEqual(len(plan.diagnostics['dense_sections']), expected)
                self.assertEqual(len(plan.diagnostics['dense_seams']), seams)
                self.assert_plan(mesh, outline, plan)

    def test_free_dense_seams_respect_angle_after_world_rounding(self):
        # Translation can round the tangent displacement up or the normal depth down.
        for rotation, translation, depth in ((0, [2.**35, 0.], 4.),
                                              (1, [0., -2.**35], 4.),
                                              (0, [0., 2.**35], 4.0000009)):
            with self.subTest(rotation=rotation, translation=translation, depth=depth):
                mesh, outline = rectangle([0, 20, 25, 45], height=24., rotation=rotation)
                mesh.nodes[:, :2] += translation
                kwargs = dict(small_element_size=6., expand_size=depth, dense_mapping_angle=30.)
                plan = plan_quad_sections(mesh, outline, 10., **kwargs)
                result = generate_layer_adaptive_mesh(mesh, outline, 10., **kwargs)
                actual = result.diagnostics["dense_mappings"]
                self.assertEqual(len(actual), 4)
                for planned, committed in zip(plan.diagnostics["dense_mappings"], actual):
                    self.assertEqual(committed["reason"], "expanded_contact")
                    np.testing.assert_array_equal(planned["target"], committed["target"])
                    displacement = np.abs(committed["target"] - mesh.nodes[committed["source_node"], :2])
                    angle = np.degrees(np.arctan2(displacement[rotation], displacement[1 - rotation]))
                    self.assertLessEqual(angle, 30. + 1e-12)
                    self.assertGreater(angle, 29.9998)
                    self.assertAlmostEqual(abs(planned["angle"]), angle, places=11)
                    self.assertAlmostEqual(abs(committed["angle"]), angle, places=11)

    def test_disabled_partition_and_explicit_old_cap_remain_available(self):
        mesh, outline = rectangle()
        plan = plan_quad_sections(mesh, outline, 1000.)
        self.assertEqual(plan.diagnostics['sections'], {'OUTER': 4, 'LINE': 4})
        self.assertEqual(plan.diagnostics['dense_sections'], [])
        result = generate_layer_adaptive_mesh(mesh, outline, 1000., max_transition_count=10)
        self.assertEqual(result.diagnostics['max_transition_count'], 10)
        mesh, outline = rectangle()
        result = generate_layer_adaptive_mesh(mesh, outline, 1000.)
        self.assertEqual(result.diagnostics['max_transition_count'], 20)
        self.assertTrue(any(d.get('max_actual_transition_count', 0) > 10
                            for d in result.diagnostics['section_details']))
        self.assertTrue(all(d.get('max_actual_transition_count', 0) <= 20
                            for d in result.diagnostics['section_details']))

    def test_group_count_filters_each_consecutive_run_before_mapping(self):
        # Runs of one, two and three short edges, separated by edges of exactly size 10.
        mesh, outline = rectangle([0, 20, 25, 35, 40, 45, 55, 60, 65, 70, 90], height=20.)
        settings = dict(small_element_size=10., small_element_group_count=np.int64(2),
                        dense_mapping_angle=0.)
        plan = plan_quad_sections(mesh, outline, 10., **settings)
        result = generate_layer_adaptive_mesh(mesh, outline, 10., **settings)
        for diagnostics in (plan.diagnostics, result.diagnostics):
            self.assertEqual(diagnostics['small_element_group_count'], 2)
            dense_spans = [sorted(mesh.nodes[[r['start_node'], r['end_node']], 0])
                           for r in diagnostics['dense_sections']]
            self.assertEqual(dense_spans, [[55., 70.], [55., 70.]])
            mapped_x = sorted(mesh.nodes[r['source_node'], 0] for r in diagnostics['dense_mappings'])
            self.assertEqual(mapped_x, [55., 55., 70., 70.])

    def test_short_dense_sections_obey_transition_cap_and_neighbors_obey_spacing(self):
        for cap in (20, 10):
            with self.subTest(cap=cap):
                mesh, outline = rectangle()
                nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
                result = generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.,
                                                      dense_mapping_angle=0., min_size_ratio=.5, max_size_ratio=1.25,
                                                      max_transition_count=cap, max_added_elements=1000)
                self.assert_built_partition(mesh, nodes, cells, result)
                diagnostics = result.diagnostics
                dense = {record['section_id'] for record in diagnostics['dense_sections']}
                self.assertEqual(len(dense), 2)
                for detail in diagnostics['section_details']:
                    if detail['kind'] != 'LINE':
                        continue
                    points = mesh.nodes[detail['outer_ids'], :2]
                    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
                    self.assertAlmostEqual(detail['min_outer_spacing'], gaps.min())
                    self.assertAlmostEqual(detail['max_outer_spacing'], gaps.max())
                    self.assertEqual(detail['size_band_active'], detail['section_id'] not in dense)
                    if detail['section_id'] in dense:
                        self.assertEqual(detail['forced_short_intervals'], [])
                        self.assertAlmostEqual(gaps.sum(), 300.)
                        self.assertLess(gaps.max(), 500.)
                        units = [diagnostics['units'][index] for index in detail['unit_ids']]
                        stage_loads = [[n for unit in units if unit['sublayer'] == stage
                                        for n in unit['sublayer_segments_per_outer']] for stage in (1, 2)]
                        inner, outer = stage_loads
                        self.assertEqual(sum(inner), 60)
                        self.assertEqual(sum(outer), len(inner))
                        self.assertEqual(detail['sublayer_max_transition_counts'], [max(inner), max(outer)])
                        self.assertLessEqual(max(map(max, stage_loads)), cap)
                        loads = [n for unit in units for n in unit['source_segments_per_outer']]
                        self.assertEqual(len(loads), len(gaps))
                        np.testing.assert_equal(loads, np.diff(np.cumsum([0, *inner])[np.cumsum([0, *outer])]))
                        self.assertEqual(detail['max_actual_transition_count'], max(loads))
                        self.assertGreater(max(loads), cap)
                        if cap == 20:
                            middle = mesh.nodes[detail['midline_ids'], :2]
                            np.testing.assert_allclose(np.linalg.norm(np.diff(middle, axis=0), axis=1), [75.] * 4)
                            self.assertAlmostEqual(detail['max_alignment_angle'], 0.)
                    else:
                        self.assertGreaterEqual(gaps.min(), 500. - 1e-8)
                        self.assertLessEqual(gaps.max(), 1250. + 1e-8)

    def test_dense_spacing_can_exceed_max_ratio_on_a_resampled_front(self):
        mesh, outline = rectangle([0., 20., 31., 42., 62.], height=40.)
        result = generate_layer_adaptive_mesh(mesh, outline, 10., small_element_size=12.,
                                              dense_mapping_angle=0., max_transition_count=1, max_size_ratio=1.)
        dense = {record['section_id'] for record in result.diagnostics['dense_sections']}
        self.assertEqual(len(dense), 2)
        for detail in result.diagnostics['section_details']:
            if detail['kind'] != 'LINE':
                continue
            points = mesh.nodes[detail['outer_ids'], :2]
            gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            self.assertLessEqual(detail['max_actual_transition_count'], 1)
            if detail['section_id'] in dense:
                np.testing.assert_allclose(gaps, [11., 11.])
            else:
                self.assertLessEqual(gaps.max(), 10. + 1e-8)

    def test_widened_sublayers_preserve_the_fractions_of_retained_source_nodes(self):
        # Exercise both a two-edge unit and actual coarsening with irregular gaps.
        for xs in ([100, 104, 120], [100, 102, 104, 112, 114, 115, 120, 127, 129]):
            with self.subTest(xs=xs):
                mesh, outline = rectangle([0, 50, *xs, 170, 220], height=100.)
                result = generate_layer_adaptive_mesh(mesh, outline, 40., expand_size=20.,
                                                      small_element_size=17., dense_mapping_angle=45.,
                                                      max_transition_count=3)
                detail, = [s for s in result.diagnostics['section_details'] if 'midline_ids' in s
                           and mesh.nodes[s['source_chains'][0][0], 1] == 100.]
                units = [result.diagnostics['units'][i] for i in detail['unit_ids']]
                source = mesh.nodes[detail['source_chains'][0], 0]
                for stage, chain in enumerate(('midline_ids', 'outer_ids'), 1):
                    front = mesh.nodes[detail[chain], 0]
                    loads = [n for u in units if u['sublayer'] == stage for n in u['sublayer_segments_per_outer']]
                    fractions = (source[np.cumsum([0, *loads])] - source[0]) / np.ptp(source)
                    np.testing.assert_allclose((front - front[0]) / np.ptp(front), fractions)
                    self.assertGreater(np.ptp(front), np.ptp(source))
                    source = front
                if len(xs) > 3:
                    self.assertLess(len(source), len(xs))

    def test_dense_sublayers_share_midline_and_split_only_the_conflicting_front(self):
        # A 100-wide ordinary section with two source edges lies between a
        # single short edge and a run of twenty, as in debug_imprint_line.
        xs = np.r_[0, 1000, 2000, 2025, 2075, 2125, np.arange(2130, 2230, 5), 3000, 4000]
        mesh, outline = rectangle(xs, height=1000)
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 800, expand_size=400,
                                              small_element_size=30, max_transition_count=10,
                                              min_size_ratio=.5, max_size_ratio=1.5)
        self.assert_built_partition(mesh, nodes, cells, result)
        details = result.diagnostics["section_details"]
        dense = [s for s in details if "midline_ids" in s]
        self.assertEqual(len(dense), 4)
        for s in dense:
            middle = s["midline_ids"]
            for ids, midpoint in ((s["left_ids"], middle[0]), (s["right_ids"], middle[-1])):
                self.assertEqual(len(ids), 3)
                self.assertEqual(ids[1], midpoint)
                np.testing.assert_allclose(mesh.nodes[midpoint], mesh.nodes[ids[::2]].mean(axis=0))
            n = len(s["source_chains"][0]) - 1
            self.assertEqual((len(middle) - 1, len(s["outer_ids"]) - 1), (1, 1) if n == 1 else (2, 2))
            units = [result.diagnostics["units"][i] for i in s["unit_ids"]]
            for layer in (1, 2):
                incidence = Counter(tuple(sorted((a, b))) for unit in units if unit["sublayer"] == layer
                                    for q in mesh.elements[unit["element_start"]:unit["element_end"]]
                                    for a, b in zip(q, np.roll(q, -1)))
                self.assertTrue(all(incidence[tuple(sorted(edge))] == 1 for edge in zip(middle, middle[1:])))
            loads = [value for unit in units for value in unit["source_segments_per_outer"]]
            self.assertEqual(sum(loads), n)
            self.assertLessEqual(max(s["sublayer_max_transition_counts"]), 10)

        splits = [s for s in details if "parity_split_node" in s]
        self.assertEqual(len(splits), 2)  # Both long faces of the fixture.
        for s in splits:
            outer = s["outer_ids"]
            self.assertEqual(s["parity_split_node"], outer[1])
            self.assertIn(outer[1], result.outline)
            np.testing.assert_allclose(np.linalg.norm(np.diff(mesh.nodes[outer], axis=0), axis=1), [50, 50])
            self.assertEqual(s["element_end"] - s["element_start"], 4)
            self.assertTrue(s["size_band_active"])
            np.testing.assert_allclose([gap["length"] for gap in s["forced_short_intervals"]], [50, 50])

    def test_dense_seam_respects_mandatory_pattern_nodes(self):
        for offset in (100, 200):
            with self.subTest(offset=offset):
                mesh, outline = rectangle()
                nodes, cells = mesh.nodes, mesh.elements
                joint = [4000, 2000 + offset]
                kwargs = dict(expand_size=400, small_element_size=10,
                              pattern_lines=[[[4000, 2000], joint], [joint, [4000, 2400]]])
                if offset == 100:
                    with self.assertRaises(QuadLayerError) as caught:
                        generate_layer_adaptive_mesh(mesh, outline, 1000, **kwargs)
                    self.assertEqual(caught.exception.code, "unmeshable_section_layout")
                    self.assertIn("seam_id", caught.exception.context)
                    self.assertIn("section_id", caught.exception.context)
                    self.assertEqual(caught.exception.context["pattern_ids"], [0, 1])
                    self.assertIs(mesh.nodes, nodes)
                    self.assertIs(mesh.elements, cells)
                else:
                    result = generate_layer_adaptive_mesh(mesh, outline, 1000, **kwargs)
                    seam, = [s for s in result.diagnostics["seam_states"] if s["pattern_ids"]]
                    self.assertEqual(seam["node_count"], 3)
                    np.testing.assert_allclose(seam["points"][1], joint)

    def test_a_wholly_dense_line_gets_the_exemption_without_any_new_seams(self):
        mesh, outline = rectangle(np.arange(0., 305., 5.))
        result = generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.)
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics['dense_seams'], [])
        self.assertEqual(len(diagnostics['dense_sections']), 2)
        for record in diagnostics['dense_sections']:
            detail = diagnostics['section_details'][record['section_id']]
            self.assertFalse(detail['size_band_active'])
            self.assertLess(detail['max_outer_spacing'], 500.)
            self.assertLessEqual(max(detail['sublayer_max_transition_counts']), 20)

    def test_dense_layer_over_budget_fails_atomically(self):
        mesh, outline = rectangle()
        nodes, cells = mesh.nodes, mesh.elements
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.,
                                         max_added_elements=1)
        self.assertEqual(caught.exception.code, 'element_limit_exceeded')
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, cells)

    def test_scale_and_near_axis_source_keep_endpoint_correspondence(self):
        reference = None
        for scale in (1e-5, 1., 1e5):
            mesh, outline = rectangle(scale=scale, shear=1e-5)
            plan = plan_quad_sections(mesh, outline, 1000. * scale,
                                     small_element_size=10. * scale, tolerance=.025 * scale,
                                     dense_mapping_angle=30.0)
            contacts = [plan.seams[r['seam_id']].points / scale
                        for r in plan.diagnostics['dense_seams']]
            if reference is None:
                reference = contacts
            else:
                np.testing.assert_allclose(contacts, reference, rtol=1e-12, atol=1e-8)
            for points in contacts:
                dx, dy = points[-1] - points[0]
                self.assertAlmostEqual(abs(dx), abs(dy) * np.tan(np.radians(30.)))

    def test_explicit_outer_anchor_mapping_is_directional_and_does_not_move_inputs(self):
        for distance, angle, reused in ((-.0625, 30., True), (.0625, 30., False),
                                        (-30., 30., False), (0., 0., True)):
            with self.subTest(distance=distance, angle=angle):
                mesh, outline = rectangle([0, 100, 105, 110, 200], height=100.)
                target = np.array([[-50., -50.], [-50., 150.], [100. + distance, 150.],
                                   [250., 150.], [250., -50.]])
                saved = target.copy()
                ids = _target_ids(mesh, target)
                plan = plan_quad_sections(mesh, outline, 50., outline_o=ids,
                                         small_element_size=10., dense_mapping_angle=angle)
                record = next(r for r in plan.diagnostics['dense_seams']
                              if np.array_equal(mesh.nodes[r['source_node'], :2], [100., 100.]))
                point = plan.seams[record['seam_id']].points[-1]
                expected_offset = distance if reused else -50. * np.tan(np.radians(angle))
                self.assertAlmostEqual(point[0], 100. + expected_offset)
                np.testing.assert_array_equal(target, saved)
                self.assertTrue(any(np.any(np.all(np.isclose(s.outer_boundary, target[2]), axis=1))
                                    for s in plan.sections))

    def test_equal_distance_anchor_choice_follows_boundary_order(self):
        mesh, outline = rectangle([0, 100, 105, 110, 200], height=100.)
        target = [[-50., -50.], [-50., 150.], [99.75, 150.], [100.25, 150.],
                  [250., 150.], [250., -50.]]
        ids = _target_ids(mesh, target)
        plan = plan_quad_sections(mesh, outline, 50., outline_o=ids,
                                 small_element_size=10., dense_mapping_angle=30.0)
        record = next(r for r in plan.diagnostics['dense_seams']
                      if np.array_equal(mesh.nodes[r['source_node'], :2], [100., 100.]))
        self.assertEqual(plan.seams[record['seam_id']].points[-1, 0], 99.75)

    def test_pattern_splits_dense_run_and_keeps_its_original_shared_seam(self):
        mesh, outline = rectangle()
        patterns = [[[4150., 2000.], [4150., 3000.]]]
        plan = plan_quad_sections(mesh, outline, 1000., small_element_size=10., pattern_lines=patterns)
        self.assertEqual(len(plan.diagnostics['dense_sections']), 3)
        self.assertEqual(plan.diagnostics['pattern_chord_count'], 1)
        self.assertEqual(sum(bool(s.pattern_ids) for s in plan.seams), 1)
        self.assert_plan(mesh, outline, plan)
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 1000., small_element_size=10.,
                                              pattern_lines=patterns)
        self.assert_built_partition(mesh, nodes, cells, result)
        section_checks.SectionIntegrationTests.assert_pattern_covered(self, mesh, patterns[0])

    def test_dense_source_after_repair_uses_prepared_snapshot_and_preserves_original(self):
        mesh, outline = _c_slot(bottom_breaks=[1., 1.5, 2., 2.1, 2.2, 2.5, 3.])
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        plan = plan_quad_sections(mesh, outline, .5, small_element_size=.2)
        self.assertIsNotNone(plan.prepared_source)
        self.assertTrue(plan.diagnostics['dense_sections'])
        self.assert_plan(mesh, outline, plan)
        result = generate_layer_adaptive_mesh(mesh, outline, .5, small_element_size=.2)
        self.assert_frozen(mesh, nodes, cells)
        self.assertEqual(plan.diagnostics['dense_sections'], result.diagnostics['dense_sections'])

    def test_inward_hole_can_be_partitioned_in_single_and_full_meshes(self):
        mesh, _ = _cells({(x, y) for x in range(8) for y in range(8)
                          if x in (0, 7) or y in (0, 7)})
        xs = np.array([0, 10, 20, 21, 22, 30, 40, 50, 60.])
        mesh.nodes[:, 0] = xs[mesh.nodes[:, 0].astype(int)]
        mesh.nodes[:, 1] *= 10.
        outlines = EditorMesh2D.outlines(mesh)
        outline = min(outlines, key=lambda o: np.ptp(mesh.nodes[o, 0]))
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        plan = plan_quad_sections(mesh, outline, 2., small_element_size=2.)
        self.assertEqual(len(plan.diagnostics['dense_sections']), 2)
        result = generate_layer_adaptive_mesh(mesh, outline, 2., small_element_size=2.)
        self.assert_built_partition(mesh, nodes, cells, result)
        # Building the exterior first must not change the hole's saved contacts.
        full_mesh = Mesh2D(nodes.copy(), cells.copy())
        full = generate_full_adaptive_mesh(
            full_mesh, [2.], small_element_size=2.,
            outer_outline=[[-10., -10.], [70., -10.], [70., 90.], [-10., 90.]])
        self.assert_frozen(full_mesh, nodes, cells)
        self.assertEqual([len(d['dense_sections']) for d in full.diagnostics['layers'][0]], [2, 2])
        incidence = Counter(tuple(sorted((int(a), int(b)))) for q in full_mesh.elements
                            for a, b in zip(q, np.roll(q, -1)))
        self.assertEqual(len(full.layer_outlines[-1]), 2)
        for boundary in full.layer_outlines[-1]:
            for a, b in zip(boundary, np.roll(boundary, -1)):
                self.assertEqual(incidence[tuple(sorted((a, b)))], 2)

    def test_invalid_parameters_are_atomic_for_all_public_entrypoints(self):
        for parameter, values in (
            ('small_element_size', [0, -1, True, np.nan, np.inf, '10', [10], 10**1000]),
            ('small_element_group_count', [-1, 2.0, True, np.bool_(False), None, '2']),
            ('dense_mapping_angle', [-1, 90, 100, None, True, np.nan, np.inf, '30', [30], 10**1000]),
            ('dense_min_gap_ratio', [-.01, 1.01, None, True, np.bool_(False), np.nan, np.inf, '0.5', 10**1000]),
        ):
            for value in values:
                for operation in ('preview', 'single', 'full'):
                    with self.subTest(parameter=parameter, value=repr(value), operation=operation):
                        mesh, outline = rectangle()
                        nodes, cells = mesh.nodes, mesh.elements
                        with self.assertRaises(QuadLayerError) as caught:
                            kwargs = {parameter: value}
                            if operation == 'full':
                                generate_full_adaptive_mesh(mesh, [1000.], outer_outline=[], **kwargs)
                            else:
                                fn = plan_quad_sections if operation == 'preview' else generate_layer_adaptive_mesh
                                fn(mesh, outline, 1000., **kwargs)
                        self.assertEqual(caught.exception.code, 'invalid_input')
                        self.assertEqual(caught.exception.context['parameter'], parameter)
                        self.assertIs(mesh.nodes, nodes)
                        self.assertIs(mesh.elements, cells)

    def test_close_dense_endpoints_keep_distinct_projections(self):
        mesh, outline = rectangle([0, 100, 100.0001, 200], height=100.)
        nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 50., small_element_size=10., tolerance=1e-6)
        self.assert_frozen(mesh, nodes, cells)
        records = [r for r in result.diagnostics['dense_mappings']
                   if mesh.nodes[r['source_node'], 1] == 100.]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r['status'] == 'mapped' for r in records))
        self.assertGreater(records[1]['target'][0] - records[0]['target'][0], 0.)

    def test_full_mesh_routes_parameters_and_preserves_fixed_interface_for_one_or_two_layers(self):
        for sizes in ([1000.], [1000., 1000.]):
            with self.subTest(sizes=sizes):
                mesh, _ = rectangle()
                nodes, cells = mesh.nodes.copy(), mesh.elements.copy()
                target = [[-4000., -4000.], [12000., -4000.], [12000., 6000.], [-4000., 6000.]]
                with patch.object(_full, '_generate_layer_adaptive_mesh',
                                  wraps=_full._generate_layer_adaptive_mesh) as fixed:
                    result = generate_full_adaptive_mesh(mesh, sizes, outer_outline=target,
                                                        small_element_size=10., small_element_group_count=3)
                self.assert_frozen(mesh, nodes, cells)
                self.assertEqual(fixed.call_args.kwargs['small_element_size'], 10.)
                self.assertEqual(result.diagnostics['small_element_group_count'], 3)
                for layer in result.diagnostics['layers']:
                    self.assertEqual(layer[0]['small_element_group_count'], 3)
                self.assertNotIn('max_transition_count', fixed.call_args.kwargs)
                self.assertTrue(result.diagnostics['layers'][0][0]['dense_sections'])
                if len(sizes) == 1:
                    records = result.diagnostics['layers'][0][0]['dense_mappings']
                    self.assertEqual(len(records), 4)
                    for record in records:
                        self.assertEqual(record['reason'], 'canonical_contact')
                        self.assertAlmostEqual(record['angle'], -30. if record['endpoint'] == 'begin' else 30.)
                incidence = Counter(tuple(sorted((int(a), int(b)))) for q in mesh.elements
                                    for a, b in zip(q, np.roll(q, -1)))
                final = result.layer_outlines[-1][0]
                for a, b in zip(final, np.roll(final, -1)):
                    self.assertEqual(incidence[tuple(sorted((a, b)))], 2)

    def test_fixed_construction_reuses_saved_contact_instead_of_a_closer_new_anchor(self):
        mesh, outline = rectangle([0, 100, 105, 110, 200], height=100.)
        target = np.array([[-50., -50.], [-50., 150.], [99.9375, 150.],
                           [250., 150.], [250., -50.]])
        ids = _target_ids(mesh, target)
        plan = plan_quad_sections(mesh, outline, 50., outline_o=ids,
                                 small_element_size=10., dense_mapping_angle=30.0)
        contacts = {(r['source_edge'], r['source_index']): plan.seams[r['seam_id']].points[-1]
                    for r in plan.diagnostics['dense_seams']}
        # Simulate the global mesher adding closer contacts after planning.
        dense_front = _full.section_outline_points(plan)
        anchor = np.array([99.9375, 150.])
        pos = int(np.argmin(np.linalg.norm(dense_front - anchor, axis=1)))
        fixed = np.insert(dense_front, pos + 1, [[100., 150.]], axis=0)
        fixed_ids = _target_ids(mesh, fixed)
        data, rebuilt = _mesher._prepare_plan(mesh, outline, 50., (), 1., 1e-3,
                                              outline_o=fixed_ids, small_element_size=10.,
                                              dense_mapping_angle=30.0,
                                              _dense_seam_targets=contacts)
        record = next(r for r in rebuilt.diagnostics['dense_seams']
                      if np.array_equal(mesh.nodes[r['source_node'], :2], [100., 100.]))
        actual = rebuilt.seams[record['seam_id']].points[-1] * data['scale'] + data['origin']
        np.testing.assert_allclose(actual, anchor, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
