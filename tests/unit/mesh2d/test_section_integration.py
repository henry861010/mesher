"""Public section preview and unit assembly agree on geometry and shared nodes."""
from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import (QuadLayerError, generate_layer_adaptive_mesh,
                                   plan_quad_sections)
from mesher.mesh2d.adapter.layer._geometry import area, collinear_interval, intervals_cover
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.adapter.layer._construction import line as _line
from tests.unit.mesh2d.test_quad_sections import _cells


class SectionIntegrationTests(unittest.TestCase):
    @staticmethod
    def _l_shape():
        return _cells({(x, y) for x in range(6) for y in range(6)
                       if x >= 3 or y < 3})

    @staticmethod
    def _small_hole():
        mesh, _ = _cells({(x, y) for x in range(4) for y in range(4)
                          if x < 1 or x >= 3 or y < 1 or y >= 3})
        outline = next(loop for loop in EditorMesh2D().outlines_by_edge(mesh)
                       if area(np.vstack([mesh.nodes[edge[:-1], :2] for edge in loop])) > 0.)
        return mesh, outline

    @staticmethod
    def _nonuniform_rectangle(rotation=0):
        xs = np.r_[0., 500., np.arange(1000., 1101., 5.), 1600., 2100.]
        mesh, outline = _cells({(x, y) for x in range(len(xs) - 1) for y in range(2)})
        mesh.nodes[:, 0] = xs[mesh.nodes[:, 0].astype(int)]
        mesh.nodes[:, 1] *= 500.
        for _ in range(rotation):
            mesh.nodes[:, :2] = mesh.nodes[:, [1, 0]] * [-1, 1]
        return mesh, outline

    def assert_pattern_covered(self, mesh, pattern):
        edges = np.asarray(sorted({tuple(sorted((int(a), int(b))))
                                   for quad in mesh.elements
                                   for a, b in zip(quad, np.roll(quad, -1))}))
        start, end = np.asarray(pattern, dtype=float)
        intervals = [interval for a, b in mesh.nodes[edges, :2]
                     if (interval := collinear_interval(start, end, a, b, 1e-9)) is not None]
        self.assertTrue(intervals_cover([(0., 1.)], intervals, 1e-9))

    def assert_built_partition(self, mesh, before_nodes, before_elements, result):
        np.testing.assert_array_equal(mesh.nodes[:len(before_nodes)], before_nodes)
        np.testing.assert_array_equal(mesh.elements[:len(before_elements)], before_elements)
        details = result.diagnostics['section_details']
        for section, following in zip(details, details[1:] + details[:1]):
            self.assertEqual(section['right_seam'], following['left_seam'])
            self.assertEqual(section['right_ids'], following['left_ids'])
        self.assertEqual(details[0]['element_start'], len(before_elements))
        self.assertEqual(details[-1]['element_end'], len(mesh.elements))
        for first, second in zip(details, details[1:]):
            self.assertEqual(first['element_end'], second['element_start'])
        added = mesh.nodes[mesh.elements[len(before_elements):], :2]
        vectors = np.roll(added, -1, axis=1) - added
        following = np.roll(vectors, -1, axis=1)
        turns = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
        self.assertTrue(np.all(turns > 0.))
        self.assertEqual(result.diagnostics['path'], 'sections')
        self.assertNotIn('collar_distance', result.diagnostics)

    def test_all_six_section_kinds_build_and_left_right_are_mirrors(self):
        left = [[1., 3.], [1., 4.6]]
        right = [[1.4, 5.], [3., 5.]]
        summaries = {}
        for expected, patterns in (('INNER', []), ('INNER_LEFT', [left]),
                                   ('INNER_RIGHT', [right]), ('INNER_BOTH', [left, right])):
            with self.subTest(kind=expected):
                mesh, outline = self._l_shape()
                before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
                plan = plan_quad_sections(mesh, outline, 1.6, pattern_lines=patterns)
                result = generate_layer_adaptive_mesh(mesh, outline, 1.6, pattern_lines=patterns)
                self.assertEqual(result.diagnostics['sections'], {'OUTER': 5, 'LINE': 6, expected: 1})
                self.assertEqual([section.kind for section in plan.sections],
                                 [section['kind'] for section in result.diagnostics['section_details']])
                self.assert_built_partition(mesh, before_nodes, before_elements, result)
                for pattern in patterns:
                    self.assert_pattern_covered(mesh, pattern)
                corner = next(section for section in result.diagnostics['section_details']
                              if section['kind'] == expected)
                summaries[expected] = (corner['element_end'] - corner['element_start'],
                                       corner['quality'], corner['size_error'])
        np.testing.assert_allclose(summaries['INNER_LEFT'], summaries['INNER_RIGHT'], rtol=1e-11, atol=1e-11)

    def test_public_plan_is_readonly_and_uses_world_coordinates(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        scale, shift = 17., np.asarray([10000., -20000.])
        mesh.nodes[:, :2] = mesh.nodes[:, :2] * scale + shift
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        plan = plan_quad_sections(mesh, outline, 8.5)
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)
        np.testing.assert_allclose(plan.front.min(axis=0), shift + [-8.5, -8.5])
        np.testing.assert_allclose(plan.front.max(axis=0), shift + [76.5, 59.5])
        self.assertEqual(plan.edge_expansion, (8.5,) * 4)
        for array in [plan.front, *(seam.points for seam in plan.seams),
                      *(section.outer_boundary for section in plan.sections)]:
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0.
        with self.assertRaises(FrozenInstanceError):
            plan.sections[0].kind = 'LINE'

    def test_test2_keeps_twelve_full_width_sections_and_frozen_source(self):
        mesh, outline = _cells({(x, y) for x in range(40) for y in range(40)
                               if x >= 20 or y < 20})
        mesh.nodes[:, :2] *= 5.
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        plan = plan_quad_sections(mesh, outline, 50.)
        self.assertEqual(plan.diagnostics['sections'], {'OUTER': 5, 'LINE': 6, 'INNER': 1})
        self.assertEqual(len(plan.sections), 12)
        self.assertEqual(plan.edge_expansion, (50.,) * 6)
        self.assertNotIn('collar_distance', plan.diagnostics)
        left_sections = [section for section in plan.sections if section.kind == 'LINE'
                         and np.all(mesh.nodes[list(section.source_chains[0]), 0] == 0.)]
        self.assertEqual(len(left_sections), 1)
        left = left_sections[0]
        self.assertEqual(len(left.source_chains[0]), 21)
        source = mesh.nodes[list(left.source_chains[0]), :2]
        np.testing.assert_array_equal(source[[0, -1]], [[0., 0.], [0., 100.]])
        np.testing.assert_array_equal(left.outer_boundary, [[-50., 0.], [-50., 100.]])
        for seam_id in (left.left_seam, left.right_seam):
            seam = plan.seams[seam_id]
            self.assertAlmostEqual(np.linalg.norm(seam.points[-1] - seam.points[0]), 50.)
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)

    def test_public_build_rejects_interior_pattern_endpoint_atomically(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=[[[1., 3.], [1., 3.25]]])
        self.assertEqual(caught.exception.code, 'unsupported_pattern_layout')
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)

    def test_local_fine_run_survives_public_expansion_in_all_directions(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                mesh, outline = self._nonuniform_rectangle(rotation)
                before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
                plan = plan_quad_sections(mesh, outline, 500.)
                np.testing.assert_array_equal(mesh.nodes, before_nodes)
                np.testing.assert_array_equal(mesh.elements, before_elements)
                self.assertFalse(plan.front.flags.writeable)
                result = generate_layer_adaptive_mesh(mesh, outline, 500.)
                self.assert_built_partition(mesh, before_nodes, before_elements, result)
                lines = [detail for detail in result.diagnostics['section_details']
                         if detail['kind'] == 'LINE' and len(detail['source_chains'][0]) == 25]
                self.assertEqual(len(lines), 2)
                for detail in lines:
                    points = mesh.nodes[detail['outer_ids'], :2].copy()
                    for _ in range(rotation):
                        points = points[:, [1, 0]] * [1, -1]
                    values = np.sort(points[:, 0])
                    for endpoint in (1000., 1100.):
                        self.assertEqual(np.count_nonzero(np.isclose(values, endpoint, atol=1e-8,
                                                                    rtol=0.)), 1)
                    dense = values[(values >= 1000. - 1e-8) & (values <= 1100. + 1e-8)]
                    self.assertLessEqual(np.max(np.diff(dense)), 10. + 1e-8)

    def test_local_fine_run_preserves_explicit_outer_anchor(self):
        mesh, outline = self._nonuniform_rectangle()
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        target = np.array([[-500., -500.], [-500., 1500.], [1007.5, 1500.],
                           [2600., 1500.], [2600., -500.]])
        original_target = target.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., new_outline=target)
        self.assert_built_partition(mesh, before_nodes, before_elements, result)
        np.testing.assert_array_equal(target, original_target)
        front = mesh.nodes[np.unique(np.concatenate(result.outline)), :2]
        for point in [*target, [1000., 1500.], [1100., 1500.]]:
            self.assertEqual(np.count_nonzero(np.linalg.norm(front - point, axis=1) < 1e-8), 1)

    def test_local_fine_run_obeys_exact_budget_and_rejects_atomically(self):
        baseline, outline = self._nonuniform_rectangle()
        count = generate_layer_adaptive_mesh(baseline, outline, 500.).diagnostics['added_elements']
        mesh, outline = self._nonuniform_rectangle()
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., max_added_elements=count)
        self.assertEqual(result.diagnostics['added_elements'], count)
        self.assert_built_partition(mesh, before_nodes, before_elements, result)
        mesh, outline = self._nonuniform_rectangle()
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 500., max_added_elements=count - 1)
        self.assertEqual(caught.exception.code, 'element_limit_exceeded')
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)

    def test_line_diagnostics_count_every_seam_pair_build(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        with patch.object(_line, "build_unit_patch", wraps=_line.build_unit_patch) as calls:
            result = generate_layer_adaptive_mesh(mesh, outline, .5)
        diagnostics = result.diagnostics
        lines = [detail for detail in diagnostics["section_details"] if detail["kind"] == "LINE"]
        self.assertEqual(diagnostics["line_planner"], "outer_first")
        self.assertEqual(diagnostics["line_unit_trial_count"], calls.call_count)
        self.assertEqual(sum(detail["unit_trial_count"] for detail in lines), calls.call_count)
        self.assertTrue(all(detail["line_planner"] == "outer_first" for detail in lines))
        self.assertEqual(diagnostics["line_local_repair_count"],
                         sum(detail["local_repair_count"] for detail in lines))
        self.assert_built_partition(mesh, before_nodes, before_elements, result)

    def test_exhausted_line_repairs_are_a_typed_atomic_failure(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        with patch.object(_line, "build_unit_patch", side_effect=ValueError("invalid unit")) as calls:
            with self.assertRaises(QuadLayerError) as caught:
                generate_layer_adaptive_mesh(mesh, outline, .5)
        self.assertEqual(caught.exception.code, "unmeshable_section_layout")
        self.assertEqual(caught.exception.context["stage"], "section_builder")
        self.assertEqual(caught.exception.context["line_planner"], "outer_first")
        self.assertEqual(caught.exception.context["unit_trial_count"], calls.call_count)
        self.assertLessEqual(calls.call_count, 4 * 2 * 8 * 4)
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)

    def test_hole_with_four_inner_both_sections_builds_without_lines(self):
        mesh, outline = self._small_hole()
        patterns = [[[2, 1], [2, 1.6]], [[3, 2], [2.4, 2]],
                    [[2, 3], [2, 2.4]], [[1, 2], [1.6, 2]]]
        before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
        plan = plan_quad_sections(mesh, outline, .6, pattern_lines=patterns)
        result = generate_layer_adaptive_mesh(mesh, outline, .6, pattern_lines=patterns)
        self.assertEqual(plan.diagnostics['sections'], {'INNER_BOTH': 4})
        self.assertEqual(result.diagnostics['sections'], {'INNER_BOTH': 4})
        self.assertEqual(len(result.diagnostics['seam_states']), 4)
        self.assert_built_partition(mesh, before_nodes, before_elements, result)
        for pattern in patterns:
            self.assert_pattern_covered(mesh, pattern)
        new_front = mesh.nodes[np.concatenate(result.outline), :2]
        np.testing.assert_allclose(new_front.min(axis=0), [1.6, 1.6])
        np.testing.assert_allclose(new_front.max(axis=0), [2.4, 2.4])

    def test_merged_chords_retain_three_and_four_mandatory_seam_nodes(self):
        for levels in ((3., 3.25, 3.5), (3., 3.125, 3.375, 3.5)):
            with self.subTest(mandatory_count=len(levels)):
                mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
                patterns = [[[1., low], [1., high]] for low, high in zip(levels, levels[1:])]
                before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
                plan = plan_quad_sections(mesh, outline, .5, pattern_lines=patterns)
                locked = [seam for seam in plan.seams if seam.pattern_ids]
                self.assertEqual(len(locked), 1)
                self.assertEqual(len(locked[0].points), len(levels))
                self.assertEqual(locked[0].pattern_ids, tuple(range(len(patterns))))
                result = generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=patterns)
                state = next(state for state in result.diagnostics['seam_states']
                             if state['seam_id'] == locked[0].id)
                self.assertIn(state['node_count'], (len(levels), len(levels) + 1))
                points = mesh.nodes[state['node_ids'], :2]
                for level in levels:
                    self.assertEqual(np.count_nonzero(np.linalg.norm(points - [1., level], axis=1) < 1e-12), 1)
                for pattern in patterns:
                    self.assert_pattern_covered(mesh, pattern)
                self.assert_built_partition(mesh, before_nodes, before_elements, result)

    def test_duplicate_pattern_ids_survive_union_and_public_build(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        patterns = [[[1., 3.], [1., 3.5]], [[1., 3.5], [1., 3.]]]
        plan = plan_quad_sections(mesh, outline, .5, pattern_lines=patterns)
        locked = [seam for seam in plan.seams if seam.pattern_ids]
        self.assertEqual(len(locked), 1)
        self.assertEqual(locked[0].pattern_ids, (0, 1))
        result = generate_layer_adaptive_mesh(mesh, outline, .5, pattern_lines=patterns)
        owners = [section for section in result.diagnostics['section_details']
                  if locked[0].id in (section['left_seam'], section['right_seam'])]
        self.assertEqual(len(owners), 2)
        self.assertTrue(all(section['pattern_ids'] == [0, 1] for section in owners))

    def test_reservation_collision_and_front_collapse_have_distinct_errors(self):
        for h, code in ((.6, 'unmeshable_section_layout'), (1., 'topology_change')):
            with self.subTest(h=h):
                mesh, outline = self._small_hole()
                before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()
                with self.assertRaises(QuadLayerError) as caught:
                    generate_layer_adaptive_mesh(mesh, outline, h)
                self.assertEqual(caught.exception.code, code)
                if code == 'unmeshable_section_layout':
                    self.assertEqual(caught.exception.context['stage'], 'section_partition')
                    self.assertIn('source_edge', caught.exception.context)
                    self.assertIn('wedge', str(caught.exception))
                else:
                    self.assertIn('closes', str(caught.exception))
                np.testing.assert_array_equal(mesh.nodes, before_nodes)
                np.testing.assert_array_equal(mesh.elements, before_elements)


if __name__ == '__main__':
    unittest.main()
