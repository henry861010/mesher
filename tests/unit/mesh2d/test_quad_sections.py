"""Corner-first geometry stays independent from unit counts and mesh creation."""
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter.layer._geometry import area
from mesher.mesh2d.adapter.layer._mesher import _front, _prepare, _validate_old_patterns
from mesher.mesh2d.adapter.layer._sections import partition_sections, section_polygon
from mesher.mesh2d.adapter.layer._types import QuadLayerError
from mesher.mesh2d.editor import EditorMesh2D


def _cells(cells):
    coordinates = sorted({(x + dx, y + dy) for x, y in cells
                          for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))})
    lookup = {point: index for index, point in enumerate(coordinates)}
    elements = [[lookup[x, y], lookup[x + 1, y], lookup[x + 1, y + 1], lookup[x, y + 1]]
                for x, y in sorted(cells)]
    mesh = Mesh2D(coordinates, np.asarray(elements, dtype=np.int32))
    return mesh, EditorMesh2D().outlines_by_edge(mesh)[0]


def _plan(mesh, outline, h, patterns=(), ratio=1.):
    data = _prepare(mesh, outline, h, patterns, ratio, 1e-3)
    _validate_old_patterns(mesh, data)
    _front(data)
    return partition_sections(data), data


class QuadSectionsTests(unittest.TestCase):
    def assert_partition(self, plan, data):
        for index, section in enumerate(plan.sections):
            self.assertEqual(section.right_seam, plan.sections[(index + 1) % len(plan.sections)].left_seam)
        self.assertTrue(all(len(seam.owners) == 2 for seam in plan.seams))
        observed = sum(area(section_polygon(section, plan.seams, data['xy'], data['tol']))
                       for section in plan.sections)
        self.assertAlmostEqual(observed, area(data['source']) - area(data['front']))

    def test_long_line_preserves_whole_source_chain(self):
        mesh, outline = _cells({(x, y) for x in range(40) for y in range(40) if x >= 20 or y < 20})
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        plan, data = _plan(mesh, outline, 10.)
        self.assertEqual(plan.diagnostics['sections'], {'OUTER': 5, 'LINE': 6, 'INNER': 1})
        left = [section for section in plan.sections if section.kind == 'LINE'
                and np.all(mesh.nodes[list(section.source_chains[0]), 0] == 0.)]
        self.assertEqual(len(left), 1)
        self.assertEqual(len(left[0].source_chains[0]), 21)
        np.testing.assert_array_equal(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.elements, elements)
        self.assert_partition(plan, data)

    def test_all_inner_variants_have_explicit_front_arms(self):
        for left, right, expected in ((False, False, 'INNER'), (True, False, 'INNER_LEFT'),
                                      (False, True, 'INNER_RIGHT'), (True, True, 'INNER_BOTH')):
            with self.subTest(expected=expected):
                mesh, outline = _cells({(x, y) for x in range(6) for y in range(6) if x >= 3 or y < 3})
                patterns = []
                if left:
                    patterns.append([[1., 3.], [1., 4.6]])
                if right:
                    patterns.append([[1.4, 5.], [3., 5.]])
                plan, data = _plan(mesh, outline, 1.6, patterns)
                inner = [section for section in plan.sections if section.kind.startswith('INNER')]
                self.assertEqual(len(inner), 1)
                self.assertEqual(inner[0].kind, expected)
                self.assertEqual(len(inner[0].outer_boundary), 1 + left + right)
                self.assert_partition(plan, data)

    def test_full_chords_split_lines_and_merge_collinear_intervals(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        patterns = [[[1, 3], [1, 3.25]], [[1, 3.25], [1, 3.5]], [[3, 3], [3, 3.5]]]
        plan, data = _plan(mesh, outline, .5, patterns)
        self.assertEqual(plan.diagnostics['pattern_chord_count'], 2)
        self.assertEqual(plan.diagnostics['sections']['LINE'], 6)
        locked = [seam for seam in plan.seams if seam.pattern_ids]
        self.assertEqual(len(locked), 2)
        self.assertTrue(any(len(seam.points) == 3 for seam in locked))
        self.assert_partition(plan, data)

    def test_finite_interior_endpoints_and_crosses_are_typed_rejections(self):
        cases = ([[[1., 3.], [1., 3.25]]],
                 [[[1., 3.1], [1., 3.4]]],
                 [[[4.2, 3.4], [4.6, 3.4]], [[4.4, 3.2], [4.4, 3.6]]],
                 [[[4.2, 3.2], [4.6, 3.2]], [[4.6, 3.2], [4.6, 3.6]],
                  [[4.6, 3.6], [4.2, 3.6]], [[4.2, 3.6], [4.2, 3.2]]])
        for patterns in cases:
            with self.subTest(patterns=patterns):
                mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
                with self.assertRaises(QuadLayerError) as caught:
                    _plan(mesh, outline, .8, patterns)
                self.assertEqual(caught.exception.code, 'unsupported_pattern_layout')

    def test_source_only_touching_reservations_are_not_silently_omitted(self):
        mesh, outline = _cells({(0, 0), (1, 0), (0, 1)})
        with self.assertRaises(QuadLayerError) as caught:
            _plan(mesh, outline, .2)
        self.assertEqual(caught.exception.code, 'unmeshable_section_layout')
        self.assertIn('wedge', str(caught.exception))

    def test_coincident_complete_interfaces_allow_no_line_sections(self):
        mesh, _ = _cells({(x, y) for x in range(4) for y in range(4)
                          if x < 1 or x >= 3 or y < 1 or y >= 3})
        outline = next(loop for loop in EditorMesh2D().outlines_by_edge(mesh)
                       if area(np.vstack([mesh.nodes[edge[:-1], :2] for edge in loop])) > 0)
        patterns = [[[2, 1], [2, 1.6]], [[3, 2], [2.4, 2]],
                    [[2, 3], [2, 2.4]], [[1, 2], [1.6, 2]]]
        plan, data = _plan(mesh, outline, .6, patterns)
        self.assertEqual(plan.diagnostics['sections'], {'INNER_BOTH': 4})
        self.assertEqual(len(plan.seams), 4)
        self.assert_partition(plan, data)

    def test_counterclockwise_hole_allows_concave_semantic_sections(self):
        mesh, _ = _cells({(x, y) for x in range(10) for y in range(10)
                          if x < 2 or x >= 8 or y < 2 or y >= 8})
        outline = next(loop for loop in EditorMesh2D().outlines_by_edge(mesh)
                       if area(np.vstack([mesh.nodes[edge[:-1], :2] for edge in loop])) > 0)
        plan, data = _plan(mesh, outline, .2)
        self.assertEqual(plan.diagnostics['sections'], {'INNER': 4, 'LINE': 4})
        self.assert_partition(plan, data)

    def test_nearest_anchor_tie_prefers_corner(self):
        mesh, outline = _cells({(x, y) for x in range(8) for y in range(8) if x >= 4 or y < 4})
        plan, _ = _plan(mesh, outline, 1.5)
        inner = next(section for section in plan.sections if section.kind == 'INNER')
        np.testing.assert_array_equal(mesh.nodes[inner.source_chains[0][0], :2], [3., 4.])
        np.testing.assert_array_equal(mesh.nodes[inner.source_chains[1][-1], :2], [4., 5.])

    def test_parallel_partial_pattern_snaps_front_beyond_h_within_ratio(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        plan, data = _plan(mesh, outline, .5, [[[1, 3.75], [2, 3.75]]], ratio=1.5)
        world_front = plan.front * data['scale'] + data['origin']
        self.assertAlmostEqual(world_front[:, 1].max(), 3.75)
        self.assertEqual(plan.diagnostics['pattern_chord_count'], 0)
        self.assert_partition(plan, data)

    def test_rotated_start_and_near_axis_coordinates(self):
        mesh, outline = _cells({(x, y) for x in range(4) for y in range(3)})
        mesh.nodes[:, 0] += mesh.nodes[:, 1] * 1e-5
        plan, data = _plan(mesh, outline[2:] + outline[:2], .5)
        self.assertEqual(plan.diagnostics['sections'], {'OUTER': 4, 'LINE': 4})
        self.assert_partition(plan, data)


if __name__ == '__main__':
    unittest.main()
