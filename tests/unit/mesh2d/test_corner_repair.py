"""An isolated step corner is filled before the normal expansion transaction."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, extend_quad_layer, plan_quad_sections
from mesher.mesh2d.adapter.layer import _mesher, _repair
from mesher.mesh2d.editor import EditorMesh2D
from tests.unit.mesh2d import test_extend_quad_layer as layer_checks
from tests.unit.mesh2d.test_adaptive_slot_repair import _test2
from tests.unit.mesh2d.test_slot_repair import _blocks


def _step(width=1., height=2., *, nonuniform=False):
    """A 3 by 3 block with the upper-left width by height rectangle missing."""
    if nonuniform:
        left = np.array([0., .2, .5, 1.]) * width
        bottom = np.array([0., .25, .6, 1.])
        upper = 1. + np.array([0., .15, .4, .65, 1.]) * height
        right = np.linspace(width, width + 2., 9)
    else:
        left = np.linspace(0., width, round(width / .25) + 1)
        bottom = np.linspace(0., 1., 5)
        upper = np.linspace(1., 1. + height, round(height / .25) + 1)
        right = np.linspace(width, width + 2., 9)
    return _blocks([(left, bottom), (right, np.r_[bottom, upper[1:]])])


class CornerRepairTests(unittest.TestCase):
    assert_layer = layer_checks.ExtendQuadLayerTests.assert_layer
    assert_pattern_covered = layer_checks.ExtendQuadLayerTests.assert_pattern_covered
    assert_unchanged = layer_checks.ExtendQuadLayerTests.assert_unchanged
    _snapshot = staticmethod(layer_checks.ExtendQuadLayerTests._snapshot)

    def run_corner(self, mesh, outline, size=1., **kwargs):
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        original_outline = [np.asarray(chain).copy() for chain in outline]
        result = extend_quad_layer(mesh, outline, size, **kwargs)
        self.assert_layer(mesh, original_outline, result, old_nodes, old_elements)
        for before, after in zip(original_outline, outline):
            np.testing.assert_array_equal(after, before)
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics['added_nodes'],
                         diagnostics['repair_added_nodes'] + diagnostics['expand_added_nodes'])
        self.assertEqual(diagnostics['added_elements'],
                         diagnostics['repair_added_elements'] + diagnostics['expand_added_elements'])
        repairs = diagnostics['corner_repairs']
        self.assertEqual(len(repairs), 1)
        repair = repairs[0]
        self.assertEqual(len(repair['source_edges']), 2)
        for edge, chain in zip(repair['source_edges'], repair['source_chains']):
            np.testing.assert_array_equal(chain, original_outline[edge])
        self.assertGreater(repair['element_end'], repair['element_start'])
        self.assertGreaterEqual(repair['element_start'], len(old_elements))
        self.assertLessEqual(repair['element_end'],
                             len(old_elements) + diagnostics['repair_added_elements'])
        return result

    def test_below_threshold_retains_legacy_path_and_empty_diagnostics(self):
        mesh, outline = _step()
        other = Mesh2D(mesh.nodes.copy(), mesh.elements.copy())
        preview = plan_quad_sections(mesh, outline, .75)
        self.assertIsNone(preview.prepared_source)
        self.assertEqual(preview.diagnostics['corner_repairs'], [])
        result = extend_quad_layer(mesh, outline, .75)
        legacy = extend_quad_layer(other, outline, .75, repair_slots=False)
        self.assertEqual(result.diagnostics['corner_repairs'], [])
        self.assertEqual(legacy.diagnostics['corner_repairs'], [])
        np.testing.assert_array_equal(mesh.nodes, other.nodes)
        np.testing.assert_array_equal(mesh.elements, other.elements)

    def test_equal_and_exceeded_threshold_fill_whole_corner_first(self):
        for size in (1., 1.2):
            with self.subTest(size=size):
                mesh, outline = _step()
                result = self.run_corner(mesh, outline, size)
                self.assertEqual(result.diagnostics['slot_repairs'], [])
                self.assertEqual(len(result.outline), 4)
                repair = result.diagnostics['corner_repairs'][0]
                cells = mesh.nodes[mesh.elements[repair['element_start']:repair['element_end']], :2]
                self.assertAlmostEqual(sum(layer_checks._area(cell) for cell in cells), 2.)
                mouth = mesh.nodes[repair['mouth_node_ids'], :2]
                self.assertTrue(np.any(np.all(np.isclose(mouth, [0., 3.]), axis=1)))
                front = mesh.nodes[layer_checks._outline_nodes(result.outline), :2]
                np.testing.assert_allclose(front.min(axis=0), [-size, -size])
                np.testing.assert_allclose(front.max(axis=0), [3. + size, 3. + size])

    def test_detection_threshold_uses_tolerance_and_actual_supports(self):
        # Inspect detection directly at the threshold: the existing section
        # partitioner need not support a vanishing unfilled corner below it.
        for shear, threshold in ((0., 1.), (-.0005, .999)):
            mesh, outline = _step()
            mesh.nodes[:, 0] += shear * mesh.nodes[:, 1]
            baseline = _mesher._prepare(mesh, outline, threshold, (), 1., 1e-3)
            epsilon = baseline['tol'] * baseline['scale']
            for delta, count in ((-1e-8, 0), (-epsilon / 2, 1), (0., 1), (1e-8, 1)):
                with self.subTest(shear=shear, delta=delta):
                    data = _mesher._prepare(mesh, outline, threshold + delta, (), 1., 1e-3)
                    _mesher._expansion_distances(data)
                    self.assertEqual(len(_repair._find_corners(data)), count)

    def test_either_incident_wall_can_reach_the_cap(self):
        for width, height in ((1., 2.), (2., 1.)):
            with self.subTest(width=width, height=height):
                mesh, outline = _step(width, height)
                result = self.run_corner(mesh, outline)
                repair = result.diagnostics['corner_repairs'][0]
                mouth = mesh.nodes[repair['mouth_node_ids'], :2]
                self.assertTrue(np.any(np.all(np.isclose(mouth, [0., 1. + height]), axis=1)))

    def test_all_cardinal_rotations_and_outline_starts_preserve_fixed_segments(self):
        for rotation in range(4):
            for shift in (0, 2, 5):
                with self.subTest(rotation=rotation, shift=shift):
                    mesh, outline = _step(nonuniform=True)
                    for _ in range(rotation):
                        mesh.nodes[:, :2] = mesh.nodes[:, [1, 0]] * [-1, 1]
                    outline = outline[shift:] + outline[:shift]
                    result = self.run_corner(mesh, outline)
                    self.assertEqual(len(result.outline), 4)
                    self.assertTrue(result.diagnostics['edge_provenance'])

    def test_near_axis_source_uses_semantic_cap_without_moving_old_nodes(self):
        mesh, outline = _step(nonuniform=True)
        mesh.nodes[:, 0] += .0005 * mesh.nodes[:, 1]
        # The left endpoint is (.0005, 1), and the upper endpoint is
        # (1.0015, 3). Their cardinal projection intersection is (.0005, 3).
        result = self.run_corner(mesh, outline, 1.01)
        repair = result.diagnostics['corner_repairs'][0]
        mouth = mesh.nodes[repair['mouth_node_ids'], :2]
        self.assertTrue(np.any(np.all(np.isclose(mouth, [.0005, 3.], atol=1e-10), axis=1)))

    def test_pattern_snapping_can_trigger_either_wall_before_expansion(self):
        cases = (
            (1., 2., [[0., 1.], [0., 3.75]]),
            (2., 1., [[-.75, 2.], [2., 2.]]),
        )
        for width, height, pattern in cases:
            with self.subTest(width=width, height=height):
                mesh, outline = _step(width, height)
                result = self.run_corner(mesh, outline, .75, expand_ratio=1.5,
                                         pattern_lines=[pattern])
                self.assert_pattern_covered(mesh, pattern)
                np.testing.assert_allclose(result.diagnostics['edge_expansion'], .75)

    def test_patterns_cross_both_cap_arms_and_keep_the_bend(self):
        # An interior guide shortens the parallel wall's expansion. Keep the
        # other wall at its full threshold so the original cap still triggers.
        for pattern, size in (([[.5, 1.], [.5, 5.]], 2.),
                              ([[-1., 2.], [1., 2.]], 1.)):
            with self.subTest(pattern=pattern):
                mesh, outline = _step()
                result = self.run_corner(mesh, outline, size, pattern_lines=[pattern])
                self.assert_pattern_covered(mesh, pattern)
                mouth = mesh.nodes[result.diagnostics['corner_repairs'][0]['mouth_node_ids'], :2]
                self.assertTrue(np.any(np.all(np.isclose(mouth, [0., 3.]), axis=1)))

    def test_preview_is_readonly_and_matches_committed_repair(self):
        mesh, outline = _step()
        before = self._snapshot(mesh)
        preview = plan_quad_sections(mesh, outline, 1., repair_max_element_size=.5)
        self.assert_unchanged(mesh, before)
        source = preview.prepared_source
        self.assertIsNotNone(source)
        self.assertEqual(source.original_node_count, mesh.node_count)
        self.assertEqual(source.original_element_count, mesh.element_count)
        self.assertEqual(len(preview.diagnostics['corner_repairs']), 1)
        for array in (source.nodes, source.elements):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        result = self.run_corner(mesh, outline, repair_max_element_size=.5)
        np.testing.assert_array_equal(mesh.nodes[:len(source.nodes)], source.nodes)
        np.testing.assert_array_equal(mesh.elements[:len(source.elements)], source.elements)
        self.assertEqual(result.diagnostics['corner_repairs'], preview.diagnostics['corner_repairs'])

    def test_repair_cap_and_world_coordinate_diagnostics(self):
        for scale, translation in ((1e-3, [100., -20.]), (1e3, [-4e5, 2e5])):
            with self.subTest(scale=scale):
                mesh, outline = _step()
                mesh.nodes[:, :2] = mesh.nodes[:, :2] * scale + translation
                preview = plan_quad_sections(mesh, outline, scale,
                                             repair_max_element_size=.5 * scale)
                repair = preview.diagnostics['corner_repairs'][0]
                self.assertAlmostEqual(repair['boundary_element_size'] / scale, .25)
                self.assertAlmostEqual(repair['max_element_size'] / scale, .5)
                mouth = np.asarray(repair['mouth_points'])
                np.testing.assert_allclose(mouth, preview.prepared_source.nodes[repair['mouth_node_ids'], :2])
                normalized = (mouth - translation) / scale
                self.assertTrue(np.any(np.all(np.isclose(normalized, [0., 3.]), axis=1)))
                polygon = (np.asarray(repair['polygon']) - translation) / scale
                self.assertAlmostEqual(layer_checks._area(polygon), 2., places=7)
                for face in repair['faces']:
                    self.assertTrue(all(0. < size <= .5 * scale * (1. + 1e-8)
                                        for size in face['ring_target_sizes']))

    def test_shared_budget_is_exact_and_failures_are_atomic(self):
        mesh, outline = _step()
        result = self.run_corner(mesh, outline)
        total = result.diagnostics['added_elements']
        repair_count = result.diagnostics['repair_added_elements']
        mesh, outline = _step()
        exact = extend_quad_layer(mesh, outline, 1., max_added_elements=total)
        self.assertEqual(exact.diagnostics['added_elements'], total)
        for limit in (total - 1, repair_count - 1):
            with self.subTest(limit=limit):
                mesh, outline = _step()
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    extend_quad_layer(mesh, outline, 1., max_added_elements=limit)
                self.assertEqual(caught.exception.code, 'element_limit_exceeded')
                self.assertGreater(caught.exception.context['required_elements'], limit)
                self.assert_unchanged(mesh, before)

    def test_cap_below_fixed_boundary_spacing_is_atomic(self):
        for api in (plan_quad_sections, extend_quad_layer):
            with self.subTest(api=api.__name__):
                mesh, outline = _step()
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    api(mesh, outline, 1., repair_max_element_size=.1)
                self.assert_unchanged(mesh, before)

    def test_opt_out_retains_original_shape_and_empty_corner_diagnostics(self):
        mesh, outline = _step()
        preview = plan_quad_sections(mesh, outline, 1., repair_slots=False)
        self.assertIsNone(preview.prepared_source)
        self.assertEqual(preview.diagnostics['corner_repairs'], [])
        result = extend_quad_layer(mesh, outline, 1., repair_slots=False)
        self.assertEqual(result.diagnostics['corner_repairs'], [])
        self.assertEqual(result.diagnostics['repair_added_elements'], 0)

    def test_post_repair_failure_rolls_back_and_detection_runs_once(self):
        for api in (plan_quad_sections, extend_quad_layer):
            with self.subTest(api=api.__name__):
                mesh, outline = _step()
                before = self._snapshot(mesh)

                def fail_front(data):
                    self.assertIn('_original_data', data)
                    raise QuadLayerError('injected expansion failure', code='topology_change')

                with patch.object(_repair, 'repair_slots', wraps=_repair.repair_slots) as repairs:
                    with patch.object(_mesher, '_front', side_effect=fail_front) as fronts:
                        with self.assertRaises(QuadLayerError) as caught:
                            api(mesh, outline, 1.)
                self.assertEqual(caught.exception.context.get('stage'), 'expansion')
                self.assertEqual(repairs.call_count, 1)
                self.assertEqual(fronts.call_count, 1)
                self.assert_unchanged(mesh, before)

    def test_touching_corner_repairs_fail_the_entire_transaction(self):
        # The two missing rectangles meet at the original convex lip (1, 2).
        cells = {(x, y) for x in range(3) for y in range(x + 1)}
        for api in (plan_quad_sections, extend_quad_layer):
            with self.subTest(api=api.__name__):
                mesh, outline, _ = layer_checks._polyomino(cells)
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    api(mesh, outline, 1.)
                self.assertEqual(caught.exception.context.get('reason'), 'interacting_repairs')
                source_edges = caught.exception.context['source_edges']
                self.assertEqual(len(source_edges), 2)
                self.assertTrue(all(len(pair) == 2 for pair in source_edges))
                self.assertNotEqual(tuple(source_edges[0]), tuple(source_edges[1]))
                self.assert_unchanged(mesh, before)

    def test_cap_cannot_touch_an_isolated_original_node(self):
        for point in ((0., 3., 0.), (0., 2., 0.), (.5, 3., 0.)):
            with self.subTest(point=point):
                mesh, outline = _step()
                mesh.replace_data(nodes=np.vstack((mesh.nodes, point)), elements=mesh.elements)
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    extend_quad_layer(mesh, outline, 1.)
                self.assert_unchanged(mesh, before)

    def test_cap_cannot_cross_or_contain_another_component(self):
        for rectangle in ((.2, .8, 1.5, 2.5), (-.2, .2, 2., 2.5)):
            with self.subTest(rectangle=rectangle):
                mesh, outline = _step()
                left, right, bottom, top = rectangle
                offset = mesh.node_count
                island = [[left, bottom, 0.], [right, bottom, 0.],
                          [right, top, 0.], [left, top, 0.]]
                mesh.replace_data(nodes=np.vstack((mesh.nodes, island)),
                                  elements=np.vstack((mesh.elements, np.arange(offset, offset + 4))))
                before = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    extend_quad_layer(mesh, outline, 1.)
                self.assert_unchanged(mesh, before)

    def test_four_independent_corners_are_filled_in_one_transaction(self):
        cells = {(x, y) for x in range(3) for y in range(3) if x == 1 or y == 1}
        mesh, outline, _ = layer_checks._polyomino(cells)
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        result = extend_quad_layer(mesh, outline, 1.)
        self.assert_layer(mesh, outline, result, old_nodes, old_elements)
        self.assertEqual(len(result.outline), 4)
        self.assertEqual(result.diagnostics['slot_repairs'], [])
        repairs = result.diagnostics['corner_repairs']
        self.assertEqual(len(repairs), 4)
        repair_ranges = [set(range(repair['element_start'], repair['element_end']))
                         for repair in repairs]
        self.assertEqual(sum(map(len, repair_ranges)), len(set.union(*repair_ranges)))
        self.assertEqual(sum(map(len, repair_ranges)), result.diagnostics['repair_added_elements'])
        front = mesh.nodes[layer_checks._outline_nodes(result.outline), :2]
        np.testing.assert_allclose(front.min(axis=0), [-1., -1.])
        np.testing.assert_allclose(front.max(axis=0), [4., 4.])

    def test_test2_two_layers_repair_slot_and_corner_in_one_transaction(self):
        mesh, outline = _test2()
        first = extend_quad_layer(mesh, outline, 30., expand_ratio=1.5,
                                  repair_max_element_size=20.)
        self.assertEqual(first.diagnostics['corner_repairs'], [])
        outline = EditorMesh2D.outlines_by_edge(mesh)[0]
        before = self._snapshot(mesh)
        preview = plan_quad_sections(mesh, outline, 100., expand_ratio=1.5)
        self.assert_unchanged(mesh, before)
        self.assertEqual(len(preview.diagnostics['slot_repairs']), 1)
        self.assertEqual(len(preview.diagnostics['corner_repairs']), 1)
        result = self.run_corner(mesh, outline, 100., expand_ratio=1.5)
        self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
        source = preview.prepared_source
        np.testing.assert_array_equal(mesh.nodes[:len(source.nodes)], source.nodes)
        np.testing.assert_array_equal(mesh.elements[:len(source.elements)], source.elements)
        repairs = result.diagnostics['slot_repairs'] + result.diagnostics['corner_repairs']
        self.assertEqual(sum(repair['element_end'] - repair['element_start'] for repair in repairs),
                         result.diagnostics['repair_added_elements'])


if __name__ == '__main__':
    unittest.main()
