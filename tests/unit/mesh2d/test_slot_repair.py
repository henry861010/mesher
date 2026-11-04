"""Slot repair preserves the old mesh and validates both construction phases."""
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, extend_quad_layer, plan_quad_sections
from mesher.mesh2d.editor import EditorMesh2D
from tests.unit.mesh2d import test_extend_quad_layer as layer_checks


def _blocks(blocks):
    """Assemble conforming rectangles independently of the adapter under test."""
    coordinates, elements, lookup = [], [], {}
    for xs, ys in blocks:
        for left, right in zip(xs[:-1], xs[1:]):
            for bottom, top in zip(ys[:-1], ys[1:]):
                cell = []
                for point in ((left, bottom), (right, bottom), (right, top), (left, top)):
                    if point not in lookup:
                        lookup[point] = len(coordinates)
                        coordinates.append(point)
                    cell.append(lookup[point])
                elements.append(cell)
    mesh = Mesh2D(coordinates, np.asarray(elements, dtype=np.int32))
    return mesh, EditorMesh2D.outlines_by_edge(mesh)[0]


def _c_slot(bottom_end=3., *, bottom_breaks=None):
    bottom_x = ([1., 1.5, 2., 2.5, 3.] if bottom_end == 3.
                else [1., 1.5, bottom_end])
    if bottom_breaks is not None:
        bottom_x = bottom_breaks
    return _blocks([
        ([0., .5, 1.], np.arange(0., 3.01, .5)),
        (bottom_x, [0., .5, 1.]),
        ([1., 1.5, 2., 2.5, 3.], [2., 2.5, 3.]),
    ])


class SlotRepairTests(unittest.TestCase):
    # Reuse the independent area, incidence, and convex clipping assertions,
    # without inheriting the unrelated test methods.
    assert_layer = layer_checks.ExtendQuadLayerTests.assert_layer
    assert_pattern_covered = layer_checks.ExtendQuadLayerTests.assert_pattern_covered
    assert_unchanged = layer_checks.ExtendQuadLayerTests.assert_unchanged
    _snapshot = staticmethod(layer_checks.ExtendQuadLayerTests._snapshot)

    def run_repair(self, mesh, outline, h=.5, **kwargs):
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        original_outline = [np.asarray(chain).copy() for chain in outline]
        result = extend_quad_layer(mesh, outline, h, **kwargs)
        self.assert_layer(mesh, original_outline, result, old_nodes, old_elements)
        for old, actual in zip(original_outline, outline):
            np.testing.assert_array_equal(actual, old)
        for edge in result.outline:
            points = mesh.nodes[edge, :2]
            self.assertTrue(np.any(np.ptp(points, axis=0) < 1e-10))
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics['added_elements'],
                         diagnostics['repair_added_elements'] + diagnostics['expand_added_elements'])
        self.assertEqual(diagnostics['added_nodes'],
                         diagnostics['repair_added_nodes'] + diagnostics['expand_added_nodes'])
        for repair in diagnostics['slot_repairs']:
            self.assertGreater(repair['element_end'], repair['element_start'])
            self.assertGreaterEqual(repair['element_start'], len(old_elements))
            self.assertLessEqual(repair['element_end'],
                                 len(old_elements) + diagnostics['repair_added_elements'])
            for record in [*repair['units'], *repair['faces']]:
                self.assertGreaterEqual(record['element_start'], repair['element_start'])
                self.assertLessEqual(record['element_end'], repair['element_end'])
        for unit in diagnostics['units']:
            self.assertGreaterEqual(unit['element_start'],
                                    len(old_elements) + diagnostics['repair_added_elements'])
        # Independently recompute quality over both repair and expansion cells.
        points = mesh.nodes[mesh.elements[len(old_elements):], :2]
        edges = np.roll(points, -1, axis=1) - points
        lengths = np.linalg.norm(edges, axis=2)
        following = np.roll(edges, -1, axis=1)
        scaled_jacobian = ((edges[..., 0] * following[..., 1]
                            - edges[..., 1] * following[..., 0])
                           / (lengths * np.roll(lengths, -1, axis=1)))
        quality = diagnostics['quality']
        self.assertAlmostEqual(quality['max_aspect_ratio'],
                               np.max(lengths.max(axis=1) / lengths.min(axis=1)))
        self.assertAlmostEqual(quality['min_scaled_jacobian'], scaled_jacobian.min())
        return result

    def test_exact_and_exceeded_closure_fill_to_original_mouth(self):
        for size in (.5, .6):
            with self.subTest(element_size=size):
                mesh, outline = _c_slot()
                result = self.run_repair(mesh, outline, size)
                self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
                self.assertEqual(len(result.outline), 4)
                mouth = np.asarray(result.diagnostics['slot_repairs'][0]['mouth_points'])
                np.testing.assert_allclose(mouth[:, 0], 3.)
                self.assertAlmostEqual(mouth[:, 1].min(), 1.)
                self.assertAlmostEqual(mouth[:, 1].max(), 2.)

    def test_open_slot_uses_unchanged_path(self):
        def refined_slot():
            return _blocks([
                (np.arange(0., 1.01, .25), np.arange(0., 3.01, .25)),
                (np.arange(1., 3.01, .25), np.arange(0., 1.01, .25)),
                (np.arange(1., 3.01, .25), np.arange(2., 3.01, .25)),
            ])

        mesh, outline = refined_slot()
        other = Mesh2D(mesh.nodes.copy(), mesh.elements.copy())
        result = extend_quad_layer(mesh, outline, .2)
        legacy = extend_quad_layer(other, outline, .2, repair_slots=False)
        np.testing.assert_array_equal(result.mesh.nodes, legacy.mesh.nodes)
        np.testing.assert_array_equal(result.mesh.elements, legacy.mesh.elements)
        self.assertFalse(result.diagnostics.get('slot_repairs'))
        plan_mesh, plan_outline = refined_slot()
        self.assertIsNone(plan_quad_sections(plan_mesh, plan_outline, .2).prepared_source)

    def test_misaligned_mouth_and_unequal_counts_in_all_directions(self):
        for bottom_end in (2.25, 2.4):
            for rotation in range(4):
                with self.subTest(bottom_end=bottom_end, rotation=rotation):
                    mesh, outline = _c_slot(bottom_end)
                    for _ in range(rotation):
                        mesh.nodes[:, :2] = mesh.nodes[:, [1, 0]] * [-1, 1]
                    # Exercise a different logical-edge start on each rotation.
                    shift = rotation % len(outline)
                    outline = outline[shift:] + outline[:shift]
                    result = self.run_repair(mesh, outline)
                    repair = result.diagnostics['slot_repairs'][0]
                    mouth = np.asarray(repair['mouth_points'])[:, :2]
                    self.assertTrue(np.all(np.ptp(mouth, axis=0) > 0))
                    self.assertTrue(result.diagnostics['edge_provenance'])

    def test_preview_contains_readonly_prepared_source_without_mutation(self):
        mesh, outline = _c_slot(2.25)
        snapshot = self._snapshot(mesh)
        plan = plan_quad_sections(mesh, outline, .5)
        self.assert_unchanged(mesh, snapshot)
        source = plan.prepared_source
        self.assertIsNotNone(source)
        self.assertEqual(source.original_node_count, mesh.node_count)
        self.assertEqual(source.original_element_count, mesh.element_count)
        self.assertGreater(len(source.elements), mesh.element_count)
        self.assertEqual(source.nodes.shape[1], 3)
        self.assertFalse(source.nodes.flags.writeable)
        self.assertFalse(source.elements.flags.writeable)
        self.assertIsInstance(source.outline, tuple)
        self.assertTrue(all(isinstance(chain, tuple) for chain in source.outline))
        for section in plan.sections:
            for chain in section.source_chains:
                self.assertTrue(all(0 <= node < len(source.nodes) for node in chain))
        with self.assertRaises(ValueError):
            source.nodes[0, 0] = 100.
        with self.assertRaises(ValueError):
            source.elements[0, 0] = 100
        for array in (source.nodes, source.elements):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            source.original_node_count = 0
        result = self.run_repair(mesh, outline)
        np.testing.assert_array_equal(mesh.nodes[:len(source.nodes)], source.nodes)
        np.testing.assert_array_equal(mesh.elements[:len(source.elements)], source.elements)
        self.assertEqual(result.diagnostics['repair_added_elements'],
                         len(source.elements) - source.original_element_count)

    def test_snapshot_and_repair_diagnostics_use_world_coordinates(self):
        for scale, translation in ((1e-3, [100., -20.]), (1e3, [-4e5, 2e5])):
            with self.subTest(scale=scale):
                mesh, outline = _c_slot(2.25)
                mesh.nodes[:, :2] = mesh.nodes[:, :2] * scale + translation
                before = mesh.nodes.copy()
                plan = plan_quad_sections(mesh, outline, .5 * scale)
                source = plan.prepared_source
                np.testing.assert_array_equal(source.nodes[:len(before)], before)
                result = self.run_repair(mesh, outline, .5 * scale)
                repair = result.diagnostics['slot_repairs'][0]
                mouth_ids = repair['mouth_node_ids']
                np.testing.assert_allclose(repair['mouth_points'], mesh.nodes[mouth_ids, :2])
                np.testing.assert_allclose(source.nodes, mesh.nodes[:len(source.nodes)])
                mouth_local = (np.asarray(repair['mouth_points']) - translation) / scale
                self.assertAlmostEqual(mouth_local[:, 1].min(), 1., places=7)
                self.assertAlmostEqual(mouth_local[:, 1].max(), 2., places=7)
                self.assertTrue(np.all(np.asarray(repair['polygon'])[:, 0] > translation[0]))

    def test_closing_slot_opt_out_and_original_xy_validation_are_atomic(self):
        for preview in (False, True):
            for distorted in (False, True):
                with self.subTest(preview=preview, distorted=distorted):
                    mesh, outline = _c_slot()
                    if distorted:
                        mesh.nodes[:, 0] += .02 * mesh.nodes[:, 1]
                    snapshot = self._snapshot(mesh)
                    call = plan_quad_sections if preview else extend_quad_layer
                    with self.assertRaises(QuadLayerError):
                        call(mesh, outline, .5, repair_slots=distorted)
                    self.assert_unchanged(mesh, snapshot)

    def test_near_axis_walls_are_filled_when_the_planned_supports_close(self):
        mesh, outline = _c_slot()
        # Pointwise widths vary, but the actual cardinal front supports
        # overlap. Repair must use that same geometry and cover the full slot.
        upper = mesh.nodes[:, 1] >= 2.
        mesh.nodes[upper, 1] += .0005 * (mesh.nodes[upper, 0] - 2.)
        snapshot = self._snapshot(mesh)
        preview = plan_quad_sections(mesh, outline, .5)
        self.assert_unchanged(mesh, snapshot)
        result = self.run_repair(mesh, outline, .5)
        self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
        np.testing.assert_array_equal(mesh.nodes[:len(preview.prepared_source.nodes)],
                                      preview.prepared_source.nodes)
        np.testing.assert_array_equal(mesh.elements[:len(preview.prepared_source.elements)],
                                      preview.prepared_source.elements)

    def test_supported_repair_patterns_have_complete_mesh_edge_coverage(self):
        cases = (
            [[[1., 1.], [3., 1.]]],                  # Existing slot wall.
            [[[2., 1.], [2., 2.]]],                  # Opposing walls.
            [[[1.5, 1.], [1.5, 2.]], [[2.5, 1.], [2.5, 2.]]],
            [[[1., 1.5], [3.5, 1.5]]],               # Through mouth into expansion.
            [[[2., 1.], [2., 2.]], [[2., 1.], [2., 2.]]],
        )
        for patterns in cases:
            with self.subTest(patterns=patterns):
                mesh, outline = _c_slot()
                result = self.run_repair(mesh, outline, pattern_lines=patterns)
                self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
                for pattern in patterns:
                    self.assert_pattern_covered(mesh, pattern)

    def test_pattern_adjusted_distances_trigger_closure_before_expansion(self):
        mesh, outline = _c_slot()
        # Default .4 + .4 leaves a .2 gap. This guide snaps both walls to .5.
        pattern = [[1., 1.5], [3.4, 1.5]]
        result = self.run_repair(mesh, outline, .4, expand_ratio=1.25,
                                 pattern_lines=[pattern])
        self.assertEqual(len(result.diagnostics['slot_repairs']), 1)
        self.assert_pattern_covered(mesh, pattern)
        # Former gap walls disappear; pattern snapping is recomputed for the
        # four remaining external logical edges, whose expansion is now .4.
        np.testing.assert_allclose(result.diagnostics['edge_expansion'], .4)

    def test_invalid_patterns_are_not_hidden_by_filling(self):
        cases = (
            ([[[1.25, 1.], [1.25, 2.]]], 'fixed_boundary_conflict'),
            ([[[1., 1.5], [2., 1.5]]], 'unsupported_pattern_layout'),
            ([[[1., 1.5], [3.5, 1.5]], [[2., 1.], [2., 2.]]], 'unsupported_pattern_layout'),
            ([[[1., 1.5], [3.5, 1.5]], [[2., 1.], [2., 1.5]]], 'unsupported_pattern_layout'),
            ([[[1.5, 1.25], [2., 1.25]], [[2., 1.25], [2., 1.75]],
              [[2., 1.75], [1.5, 1.75]], [[1.5, 1.75], [1.5, 1.25]]],
             None),
        )
        for patterns, code in cases:
            with self.subTest(patterns=patterns):
                mesh, outline = _c_slot()
                snapshot = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    extend_quad_layer(mesh, outline, .5, pattern_lines=patterns)
                if code is not None:
                    self.assertEqual(caught.exception.code, code)
                self.assert_unchanged(mesh, snapshot)

    def test_two_independent_slots_are_filled_in_one_transaction(self):
        # Right-facing comb, with two unit-width gaps and a thick middle arm.
        mesh, outline, _ = layer_checks._polyomino({(x, y) for x in range(6) for y in range(10)
                                      if x < 2 or y < 2 or 4 <= y < 6 or y >= 8})
        mesh.nodes[:, :2] *= .5
        result = self.run_repair(mesh, outline)
        self.assertEqual(len(result.diagnostics['slot_repairs']), 2)
        self.assertEqual(len(result.outline), 4)
        repair_total = result.diagnostics['repair_added_elements']
        mesh, outline, _ = layer_checks._polyomino({(x, y) for x in range(6) for y in range(10)
                                                   if x < 2 or y < 2 or 4 <= y < 6 or y >= 8})
        mesh.nodes[:, :2] *= .5
        snapshot = self._snapshot(mesh)
        with self.assertRaises(QuadLayerError) as caught:
            extend_quad_layer(mesh, outline, .5, max_added_elements=repair_total - 1)
        self.assertEqual(caught.exception.code, 'element_limit_exceeded')
        self.assertGreater(caught.exception.context['required_elements'], repair_total - 1)
        self.assert_unchanged(mesh, snapshot)

    def test_keyhole_and_other_component_collisions_are_atomic(self):
        cells = {(x, y) for x in range(6) for y in range(5)
                 if not (2 <= x < 4 and 1 <= y < 4 or x >= 4 and y == 2)}
        keyhole, outline, _ = layer_checks._polyomino(cells)
        cases = [(keyhole, outline)]
        mesh, outline = _c_slot()
        offset = mesh.node_count
        # An isolated Quad4 occupies the fill region; joining it is unsupported.
        mesh.replace_data(nodes=np.vstack((mesh.nodes, [[1.7, 1.2, 0], [2.2, 1.2, 0],
                                                        [2.2, 1.8, 0], [1.7, 1.8, 0]])),
                          elements=np.vstack((mesh.elements, np.arange(offset, offset + 4))))
        cases.append((mesh, outline))
        for mesh, source in cases:
            with self.subTest(nodes=mesh.node_count):
                snapshot = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError):
                    extend_quad_layer(mesh, source, .5)
                self.assert_unchanged(mesh, snapshot)

    def test_rebuild_retains_selected_loop_when_another_component_sorts_first(self):
        mesh, outline = _c_slot()
        offset = mesh.node_count
        mesh.replace_data(nodes=np.vstack((mesh.nodes, [[-10., -10., 0], [-9., -10., 0],
                                                        [-9., -9., 0], [-10., -9., 0]])),
                          elements=np.vstack((mesh.elements, np.arange(offset, offset + 4))))
        result = self.run_repair(mesh, outline)
        front = mesh.nodes[[node for chain in result.outline for node in chain], :2]
        np.testing.assert_allclose(front.min(axis=0), [-.5, -.5])
        np.testing.assert_allclose(front.max(axis=0), [3.5, 3.5])

    def test_original_mesh_receives_exactly_one_commit(self):
        mesh, outline = _c_slot()
        original_replace = Mesh2D.replace_data
        commits = []

        def record(candidate, *args, **kwargs):
            if candidate is mesh:
                commits.append(True)
            return original_replace(candidate, *args, **kwargs)

        with patch.object(Mesh2D, 'replace_data', record):
            self.run_repair(mesh, outline)
        self.assertEqual(len(commits), 1)

    def test_budget_counts_both_phases_and_exact_cap_is_accepted(self):
        mesh, outline = _c_slot()
        result = extend_quad_layer(mesh, outline, .5)
        total = result.diagnostics['added_elements']
        repair_count = result.diagnostics['repair_added_elements']
        self.assertGreater(repair_count, 0)
        self.assertGreater(total, repair_count)
        mesh, outline = _c_slot()
        exact = self.run_repair(mesh, outline, max_added_elements=total)
        self.assertEqual(exact.diagnostics['added_elements'], total)
        for limit in (total - 1, repair_count, 1):
            with self.subTest(limit=limit):
                mesh, outline = _c_slot()
                snapshot = self._snapshot(mesh)
                with self.assertRaises(QuadLayerError) as caught:
                    extend_quad_layer(mesh, outline, .5, max_added_elements=limit)
                self.assertEqual(caught.exception.code, 'element_limit_exceeded')
                self.assertGreater(caught.exception.context['required_elements'], limit)
                self.assert_unchanged(mesh, snapshot)

    def test_failure_after_repair_never_commits_original_mesh(self):
        from mesher.mesh2d.adapter.layer import _mesher

        for name in ('fill_sections', '_validate_patch'):
            with self.subTest(stage=name):
                mesh, outline = _c_slot()
                snapshot = self._snapshot(mesh)
                with patch.object(_mesher, name, side_effect=ValueError('injected late failure')):
                    with self.assertRaisesRegex(ValueError, 'injected late failure'):
                        extend_quad_layer(mesh, outline, .5)
                self.assert_unchanged(mesh, snapshot)


if __name__ == '__main__':
    unittest.main()
