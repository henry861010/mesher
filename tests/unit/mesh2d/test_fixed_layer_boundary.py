"""The full-mesh connector consumes an exact, globally generated interface."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.adapter.layer import _mesher
from mesher.mesh2d.adapter.layer._construction import line
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh, generate_rectilinear_mesh_optimal
from tests.unit.mesh2d.test_extend_quad_layer import _polyomino
from tests.unit.mesh2d.test_slot_repair import _c_slot


def _outline_ids(outline):
    return np.concatenate([np.asarray(chain, dtype=np.int64)[:-1] for chain in outline])


def _global_target(mesh, outline, size, *, patterns=(), inward=False):
    plan = plan_quad_sections(mesh, outline, size, pattern_lines=patterns, expand_ratio=1.5)
    front = np.concatenate([section.outer_boundary[:-1] for section in plan.sections
                            if len(section.outer_boundary) > 1])
    if inward:
        grid = generate_rectilinear_mesh_optimal(size, front, pattern_lines=patterns)
        target = grid.mesh.nodes[grid.outer_outline, :2]
    else:
        lower, upper = front.min(axis=0) - 2 * size, front.max(axis=0) + 2 * size
        outer = np.array([lower, [upper[0], lower[1]], upper, [lower[0], upper[1]]])
        grid = generate_rectilinear_mesh_optimal(size, outer, holes=[front], pattern_lines=patterns)
        target = grid.mesh.nodes[grid.hole_outlines[0], :2]
    return target.copy(), plan


class FixedLayerBoundaryTests(unittest.TestCase):
    def assert_exact_interface(self, mesh, result, target):
        if np.array_equal(target[0], target[-1]):
            target = target[:-1]
        ids = _outline_ids(result.outline)
        actual = mesh.nodes[ids, :2]
        self.assertEqual(len(ids), len(target))
        self.assertEqual(len(set(map(int, ids))), len(ids))
        self.assertEqual(set(map(tuple, actual)), set(map(tuple, target)))

        def edges(points):
            return {frozenset((tuple(a), tuple(b)))
                    for a, b in zip(points, np.roll(points, -1, axis=0))}

        self.assertEqual(edges(actual), edges(target))

    def assert_atomic_failure(self, mesh, operation):
        nodes, elements = mesh.nodes, mesh.elements
        before_nodes, before_elements = nodes.copy(), elements.copy()
        with self.assertRaises(QuadLayerError) as caught:
            operation()
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, before_nodes)
        np.testing.assert_array_equal(mesh.elements, before_elements)
        return caught.exception

    def test_global_pattern_and_parity_nodes_are_consumed_without_resampling(self):
        mesh = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        outline = EditorMesh2D.outlines_by_edge(mesh)[0]
        patterns = [[[-5, 1.27], [-3, 1.27]]]
        target, plan = _global_target(mesh, outline, 2, patterns=patterns)
        self.assertGreater(len(target), sum(len(s.outer_boundary) - 1 for s in plan.sections))
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        original_target = target.copy()
        with patch.object(line, "sample_outer", side_effect=AssertionError("fixed target was sampled")):
            result = _mesher._generate_layer_adaptive_mesh(
                mesh, outline, 2, new_outline=target, pattern_lines=patterns,
                expand_ratio=1.5, fixed_outer_boundary=True)
        self.assertIs(result.mesh, mesh)
        self.assert_exact_interface(mesh, result, target)
        np.testing.assert_array_equal(mesh.nodes[:len(old_nodes)], old_nodes)
        np.testing.assert_array_equal(mesh.elements[:len(old_elements)], old_elements)
        np.testing.assert_array_equal(target, original_target)

    def test_world_coordinates_winding_start_and_optional_closure_are_exact(self):
        source = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        source.nodes[:, :2] = source.nodes[:, :2] * .73 + [.137, -1.13]
        outline = EditorMesh2D.outlines_by_edge(source)[0]
        target, _ = _global_target(source, outline, 1.46)
        for reverse, shift, closed in ((False, 0, False), (True, 5, True), (False, 3, True)):
            with self.subTest(reverse=reverse, shift=shift, closed=closed):
                mesh = Mesh2D(source.nodes, source.elements)
                supplied = np.roll(target[::-1] if reverse else target, shift, axis=0)
                if closed:
                    supplied = np.vstack((supplied, supplied[0]))
                result = _mesher._generate_layer_adaptive_mesh(
                    mesh, outline, 1.46, new_outline=supplied, fixed_outer_boundary=True)
                self.assert_exact_interface(mesh, result, supplied)

    def test_inward_final_layer_matches_center_fill_after_decreasing_size(self):
        cells = {(x, y) for x in range(12) for y in range(12)
                 if x < 4 or x >= 9 or y < 4 or y >= 9}
        mesh, _, lookup = _polyomino(cells)
        outline = [[lookup[x, 4] for x in range(4, 10)],
                   [lookup[9, y] for y in range(4, 10)],
                   [lookup[x, 9] for x in range(9, 3, -1)],
                   [lookup[4, y] for y in range(9, 3, -1)]]
        outline = generate_layer_adaptive_mesh(mesh, outline, .4).outline
        patterns = [[[5, 5.27], [7, 5.27]]]
        target, _ = _global_target(mesh, outline, .2, patterns=patterns, inward=True)
        result = _mesher._generate_layer_adaptive_mesh(
            mesh, outline, .2, new_outline=target, pattern_lines=patterns,
            fixed_outer_boundary=True)
        self.assert_exact_interface(mesh, result, target)

    def test_repaired_source_and_fixed_global_target_share_one_transaction(self):
        mesh, outline = _c_slot()
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        target, _ = _global_target(mesh, outline, .5)
        result = _mesher._generate_layer_adaptive_mesh(
            mesh, outline, .5, new_outline=target, fixed_outer_boundary=True)
        self.assertTrue(result.diagnostics["slot_repairs"])
        self.assert_exact_interface(mesh, result, target)
        np.testing.assert_array_equal(mesh.nodes[:len(old_nodes)], old_nodes)
        np.testing.assert_array_equal(mesh.elements[:len(old_elements)], old_elements)

    def test_fixed_budget_uses_supplied_counts_instead_of_element_size(self):
        source = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        outline = EditorMesh2D.outlines_by_edge(source)[0]
        target, _ = _global_target(source, outline, 2)
        baseline = Mesh2D(source.nodes, source.elements)
        count = _mesher._generate_layer_adaptive_mesh(
            baseline, outline, 2, new_outline=target,
            repair_slots=False, fixed_outer_boundary=True).diagnostics["added_elements"]
        mesh = Mesh2D(source.nodes, source.elements)
        result = _mesher._generate_layer_adaptive_mesh(
            mesh, outline, 1e-6, new_outline=target, repair_slots=False,
            max_added_elements=count, fixed_outer_boundary=True)
        self.assertEqual(result.diagnostics["added_elements"], count)
        self.assert_exact_interface(mesh, result, target)
        rejected = Mesh2D(source.nodes, source.elements)
        error = self.assert_atomic_failure(rejected, lambda: _mesher._generate_layer_adaptive_mesh(
            rejected, outline, 1e-6, new_outline=target, repair_slots=False,
            max_added_elements=count - 1, fixed_outer_boundary=True))
        self.assertEqual(error.code, "element_limit_exceeded")

    def test_missing_section_contacts_and_odd_target_are_atomic(self):
        source = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        outline = EditorMesh2D.outlines_by_edge(source)[0]
        target, plan = _global_target(source, outline, 2)
        # Remove a straight-edge point, retaining the geometry and making parity odd.
        index = next(i for i in range(len(target))
                     if np.count_nonzero(target[i - 1] != target[(i + 1) % len(target)]) == 1)
        for supplied in (plan.front, np.delete(target, index, axis=0)):
            mesh = Mesh2D(source.nodes, source.elements)
            error = self.assert_atomic_failure(mesh, lambda: _mesher._generate_layer_adaptive_mesh(
                mesh, outline, 2, new_outline=supplied, fixed_outer_boundary=True))
            self.assertEqual(error.context["stage"], "fixed_boundary_plan")

    def test_missing_or_reordered_constructed_boundary_is_rejected(self):
        source = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        outline = EditorMesh2D.outlines_by_edge(source)[0]
        target, _ = _global_target(source, outline, 2)
        original_fill = _mesher.fill_sections
        for mode in ("missing", "reordered"):
            with self.subTest(mode=mode):
                mesh = Mesh2D(source.nodes, source.elements)

                def damaged(*args, **kwargs):
                    nodes, quads, boundary, diagnostics = original_fill(*args, **kwargs)
                    boundary = [list(chain) for chain in boundary]
                    chain = next(chain for chain in boundary if len(chain) >= 4)
                    if mode == "missing":
                        del chain[1]
                    else:
                        chain[1], chain[2] = chain[2], chain[1]
                    return nodes, quads, boundary, diagnostics

                with patch.object(_mesher, "fill_sections", side_effect=damaged):
                    error = self.assert_atomic_failure(mesh, lambda: _mesher._generate_layer_adaptive_mesh(
                        mesh, outline, 2, new_outline=target, fixed_outer_boundary=True))
                self.assertEqual(error.context["stage"], "fixed_boundary_validation")

    def test_canonical_world_coordinates_are_revalidated_before_commit(self):
        mesh = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        outline = EditorMesh2D.outlines_by_edge(mesh)[0]
        target, _ = _global_target(mesh, outline, 2)
        original_validator = _mesher._validate_patch
        calls = []

        def validate(*args):
            calls.append(args[1].copy())
            if len(calls) == 2:
                raise QuadLayerError("injected restored-coordinate validation failure")
            return original_validator(*args)

        with patch.object(_mesher, "_validate_patch", side_effect=validate):
            error = self.assert_atomic_failure(mesh, lambda: _mesher._generate_layer_adaptive_mesh(
                mesh, outline, 2, new_outline=target, fixed_outer_boundary=True))
        self.assertEqual(len(calls), 2)
        self.assertEqual(error.context["stage"], "fixed_boundary_world_validation")

    def test_private_mode_requires_an_explicit_target(self):
        mesh = generate_rectilinear_mesh(1, [0, 4], [0, 4])
        outline = EditorMesh2D.outlines_by_edge(mesh)[0]
        error = self.assert_atomic_failure(mesh, lambda: _mesher._generate_layer_adaptive_mesh(
            mesh, outline, 2, fixed_outer_boundary=True))
        self.assertEqual(error.context["parameter"], "new_outline")
        with self.assertRaises(TypeError):
            generate_layer_adaptive_mesh(mesh, outline, 2, fixed_outer_boundary=True)


if __name__ == "__main__":
    unittest.main()
