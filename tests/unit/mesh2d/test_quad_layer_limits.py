"""Element limits guard layer output without charging exploratory trials."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, extend_quad_layer
from mesher.mesh2d.adapter.layer import _corners, _line, _section_mesh
from mesher.mesh2d.adapter.layer._limits import (
    ElementLimitExceeded,
    outer_segment_counts,
    preflight_element_limit,
    section_element_lower_bound,
)


def _rectangle():
    mesh = Mesh2D([[0, 0], [4, 0], [4, 3], [0, 3]],
                  np.asarray([[0, 1, 2, 3]], dtype=np.int64))
    return mesh, [[0, 3], [3, 2], [2, 1], [1, 0]]


def _line_fixture():
    source = np.column_stack((np.arange(9), np.zeros(9), np.zeros(9)))
    outer = np.asarray([[0., 4, 0], [8., 4, 0]])
    return source, outer, np.asarray((source[0], outer[0])), np.asarray((source[-1], outer[-1]))


class QuadLayerLimitTests(unittest.TestCase):
    def assert_unchanged(self, mesh, nodes, elements):
        np.testing.assert_array_equal(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.elements, elements)

    def test_extreme_refinement_is_rejected_before_any_section_construction(self):
        mesh, outline = _rectangle()
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        with patch.object(_section_mesh, "plan_line") as line_builder, \
                patch.object(_section_mesh, "plan_corner") as corner_builder, \
                patch.object(_line, "sample_outer") as sampler:
            with self.assertRaises(ElementLimitExceeded) as caught:
                extend_quad_layer(mesh, outline, 1e-8, max_added_elements=1000)
        self.assertEqual(caught.exception.code, "element_limit_exceeded")
        self.assertEqual(caught.exception.context["stage"], "element_count_preflight")
        self.assertEqual(caught.exception.context["count_kind"], "lower_bound")
        self.assertGreater(caught.exception.context["required_elements"], 1000)
        line_builder.assert_not_called()
        corner_builder.assert_not_called()
        sampler.assert_not_called()
        self.assert_unchanged(mesh, nodes, elements)

    def test_exact_cap_and_numpy_integer_are_accepted(self):
        baseline, outline = _rectangle()
        count = extend_quad_layer(baseline, outline, .5).diagnostics["added_elements"]
        mesh, outline = _rectangle()
        result = extend_quad_layer(mesh, outline, .5, max_added_elements=np.int64(count))
        self.assertEqual(result.diagnostics["added_elements"], count)
        # Old elements are outside the budget.
        self.assertEqual(len(mesh.elements), count + 1)
        self.assertEqual(result.diagnostics["max_added_elements"], count)

    def test_selected_ring_over_limit_fails_before_global_node_mapping(self):
        baseline, outline = _rectangle()
        count = extend_quad_layer(baseline, outline, .5).diagnostics["added_elements"]
        mesh, outline = _rectangle()
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        with patch.object(_section_mesh, "_NodePool") as pool:
            with self.assertRaises(ElementLimitExceeded) as caught:
                extend_quad_layer(mesh, outline, .5, max_added_elements=count - 1)
        self.assertEqual(caught.exception.context["stage"], "selected_layer")
        self.assertEqual(caught.exception.context["required_elements"], count)
        pool.assert_not_called()
        self.assert_unchanged(mesh, nodes, elements)

    def test_alternative_seam_candidates_are_not_added_together(self):
        baseline, outline = _rectangle()
        count = extend_quad_layer(baseline, outline, .5).diagnostics["added_elements"]
        attempted_counts = []
        original_line, original_corner = _section_mesh.plan_line, _section_mesh.plan_corner

        def counted(builder):
            def call(*args, **kwargs):
                candidates = builder(*args, **kwargs)
                attempted_counts.extend(len(candidate.quads) for candidate in candidates)
                return candidates
            return call

        mesh, outline = _rectangle()
        with patch.object(_section_mesh, "plan_line", side_effect=counted(original_line)), \
                patch.object(_section_mesh, "plan_corner", side_effect=counted(original_corner)):
            result = extend_quad_layer(mesh, outline, .5, max_added_elements=count)
        self.assertGreater(sum(attempted_counts), count)
        self.assertEqual(result.diagnostics["added_elements"], count)

    def test_line_count_filter_prevents_sampling_and_preserves_typed_error(self):
        args = _line_fixture()
        with patch.object(_line, "sample_outer") as sampler:
            with self.assertRaises(ElementLimitExceeded):
                _line.plan_line(*args, 1e-8, 1e-10, max_added_elements=10)
        sampler.assert_not_called()

    def test_line_actual_count_is_checked_before_section_assembly(self):
        args = _line_fixture()
        # Its boundary lower bound is 5, but the direct units create more cells.
        with patch.object(_line, "assemble_section") as assembly:
            with self.assertRaises(ElementLimitExceeded) as caught:
                _line.plan_line(*args, 4., 1e-10, max_added_elements=5)
        self.assertEqual(caught.exception.context["count_kind"], "actual")
        assembly.assert_not_called()

    def test_only_oversized_outer_count_is_removed_before_sampling(self):
        # Counts 2 and 4 have lower bounds 5 and 6 respectively.
        with patch.object(_line, "sample_outer", wraps=_line.sample_outer) as sampler:
            with self.assertRaises(ElementLimitExceeded):
                _line.plan_line(*_line_fixture(), 2.5, 1e-10, max_added_elements=5)
        self.assertEqual([call.args[1] for call in sampler.call_args_list], [2])

    def test_corner_actual_limit_is_typed_and_can_try_another_count(self):
        source = (np.asarray([[0., 0, 0]]),)
        outer = np.asarray([[0., 1, 0], [1., 1, 0], [1., 0, 0]])
        left = np.asarray([[0., 0, 0], [0., 1, 0]])
        right = np.asarray([[0., 0, 0], [1., 0, 0]])
        args = ("OUTER", source, outer, left, right, outer[1])
        with patch.object(_corners, "assemble_section") as assembly:
            with self.assertRaises(ElementLimitExceeded) as caught:
                _corners.plan_corner(*args, .5, 1e-10, max_added_elements=2)
        self.assertEqual(caught.exception.context["count_kind"], "actual")
        assembly.assert_not_called()
        # Six outer segments produce 5 quads, while four produce 3 quads.
        candidates = _corners.plan_corner(*args, .4, 1e-10, max_added_elements=3)
        self.assertEqual([len(candidate.quads) for candidate in candidates], [3])

    def test_rejected_cut_trials_do_not_consume_candidate_budget(self):
        args = _line_fixture()
        original = _line.build_unit_patch
        rejected = []

        def fail_first(sides, *values, **kwargs):
            if not rejected:
                rejected.append(True)
                raise ValueError("injected invalid cut")
            return original(sides, *values, **kwargs)

        # The alternate cut has a different topology; establish its own exact cap.
        with patch.object(_line, "build_unit_patch", side_effect=fail_first):
            alternate = _line.plan_line(*args, 4., 1e-10, best_only=True)[0]
        rejected.clear()
        with patch.object(_line, "build_unit_patch", side_effect=fail_first):
            actual = _line.plan_line(*args, 4., 1e-10, best_only=True,
                                     max_added_elements=len(alternate.quads))[0]
        self.assertEqual(len(actual.quads), len(alternate.quads))
        self.assertGreater(actual.diagnostics["unit_trial_count"], len(actual.units))

    def test_inner_corner_with_zero_outer_segments_is_counted(self):
        incoming = np.asarray([[0., 1, 0], [1., 1, 0]])
        outgoing = np.asarray([[1., 1, 0], [1., 2, 0]])
        outer = np.asarray([[0., 2, 0]])
        left = np.asarray([[0., 1, 0], [0., 2, 0]])
        right = np.asarray([[1., 2, 0], [0., 2, 0]])
        self.assertEqual(outer_segment_counts(2, 2, 2, 0., 1., 0), [0])
        self.assertEqual(outer_segment_counts(2, 2, 3, 0., 1., 0), [])
        self.assertEqual(section_element_lower_bound(2, 0, 2, 2), 1)
        result = _corners.plan_corner("INNER", (incoming, outgoing), outer, left, right,
                                      outer[0], 1., 1e-10, max_added_elements=1)
        self.assertEqual(len(result[0].quads), 1)
        section = SimpleNamespace(kind="INNER", source_chains=((0, 1), (1, 2)),
                                  outer_boundary=outer[:, :2], left_seam=0, right_seam=1)
        plan = SimpleNamespace(sections=(section,), seams=(
            SimpleNamespace(points=left[:, :2]), SimpleNamespace(points=right[:, :2])))
        preflight_element_limit(plan, {"max_added_elements": 1, "tol": 1e-10, "h": 1.})

    def test_invalid_limits_are_input_errors_and_leave_mesh_unchanged(self):
        for value in (True, np.bool_(False), None, 0, -1, 1.0, "10", float("inf")):
            with self.subTest(value=value):
                mesh, outline = _rectangle()
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                with self.assertRaises(QuadLayerError) as caught:
                    extend_quad_layer(mesh, outline, .5, max_added_elements=value)
                self.assertEqual(caught.exception.code, "invalid_input")
                self.assert_unchanged(mesh, nodes, elements)

    def test_geometry_failures_keep_their_original_error_code(self):
        mesh, outline = _rectangle()
        with self.assertRaises(QuadLayerError) as caught:
            extend_quad_layer(mesh, outline, .5, pattern_lines=[[[1., 3.], [1., 3.2]]])
        self.assertEqual(caught.exception.code, "fixed_boundary_conflict")


if __name__ == "__main__":
    unittest.main()
