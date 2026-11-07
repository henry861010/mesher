"""Public adaptive APIs apply LINE limits transactionally at their intended layers."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter import (
    QuadLayerError,
    generate_layer_adaptive_mesh,
    plan_quad_sections,
)
from tests.unit.mesh2d import test_full_adaptive_mesh as full_existing
from tests.unit.mesh2d import test_section_integration as section_existing
from mesher.mesh2d.adapter.layer._construction import line
from tests.unit.mesh2d.test_extend_quad_layer import _target_ids


class LinePolicyIntegrationTests(unittest.TestCase):
    def assert_unchanged(self, mesh, snapshot):
        full_existing.FullAdaptiveMeshTests.assert_unchanged(self, mesh, snapshot)

    def assert_policy(self, mesh, result, size, ratio=.5, limit=10):
        lines = [detail for detail in result.diagnostics["section_details"]
                 if detail["kind"] == "LINE"]
        self.assertTrue(lines)
        units = result.diagnostics["units"]
        for detail in lines:
            points = mesh.nodes[detail["outer_ids"], :2]
            spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
            self.assertTrue(np.all(spacing >= ratio * size - 1e-8), spacing)
            self.assertTrue(np.all(spacing <= size + 1e-8), spacing)
            for index in detail["unit_ids"]:
                source, _, outer, _ = np.asarray(units[index]["boundary_counts"]) - 1
                self.assertTrue(source == 1 or (outer == 1 and source <= limit))

    def test_ten_segment_cap_is_conforming_and_frozen_in_all_directions(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle(rotation)
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                result = generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=10)
                self.assert_policy(mesh, result, 500.)
                section_existing.SectionIntegrationTests.assert_built_partition(
                    self, mesh, nodes, elements, result)

    def test_explicit_target_anchor_is_preserved_with_new_spacing_constraints(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        target = np.array([[-500., -500.], [-500., 1500.], [1007.5, 1500.],
                           [2600., 1500.], [2600., -500.]])
        original = target.copy()
        ids = _target_ids(mesh, target)
        nodes = mesh.nodes.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., outline_o=ids, max_transition_count=10)
        self.assert_policy(mesh, result, 500.)
        np.testing.assert_array_equal(target, original)
        front = mesh.nodes[np.unique(result.outline), :2]
        for point in target:
            self.assertEqual(np.count_nonzero(np.linalg.norm(front - point, axis=1) < 1e-8), 1)
        section_existing.SectionIntegrationTests.assert_built_partition(
            self, mesh, nodes, elements, result)

    def test_custom_transition_limit_is_observed_in_public_unit_diagnostics(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=3)
        self.assert_policy(mesh, result, 500., limit=3)

    def test_seam_candidates_share_one_density_profile_per_line(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        with patch.object(line, "LineSampling", wraps=line.LineSampling) as profiles:
            result = generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=10)
        lines = [detail for detail in result.diagnostics["section_details"]
                 if detail["kind"] == "LINE"]
        self.assertEqual(profiles.call_count, len(lines))
        self.assertTrue(any(len(detail["feasible_seam_pairs"]) > 1 for detail in lines))

    def test_short_mandatory_anchor_interval_is_reported_in_world_coordinates(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        target = np.array([[-500., -500.], [-500., 1500.], [1000., 1500.],
                           [1100., 1500.], [2600., 1500.], [2600., -500.]])
        original = target.copy()
        ids = _target_ids(mesh, target)
        nodes = mesh.nodes.copy()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., outline_o=ids, max_transition_count=10)
        details = [detail for detail in result.diagnostics["section_details"]
                   if detail.get("forced_short_intervals")]
        self.assertEqual(len(details), 1)
        exception, = details[0]["forced_short_intervals"]
        np.testing.assert_allclose(exception["start"], [1000., 1500.], atol=1e-8)
        np.testing.assert_allclose(exception["end"], [1100., 1500.], atol=1e-8)
        self.assertAlmostEqual(exception["length"], 100.)
        front = mesh.nodes[details[0]["outer_ids"], :2]
        middle = front[(front[:, 0] >= 1000. - 1e-8) & (front[:, 0] <= 1100. + 1e-8)]
        np.testing.assert_allclose(middle, [[1000., 1500.], [1100., 1500.]], atol=1e-8)
        for index in details[0]["unit_ids"]:
            self.assertLessEqual(result.diagnostics["units"][index]["boundary_counts"][0] - 1, 10)
        np.testing.assert_array_equal(target, original)
        section_existing.SectionIntegrationTests.assert_built_partition(
            self, mesh, nodes, elements, result)

    def test_incompatible_limits_are_typed_and_do_not_commit_any_mesh_arrays(self):
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        snapshot = full_existing.FullAdaptiveMeshTests._snapshot(mesh)
        # Geometric preview remains available even though resampling cannot
        # place enough outer segments within the requested size interval.
        preview = plan_quad_sections(mesh, outline, 500.)
        self.assertTrue(preview.sections)
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 500., max_transition_count=2)
        self.assertEqual(caught.exception.code, "line_constraints_infeasible")
        self.assert_unchanged(mesh, snapshot)

    def test_budget_accounts_for_policy_output_and_failure_is_atomic(self):
        baseline, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        count = generate_layer_adaptive_mesh(baseline, outline, 500., max_transition_count=10).diagnostics["added_elements"]
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        result = generate_layer_adaptive_mesh(mesh, outline, 500., max_added_elements=count, max_transition_count=10)
        self.assertEqual(result.diagnostics["added_elements"], count)
        self.assert_policy(mesh, result, 500.)
        mesh, outline = section_existing.SectionIntegrationTests._nonuniform_rectangle()
        snapshot = full_existing.FullAdaptiveMeshTests._snapshot(mesh)
        with self.assertRaises(QuadLayerError) as caught:
            generate_layer_adaptive_mesh(mesh, outline, 500., max_added_elements=count - 1, max_transition_count=10)
        self.assertEqual(caught.exception.code, "element_limit_exceeded")
        self.assert_unchanged(mesh, snapshot)

if __name__ == "__main__":
    unittest.main()
