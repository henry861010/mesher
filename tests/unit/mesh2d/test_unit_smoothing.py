"""Regressions for quality-guarded smoothing of selected adapter units."""

import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter.layer import _mesher
from mesher.mesh2d.adapter.layer._construction import smoothing
from mesher.mesh2d.adapter.layer._construction.strip import build_strip
from mesher.mesh2d.adapter.layer._construction.units import build_unit_patch
from mesher.mesh2d.adapter.layer._types import QuadLayerError
from mesher.mesh2d.quality import MeshQualityChecker
from tests.unit.mesh2d.test_build_unit_1_x_y_z import _sketch_patch
from tests.unit.mesh2d.test_dense_sections import rectangle


def smooth(nodes, quads, boundary, score, fixed=(), tol=.001):
    unit = {"quality": score, "_polygon_ids": boundary, "_fixed_interior": fixed,
            "element_start": 5, "element_end": 5 + len(quads)}
    return smoothing.smooth_selected_units(nodes, quads, [unit], element_offset=5,
                                           original_count=0, tol=tol)


def quality(nodes, quads):
    checker = MeshQualityChecker(Mesh2D(nodes, quads))
    angles = checker.calculate_scaled_jacobian()
    np.testing.assert_equal(angles.invalid_indices.size, 0)
    return angles.summary.minimum, checker.calculate_aspect_ratio().summary.maximum


def dense_layer():
    mesh, outline = rectangle([0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 34, 42], height=20)
    return _mesher.generate_layer_adaptive_mesh(mesh, outline, 10)


class UnitSmoothingTests(unittest.TestCase):
    def test_fixed_short_edge_wedge_opens_in_rotated_and_translated_frames(self):
        # Reconstruct the first screenshot, with screen Y pointing down.
        xy = np.array([[104, -96], [104, -283], [104, -727], [104, -783],
                       [882, -1205], [882, -273]], dtype=float)
        for angle in (0., .8):
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            points = xy @ rotation + [1e6, -2e6]
            unit = build_unit_patch((points[:4], points[[3, 4]], points[[4, 5]], points[[5, 0]]), .001)
            before = unit.nodes.copy()
            j0, a0 = quality(before - before[0], unit.quads)
            moved = smooth(unit.nodes, unit.quads, range(6), unit.quality)
            j1, a1 = quality(unit.nodes - before[0], unit.quads)
            self.assertIn(7, moved)
            np.testing.assert_allclose((unit.nodes[7, :2] - before[7, :2]) @ rotation.T,
                                       [0, 74.53324], atol=1e-4)
            np.testing.assert_array_equal(unit.nodes[:6], before[:6])
            self.assertAlmostEqual(j0, .12146, places=5)
            self.assertAlmostEqual(j1, .23559, places=5)
            self.assertAlmostEqual(a0, a1, places=8)

    def test_two_short_segments_next_to_a_corner(self):
        # The long left side and two short segments at its foot in screenshot 3.
        xy = np.array([[55, -31], [55, -1001], [55, -1064], [55, -1117],
                       [862, -1092], [862, -125]], dtype=float)
        unit = build_unit_patch((xy[:4], xy[[3, 4]], xy[[4, 5]], xy[[5, 0]]), .001)
        j0, a0 = quality(unit.nodes, unit.quads)
        smooth(unit.nodes, unit.quads, range(6), unit.quality)
        j1, a1 = quality(unit.nodes, unit.quads)
        np.testing.assert_array_equal(unit.nodes[:6, :2], xy)
        self.assertGreater(j1, j0 + .06)
        self.assertLess(a1, a0 - .9)

    def test_quarter_step_when_half_step_worsens_another_corner(self):
        nodes = np.array([[0, 0], [16, 0], [20, 0], [20, 16], [20, 20],
                          [8, 20], [0, 20], [0, 15], [9, 16]], dtype=float)
        quads = np.array([[0, 1, 8, 7], [1, 2, 3, 8], [8, 3, 4, 5], [7, 8, 5, 6]])
        before = nodes.copy()
        displacement = nodes[[1, 3, 5, 7]].mean(axis=0) - before[8]
        # Exactly four is below the activation condition.
        self.assertFalse(smooth(nodes, quads, range(8), 4.))
        np.testing.assert_array_equal(nodes, before)
        self.assertEqual(set(smooth(nodes, quads, range(8), 4.37)), {8})
        np.testing.assert_array_equal(nodes[8], before[8] + .25 * displacement)
        np.testing.assert_array_equal(nodes[:8], before[:8])

    def test_short_edge_rejects_both_otherwise_improving_steps(self):
        nodes = np.array([[11, -4], [9, -2], [-3, 2], [-3, -1],
                          [0, -3], [9, -6], [0, 0]], dtype=float)
        quads = np.array([[6, 0, 1, 2], [6, 2, 3, 4], [6, 4, 5, 0]])
        # Shortest edge is 2.828 originally, then 2.544 or 2.668.
        before = nodes.copy()
        self.assertFalse(smooth(nodes, quads, range(6), 4.48, tol=2.7))
        np.testing.assert_array_equal(nodes, before)
        self.assertEqual(set(smooth(nodes, quads, range(6), 4.48, tol=2.5)), {6})

    def test_cached_sweep_preserves_its_shared_midpoints(self):
        mesh, edges = _sketch_patch()
        sides = tuple(mesh.nodes[edge] for edge in edges)
        cache = {}
        for _ in range(2):
            unit = build_unit_patch(sides, .001, _cache=cache)
            self.assertEqual(unit.method, "sweep")
            fixed = list(unit.fixed_interior)
            self.assertEqual(len(fixed), 2)
            before = unit.nodes[fixed].copy()
            moved = smooth(unit.nodes, unit.quads, range(len(mesh.nodes)), unit.quality, fixed)
            self.assertTrue(set(fixed).isdisjoint(moved))
            np.testing.assert_array_equal(unit.nodes[fixed], before)

    def test_paired_unit_keeps_its_private_transition_straight(self):
        source = np.column_stack(([0, 1, 2, 3, 8, 9, 10], np.zeros((7, 2))))
        outer = np.array([[0., 4, 0], [6., 4, 0], [10., 4, 0]])
        left, right = [source[0], outer[0]], [source[-1], outer[-1]]
        section = build_strip(source, outer, left, right, [(0, 0), (6, 2)], 0., .001,
                              cache={}, stats={}, cap=6)
        record, = section.units
        midpoint, = record["_fixed_interior"]
        moved = smoothing.smooth_selected_units(section.nodes, section.quads, section.units,
                                                element_offset=0, original_count=0, tol=.001)
        self.assertTrue(moved)
        self.assertNotIn(midpoint, moved)
        np.testing.assert_array_equal(section.nodes[midpoint], [7, 2, 0])

    def test_dense_layer_improves_with_identical_connectivity_and_seams(self):
        with patch.object(smoothing, "_THRESHOLD", np.inf):
            before = dense_layer()
        after = dense_layer()
        np.testing.assert_array_equal(after.mesh.elements, before.mesh.elements)
        self.assertEqual(after.outline, before.outline)
        self.assertEqual(after.diagnostics["seam_states"], before.diagnostics["seam_states"])
        for old, new in zip(before.diagnostics["units"], after.diagnostics["units"], strict=True):
            self.assertEqual(old["polygon"], new["polygon"])
            self.assertNotIn("_fixed_interior", new)
        count = after.diagnostics["added_elements"]
        added = after.mesh.elements[-count:]
        j0, a0 = quality(before.mesh.nodes, added)
        j1, a1 = quality(after.mesh.nodes, added)
        self.assertGreaterEqual(j1, j0 - 1e-12)
        self.assertLess(a1, a0 - 1)

    def test_layer_restores_original_points_if_final_geometry_rejects_a_move(self):
        with patch.object(smoothing, "_THRESHOLD", np.inf):
            before = dense_layer()
        validate = _mesher._validate_patch

        def reject_moved(mesh, nodes, quads, outline, data):
            if not np.array_equal(data["_represented_world_nodes"], before.mesh.nodes):
                raise QuadLayerError("nonincident edge contact")
            return validate(mesh, nodes, quads, outline, data)

        with patch.object(_mesher, "_validate_patch", side_effect=reject_moved) as check:
            after = dense_layer()
        self.assertEqual(check.call_count, 2)
        np.testing.assert_array_equal(after.mesh.nodes, before.mesh.nodes)
        np.testing.assert_array_equal(after.mesh.elements, before.mesh.elements)
        self.assertEqual(after.diagnostics["quality"], before.diagnostics["quality"])
