"""Whole-cell ownership and exact coarse/fine seams."""
import unittest

import numpy as np

from mesher.mesh2d.imprint_line.mesher import _prepare_plan, _assemble
from . import test_imprint_lines as fixtures


class WholeCellTests(unittest.TestCase):
    options = dict(outer_outline=[[1000, -2000], [8000, -2000], [8000, 3000], [1000, 3000]],
                   tag="mesh_snap", min_transition_size=20)

    def mesh(self, pairs, patterns=()):
        args = _prepare_plan(1000, pairs, patterns, **self.options)
        return _assemble(*args), args[1]

    def test_corner_25_reconnects_to_the_original_grid(self):
        p = fixtures.pair(0, (3975, -25, 4025, 25), np.arange(-25, 26, 5), 1000)
        base, reference = self.mesh([])
        mesh, plan = self.mesh([p])
        left = next(t for t in plan.transitions if t.side == 0)
        self.assertEqual(left.inner, 2000)
        self.assertEqual(set(left.cells), {(1000, y, 2000, y + 1000) for y in (-1000, 0)})
        self.assertFalse(np.any(np.isin(mesh.nodes[:, 0], [2975, 5025])))
        np.testing.assert_array_equal(mesh.nodes[mesh.nodes[:, 0] == 1000, 1],
                                      base.nodes[base.nodes[:, 0] == 1000, 1])
        for y in range(-25, 26, 5):
            fixtures.ImprintTests.assert_segment(self, mesh, 0, y, 2000, 6000)
        for x in (3975, 4025):
            self.assertTrue(all(np.any(np.all(mesh.nodes[:, :2] == (x, y), axis=1))
                                for y in range(-25, 26, 5)))
        # Check actual elements, including the connectivity of distant coarse cells.
        polygons = {tuple(map(tuple, points)) for points in mesh.nodes[mesh.elements, :2]}
        for x0, y0, x1, y1 in reference.rectangles:
            if y0 >= 1000 or y1 <= -1000 or x0 >= 7000:
                self.assertIn(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), polygons)

    def test_exact_edge_and_corner_touch_claim_no_neighbour(self):
        p = fixtures.pair(0, (4000, 0, 4500, 250), [0, 125, 250], 1000)
        mesh, plan = self.mesh([p])
        end = next(t for t in plan.transitions if t.side == 0)
        self.assertEqual(end.cells, ((2000, 0, 3000, 1000),))
        levels = mesh.nodes[(mesh.nodes[:, 0] == 2000) & (mesh.nodes[:, 1] >= 0)
                            & (mesh.nodes[:, 1] <= 1000), 1]
        np.testing.assert_array_equal(levels, [0, 1000])

    def test_parallel_and_crossed_adapters_share_one_coarse_cell(self):
        first = fixtures.pair(0, (3975, 75, 4025, 125), [75, 100, 125], 900)
        for other in (fixtures.pair(0, (3900, 275, 3950, 325), [275, 300, 325], 850),
                      fixtures.pair(1, (2200, 1200, 2250, 1250), [2200, 2225, 2250], 100)):
            with self.subTest(direction=other["direction"]):
                mesh, plan = self.mesh([first, other])
                box = (2000, 0, 3000, 1000)
                self.assertEqual(sum(box in t.cells for t in plan.transitions), 2)
                self.assertFalse(np.any(np.all(plan.rectangles == box, axis=1)))
                vertices = mesh.nodes[mesh.elements, :2]
                local = vertices[np.all((vertices >= box[:2]) & (vertices <= box[2:]), axis=(1, 2))]
                delta = local - local[:, :1]
                following = np.roll(delta, -1, axis=1)
                area = (delta[..., 0] * following[..., 1] - delta[..., 1] * following[..., 0]).sum() / 2
                self.assertAlmostEqual(area, 1000**2)
                reverse, _ = self.mesh([other, first])
                np.testing.assert_array_equal(mesh.nodes, reverse.nodes)
                np.testing.assert_array_equal(mesh.elements, reverse.elements)

    def test_nearby_sizing_line_moves_only_at_the_required_contact(self):
        options = dict(outer_outline=[[0, 0], [12, 0], [12, 12], [0, 12]],
                       tag="mesh_snap", min_transition_size=.5)
        patterns = [[[11, 6.0005], [11.5, 6.0005]]]
        base = _assemble(*_prepare_plan(2, [], patterns, **options))
        control = fixtures.pair(0, (4, 4, 6, 8), [4, 5, 6.0012, 7, 8])
        mesh = _assemble(*_prepare_plan(2, [control], patterns, **options))
        # The reference sizing line is only 0.000333 away from the control.
        reference_y = base.nodes[base.nodes[:, 0] == 12, 1]
        self.assertTrue(np.any((reference_y > 8) & (reference_y < 8.001)))
        np.testing.assert_array_equal(mesh.nodes[mesh.nodes[:, 0] == 12, 1], reference_y)
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 8, 2, 8)

    def test_two_fixed_lines_within_tolerance_of_one_coarse_row(self):
        options = dict(outer_outline=[[0, 0], [18, 0], [18, 16], [0, 16]],
                       tag="mesh_snap", min_transition_size=.4)
        horizontal = fixtures.pair(0, (6.00002, 4.00072, 8.00002, 8.00072), expand=1.28)
        vertical = fixtures.pair(1, (10.00072, 7.99970, 12.00072, 11.99970), expand=1.85)
        mesh = _assemble(*_prepare_plan(2, [horizontal, vertical], **options))
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 8.00072, 6.00002 - 1.28, 8.00002 + 1.28)
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 7.99970, 10.00072, 12.00072)
        np.testing.assert_array_equal(mesh.nodes[mesh.nodes[:, 0] == 0, 1], np.arange(0, 17, 2))

    def test_remote_control_cannot_pull_an_inward_face_past_the_requested_end(self):
        control = fixtures.pair(0, (4, 2, 6, 4), expand=(0, 2.0002))
        remote = fixtures.pair(1, (8.0005, 8, 10, 10), expand=0)
        args = _prepare_plan(2, [control, remote], outer_outline=[[0, 0], [14, 0], [14, 14], [0, 14]],
                             tag="mesh_snap", min_transition_size=.5)
        mesh = _assemble(*args)
        right = next(t for t in args[1].transitions if t.axis == 0 and t.side == 1)
        self.assertEqual(right.cells, ((10, 2, 12, 4),))
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 2.4, 6, 10)

    def test_real_pattern_at_25_survives_the_takeover(self):
        p = fixtures.pair(0, (3975, -25, 4025, 25), np.arange(-25, 26, 5), 1000)
        mesh, _ = self.mesh([p], [[[1000, 25], [3975, 25]]])
        fixtures.ImprintTests.assert_segment(self, mesh, 0, 25, 1000, 3975)
        self.assertTrue(np.any(np.all(mesh.nodes[:, :2] == (3975, 25), axis=1)))
