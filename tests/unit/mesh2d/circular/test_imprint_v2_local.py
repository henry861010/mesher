"""Topology, isolation, guide barriers, and bounded-work regression tests."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.circular.imprint_v2 import imprint_circle
from mesher.mesh2d.circular.imprint_v2.local import _circle_patch_rows, _run_local_imprint
from mesher.mesh2d.circular.imprint_v2.utils import _to_circle


def run(mesh, **overrides):
    args = dict(center=[0., 0.], radius=1., projection_tolerance=0.,
                guide_tolerance=0., merge_tolerance=0., minimum_area=0.)
    args.update(overrides)
    return _run_local_imprint(mesh, **args)


class GuideBarrierTests(unittest.TestCase):
    def test_blocks_both_axes_in_both_directions_and_preserves_z(self):
        for axis in (0, 1):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    moving = [0., 0., 7.]
                    fixed = [0., 0., 9.]
                    moving[axis], fixed[axis] = sign * 1.05, sign * 1.02
                    parallel = [[0., 0.], [0., 0.]]
                    parallel[0][axis], parallel[1][axis] = -2., 2.
                    cross = [[0., 0.], [0., 0.]]
                    cross[0][axis] = cross[1][axis] = fixed[axis]
                    cross[0][1-axis], cross[1][1-axis] = -.2, .2
                    mesh = Mesh2D([moving, fixed], np.empty((0, 4), dtype=np.int32))
                    _to_circle(mesh, [0., 0.], 1., .1, 0., [parallel, cross])
                    np.testing.assert_array_equal(mesh.nodes, [moving, fixed])

    def test_barrier_outside_path_does_not_block(self):
        for fixed_x in (1.1, .98):
            mesh = Mesh2D([[1.05, 0.], [fixed_x, 0.]], np.empty((0, 4), dtype=np.int32))
            _to_circle(mesh, [0., 0.], 1., .1, 0., [
                [[-2., 0.], [2., 0.]], [[fixed_x, -.1], [fixed_x, .1]],
            ])
            np.testing.assert_allclose(mesh.nodes[0, :2], [1., 0.])

    def test_one_axis_node_is_not_a_barrier(self):
        mesh = Mesh2D([[1.05, 0.], [1.02, 0.]], np.empty((0, 4), dtype=np.int32))
        _to_circle(mesh, [0., 0.], 1., .1, 0., [[[-2., 0.], [2., 0.]]])
        np.testing.assert_allclose(mesh.nodes[:, 0], [1., 1.])

    def test_geometric_guide_intersection_without_mesh_node_does_not_block(self):
        mesh = Mesh2D([[1.05, 0.]], np.empty((0, 4), dtype=np.int32))
        _to_circle(mesh, [0., 0.], 1., .1, 0., [
            [[-2., 0.], [2., 0.]], [[1.02, -.1], [1.02, .1]],
        ])
        np.testing.assert_allclose(mesh.nodes[0, :2], [1., 0.])

    def test_target_merging_cannot_move_a_blocked_node(self):
        mesh = Mesh2D([[1.05, 0.], [1.02, 0.], [1.01, .02]],
                      np.empty((0, 4), dtype=np.int32))
        _to_circle(mesh, [0., 0.], 1., .1, 0., [
            [[-2., 0.], [2., 0.]], [[1.02, -.1], [1.02, .1]],
        ])
        np.testing.assert_array_equal(mesh.nodes[0, :2], [1.05, 0.])
        self.assertAlmostEqual(np.linalg.norm(mesh.nodes[2, :2]), 1.)

    def test_can_reach_fixed_circle_anchor(self):
        mesh = Mesh2D([[.95, 0., 7.], [1., 0., 9.]], np.empty((0, 4), dtype=np.int32))
        _to_circle(mesh, [0., 0.], 1., .1, 0., [
            [[-2., 0.], [2., 0.]], [[1., -.2], [1., .2]],
        ])
        np.testing.assert_allclose(mesh.nodes[:, :2], [[1., 0.], [1., 0.]])
        np.testing.assert_array_equal(mesh.nodes[:, 2], [7., 9.])

    @staticmethod
    def remote_barrier_mesh():
        return Mesh2D(
            [[.95, 0.], [.9, -.2], [.9, .2], [.5, 0.], [.55, .03], [.5, .05]],
            [[0, 2, 1, 1], [3, 4, 5, 5]],
        )

    @staticmethod
    def remote_guides():
        # The finite horizontal guide admits only the opposite circle root.
        return [[[-1., 0.], [.96, 0.]], [[.5, -.1], [.5, .1]]]

    def test_finds_barrier_outside_band_and_element_selection(self):
        mesh = self.remote_barrier_mesh()
        _to_circle(mesh, [0., 0.], 1., .1, 0., self.remote_guides(), indices=[0])
        np.testing.assert_array_equal(mesh.nodes[0, :2], [.95, 0.])
        np.testing.assert_array_equal(mesh.nodes[3, :2], [.5, 0.])

    def test_local_patch_uses_read_only_remote_corridor(self):
        mesh = self.remote_barrier_mesh()
        remote = mesh.nodes[mesh.elements[1]].copy()
        stats = {}
        run(mesh, projection_tolerance=.1, guide_segments=self.remote_guides(), stats=stats)
        self.assertEqual(stats['patch_elements'], 1)
        self.assertTrue(np.any(np.all(mesh.nodes[:, :2] == [.95, 0.], axis=1)))
        np.testing.assert_array_equal(mesh.nodes[mesh.elements[1]], remote)


class LocalPatchTests(unittest.TestCase):
    def test_broad_phase_keeps_edges_whose_vertices_miss_band(self):
        mesh = Mesh2D([[.5, -2.], [.5, 2.], [2., 2.], [2., -2.]], [[0, 1, 2, 3]])
        self.assertTrue(np.all(np.abs(np.linalg.norm(mesh.nodes[:, :2], axis=1) - 1.) > .1))
        np.testing.assert_array_equal(_circle_patch_rows(mesh.nodes, mesh.elements, np.zeros(2), 1., .1), [0])

    def test_no_candidate_is_true_no_op_and_validates_guides(self):
        mesh = generate_rectilinear_mesh(1., [10., 12.], [10., 12.])
        original_nodes, original_elements = mesh.nodes, mesh.elements
        run(mesh)
        self.assertIs(mesh.nodes, original_nodes)
        self.assertIs(mesh.elements, original_elements)
        with self.assertRaises(ValueError):
            run(mesh, guide_segments=[[[0., 0.], [1., 1.]]])
        self.assertIs(mesh.nodes, original_nodes)

    def test_remote_folded_clockwise_and_duplicate_geometry_is_preserved(self):
        nodes = [[0., 0.], [2., 0.], [0., 2.],
                 [10., 0.], [11., 1.], [10., 1.], [11., 0.],
                 [20., 0.], [20., 1.], [21., 0.],
                 [20., 0.], [20., 1.], [21., 0.]]
        elements = [[0, 1, 2, 2], [3, 4, 5, 6], [7, 8, 9, 9], [10, 11, 12, 12]]
        mesh = Mesh2D(nodes, elements)
        before = mesh.nodes[mesh.elements[1:]].copy()
        run(mesh, merge_tolerance=.01)
        remote = mesh.elements[np.min(mesh.nodes[mesh.elements, 0], axis=1) >= 10.]
        np.testing.assert_array_equal(mesh.nodes[remote], before)
        self.assertEqual(len(np.unique(remote[-2:])), 6)

    def test_transitive_merge_closure_reaches_disconnected_remote_rows(self):
        nodes, elements = [], []
        for i in range(12):
            x = .95 + .06 * i
            start = len(nodes)
            nodes.extend([[x, 0., i*10.], [x+.01, .5, i*10.+1], [x+.01, 1., i*10.+2]])
            elements.append([start, start+1, start+2, start+2])
        mesh = Mesh2D(nodes, elements)
        stats = {}
        run(mesh, merge_tolerance=.07, stats=stats)
        self.assertLess(stats['core_elements'], 12)
        self.assertEqual(stats['patch_elements'], 12)
        # All equivalent families retain the first global member's Z.
        self.assertLessEqual(float(mesh.nodes[:, 2].max()), 2.)

    def test_cross_boundary_equivalence_retains_lowest_global_id_and_z(self):
        for tolerance in (0., .01):
            with self.subTest(tolerance=tolerance):
                mesh = Mesh2D(
                    [[0., 2., 40.], [0., 3., 41.], [-1., 3., 42.],
                     [0., 0., 3.], [2., 0., 4.], [0., 2., 5.]],
                    [[0, 1, 2, 2], [3, 4, 5, 5]],
                )
                stats = {}
                run(mesh, merge_tolerance=tolerance, stats=stats)
                self.assertEqual(stats['core_elements'], 1)
                self.assertEqual(stats['patch_elements'], 2)
                shared = mesh.nodes[np.all(mesh.nodes[:, :2] == [0., 2.], axis=1)]
                np.testing.assert_array_equal(shared, [[0., 2., 40.]])

    def test_seam_shared_intersections_are_unique_and_z_is_interpolated(self):
        mesh = generate_rectilinear_mesh(.2, [-3., 3.], [-3., 3.])
        mesh.nodes[:, 2] = mesh.nodes[:, 0] + 2 * mesh.nodes[:, 1]
        run(mesh, radius=1.13, projection_tolerance=.02, merge_tolerance=.001)
        np.testing.assert_allclose(mesh.nodes[:, 2], mesh.nodes[:, 0] + 2 * mesh.nodes[:, 1], atol=.1)
        # Exact XY duplicates would reveal independent edge insertion at a seam.
        self.assertEqual(len(np.unique(mesh.nodes[:, :2], axis=0)), mesh.node_count)
        if importlib.util.find_spec('shapely'):
            from shapely.geometry import Polygon
            from shapely import union_all
            polygons = [Polygon(mesh.nodes[row[:3] if row[2] == row[3] else row, :2]) for row in mesh.elements]
            self.assertTrue(all(p.is_valid for p in polygons))
            covered = union_all(polygons)
            self.assertAlmostEqual(covered.area, 36., places=8)
            self.assertAlmostEqual(sum(p.area for p in polygons), covered.area, places=8)

    def test_split_failure_after_local_mutation_is_atomic(self):
        mesh = Mesh2D([[0., 0.], [2., 0.], [0., 2.]], [[0, 1, 2, 2]])
        nodes, elements = mesh.nodes, mesh.elements
        before = nodes.copy()
        def fail(local, *args):
            local.nodes[:] = 42.
            raise ValueError('forced split failure')
        with patch('mesher.mesh2d.circular.imprint_v2.local._split_circle', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'forced split'):
                run(mesh)
        self.assertIs(mesh.nodes, nodes)
        self.assertIs(mesh.elements, elements)
        np.testing.assert_array_equal(mesh.nodes, before)

    def test_far_away_padding_does_not_increase_geometry_work(self):
        near = generate_rectilinear_mesh(.2, [-3., 3.], [-3., 3.])
        stats = []
        for size in (10, 100):
            far = generate_rectilinear_mesh(1., [100., 100.+size], [100., 100.+size])
            mesh = Mesh2D(np.concatenate((near.nodes, far.nodes)), np.concatenate((near.elements, far.elements+near.node_count)))
            current = {}
            run(mesh, radius=1.13, projection_tolerance=.02, merge_tolerance=.001, stats=current)
            stats.append(current)
        self.assertEqual(stats[0]['patch_elements'], stats[1]['patch_elements'])
        self.assertEqual(stats[0]['patch_nodes'], stats[1]['patch_nodes'])
        self.assertLess(stats[1]['patch_elements'], stats[1]['global_elements'] // 10)

    def test_case_04_has_no_holes_overlap_or_invalid_elements(self):
        if not importlib.util.find_spec('shapely'):
            self.skipTest('requires optional Shapely diagnostics')
        filename = Path(__file__).resolve().parents[4] / 'examples/failures/imprint_v2_holes/diagnose.py'
        spec = importlib.util.spec_from_file_location('local_diagnostics', filename)
        diagnostics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostics)
        case = diagnostics.load_cases()[3]
        snapshots = diagnostics.execute_stages(case['parameters'])
        np.testing.assert_array_equal(snapshots['projection'].nodes[[12496, 12497]], snapshots['original'].nodes[[12496, 12497]])
        metrics, _ = diagnostics.measure(snapshots['split'], diagnostics.box(-320000., -320000., 320000., 320000.), 40.96)
        self.assertEqual(metrics['invalid_element_rows'], [])
        self.assertEqual(metrics['closed_hole_count'], 0)
        self.assertLessEqual(metrics['missing_area'], 40.96)
        self.assertLessEqual(metrics['overlap_area'], 40.96)
