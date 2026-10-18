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
from mesher.mesh2d.circular.imprint_v2.utils import _blocked_radial_paths, _to_circle
from mesher.mesh2d.circular.utils.pattern_segments import _PatternGuideSet


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


class RadialGuideBarrierTests(unittest.TestCase):
    @staticmethod
    def project(
        nodes, guides, guide_tolerance=0., elements=None, indices=None,
        projection_tolerance=.1,
    ):
        if elements is None:
            elements = np.empty((0, 4), dtype=np.int32)
        mesh = Mesh2D(nodes, elements)
        _to_circle(mesh, [0., 0.], 1., projection_tolerance, guide_tolerance, guides, indices)
        return mesh

    def test_blocks_both_axes_inward_and_outward_without_a_guide_node(self):
        for axis in (0, 1):
            for sign in (-1, 1):
                for distance in (.95, 1.05):
                    with self.subTest(axis=axis, sign=sign, distance=distance):
                        source = np.array([0., 0., 7.])
                        source[axis] = sign * distance
                        guide = np.zeros((2, 2))
                        guide[:, axis] = sign * (1. + distance) / 2.
                        guide[:, 1-axis] = [-.2, .2]
                        mesh = self.project([source], [guide])
                        np.testing.assert_array_equal(mesh.nodes, [source])

    def test_diagonal_path_checks_finite_guides_without_circle_roots(self):
        source = [0.63, 0.84, 8.]
        # This entire segment is outside the circle; the radial path crosses it.
        guides = [[[.615, .81], [.615, .83]]]
        mesh = self.project([source], guides)
        np.testing.assert_array_equal(mesh.nodes, [source])

    def test_extended_endpoints_use_guide_tolerance_on_both_axes_and_ends(self):
        extension, tolerance = .125, .0078125
        for offset, blocked in (
            (0., True), (extension, True), (extension + tolerance / 2., True),
            (extension + tolerance, True), (extension + 2. * tolerance, False),
        ):
            for axis in (0, 1):
                for sign in (-1, 1):
                    with self.subTest(offset=offset, axis=axis, sign=sign):
                        source = np.array([0., 0., 7.])
                        source[axis] = 1.05
                        guide = np.zeros((2, 2))
                        guide[:, axis] = 1.025
                        guide[:, 1-axis] = [sign * offset, sign * .5]
                        # Duplicates and reversed endpoints have the same effect.
                        mesh = self.project(
                            [source], [guide[::-1], guide], tolerance,
                            projection_tolerance=extension,
                        )
                        expected = source.copy()
                        if not blocked:
                            expected[axis] = 1.
                        np.testing.assert_allclose(mesh.nodes, [expected], rtol=0, atol=1e-15)

    def test_extension_tracks_projection_tolerance(self):
        source = [1.01, 0., 7.]
        guide = [[[1.005, .05], [1.005, .06]]]
        for tolerance, blocked in ((.02, False), (.1, True)):
            with self.subTest(projection_tolerance=tolerance):
                mesh = self.project([source], guide, projection_tolerance=tolerance)
                expected = source if blocked else [1., 0., 7.]
                np.testing.assert_allclose(mesh.nodes, [expected], rtol=0, atol=1e-15)

    def test_zero_extension_retains_finite_barriers_and_shared_metadata(self):
        starts, ends = np.array([[1.01, 0.]]), np.array([[1., 0.]])
        for lower, blocked in ((-.01, True), (.05, False)):
            with self.subTest(lower=lower):
                guides = _PatternGuideSet.from_values(
                    [[[1.005, lower], [1.005, .1]]], coordinate_scale=1.,
                )
                segments = guides.segments.copy()
                bounds = guides.lower_bounds.copy(), guides.upper_bounds.copy()
                actual = _blocked_radial_paths(
                    starts, ends, np.array([True]), guides, np.array([1e-12]),
                    1e-12, extension_length=0.,
                )
                np.testing.assert_array_equal(actual, [blocked])
                actual = _blocked_radial_paths(
                    starts, ends, np.array([True]), guides, np.array([1e-12]),
                    1e-12, extension_length=.1,
                )
                np.testing.assert_array_equal(actual, [True])
                np.testing.assert_array_equal(guides.segments, segments)
                np.testing.assert_array_equal(guides.lower_bounds, bounds[0])
                np.testing.assert_array_equal(guides.upper_bounds, bounds[1])

    def test_diagonal_crossing_beyond_original_endpoint_is_blocked(self):
        source = [0.63, 0.84, 8.]
        mesh = self.project([source], [[[.615, .90], [.615, 1.]]])
        np.testing.assert_array_equal(mesh.nodes, [source])

    def test_extended_barriers_do_not_extend_node_membership_or_circle_roots(self):
        source = [1.05, 0., 8.]
        guides = _PatternGuideSet.from_values(
            [[[.95, 0.], [.96, 0.]]], coordinate_scale=1.,
        )
        # The start lies only on the extension, so it remains a free node.
        # Its radial path overlaps the extended guide and must be blocked,
        # rather than acquiring the extended segment's circle root [1, 0].
        mesh = self.project([source], guides)
        np.testing.assert_array_equal(mesh.nodes, [source])

    def test_only_numerical_tolerance_excludes_path_endpoints(self):
        for fixed, blocked in ((1., False), (1. + 1e-14, False), (1.001, True)):
            with self.subTest(fixed=fixed):
                source = [1.05, 0., 9.]
                mesh = self.project([source], [[[fixed, -.2], [fixed, .2]]], .005)
                expected = source if blocked else [1., 0., 9.]
                np.testing.assert_allclose(mesh.nodes, [expected], rtol=0, atol=1e-15)

    def test_guides_beyond_the_path_and_parallel_segments_do_not_block(self):
        source = [1.05, 0., 7.]
        guides = [
            [[.98, -.1], [.98, .1]], [[1.1, -.1], [1.1, .1]],
            [[1.01, .03], [1.03, .03]],
        ]
        mesh = self.project([source], guides, .005)
        np.testing.assert_allclose(mesh.nodes, [[1., 0., 7.]], rtol=0, atol=1e-15)

    def test_collinear_overlap_uses_extended_bounds_and_allows_target_contact(self):
        for source_x, bounds, blocked in (
            (1.05, [1.01, 1.03], True), (.95, [.97, .99], True),
            (1.05, [1.1, 1.13], True), (1.05, [.7, .9], False),
            (.95, [1.1, 1.3], False),
        ):
            with self.subTest(source_x=source_x, bounds=bounds):
                source = [source_x, 0., 6.]
                mesh = self.project([source], [[[bounds[0], 0.], [bounds[1], 0.]]], .005)
                expected = source if blocked else [1., 0., 6.]
                np.testing.assert_allclose(mesh.nodes, [expected], rtol=0, atol=1e-15)

    def test_blocked_node_cannot_participate_in_target_merging(self):
        source = [1.05, 0., 7.]
        neighbour = [1.05 * np.cos(-.03), 1.05 * np.sin(-.03), 9.]
        mesh = self.project([source, neighbour], [[[1.025, .09], [1.025, .1]]])
        np.testing.assert_array_equal(mesh.nodes[0], source)
        np.testing.assert_allclose(mesh.nodes[1], [np.cos(-.03), np.sin(-.03), 9.])

    def test_selection_limits_movement_and_repeated_projection_is_stable(self):
        nodes = [[1.05, 0., 7.], [1.04, .1, 8.],
                 [1.05 * np.cos(-.03), 1.05 * np.sin(-.03), 9.], [2., 2., 10.]]
        guides = [[[1.025, .09], [1.025, .1]]]
        mesh = self.project(nodes, guides, elements=[[0, 2, 3, 3]], indices=[0])
        np.testing.assert_array_equal(mesh.nodes[[0, 1, 3]], np.asarray(nodes)[[0, 1, 3]])
        np.testing.assert_allclose(mesh.nodes[2], [np.cos(-.03), np.sin(-.03), 9.])
        projected = mesh.nodes.copy()
        _to_circle(mesh, [0., 0.], 1., .1, 0., guides, indices=[0])
        np.testing.assert_array_equal(mesh.nodes, projected)

    def test_original_radial_path_is_checked_before_target_merging_only(self):
        anchor = [np.cos(.02), np.sin(.02)]
        guides = [
            [[.9999, .005], [.9999, .03]],
            [[.998, anchor[1]], [.99985, anchor[1]]],
        ]
        mesh = self.project([[1.05, 0.], anchor], guides)
        # The radial target [1, 0] is safe. The existing merge to the guide
        # anchor crosses the vertical guide and is deliberately not rechecked.
        np.testing.assert_allclose(mesh.nodes[:, :2], [anchor, anchor])

    def test_sparse_pairs_are_batched_even_when_one_path_spans_many_guides(self):
        guides = [[[x, .13], [x, .14]] for x in np.linspace(1.01, 1.04, 9)]
        guides.append([[1.025, .09], [1.025, .1]])
        nodes = [[1.05, 0.], [0., 1.05], [1.05 * np.cos(-.02), 1.05 * np.sin(-.02)]]
        with patch('mesher.mesh2d.circular.imprint_v2.utils._GUIDE_PAIR_BATCH_SIZE', 2):
            mesh = self.project(nodes, guides)
        np.testing.assert_array_equal(mesh.nodes[0, :2], nodes[0])
        np.testing.assert_allclose(mesh.nodes[1, :2], [0., 1.])
        np.testing.assert_allclose(mesh.nodes[2, :2], [np.cos(-.02), np.sin(-.02)])

    def test_large_offset_circle_uses_local_path_geometry(self):
        center = np.array([1e9, -1e9])
        source = center + [1.05, 0.]
        mesh = Mesh2D([source], np.empty((0, 4), dtype=np.int32))
        guide = center + [[1.025, -.2], [1.025, .2]]
        _to_circle(mesh, center, 1., .1, 0., [guide])
        np.testing.assert_array_equal(mesh.nodes[:, :2], [source])


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

    def test_case_12_radial_crossing_is_blocked_and_preserves_coverage(self):
        if not importlib.util.find_spec('shapely'):
            self.skipTest('requires optional Shapely diagnostics')
        filename = Path(__file__).resolve().parents[4] / 'examples/failures/imprint_v2_holes/diagnose.py'
        spec = importlib.util.spec_from_file_location('radial_diagnostics', filename)
        diagnostics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostics)
        case = next(case for case in diagnostics.load_cases() if case['id'] == 'case_12')
        snapshots = diagnostics.execute_stages(case['parameters'])
        projected = snapshots['projection']
        np.testing.assert_array_equal(projected.nodes[17751], snapshots['original'].nodes[17751])
        element = projected.elements[17486]
        self.assertTrue(diagnostics.Polygon(projected.nodes[element, :2]).is_valid)
        metrics, _ = diagnostics.measure(
            snapshots['split'], diagnostics.box(-320000., -320000., 320000., 320000.), 40.96,
        )
        self.assertEqual(metrics['invalid_element_rows'], [])
        self.assertEqual(metrics['closed_hole_count'], 0)
        self.assertLessEqual(metrics['missing_area'], 40.96)
        self.assertLessEqual(metrics['overlap_area'], 40.96)

    def test_case_16_crossing_beyond_guide_endpoint_is_blocked(self):
        if not importlib.util.find_spec('shapely'):
            self.skipTest('requires optional Shapely diagnostics')
        filename = Path(__file__).resolve().parents[4] / 'examples/failures/imprint_v2_holes/diagnose.py'
        spec = importlib.util.spec_from_file_location('extended_radial_diagnostics', filename)
        diagnostics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostics)
        case = next(case for case in diagnostics.load_cases() if case['id'] == 'case_16')
        snapshots = diagnostics.execute_stages(case['parameters'])
        projected = snapshots['projection']
        np.testing.assert_array_equal(projected.nodes[10277], snapshots['original'].nodes[10277])
        element = projected.elements[10203]
        self.assertTrue(diagnostics.Polygon(projected.nodes[element, :2]).is_valid)
        metrics, _ = diagnostics.measure(
            snapshots['split'], diagnostics.box(-320000., -320000., 320000., 320000.), 40.96,
        )
        self.assertEqual(metrics['invalid_element_rows'], [])
        self.assertEqual(metrics['closed_hole_count'], 0)
        self.assertLessEqual(metrics['missing_area'], 40.96)
        self.assertLessEqual(metrics['overlap_area'], 40.96)
