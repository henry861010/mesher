import unittest
from unittest.mock import patch

import numpy as np

import mesher.mesh2d.circular.geometry as geometry


class CandidateEdgePairTests(unittest.TestCase):
    @staticmethod
    def _edge_items(edges):
        return [
            ((min(start, end), max(start, end)), (start, end))
            for start, end in edges
        ]

    @staticmethod
    def _exhaustive_candidates(xy, edge_items, tolerance):
        candidates = []
        for first_position, (first_key, first_edge) in enumerate(edge_items):
            first_start, first_end = xy[list(first_edge)]
            for second_position in range(first_position + 1, len(edge_items)):
                second_key, second_edge = edge_items[second_position]
                if set(first_key).intersection(second_key):
                    continue
                second_start, second_end = xy[list(second_edge)]
                if (
                    max(
                        min(first_start[0], first_end[0]),
                        min(second_start[0], second_end[0]),
                    )
                    > min(
                        max(first_start[0], first_end[0]),
                        max(second_start[0], second_end[0]),
                    )
                    + tolerance
                    or max(
                        min(first_start[1], first_end[1]),
                        min(second_start[1], second_end[1]),
                    )
                    > min(
                        max(first_start[1], first_end[1]),
                        max(second_start[1], second_end[1]),
                    )
                    + tolerance
                ):
                    continue
                candidates.append((first_position, second_position))
        return candidates

    def test_matches_exhaustive_bounds_and_order_across_blocks(self):
        generator = np.random.default_rng(20260822)
        xy = generator.uniform(-10.0, 10.0, size=(278, 2))
        edges = [(position, position + 1) for position in range(277)]
        edge_items = self._edge_items(edges)
        tolerance = 1.0e-6

        expected = self._exhaustive_candidates(xy, edge_items, tolerance)
        actual = list(
            geometry._iter_candidate_edge_pairs(
                xy,
                edge_items,
                tolerance,
                block_size=256,
            )
        )

        self.assertEqual(actual, expected)

    def test_includes_exact_tolerance_gap_and_excludes_larger_gap(self):
        tolerance = 0.25
        first_edge = np.array([[0.0, 0.0], [1.0, 0.0]])

        at_tolerance_xy = np.vstack(
            (
                first_edge,
                [[1.0 + tolerance, 0.0], [2.0, 0.0]],
            )
        )
        edge_items = self._edge_items([(0, 1), (2, 3)])
        self.assertEqual(
            list(
                geometry._iter_candidate_edge_pairs(
                    at_tolerance_xy,
                    edge_items,
                    tolerance,
                )
            ),
            [(0, 1)],
        )

        beyond_tolerance_xy = at_tolerance_xy.copy()
        beyond_tolerance_xy[2:, 0] = np.nextafter(
            beyond_tolerance_xy[2:, 0],
            np.inf,
        )
        self.assertEqual(
            list(
                geometry._iter_candidate_edge_pairs(
                    beyond_tolerance_xy,
                    edge_items,
                    tolerance,
                )
            ),
            [],
        )

    def test_excludes_edges_that_share_a_node(self):
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        edge_items = self._edge_items([(0, 1), (1, 2)])

        candidates = list(
            geometry._iter_candidate_edge_pairs(
                xy,
                edge_items,
                tolerance=0.0,
            )
        )

        self.assertEqual(candidates, [])


class ValidateGeneratedStripTests(unittest.TestCase):
    @staticmethod
    def _two_quads(first, second):
        return (
            np.vstack((first, second)).astype(np.float64),
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                dtype=np.int64,
            ),
        )

    def test_rejects_cross_touch_and_collinear_overlap(self):
        cases = {
            "cross": (
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                [[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]],
            ),
            "touch": (
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
            ),
            "collinear overlap": (
                [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
                [[1.0, -1.0], [3.0, -1.0], [3.0, 0.0], [1.0, 0.0]],
            ),
        }
        for name, (first, second) in cases.items():
            with self.subTest(name=name):
                xy, elements = self._two_quads(first, second)
                with self.assertRaisesRegex(
                    ValueError,
                    "non-neighbouring edges.*cross",
                ):
                    geometry._validate_generated_strip(
                        xy,
                        elements,
                        tolerance=1.0e-12,
                        angular_tolerance=1.0e-12,
                    )

    def test_allows_elements_that_share_only_one_node(self):
        xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        elements = np.array(
            [[0, 1, 2, 2], [0, 3, 4, 4]],
            dtype=np.int64,
        )

        geometry._validate_generated_strip(
            xy,
            elements,
            tolerance=1.0e-12,
            angular_tolerance=1.0e-12,
        )

    def test_respects_linear_tolerance_in_broad_and_narrow_phases(self):
        gap = 5.0e-4
        first = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        second = first + [1.0 + gap, 0.0]
        xy, elements = self._two_quads(first, second)

        geometry._validate_generated_strip(
            xy,
            elements,
            tolerance=4.0e-4,
            angular_tolerance=1.0e-12,
        )
        with self.assertRaisesRegex(
            ValueError,
            "non-neighbouring edges.*cross",
        ):
            geometry._validate_generated_strip(
                xy,
                elements,
                tolerance=6.0e-4,
                angular_tolerance=1.0e-12,
            )

    def test_skips_exact_tests_for_disjoint_edge_bounds(self):
        square_count = 32
        nodes = []
        elements = []
        for position in range(square_count):
            node_start = len(nodes)
            x = 3.0 * position
            nodes.extend(
                ([x, 0.0], [x + 1.0, 0.0], [x + 1.0, 1.0], [x, 1.0])
            )
            elements.append(
                [node_start, node_start + 1, node_start + 2, node_start + 3]
            )

        with patch.object(
            geometry,
            "_segments_intersect_xy",
            return_value=False,
        ) as exact_intersection:
            geometry._validate_generated_strip(
                np.asarray(nodes, dtype=np.float64),
                np.asarray(elements, dtype=np.int64),
                tolerance=1.0e-12,
                angular_tolerance=1.0e-12,
            )

        self.assertEqual(exact_intersection.call_count, 0)

    def test_detects_crossing_beyond_the_first_broad_phase_block(self):
        triangle_count = 90
        base_triangle = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        nodes = np.vstack(
            [
                base_triangle + [3.0 * position, 0.0]
                for position in range(triangle_count)
            ]
        )
        elements = np.asarray(
            [
                [
                    3 * position,
                    3 * position + 1,
                    3 * position + 2,
                    3 * position + 2,
                ]
                for position in range(triangle_count)
            ],
            dtype=np.int64,
        )

        geometry._validate_generated_strip(
            nodes,
            elements,
            tolerance=1.0e-12,
            angular_tolerance=1.0e-12,
        )

        crossing_nodes = nodes.copy()
        crossing_nodes[-3:] = crossing_nodes[-6:-3]
        with self.assertRaisesRegex(
            ValueError,
            "non-neighbouring edges.*cross",
        ):
            geometry._validate_generated_strip(
                crossing_nodes,
                elements,
                tolerance=1.0e-12,
                angular_tolerance=1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()
