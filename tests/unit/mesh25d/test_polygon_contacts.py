"""Exact vertex contact is topology, not a sub-tolerance gap."""

from copy import deepcopy
from itertools import product
import unittest

from mesher.mesh25d.domain import SymmetryMode
from mesher.mesh25d.controls import _normalize_xy_controls
from mesher.mesh25d.geometry.context import prepare_geometry
from mesher.mesh25d.geometry.normalization import (
    _polygon_loop_topology, _validate_polygon_topology,
)
from mesher.tolerance import GeometryToleranceError, resolve_tolerance


def square(x, y, size):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


def shifted(loop, x, y):
    return [[px + x, py + y] for px, py in loop]


class PolygonContactTests(unittest.TestCase):
    policy = resolve_tolerance(.001)
    shell = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 2]]
    hole = [[0, 2], [1, 1], [2, 2], [1, 3]]

    def topology(self, loops):
        return _polygon_loop_topology(loops, self.policy, "test.polygon")

    def assert_invalid(self, loops, reason="polygon_loop_contact"):
        with self.assertRaises(GeometryToleranceError) as caught:
            self.topology(loops)
        self.assertEqual(caught.exception.code, "invalid_geometry")
        if reason is not None:
            self.assertEqual(caught.exception.context["reason"], reason)

    def test_tangent_hole_order_winding_start_and_explicit_closure(self):
        for reverse_order, reverse_winding, rotate, closed in product((False, True), repeat=4):
            loops = deepcopy([self.shell, self.hole])
            if reverse_order:
                loops.reverse()
            for index, loop in enumerate(loops):
                if reverse_winding:
                    loop = loop[::-1]
                if rotate:
                    loop = loop[1:] + loop[:1]
                loops[index] = loop + loop[:1] if closed else loop
            with self.subTest(order=reverse_order, winding=reverse_winding, rotate=rotate, closed=closed):
                state = self.topology(loops)
                self.assertEqual(len(state.nesting), 1)
                self.assertEqual(len(state.contacts), 1)

    def test_islands_and_multiple_point_intersections(self):
        cases = [
            ([square(0, 0, 2), square(2, 2, 2)], 1),
            ([[[0, 0], [4, 0], [4, 4], [3, 4], [3, 1], [1, 1], [1, 4], [0, 4]],
              [[1, 4], [2, 5], [3, 4]]], 2),
        ]
        for loops, count in cases:
            with self.subTest(loops=loops):
                state = self.topology(loops)
                self.assertEqual(state.nesting, frozenset())
                self.assertEqual(len(state.contacts), count)

    def test_multiple_tangent_holes_and_nested_islands(self):
        shell = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 8], [0, 2]]
        holes = [self.hole, shifted(self.hole, 0, 6)]
        self.assertEqual(len(self.topology([shell, *holes]).contacts), 2)
        self.assertEqual(len(self.topology([square(0, 0, 20), square(2, 2, 4), square(6, 6, 4)]).contacts), 1)
        nested = [square(0, 0, 20), square(2, 2, 16), square(4, 4, 4), square(8, 8, 4)]
        self.assertEqual(len(self.topology(nested).contacts), 1)

    def test_hole_chain_that_disconnects_interior_is_rejected(self):
        shell = [[0, 0], [10, 0], [10, 5], [10, 10], [0, 10], [0, 5]]
        left = [[0, 5], [2.5, 2.5], [5, 5], [2.5, 7.5]]
        right = shifted(left, 5, 0)
        self.assert_invalid([shell, left, right])

    def test_edge_contact_crossings_and_vertex_on_edge_are_rejected(self):
        cases = [
            [square(0, 0, 4), square(0, 1, 1)],  # shared edge
            [square(0, 0, 4), self.hole],  # shell lacks the touching vertex
            [square(0, 0, 4), square(2, 2, 4)],  # crossing in edges
            [[[0, 0], [4, 0], [4, 2], [4, 4], [2, 4], [0, 4]],
             [[2, 2], [4, 2], [6, 2], [6, 6], [2, 6], [2, 4]]],  # crossing at vertices
            [square(0, 0, 4), square(0, 0, 4)],  # duplicate loop
        ]
        for loops in cases:
            with self.subTest(loops=loops):
                self.assert_invalid(loops)

    def test_single_loop_self_contact_is_rejected(self):
        self.assert_invalid([[[0, 0], [2, 0], [2, 2], [4, 2], [4, 4],
                              [2, 4], [2, 2], [0, 2]]], reason=None)

    def test_positive_gap_is_not_promoted_to_contact(self):
        for distance in (.0004, .001):
            with self.subTest(distance=distance), self.assertRaises(GeometryToleranceError) as caught:
                self.topology([square(-4, 0, 4), square(distance, 1, 1)])
            self.assertEqual(caught.exception.code, "feature_below_resolution")
            self.assertEqual(caught.exception.context["reason"], "polygon_loop_gap")

    def test_contact_identity_survives_coordinate_motion(self):
        loops = [self.shell, self.hole]
        moved = [shifted(loop, .0004, .0004) for loop in loops]
        _validate_polygon_topology(loops, moved, self.policy, "test.polygon",
                                   original_state=self.topology(loops))

    def test_sizing_regions_keep_the_existing_separation_contract(self):
        controls = [{"element_size": .5, "region": {
            "type": "POLYGON", "dim": [square(0, 0, 2), square(2, 2, 2)],
        }}]
        original = deepcopy(controls)
        with self.assertRaises(GeometryToleranceError) as caught:
            _normalize_xy_controls(controls, ["controls[0]"], {"bodies": []}, self.policy)
        self.assertEqual(caught.exception.code, "invalid_geometry")
        self.assertEqual(caught.exception.context["reason"], "polygon_loop_contact")
        self.assertEqual(caught.exception.context["stage"], "control_normalization")
        self.assertEqual(controls, original)

    def test_new_lost_and_reassigned_contact_are_rejected(self):
        touching = [square(0, 0, 2), square(2, 2, 2)]
        separate = [square(0, 0, 2), square(3, 3, 2)]
        reassigned = [square(0, 0, 2), square(-2, -2, 2)]
        for before, after in ((separate, touching), (touching, separate), (touching, reassigned)):
            with self.subTest(after=after), self.assertRaises(GeometryToleranceError) as caught:
                _validate_polygon_topology(before, after, self.policy, "test.polygon")
            self.assertEqual(caught.exception.code, "feature_below_resolution")
            self.assertEqual(caught.exception.context["reason"], "polygon_contact_change")
            self.assertNotEqual(caught.exception.context["original_contacts"],
                                caught.exception.context["normalized_contacts"])

    def test_point_becoming_edge_and_containment_change_are_rejected(self):
        for before, after, reason in (
            ([square(0, 0, 2), square(2, 2, 2)],
             [square(0, 0, 2), square(2, 0, 2)], "polygon_loop_contact"),
            ([square(0, 0, 10), square(2, 2, 2)],
             [square(0, 0, 10), square(12, 2, 2)], "polygon_containment_change"),
        ):
            with self.subTest(reason=reason), self.assertRaises(GeometryToleranceError) as caught:
                _validate_polygon_topology(before, after, self.policy, "test.polygon")
            self.assertEqual(caught.exception.code, "feature_below_resolution")
            self.assertEqual(caught.exception.context["reason"], reason)

    def test_actual_normalization_preserves_contact_input_and_is_idempotent(self):
        loops = [shifted(loop, .0004, .0004) for loop in (square(0, 0, 2), square(2, 2, 2))]
        container = {"bodies": [
            {"material": "UF", "geometry": {"type": "PolygonGeometry", "thk": 1,
                "polys": [[[x, y, 0] for x, y in loop] for loop in loops]}},
            {"material": "UF", "geometry": {"type": "BoxGeometry", "thk": 1,
                "bottom_left": [-2, -2, -1], "top_right": [2, 2, -1]}},
        ]}
        original = deepcopy(container)
        first = prepare_geometry(container, SymmetryMode.FULL)
        normalized = first.container["bodies"][0]["geometry"]["polys"]
        self.assertEqual(normalized[0][2], [2, 2, 0])
        self.assertEqual(normalized[1][0], [2, 2, 0])
        self.assertEqual(container, original)
        second = prepare_geometry(first.container, SymmetryMode.FULL)
        self.assertEqual(second.container, first.container)


if __name__ == "__main__":
    unittest.main()
