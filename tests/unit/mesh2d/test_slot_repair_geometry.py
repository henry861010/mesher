"""Defensive repair topology guards and an interacting-slot integration case."""

import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import QuadLayerError, extend_quad_layer
from mesher.mesh2d.adapter.layer._mesher import _expansion_distances, _prepare
from mesher.mesh2d.adapter.layer._repair import (
    _Candidate, _Slot, _candidate, _check_slot_conflict, _find_slots, _regroup,
    repair_slots,
)
from tests.unit.mesh2d.test_extend_quad_layer import _polyomino


def _region(points, node_ids, edges=(0, 1, 2)):
    ids = np.asarray(node_ids, dtype=np.int64)
    slot = _Slot(edges, np.array([-1., 0.]), 0, 0., (0,))
    return _Candidate(slot, (ids[:2], ids[1:3], ids[2:]), ids,
                      np.asarray(points, dtype=float))


class SlotRepairGeometryTests(unittest.TestCase):
    def test_shared_consumed_segment_is_rejected_before_geometry_selection(self):
        first = _region([[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 2, 3])
        second = _region([[0, 1], [1, 1], [1, 2], [0, 2]], [3, 2, 4, 5], (3, 4, 5))
        # This is a defensive check: today's strict C/U turn pattern prevents
        # two detected slots from consuming one original segment.
        with self.assertRaises(QuadLayerError) as caught:
            _check_slot_conflict(second, [first], 1e-10)
        self.assertEqual(caught.exception.context["reason"], "interacting_slots")
        self.assertIn("same original source segment", str(caught.exception))

    def test_crossing_and_touching_mouth_regions_are_rejected(self):
        first = _region([[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 2, 3])
        # Each polygon is a valid CCW rectangle.  The first pair has crossing
        # mouth segments; the second pair touches at one shared old lip node.
        cases = (
            _region([[.5, .5], [.5, 1.5], [-.5, 1.5], [-.5, .5]],
                    [4, 5, 6, 7], (3, 4, 5)),
            _region([[0, 0], [-1, 0], [-1, -1], [0, -1]],
                    [0, 4, 5, 6], (3, 4, 5)),
        )
        for candidate in cases:
            with self.subTest(polygon=candidate.polygon.tolist()):
                with self.assertRaises(QuadLayerError) as caught:
                    _check_slot_conflict(candidate, [first], 1e-10)
                self.assertEqual(caught.exception.context["reason"], "interacting_slots")
        disjoint = _region([[2, 0], [3, 0], [3, 1], [2, 1]], [4, 5, 6, 7], (3, 4, 5))
        _check_slot_conflict(disjoint, [first], 1e-10)

    def test_adjacent_perpendicular_slots_fail_the_entire_transaction(self):
        # Two width-one slots open north and east around the same convex lip
        # at (0, 0).  Their completed mouth regions touch at that vertex.
        cells = {(x, y) for x in range(-3, 2) for y in range(-3, 2)
                 if not (x >= 0 and y >= 0)
                 and not (-2 <= x < 0 and y == 0)
                 and not (x == 0 and -2 <= y < 0)}
        mesh, outline, _ = _polyomino(cells)
        old_nodes, old_elements = mesh.nodes.copy(), mesh.elements.copy()
        with self.assertRaises(QuadLayerError) as caught:
            extend_quad_layer(mesh, outline, .5)
        self.assertEqual(caught.exception.code, "unsupported_slot_repair")
        self.assertEqual(caught.exception.context["reason"], "interacting_slots")
        np.testing.assert_array_equal(mesh.nodes, old_nodes)
        np.testing.assert_array_equal(mesh.elements, old_elements)

    def test_actual_repair_boundary_rejects_nonmonotone_semantic_direction(self):
        cells = {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)}
        mesh, outline, _ = _polyomino(cells)
        data = _prepare(mesh, outline, .5, (), 1., 1e-3)
        _expansion_distances(data)
        normalized = Mesh2D(np.column_stack((data["xy"], np.zeros(len(mesh.nodes)))), mesh.elements)
        result = repair_slots(normalized, data)
        self.assertIsNotNone(result)
        slot = _find_slots(data)[0]
        selected = _candidate(slot, slot.candidates[0], data)
        mouth = np.asarray(result.diagnostics["slot_repairs"][0]["mouth_node_ids"])
        quads = result.mesh.elements[len(mesh.elements):]
        invalid = dict(data)
        invalid["directions"] = data["directions"].copy()
        # Use a real, validated repair mesh and real mouth IDs.  Only corrupt
        # the inherited semantic direction, simulating bad repair provenance.
        invalid["directions"][slot.edges[1]] *= -1
        original_nodes, original_elements = normalized.nodes.copy(), normalized.elements.copy()
        with self.assertRaises(QuadLayerError) as caught:
            _regroup(normalized, invalid, result.mesh.nodes, quads, [selected], [mouth])
        self.assertEqual(caught.exception.context["reason"], "nonmonotone_repaired_chain")
        np.testing.assert_array_equal(normalized.nodes, original_nodes)
        np.testing.assert_array_equal(normalized.elements, original_elements)


if __name__ == "__main__":
    unittest.main()
