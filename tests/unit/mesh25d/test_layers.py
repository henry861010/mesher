import copy
import unittest

from mesher.mesh25d.plan3d import plan_layers


def interval(start, end, size):
    # The planner must not interpret the method name.
    return {"start": start, "end": end, "element_size": size, "method": "opaque"}


class LayerPlanningTests(unittest.TestCase):
    def test_overlaps_use_smallest_active_size_and_preserve_event_order(self):
        end = {"z": 10, "type": 0, "marker": "terminal"}
        start = {"z": 0, "type": 3, "marker": "start"}
        convert = {"z": 5, "type": 1, "marker": "convert"}
        stop = {"z": 5, "type": 0, "marker": "stop"}
        assignments = [end, convert, start, stop]
        controls = [interval(1, 8, 2), interval(3, 6, 1), interval(0, 10, 20)]
        before = copy.deepcopy((assignments, controls))

        infos = plan_layers(assignments, controls, element_size=4).layer_infos

        self.assertEqual([info["z"] for info in infos], [0, 1, 3, 5, 6, 8, 10])
        self.assertEqual([info["element_size"] for info in infos], [4, 2, 1, 1, 2, 4, 4])
        self.assertEqual(infos[3]["assignments"], [stop, convert])
        self.assertIs(infos[-1]["assignments"][0], end)
        self.assertEqual((assignments, controls), before)

    def test_points_duplicate_boundaries_and_outside_controls_do_not_extend_model(self):
        assignments = [{"z": 0, "type": 3}, {"z": 10, "type": 0}]
        controls = [
            interval(-3, 2, 1), interval(8, 20, 2),
            interval(-5, -1, 0.1), interval(11, 12, 0.1),
            interval(5, 5, 0.01), interval(5, 5, 0.01),
            interval(-1, -1, 0.01), interval(11, 11, 0.01),
            interval(0, 0, 0.01), interval(10, 10, 0.01),
        ]
        infos = plan_layers(assignments, controls, element_size=4).layer_infos
        self.assertEqual([info["z"] for info in infos], [0, 2, 5, 8, 10])
        self.assertEqual([info["element_size"] for info in infos], [1, 4, 4, 2, 4])

    def test_close_geometry_boundaries_are_preserved(self):
        assignments = [{"z": z, "type": 3} for z in (0, 0.001, 1)]
        infos = plan_layers(assignments, [], element_size=1).layer_infos
        self.assertEqual([info["z"] for info in infos], [0, 0.001, 1])

    def test_no_material_events_produce_no_layers(self):
        self.assertEqual(plan_layers([], [interval(0, 10, 1)], element_size=1).layer_infos, [])
