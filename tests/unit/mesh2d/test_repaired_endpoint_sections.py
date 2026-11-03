"""Staged repairs can share a whole-edge corner reservation without a gap."""
import unittest

import numpy as np

from mesher.mesh2d.adapter import QuadLayerError, extend_quad_layer
from mesher.mesh2d.adapter.layer import _mesher, _sections
from mesher.mesh2d.adapter.layer._geometry import area
from mesher.mesh2d.editor import EditorMesh2D
from tests.unit.mesh2d.test_adaptive_slot_repair import _test2
from tests.unit.mesh2d.test_quad_sections import _cells


class RepairedEndpointSectionsTests(unittest.TestCase):
    def test_repaired_step_keeps_exact_front_and_frozen_source_in_all_rotations(self):
        rotation = np.asarray(((0., -1.), (1., 0.)))
        for turns in range(4):
            with self.subTest(turns=turns):
                mesh, outline = _test2()
                mesh.nodes[:, :2] = mesh.nodes[:, :2] @ np.linalg.matrix_power(rotation, turns).T
                extend_quad_layer(mesh, outline, 30., expand_ratio=1.5,
                                  repair_max_element_size=20.)
                outline = EditorMesh2D.outlines_by_edge(mesh)[0]
                nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
                data, plan = _mesher._prepare_plan(mesh, outline, 100., (), 1.5, 1e-3)
                np.testing.assert_array_equal(mesh.nodes, nodes)
                np.testing.assert_array_equal(mesh.elements, elements)
                self.assertEqual(len(plan.diagnostics["slot_repairs"]), 1)
                self.assertEqual(len(plan.diagnostics["corner_repairs"]), 1)
                original_sections = dict(data)
                original_sections.pop("_original_data")
                with self.assertRaises(QuadLayerError) as caught:
                    _sections.partition_sections(original_sections)
                self.assertIn("wedge", str(caught.exception))
                front_data = dict(data)
                _mesher._front(front_data)
                np.testing.assert_array_equal(plan.front, front_data["front"])
                covered = sum(area(_sections.section_polygon(section, plan.seams,
                                                            data["xy"], data["tol"]))
                              for section in plan.sections)
                self.assertAlmostEqual(covered, area(data["source"]) - area(plan.front))
                self.assertTrue(all(len(seam.owners) == 2 for seam in plan.seams))
                inner = next(section for section in plan.sections if section.kind.startswith("INNER"))
                shared = [plan.seams[inner.left_seam], plan.seams[inner.right_seam]]
                self.assertTrue(any(np.array_equal(seam.points[-1], plan.front[inner.corner])
                                    and np.all(np.abs(seam.points[-1] - seam.points[0]) > data["tol"])
                                    for seam in shared))

    def test_shared_endpoint_retains_complete_pattern_and_mandatory_anchor(self):
        mesh, outline = _cells({(0, 0), (1, 0), (1, 1)})
        patterns = [[[0., 1.], [0., 1.1]], [[0., 1.1], [0., 1.2]]]
        data = _mesher._prepare(mesh, outline, .2, patterns, 1., 1e-3)
        _mesher._validate_old_patterns(mesh, data)
        _mesher._front(data)
        data["_original_data"] = dict(data)
        plan = _sections.partition_sections(data)
        self.assertEqual(plan.diagnostics["sections"]["INNER_BOTH"], 1)
        locked = [seam for seam in plan.seams if seam.pattern_ids]
        self.assertEqual(len(locked), 1)
        self.assertEqual(locked[0].pattern_ids, (0, 1))
        world = locked[0].points * data["scale"] + data["origin"]
        np.testing.assert_allclose(world, [[0., 1.], [0., 1.1], [0., 1.2]])
        self.assertEqual(len(locked[0].owners), 2)


if __name__ == "__main__":
    unittest.main()
