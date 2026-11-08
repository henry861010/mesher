import unittest

import numpy as np

from mesher.mesh3d.extrusion import Dragger, START_DENSITY
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.tolerance import GeometryToleranceError


class PolygonFaceSelectionTests(unittest.TestCase):
    def test_unresolved_z_subdivision_is_rejected_before_buffer_changes(self):
        for begin, end, size, code in (
            (0.0, 1.0, .001, "feature_below_resolution"),
            (0.0, 1.0, .0005, "feature_below_resolution"),
            (1e9, 1e9 + .0100001, .001000005, "numeric_resolution_exceeded"),
        ):
            dragger = Dragger()
            dragger.set_2D([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
            dragger.element_2D_comp[:] = 1
            before = {name: value.copy() for name, value in vars(dragger).items()
                      if isinstance(value, np.ndarray)}
            with self.subTest(begin=begin, size=size), self.assertRaises(GeometryToleranceError) as caught:
                dragger._drag(size, begin, end)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(dragger.node_num, 0)
            self.assertEqual(dragger.element_num, 0)
            for name, expected in before.items():
                np.testing.assert_array_equal(getattr(dragger, name), expected)

    def test_rejected_source_geometry_preserves_previous_dragger_state(self):
        dragger = Dragger()
        dragger.set_2D([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
        before = {name: value.copy() for name, value in vars(dragger).items()
                  if isinstance(value, np.ndarray)}
        for nodes in ([[0, 0], [.0004, 0], [.0004, 1], [0, 1]],
                      [[0, 0], [1, 0], [.1, .1], [0, 1]],
                      [[0, 0], [1, 1], [1, 0], [0, 1]]):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                dragger.set_2D(nodes, [[0, 1, 2, 3]])
            for name, expected in before.items():
                np.testing.assert_array_equal(getattr(dragger, name), expected)

    def test_partial_hole_entry_is_rejected_for_either_loop_winding(self):
        dragger = Dragger()
        # First quad enters the hole with two corners; the second only
        # touches its boundary and remains in the eligible material region.
        dragger.set_2D(
            [[1, 1], [3, 1], [3, 3], [1, 3], [0, 2], [2, 2], [2, 4], [0, 4]],
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )
        outer = [[0, 0], [6, 0], [6, 6], [0, 6]]
        hole = [[2, 2], [4, 2], [4, 4], [2, 4]]
        for loops in ([outer, hole], [hole[::-1], outer[::-1]]):
            with self.subTest(loops=loops):
                mask = dragger._search_faces({"type": "POLYGON", "dim": loops})
                np.testing.assert_array_equal(mask, [False, True])

    def test_empty_polygon_selects_no_elements(self):
        dragger = Dragger()
        dragger.set_2D([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
        np.testing.assert_array_equal(
            dragger._search_faces({"type": "POLYGON", "dim": []}), [False]
        )

    def test_precomputed_density_face_is_not_inset_twice(self):
        dragger = Dragger()
        mesh = generate_rectilinear_mesh(0.25, [0.25, 0.75], [0.25, 0.75])
        dragger.set_2D(mesh.nodes, mesh.elements)
        dragger._organize([{
            "face": {"type": "BOX", "dim": [0, 0, 1, 1]},
            "type": START_DENSITY,
            "areas": [{"face": {"type": "BOX", "dim": [0.25, 0.25, 0.75, 0.75]},
                       "koz": 0.25, "priority": 1.5, "density": 100, "material": "Cu"}],
        }])
        np.testing.assert_array_equal(dragger.element_2D_comp,
                                      np.full(4, dragger.comps["Cu"]))
        np.testing.assert_array_equal(dragger.element_2D_density_occupy, np.full(4, 1.5))


if __name__ == "__main__":
    unittest.main()
