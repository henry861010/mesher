import copy
import unittest

import numpy as np
from shapely.geometry import Polygon

from mesher.process_flow.translation.koz import inset_face, koz_value
from mesher.process_flow.translation.standard_v1 import StandardV1Translator


CONTROL = {"globalElementSize": 1.0, "controls": []}


def _container():
    return dict(bodies=[], bumps=[], circuits=[], vias=[], children=[])


def _polygon(*loops):
    return {"type": "POLYGON", "dim": [list(loop) for loop in loops]}


def _region(face):
    region = Polygon()
    for loop in face["dim"]:
        region = region.symmetric_difference(Polygon(loop))
    return region


class KozTranslationTests(unittest.TestCase):
    def test_box_and_circle_insets_and_empty_regions(self):
        box = {"type": "BOX", "dim": [4, 3, 0, 0]}
        circle = {"type": "CIRCLE", "dim": [1, 2, 3]}
        self.assertEqual(inset_face(box, 0.5)["dim"], [0.5, 0.5, 3.5, 2.5])
        self.assertEqual(inset_face(circle, 0.5)["dim"], [1, 2, 2.5])
        self.assertIsNone(inset_face(box, 1.5))
        self.assertIsNone(inset_face(circle, 3))
        self.assertIsNone(inset_face(circle, 4))

    def test_polygon_holes_and_winding_are_independent(self):
        outer = [[0, 0], [10, 0], [10, 10], [0, 10]]
        hole = [[4, 4], [6, 4], [6, 6], [4, 6]]
        expected = Polygon(
            [[1, 1], [9, 1], [9, 9], [1, 9]],
            [[[3, 3], [7, 3], [7, 7], [3, 7]]],
        )
        for loops in ((outer, hole), (hole[::-1], outer[::-1])):
            with self.subTest(loops=loops):
                result = inset_face(_polygon(*loops), 1)
                self.assertTrue(_region(result).equals(expected))

    def test_concave_polygon_keeps_mitre_corners(self):
        face = _polygon([[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]])
        expected = Polygon([[0.5, 0.5], [5.5, 0.5], [5.5, 1.5],
                            [1.5, 1.5], [1.5, 5.5], [0.5, 5.5]])
        self.assertTrue(_region(inset_face(face, 0.5)).equals(expected))

    def test_polygon_inset_can_split_into_two_regions(self):
        face = _polygon([[0, 0], [3, 0], [3, 1.25], [5, 1.25], [5, 0],
                         [8, 0], [8, 3], [5, 3], [5, 1.75], [3, 1.75],
                         [3, 3], [0, 3]])
        result = inset_face(face, 0.5)
        self.assertEqual(len(result["dim"]), 2)
        self.assertEqual(_region(result).geom_type, "MultiPolygon")
        self.assertAlmostEqual(_region(result).area, 8.0)

    def test_polygon_inset_can_disappear_or_keep_a_nested_island(self):
        small = _polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertIsNone(inset_face(small, 0.5))
        face = _polygon(
            [[0, 0], [20, 0], [20, 20], [0, 20]],
            [[3, 3], [17, 3], [17, 17], [3, 17]],
            [[6, 6], [14, 6], [14, 14], [6, 14]],
        )
        result = _region(inset_face(face, 1))
        self.assertEqual(result.geom_type, "MultiPolygon")
        self.assertAlmostEqual(result.area, 104.0)

    def test_rejects_invalid_koz_and_polygon_loops(self):
        for value in (-1, float("nan"), float("inf"), None, "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "koz"):
                koz_value({"koz": value})
        with self.assertRaisesRegex(ValueError, "non-self-intersecting"):
            inset_face(_polygon([[0, 0], [2, 2], [2, 0], [0, 2]]), 0.1)

    def test_recursively_collects_both_faces_for_all_feature_and_shape_types(self):
        geometries = [
            {"type": "BoxGeometry", "bottom_left": [0, 0, 0],
             "top_right": [4, 4, 0], "thk": 1},
            {"type": "CylinderGeometry", "center": [2, 2, 0],
             "bottom_radius": 2, "thk": 1},
            {"type": "PolygonGeometry", "polys": [
                [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]], "thk": 1},
        ]
        translator = StandardV1Translator()
        for field in ("bumps", "circuits", "vias"):
            for geometry in geometries:
                with self.subTest(field=field, shape=geometry["type"]):
                    child = _container()
                    child[field] = [{"geometry": geometry, "koz": 0.5,
                                     "material": "Cu", "density": 100}]
                    container = _container()
                    container["children"] = [child]
                    before = copy.deepcopy(container)
                    base, faces = translator.get_2D_pattern(container, CONTROL)
                    self.assertEqual(len(faces), 1)
                    layers = translator.get_3D_pattern(copy.deepcopy(container), CONTROL)
                    area = layers[0]["assignments"][0]["areas"][0]
                    self.assertEqual(area["face"], faces[0])
                    self.assertEqual(inset_face(base, 0.5), faces[0])
                    self.assertEqual(container, before)

    def test_zero_missing_and_duplicate_koz_faces(self):
        container = _container()
        feature = {"geometry": {"type": "BoxGeometry", "bottom_left": [0, 0, 0],
                   "top_right": [4, 4, 0], "thk": 1}, "material": "Cu", "density": 100}
        container["vias"] = [feature, dict(feature, koz=0)]
        translator = StandardV1Translator()
        self.assertEqual(translator.get_2D_pattern(container, CONTROL)[1], [])
        container["vias"] = [dict(feature, koz=0.5), dict(feature, koz=0.5)]
        self.assertEqual(len(translator.get_2D_pattern(container, CONTROL)[1]), 1)

    def test_koz_never_replaces_original_polygon_base_with_mixed_loop_winding(self):
        container = _container()
        loops = [
            [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
            [[6, 0, 0], [6, 4, 0], [10, 4, 0], [10, 0, 0]],
        ]
        container["vias"] = [{"geometry": {"type": "PolygonGeometry", "polys": loops,
                                            "thk": 1}, "koz": 0.5}]
        base, faces = StandardV1Translator().get_2D_pattern(container, CONTROL)
        self.assertEqual(base["dim"], [[[x, y] for x, y, _ in loop] for loop in loops])
        self.assertEqual(len(faces), 1)

    def test_thin_koz_is_not_deduplicated_and_empty_area_is_explicit(self):
        container = _container()
        feature = {"geometry": {"type": "BoxGeometry", "bottom_left": [0, 0, 0],
                   "top_right": [1, 1, 0], "thk": 1}, "material": "Cu", "density": 100}
        container["vias"] = [dict(feature, koz=1e-7)]
        translator = StandardV1Translator()
        _, faces = translator.get_2D_pattern(container, CONTROL)
        np.testing.assert_allclose(faces[0]["dim"], [1e-7, 1e-7, 1-1e-7, 1-1e-7])
        container["vias"] = [dict(feature, koz=0.5)]
        self.assertEqual(translator.get_2D_pattern(container, CONTROL)[1], [])
        area = translator.get_3D_pattern(container, CONTROL)[0]["assignments"][0]["areas"][0]
        self.assertEqual(area["face"], {"type": "POLYGON", "dim": []})


if __name__ == "__main__":
    unittest.main()
