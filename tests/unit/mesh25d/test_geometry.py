import unittest

from mesher.mesh25d.geometry.patterns import extract_pattern
from mesher.mesh25d.geometry.assignments import extract_assignments
from mesher.mesh25d.plan3d import plan_layers


MESH_CONTROL = {"globalElementSize": 1.0, "controls": []}


class GeometryExtractionTests(unittest.TestCase):
    def test_translates_box_geometry_to_face_and_layers(self):
        container = {
            "bodies": [
                {
                    "geometry": {
                        "type": "BoxGeometry",
                        "bottom_left": [0.0, 0.0, 0.0],
                        "top_right": [2.0, 1.0, 0.0],
                        "thk": 1.0,
                    },
                    "material": "Si",
                }
            ],
            "vias": [],
            "circuits": [],
            "bumps": [],
            "children": [],
        }

        pattern = extract_pattern(container)

        base_face, faces = pattern.base_face, pattern.faces
        layers = plan_layers(extract_assignments(container), [], element_size=MESH_CONTROL["globalElementSize"]).layer_infos

        self.assertEqual(base_face, {"type": "BOX", "dim": [0.0, 0.0, 2.0, 1.0]})
        self.assertEqual(faces, [])
        self.assertEqual([layer["z"] for layer in layers], [0.0, 1.0])

    def test_accepts_multiple_circles_when_another_shape_is_the_base_face(self):
        container = {
            "bodies": [
                {
                    "geometry": {
                        "type": "BoxGeometry",
                        "bottom_left": [-10.0, -10.0, 0.0],
                        "top_right": [10.0, 10.0, 0.0],
                        "thk": 1.0,
                    },
                    "material": "Si",
                },
                {
                    "geometry": {
                        "type": "CylinderGeometry",
                        "center": [-4.0, 0.0, 1.0],
                        "bottom_radius": 2.0,
                        "thk": 1.0,
                    },
                    "material": "Cu",
                },
                {
                    "geometry": {
                        "type": "CylinderGeometry",
                        "center": [4.0, 0.0, 1.0],
                        "bottom_radius": 3.0,
                        "thk": 1.0,
                    },
                    "material": "Cu",
                },
            ],
            "vias": [],
            "circuits": [],
            "bumps": [],
            "children": [],
        }
        pattern = extract_pattern(container)
        base_face, faces = pattern.base_face, pattern.faces

        self.assertEqual(base_face, {"type": "BOX", "dim": [-10.0, -10.0, 10.0, 10.0]})
        self.assertEqual(
            faces,
            [
                {"type": "CIRCLE", "dim": [-4.0, 0.0, 2.0]},
                {"type": "CIRCLE", "dim": [4.0, 0.0, 3.0]},
            ],
        )

    def test_collects_circle_faces_from_every_container_item_type(self):
        def circle(center_x):
            return {
                "geometry": {
                    "type": "CylinderGeometry",
                    "center": [center_x, 0.0, 0.0],
                    "bottom_radius": 1.0,
                    "thk": 1.0,
                }
            }

        container = {
            "bodies": [
                {
                    "geometry": {
                        "type": "BoxGeometry",
                        "bottom_left": [-10.0, -10.0, 0.0],
                        "top_right": [10.0, 10.0, 0.0],
                        "thk": 1.0,
                    }
                },
                circle(-6.0),
            ],
            "vias": [circle(-2.0)],
            "circuits": [circle(2.0)],
            "bumps": [circle(6.0)],
            "children": [],
        }
        pattern = extract_pattern(container)
        base_face, faces = pattern.base_face, pattern.faces

        self.assertEqual(base_face["type"], "BOX")
        self.assertEqual(
            faces,
            [
                {"type": "CIRCLE", "dim": [-6.0, 0.0, 1.0]},
                {"type": "CIRCLE", "dim": [-2.0, 0.0, 1.0]},
                {"type": "CIRCLE", "dim": [2.0, 0.0, 1.0]},
                {"type": "CIRCLE", "dim": [6.0, 0.0, 1.0]},
            ],
        )


if __name__ == "__main__":
    unittest.main()
