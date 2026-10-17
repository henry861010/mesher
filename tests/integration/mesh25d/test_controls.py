import copy
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh25d import build_mesh
from mesher.mesh25d.controls import resolve_controls
from mesher.mesh25d.domain import SymmetryMode
from mesher.mesh25d.geometry.assignments import extract_assignments
from mesher.mesh25d.geometry.context import prepare_geometry
from mesher.meshcontrol import get_control_obj
from .test_builder import _box_structure, _mesh_control


def section(method, start=0, end=35, size=10):
    return {"method": method, "coordinateMode": "absolute", "label": method,
            "elementSize": size, "startZ": start, "endZ": end}


class ControlIntegrationTests(unittest.TestCase):
    def test_section_methods_produce_their_own_remainder_layers(self):
        structure = _box_structure()
        structure["root"]["bodies"][0]["geometry"]["thk"] = 35
        before = copy.deepcopy(structure)
        for method, expected in (
            ("Z_SECTION_AVG", np.linspace(0, 35, 4)),
            ("Z_SECTION_TOP", [0, 10, 20, 30, 35]),
            ("Z_SECTION_BOT", [0, 5, 15, 25, 35]),
            ("Z_SECTION_CENTER", [0, 10, 15, 25, 35]),
        ):
            with self.subTest(method=method):
                control = _mesh_control(10, controls=[section(method)])
                snapshot = copy.deepcopy(control)
                mesh = build_mesh(structure, control)
                np.testing.assert_allclose(np.unique(mesh.nodes[:, 2]), expected)
                self.assertEqual(structure, before)
                self.assertEqual(control, snapshot)

    def test_point_adds_a_boundary_without_refining_other_intervals(self):
        structure = _box_structure()
        control = {"method": "Z_POINT", "coordinateMode": "absolute", "label": "Cut", "z": 0.3}
        mesh = build_mesh(structure, _mesh_control(1, controls=[control]))
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 2]), [0, 0.3, 1])

    def test_missing_reference_is_an_error_in_a_reduced_model(self):
        control = {"method": "Z_POINT", "coordinateMode": "reference", "label": "Typo",
                   "reference": {"kind": "body", "id": "missing"},
                   "z": {"anchor": "z_min", "offset": 0}}
        with self.assertRaisesRegex(ValueError, r"controls\[0\].*Typo.*missing"):
            build_mesh(_box_structure(), _mesh_control(1, "upper_right_quarter", [control]))

    def test_retained_container_reference_uses_its_retained_subtree_bounds(self):
        structure = _box_structure()
        structure["root"]["bodies"][0]["geometry"]["thk"] = 4
        parent = copy.deepcopy(structure["root"])
        parent["id"] = "parent"
        parent["bodies"][0]["geometry"].update(bottom_left=[-2, -2, 0], top_right=[0, 0, 0], thk=1)
        child = copy.deepcopy(structure["root"])
        child["bodies"][0]["geometry"].update(bottom_left=[1.5, 1.5, 2], top_right=[2, 2, 2], thk=1)
        parent["children"] = [child]
        structure["root"]["children"] = [parent]
        control = {"method": "Z_POINT", "coordinateMode": "reference", "label": "Parent",
                   "reference": {"kind": "container", "id": "parent"},
                   "z": {"anchor": "z_min", "offset": 0.5}}
        before = copy.deepcopy(structure)
        for mode, expected in ((SymmetryMode.FULL, 0.5), (SymmetryMode.UPPER_RIGHT_QUARTER, 2.5)):
            with self.subTest(mode=mode):
                context = prepare_geometry(structure["root"], mode)
                resolved = resolve_controls(structure["root"], context.container, [control], global_element_size=1)
                self.assertEqual(resolved.z[0]["start"], expected)
        self.assertEqual(structure, before)

    def test_xy_projection_preserves_polygon_holes_and_is_detached(self):
        loops = [[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 4}],
                 [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 1, "y": 2}]]
        entry = {"method": "3D_REGION", "coordinateMode": "absolute", "label": "Region",
                 "elementSize": 0.1, "region": {"type": "polygon", "loops": loops, "startZ": 1, "endZ": 2}}
        before = copy.deepcopy(entry)
        obj = get_control_obj(None, [entry])[0]
        xy = obj.get_25D_xy_controls()
        self.assertEqual(xy, [{"region": {"type": "POLYGON", "dim": [
            [[0, 0], [4, 0], [0, 4]], [[1, 1], [2, 1], [1, 2]],
        ]}, "element_size": 0.1}])
        xy[0]["region"]["dim"][0][0][0] = 99
        self.assertEqual(obj.get_25D_xy_controls()[0]["region"]["dim"][0][0], [0, 0])
        self.assertEqual(obj.get_25D_z_controls(), [])
        self.assertEqual(entry, before)

    def test_xy_control_rejects_before_algorithms_with_original_source_index(self):
        structure = _box_structure()
        structure["root"]["bodies"][0]["id"] = "body"
        point = {"method": "Z_POINT", "coordinateMode": "absolute", "label": "Cut", "z": 0.5}
        region = {"method": "3D_REGION", "coordinateMode": "reference", "label": "Local sizing",
                  "reference": {"kind": "body", "id": "body"}, "elementSize": 0.1,
                  "region": {"type": "box", "min": {axis: {"anchor": f"{axis}_min", "offset": 0} for axis in "xyz"},
                             "max": {axis: {"anchor": f"{axis}_max", "offset": 0} for axis in "xyz"}}}
        before = copy.deepcopy(structure)
        with patch("mesher.mesh25d.plan2d.generate_rectilinear_mesh") as grid, patch("mesher.mesh25d.plan3d.Dragger") as dragger:
            with self.assertRaisesRegex(NotImplementedError, r"XY.*controls\[1\].*Local sizing"):
                build_mesh(structure, _mesh_control(1, controls=[point, region]))
        grid.assert_not_called()
        dragger.assert_not_called()
        self.assertEqual(structure, before)

    def test_assignment_extraction_respects_bump_stage(self):
        container = _box_structure()["root"]
        feature = {"geometry": copy.deepcopy(container["bodies"][0]["geometry"]),
                   "material": "Cu", "density": 100}
        container["bumps"] = [{**feature, "id": "joined", "stage": "joined"},
                              {**feature, "id": "unjoined", "stage": "unjoined"}]
        events = extract_assignments(container)
        bumps = [event for event in events if event["diagnostic"]["featureType"] == "bump"]
        self.assertEqual([event["diagnostic"]["sourceRef"] for event in bumps], ["joined", "joined"])
