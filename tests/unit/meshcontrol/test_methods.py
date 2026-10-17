import copy
import unittest
from unittest.mock import patch

from mesher.meshcontrol.methods import (
    MeshControl,
    MeshControl_3dRegion,
    MeshControl_Zpoint,
    MeshControl_ZsectionAvg,
    MeshControl_ZsectionTop,
    MeshControl_ZsectionBot,
    MeshControl_ZsectionCenter,
    ZSectionControl,
    _container_bounding_geometry,
    _geometry_bounding,
    get_control_obj,
)


def box(bottom=(0, 0, 0), top=(2, 3, 0), thickness=1):
    return {
        "type": "BoxGeometry",
        "bottom_left": list(bottom),
        "top_right": list(top),
        "thk": thickness,
    }


class ContainerBoundingGeometryTests(unittest.TestCase):
    def test_single_box_uses_bottom_face_and_thickness(self):
        geometry = box((1, 2, 3), (4, 6, 3), 5)
        self.assertEqual(
            _container_bounding_geometry({"bodies": [{"geometry": geometry}]}),
            geometry,
        )

    def test_mixed_features_and_descendants_contribute_to_every_axis(self):
        container = {
            "bodies": [{"geometry": box((-2, -1, 1), (1, 2, 1), 3)}],
            "vias": [{"geometry": {
                "type": "CylinderGeometry", "center": [5, 0, -3],
                "bottom_radius": 2, "thk": 2,
            }}],
            "circuits": [{"geometry": {
                "type": "PolygonGeometry", "thk": 3, "polys": [
                    [[0, 0, 2], [2, 0, 2], [0, 1, 2]],
                    [[-8, 3, 2], [-7, 3, 2], [-8, 4, 2]],
                ],
            }}],
            "bumps": [{"geometry": box((0, -6, 5), (1, -5, 5), 4)}],
            "children": [{}, {"children": [{
                "bodies": [{"geometry": box((8, 5, 10), (12, 7, 10), 2)}],
            }]}],
        }
        before = copy.deepcopy(container)

        self.assertEqual(_container_bounding_geometry(container), {
            "type": "BoxGeometry", "bottom_left": [-8, -6, -3],
            "top_right": [12, 7, -3], "thk": 15,
        })
        self.assertEqual(container, before)

    def test_accepts_geometry_only_in_descendants_and_nullable_lists(self):
        container = {
            "bodies": None, "vias": None, "circuits": None, "bumps": None,
            "children": [{"children": None}, {"children": [{
                "vias": [{"geometry": box()}],
            }]}],
        }
        self.assertEqual(_container_bounding_geometry(container), box())

    def test_each_feature_field_is_included(self):
        for field in ("bodies", "vias", "circuits", "bumps"):
            with self.subTest(field=field):
                self.assertEqual(
                    _container_bounding_geometry({field: [{"geometry": box()}]}),
                    box(),
                )

    def test_reversed_box_corners_and_negative_thickness_are_normalized(self):
        geometry = box((4, 3, 5), (-2, -1, 5), -8)
        self.assertEqual(
            _container_bounding_geometry({"bodies": [{"geometry": geometry}]}),
            box((-2, -1, -3), (4, 3, -3), 8),
        )

    def test_cylinder_and_polygon_negative_thickness(self):
        geometries = [
            {"type": "CylinderGeometry", "center": [-2, -3, 2],
             "bottom_radius": 2, "thk": -5},
            {"type": "PolygonGeometry", "polys": [
                [[-4, -5, 2], [0, -5, 2], [0, -1, 2]],
            ], "thk": -5},
        ]
        for geometry in geometries:
            with self.subTest(kind=geometry["type"]):
                self.assertEqual(
                    _container_bounding_geometry({"bodies": [{"geometry": geometry}]}),
                    box((-4, -5, -3), (0, -1, -3), 5),
                )

    def test_preserves_coordinate_precision_and_accepts_zero_thickness(self):
        geometry = box((-0.123456789, 0.234567891, 0.123456789),
                       (1.234567891, 2.345678912, 0.123456789), 0)
        self.assertEqual(
            _container_bounding_geometry({"bodies": [{"geometry": geometry}]}),
            geometry,
        )
        geometry["thk"] = 0.000000123
        result = _container_bounding_geometry({"bodies": [{"geometry": geometry}]})
        self.assertEqual(result["bottom_left"][2], geometry["bottom_left"][2])
        self.assertAlmostEqual(result["thk"], geometry["thk"], places=15)

    def test_empty_subtrees_raise(self):
        for container in ({}, {"children": [{}, {"bodies": None}]}):
            with self.subTest(container=container), self.assertRaisesRegex(
                ValueError, "does not contain any geometry"
            ):
                _container_bounding_geometry(container)

    def test_unsupported_geometries_raise(self):
        for kind in ("ConeGeometry", "UnknownGeometry"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, kind):
                _container_bounding_geometry({"bodies": [{"geometry": {"type": kind}}]})

    def test_malformed_containers_and_features_raise_with_context(self):
        cases = [
            (None, "container"), ({"children": {}}, "children"),
            ({"children": [None]}, r"children\[0\]"),
            ({"bodies": {}}, "bodies"), ({"vias": [None]}, r"vias\[0\]"),
            ({"bumps": [{}]}, "geometry"),
            ({"circuits": [{"geometry": None}]}, "geometry"),
        ]
        for container, message in cases:
            with self.subTest(container=container), self.assertRaisesRegex(ValueError, message):
                _container_bounding_geometry(container)

    def test_invalid_geometry_fields_raise(self):
        geometries = [
            {}, {**box(), "bottom_left": [0, 0]},
            {**box(), "top_right": [1, float("nan"), 0]},
            {**box(), "thk": float("inf")}, {**box(), "thk": True},
            {"type": "CylinderGeometry", "center": [0, 0, 0],
             "bottom_radius": -1, "thk": 1},
            {"type": "PolygonGeometry", "polys": [], "thk": 1},
            {"type": "PolygonGeometry", "polys": [[]], "thk": 1},
        ]
        for geometry in geometries:
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                _container_bounding_geometry({"bodies": [{"geometry": geometry}]})


class MeshControl3dRegionTests(unittest.TestCase):
    def setUp(self):
        self.geometry = box((10, 20, 30), (40, 60, 30), 50)

    def control(self, region, mode="reference"):
        return {"method": "3D_REGION", "coordinateMode": mode,
                "elementSize": 5, "region": region}

    def box_region(self):
        return {
            "type": "box",
            "min": {"x": {"anchor": "x_min", "offset": -2},
                    "y": {"anchor": "y_max", "offset": -35},
                    "z": {"anchor": "z_min", "offset": 0.25}},
            "max": {"x": {"anchor": "x_max", "offset": 3},
                    "y": {"anchor": "y_min", "offset": 45},
                    "z": {"anchor": "z_max", "offset": -0.5}},
        }

    def test_box_resolves_each_axis_anchor_and_signed_offset(self):
        control = self.control(self.box_region())
        before = copy.deepcopy(control)
        geometry_before = copy.deepcopy(self.geometry)
        result = MeshControl_3dRegion(control, self.geometry)
        self.assertEqual(result.region, {
            "type": "box", "min": {"x": 8, "y": 25, "z": 30.25},
            "max": {"x": 43, "y": 65, "z": 79.5},
        })
        self.assertEqual(control, before)
        self.assertEqual(self.geometry, geometry_before)
        second = MeshControl_3dRegion(control, box((0, 0, 0), (5, 10, 0), 20))
        self.assertEqual(second.region["min"], {"x": -2, "y": -25, "z": 0.25})

    def test_polygon_resolves_all_loops_and_z_limits(self):
        def point(x_anchor, x_offset, y_anchor, y_offset):
            return {"x": {"anchor": x_anchor, "offset": x_offset},
                    "y": {"anchor": y_anchor, "offset": y_offset}}

        control = self.control({
            "type": "polygon", "loops": [
                [point("x_min", -1, "y_min", 0),
                 point("x_max", 2, "y_min", 0),
                 point("x_max", 2, "y_max", 3)],
                [point("x_min", 5, "y_min", 5),
                 point("x_min", 10, "y_min", 5),
                 point("x_min", 10, "y_min", 10)],
            ], "startZ": {"anchor": "z_max", "offset": -48},
            "endZ": {"anchor": "z_min", "offset": 49},
        })
        before = copy.deepcopy(control)
        result = MeshControl_3dRegion(control, self.geometry)
        self.assertEqual(result.region, {
            "type": "polygon", "loops": [
                [{"x": 9, "y": 20}, {"x": 42, "y": 20}, {"x": 42, "y": 63}],
                [{"x": 15, "y": 25}, {"x": 20, "y": 25}, {"x": 20, "y": 30}],
            ], "startZ": 32, "endZ": 79,
        })
        self.assertEqual(control, before)

    def test_absolute_regions_work_without_geometry_and_are_detached(self):
        regions = [
            {"type": "box", "min": {"x": 1, "y": 2, "z": 3},
             "max": {"x": 4, "y": 5, "z": 6}},
            {"type": "polygon", "loops": [[
                {"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 0, "y": 3},
            ]], "startZ": -2, "endZ": 5},
        ]
        for region in regions:
            with self.subTest(shape=region["type"]):
                result = MeshControl_3dRegion(self.control(region, "absolute"))
                self.assertEqual(result.region, region)
                point = (result.region["min"] if region["type"] == "box"
                         else result.region["loops"][0][0])
                point["x"] = 999
                self.assertNotEqual(result.region, region)

    def test_reference_uses_normalized_geometry_bounds(self):
        region = {"type": "box", **{
            bound: {axis: {"anchor": f"{axis}_{bound}", "offset": 0}
                    for axis in "xyz"}
            for bound in ("min", "max")
        }}
        geometries = [
            box((4, 3, 5), (-2, -1, 5), -8),
            {"type": "CylinderGeometry", "center": [1, 1, 5],
             "bottom_radius": 3, "thk": -8},
            {"type": "PolygonGeometry", "thk": -8, "polys": [
                [[-2, -1, 5], [4, -1, 5], [4, 3, 5]],
            ]},
        ]
        for geometry in geometries:
            with self.subTest(kind=geometry["type"]):
                result = MeshControl_3dRegion(self.control(region), geometry)
                self.assertEqual(result.region["min"]["x"], -2)
                self.assertEqual(result.region["max"]["x"], 4)
                self.assertEqual(result.region["min"]["z"], -3)
                self.assertEqual(result.region["max"]["z"], 5)

    def test_reference_requires_geometry(self):
        with self.assertRaisesRegex(ValueError, "reference geometry"):
            MeshControl_3dRegion(self.control(self.box_region()))

    def test_reference_rejects_wrong_axis_or_unknown_anchor(self):
        for anchor in ("y_min", "x_center"):
            control = self.control(self.box_region())
            control["region"]["min"]["x"]["anchor"] = anchor
            with self.subTest(anchor=anchor), self.assertRaisesRegex(ValueError, "anchor"):
                MeshControl_3dRegion(control, self.geometry)

    def test_reference_rejects_unsupported_region_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported region type"):
            MeshControl_3dRegion(self.control({"type": "sphere"}), self.geometry)


SECTION_CLASSES = (
    MeshControl_ZsectionAvg, MeshControl_ZsectionTop,
    MeshControl_ZsectionBot, MeshControl_ZsectionCenter,
)


def section_control(method="Z_SECTION_AVG", start=0, end=35, size=10):
    return {"method": method, "coordinateMode": "absolute", "label": "Focus",
            "elementSize": size, "startZ": start, "endZ": end}


def point_control(z=5):
    return {"method": "Z_POINT", "coordinateMode": "absolute", "label": "Slice", "z": z}


def reference_control(control, reference):
    result = copy.deepcopy(control)
    result.update(coordinateMode="reference", reference=reference)
    for field in ("z", "startZ"):
        if field in result:
            result[field] = {"anchor": "z_min", "offset": 0}
    if "endZ" in result:
        result["endZ"] = {"anchor": "z_max", "offset": 0}
    return result


class ZSectionControlTests(unittest.TestCase):
    def intervals(self, cls, start=0, end=35, size=10, min_dis=0.01):
        obj = cls(section_control(cls.METHOD, start, end, size))
        controls = obj.get_25D_z_controls(min_dis)
        self.assertEqual(controls[0]["start"], start)
        self.assertEqual(controls[-1]["end"], end)
        for index, control in enumerate(controls):
            self.assertEqual(control["method"], cls.METHOD)
            self.assertEqual(control["element_size"], size)
            self.assertLess(control["start"], control["end"])
            if index:
                self.assertEqual(controls[index - 1]["end"], control["start"])
        return [(item["start"], item["end"]) for item in controls]

    def test_remainder_placement_and_center_odd_element_count(self):
        for cls, expected in (
            (MeshControl_ZsectionAvg, [(2, 37)]),
            (MeshControl_ZsectionTop, [(2, 32), (32, 37)]),
            (MeshControl_ZsectionBot, [(2, 7), (7, 37)]),
            (MeshControl_ZsectionCenter, [(2, 12), (12, 17), (17, 37)]),
        ):
            with self.subTest(cls=cls):
                self.assertEqual(self.intervals(cls, 2, 37), expected)

    def test_center_short_interval_and_even_element_count(self):
        for length, expected in (
            (15, [(0, 5), (5, 15)]),
            (25, [(0, 10), (10, 15), (15, 25)]),
            (45, [(0, 20), (20, 25), (25, 45)]),
        ):
            with self.subTest(length=length):
                self.assertEqual(self.intervals(MeshControl_ZsectionCenter, end=length), expected)

    def test_exact_division_and_interval_smaller_than_size_remain_single(self):
        for cls in SECTION_CLASSES:
            for length in (5, 10, 20, 30):
                with self.subTest(cls=cls, length=length):
                    self.assertEqual(self.intervals(cls, end=length, min_dis=0), [(0, length)])

    def test_floating_division_boundaries_do_not_create_sliver_sections(self):
        for cls in SECTION_CLASSES:
            for length in (0.3, 0.3 - 1e-14, 0.3 + 1e-14):
                with self.subTest(cls=cls, length=length):
                    self.assertEqual(self.intervals(cls, end=length, size=0.1, min_dis=0), [(0, length)])

    def test_tolerance_suppresses_only_small_remainder(self):
        for cls in SECTION_CLASSES[1:]:
            with self.subTest(cls=cls):
                self.assertEqual(self.intervals(cls, end=30.005), [(0, 30.005)])
                self.assertGreater(len(self.intervals(cls, end=30.005, min_dis=0.001)), 1)
                self.assertGreater(len(self.intervals(cls, end=35, min_dis=5)), 1)

    def test_reference_coordinates_and_input_immutability(self):
        geometry = box((0, 0, 10), (2, 3, 10), 40)
        for cls in SECTION_CLASSES:
            entry = reference_control(section_control(cls.METHOD), {"kind": "root"})
            entry["startZ"]["offset"] = -2
            entry["endZ"]["offset"] = 3
            before = copy.deepcopy((entry, geometry))
            with self.subTest(cls=cls):
                obj = cls(entry, geometry)
                self.assertEqual((obj.start_z, obj.end_z), (8, 53))
                obj.get_25D_z_controls()
                self.assertEqual((entry, geometry), before)

    def test_rejects_invalid_sizes_and_intervals(self):
        for cls in SECTION_CLASSES:
            for size in (0, -1, True, None, float("nan"), float("inf")):
                with self.subTest(cls=cls, size=size), self.assertRaisesRegex(ValueError, "elementSize"):
                    cls(section_control(cls.METHOD, size=size))
            for start, end in ((5, 5), (6, 5), (float("nan"), 5), (0, float("inf")), (-1e308, 1e308)):
                with self.subTest(cls=cls, start=start, end=end), self.assertRaises(ValueError):
                    cls(section_control(cls.METHOD, start, end))

    def test_rejects_invalid_min_distance(self):
        for cls in SECTION_CLASSES:
            obj = cls(section_control(cls.METHOD))
            for value in (-1, float("nan"), float("inf"), True, None):
                with self.subTest(cls=cls, value=value), self.assertRaisesRegex(ValueError, "min_dis"):
                    obj.get_25D_z_controls(value)

    def test_rejects_collapsed_or_overflowing_reference_interval(self):
        entry = reference_control(section_control(), {"kind": "root"})
        entry["startZ"]["offset"] = 2
        with self.assertRaisesRegex(ValueError, "startZ < endZ"):
            MeshControl_ZsectionAvg(entry, box())
        entry["startZ"]["offset"] = 1e308
        with self.assertRaisesRegex(ValueError, "finite"):
            MeshControl_ZsectionAvg(entry, box((0, 0, 1e308), (1, 1, 1e308), 0))


class MeshControlInterfaceTests(unittest.TestCase):
    def test_base_interfaces_are_abstract(self):
        for cls in (MeshControl, ZSectionControl):
            with self.subTest(cls=cls), self.assertRaises(TypeError):
                cls(section_control())

    def test_each_z_control_explicitly_has_no_xy_controls(self):
        for cls in SECTION_CLASSES:
            with self.subTest(cls=cls):
                self.assertEqual(cls(section_control(cls.METHOD)).get_25D_xy_controls(), [])
        self.assertEqual(MeshControl_Zpoint(point_control()).get_25D_xy_controls(), [])

    def test_point_uses_global_size_without_requiring_local_size(self):
        entry = point_control(-2.5)
        obj = MeshControl_Zpoint(entry)
        self.assertEqual(obj.get_25D_z_controls(global_element_size=20), [
            {"element_size": 20, "method": "Z_POINT", "start": -2.5, "end": -2.5},
        ])
        self.assertNotIn("elementSize", entry)
        for size in (None, 0, -1, True, float("nan"), float("inf")):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "global_element_size"):
                obj.get_25D_z_controls(global_element_size=size)

    def test_point_reference_supports_both_z_anchors_and_offsets(self):
        for anchor, offset, expected in (("z_min", -2, 8), ("z_max", 3, 53)):
            entry = reference_control(point_control(), {"kind": "root"})
            entry["z"] = {"anchor": anchor, "offset": offset}
            with self.subTest(anchor=anchor):
                self.assertEqual(MeshControl_Zpoint(entry, box((0, 0, 10), (1, 1, 10), 40)).z, expected)

    def test_point_rejects_nonfinite_and_wrong_axis_coordinates(self):
        for z in (float("nan"), float("inf"), True):
            with self.subTest(z=z), self.assertRaisesRegex(ValueError, "z"):
                MeshControl_Zpoint(point_control(z))
        entry = reference_control(point_control(), {"kind": "root"})
        entry["z"]["anchor"] = "x_min"
        with self.assertRaisesRegex(ValueError, "anchor"):
            MeshControl_Zpoint(entry, box())

    def test_sections_reject_wrong_axis_and_legacy_bare_anchors(self):
        for cls in SECTION_CLASSES:
            for anchor in ("x_min", "min", "z_center"):
                entry = reference_control(section_control(cls.METHOD), {"kind": "root"})
                entry["startZ"]["anchor"] = anchor
                with self.subTest(cls=cls, anchor=anchor), self.assertRaisesRegex(ValueError, "anchor"):
                    cls(entry, box())

    def test_each_reference_control_requires_geometry(self):
        for cls in SECTION_CLASSES:
            entry = reference_control(section_control(cls.METHOD), {"kind": "root"})
            with self.subTest(cls=cls), self.assertRaisesRegex(ValueError, "reference geometry"):
                cls(entry)
        with self.assertRaisesRegex(ValueError, "reference geometry"):
            MeshControl_Zpoint(reference_control(point_control(), {"kind": "root"}))

    def test_geometry_bounds_are_resolved_once_per_control_initialization(self):
        region_entry = MeshControl3dRegionTests().control(MeshControl3dRegionTests().box_region())
        cases = [(cls, reference_control(section_control(cls.METHOD), {"kind": "root"}))
                 for cls in SECTION_CLASSES]
        cases += [(MeshControl_Zpoint, reference_control(point_control(), {"kind": "root"})),
                  (MeshControl_3dRegion, region_entry)]
        for cls, entry in cases:
            with self.subTest(cls=cls), patch(
                "mesher.meshcontrol.methods._geometry_bounding",
                wraps=_geometry_bounding,
            ) as bounding:
                cls(entry, box())
                self.assertEqual(bounding.call_count, 1)

    def test_region_projects_to_xy_without_z_constraints(self):
        entry = {"method": "3D_REGION", "coordinateMode": "absolute", "elementSize": 5,
                 "region": {"type": "box", "min": {"x": 0, "y": 0, "z": 0},
                            "max": {"x": 1, "y": 1, "z": 1}}}
        obj = MeshControl_3dRegion(entry)
        self.assertEqual(obj.get_25D_z_controls(global_element_size=10), [])
        self.assertEqual(obj.get_25D_xy_controls(), [{
            "region": {"type": "BOX", "dim": [0, 0, 1, 1]},
            "element_size": 5,
        }])
        for bad in (-1, float("inf")):
            with self.assertRaisesRegex(ValueError, "min_dis"):
                obj.get_25D_z_controls(bad)
        entry["region"]["min"]["x"] = float("nan")
        with self.assertRaisesRegex(ValueError, "region.min.x"):
            MeshControl_3dRegion(entry)


class MeshControlFactoryTests(unittest.TestCase):
    def test_dispatches_all_methods_and_keeps_input_order(self):
        entries = [section_control(cls.METHOD) for cls in SECTION_CLASSES]
        entries += [point_control(), {"method": "3D_REGION", "coordinateMode": "absolute",
                    "elementSize": 5, "region": {"type": "box",
                    "min": {"x": 0, "y": 0, "z": 0}, "max": {"x": 1, "y": 1, "z": 1}}}]
        before = copy.deepcopy(entries)
        # Absolute controls must not inspect even an absent container.
        objects = get_control_obj(None, entries)
        self.assertEqual([type(obj) for obj in objects], [*SECTION_CLASSES, MeshControl_Zpoint, MeshControl_3dRegion])
        self.assertEqual(entries, before)
        self.assertEqual(get_control_obj(None, []), [])

    def test_root_reference_bounds_include_descendants(self):
        container = {"bodies": [{"geometry": box((0, 0, 10), (1, 1, 10), 2)}],
                     "children": [{"vias": [{"geometry": box((0, 0, -3), (1, 1, -3), 20)}]}]}
        entry = reference_control(section_control(), {"kind": "root"})
        before = copy.deepcopy((container, entry))
        objects = get_control_obj(container, [entry])
        self.assertEqual([(obj.start_z, obj.end_z) for obj in objects], [(-3, 17)])
        self.assertEqual((container, entry), before)

    def test_all_feature_kinds_match_every_item_in_preorder(self):
        for kind, field in (("body", "bodies"), ("via", "vias"), ("circuit", "circuits"), ("bump", "bumps")):
            def feature(z, item_id="target"):
                return {"id": item_id, "geometry": box((0, 0, z), (1, 1, z), 1)}
            container = {field: [feature(1), feature(2, "other"), feature(3)], "children": [
                {field: [feature(4)], "children": [{field: [feature(5)]}]},
                {field: [feature(6)]},
            ]}
            before = copy.deepcopy(container)
            entry = reference_control(point_control(), {"kind": kind, "id": "target"})
            with self.subTest(kind=kind):
                self.assertEqual([obj.z for obj in get_control_obj(container, [entry])], [1, 3, 4, 5, 6])
                self.assertEqual(container, before)

    def test_nested_containers_match_ancestors_and_descendants_without_pruning(self):
        container = {"key": "same", "bodies": [{"geometry": box((0, 0, 1), (1, 1, 1), 1)}],
                     "children": [{"key": "same", "children": [{"key": "same", "bodies": [
                         {"geometry": box((0, 0, 5), (1, 1, 5), 2)}]}]}]}
        entry = reference_control(section_control(), {"kind": "container", "key": "same"})
        self.assertEqual([(obj.start_z, obj.end_z) for obj in get_control_obj(container, [entry])],
                         [(1, 7), (5, 7), (5, 7)])

    def test_key_and_id_must_both_match(self):
        entries = [{"key": key, "id": item_id, "geometry": box((0, 0, z), (1, 1, z), 1)}
                   for z, key, item_id in ((1, "a", "x"), (2, "a", "y"), (3, "b", "x"), (4, "a", "x"))]
        entry = reference_control(point_control(), {"kind": "body", "key": "a", "id": "x"})
        self.assertEqual([obj.z for obj in get_control_obj({"bodies": entries}, [entry])], [1, 4])

    def test_nullable_and_missing_lists_are_empty_during_search(self):
        container = {"bodies": None, "children": [{"bodies": [{"id": "match", "geometry": box()}],
                                                   "children": None}]}
        entry = reference_control(point_control(), {"kind": "body", "id": "match"})
        self.assertEqual([obj.z for obj in get_control_obj(container, [entry])], [0])

    def test_missing_reference_target_reports_control_index_label_and_selector(self):
        entry = reference_control(point_control(), {"kind": "body", "id": "missing"})
        with self.assertRaisesRegex(ValueError, r"controls\[1\].*Slice.*reference.*missing"):
            get_control_obj({}, [point_control(), entry])
        del entry["label"]
        with self.assertRaisesRegex(ValueError, r"controls\[0\].*missing"):
            get_control_obj({}, [entry])

    def test_missing_reference_unknown_kind_mode_and_method_are_errors(self):
        for entry, message in (
            ({**point_control(), "method": "UNKNOWN"}, "method"),
            ({**point_control(), "coordinateMode": "relative"}, "coordinateMode"),
            ({**point_control(), "coordinateMode": "reference"}, "reference"),
            (reference_control(point_control(), {"kind": "unknown"}), "reference kind"),
        ):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, message):
                get_control_obj({}, [entry])

    def test_direct_constructor_rejects_wrong_method_and_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "method"):
            MeshControl_ZsectionTop(section_control())
        with self.assertRaisesRegex(ValueError, "coordinateMode"):
            MeshControl_Zpoint({**point_control(), "coordinateMode": "other"})

    def test_nested_geometry_errors_keep_full_path(self):
        container = {"children": [{"bodies": [{"geometry": {**box(), "thk": float("inf")}}]}]}
        with self.assertRaisesRegex(ValueError, r"container.children\[0\].bodies\[0\].geometry.thk"):
            _container_bounding_geometry(container)


if __name__ == "__main__":
    unittest.main()
