"""Tests for the standalone, uniform-coordinate process-flow JSON contract."""
import copy
import unittest

from mesher.contracts.process_flow_2_5d import mesh_control_settings, validate_mesh_control


def document(*controls):
    return {"schemaVersion": "1.0.0", "unitSystem": "um", "mesher": "process_flow_2_5d",
            "globalElementSize": 500, "symmetry": "full", "controls": list(controls)}


def location(axis, bound, mode, number=0):
    return {"anchor": f"{axis}_{bound}", "offset": number} if mode == "reference" else number


def control(method="Z_SECTION_AVG", mode="reference", shape="box"):
    entry = {"method": method, "coordinateMode": mode, "label": "Focus"}
    if mode == "reference":
        entry["reference"] = {"kind": "root"}
    if method == "Z_POINT":
        entry["z"] = location("z", "min", mode)
    else:
        entry["elementSize"] = 5
        if method != "3D_REGION":
            entry.update(startZ=location("z", "min", mode), endZ=location("z", "max", mode, 100))
        elif shape == "box":
            entry["region"] = {"type": "box", **{
                bound: {a: location(a, bound, mode, value) for a in "xyz"}
                for bound, value in (("min", 0), ("max", 100))}}
        else:
            entry["region"] = {"type": "polygon", "loops": [[
                {"x": location("x", x, mode, nx), "y": location("y", y, mode, ny)}
                for x, y, nx, ny in (("min", "min", 0, 0), ("max", "min", 100, 0), ("min", "max", 0, 100))
            ]], "startZ": location("z", "min", mode), "endZ": location("z", "max", mode, 100)}
    return entry


class MeshControlContractTests(unittest.TestCase):
    def test_accepts_all_methods_and_both_coordinate_modes_without_mutation(self):
        entries = []
        for mode in ("reference", "absolute"):
            entries.extend(control(method, mode) for method in (
                "Z_SECTION_AVG", "Z_SECTION_TOP", "Z_SECTION_BOT", "Z_SECTION_CENTER", "Z_POINT"))
            entries.extend(control("3D_REGION", mode, shape) for shape in ("box", "polygon"))
        value = document(*entries)
        before = copy.deepcopy(value)
        self.assertEqual(mesh_control_settings(value), (500.0, "full", 14))
        self.assertEqual(value, before)

    def test_empty_controls_and_all_symmetry_modes(self):
        for symmetry in ("full", "upper_half", "right_half", "upper_right_quarter"):
            value = {**document(), "symmetry": symmetry}
            self.assertEqual(mesh_control_settings(value), (500.0, symmetry, 0))

    def test_reference_kinds_and_intersected_key_and_id_are_accepted(self):
        references = [
            {"kind": "root"}, {"kind": "container", "key": "hbm"},
            {"kind": "body", "key": "soc.si", "id": "si"},
            *({"kind": kind, "id": "feature"} for kind in ("body", "container", "via", "circuit", "bump")),
        ]
        for reference in references:
            with self.subTest(reference=reference):
                entry = control()
                entry["reference"] = reference
                validate_mesh_control(document(entry))

    def test_rejects_missing_mode_legacy_relative_mode_and_renamed_method(self):
        for mutation, field in (
            (lambda c: c.pop("coordinateMode"), "coordinateMode"),
            (lambda c: c.update(coordinateMode="relative"), "coordinateMode"),
            (lambda c: c.update(coordinateMode=True), "coordinateMode"),
            (lambda c: c.update(method="REGION_SIZE"), "method"),
        ):
            entry = control()
            mutation(entry)
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, field):
                validate_mesh_control(document(entry))

    def test_requires_reference_in_reference_mode_and_forbids_it_in_absolute_mode(self):
        for mode in ("reference", "absolute"):
            entry = control(mode=mode)
            if mode == "reference":
                entry.pop("reference")
            else:
                entry["reference"] = {"kind": "root"}
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "reference"):
                validate_mesh_control(document(entry))

    def test_rejects_malformed_reference_selectors(self):
        for reference in (None, [], {}, {"kind": "unknown"}, {"kind": "root", "id": "root"},
                          {"kind": "via", "key": "via"}, {"kind": "body"},
                          {"kind": "body", "id": ""}, {"kind": "body", "id": None},
                          {"kind": "body", "id": "si", "key": None}, {"kind": "container", "key": " "}):
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, "reference"):
                validate_mesh_control(document({**control(), "reference": reference}))

    def test_labels_are_required_nonblank_strings_and_point_forbids_size(self):
        for method in ("Z_POINT", "Z_SECTION_AVG", "3D_REGION"):
            for label in (None, "", " ", 12):
                with self.subTest(method=method, label=label), self.assertRaisesRegex(ValueError, "label"):
                    validate_mesh_control(document({**control(method), "label": label}))
            entry = control(method)
            entry.pop("label")
            with self.assertRaisesRegex(ValueError, "label"):
                validate_mesh_control(document(entry))
        with self.assertRaisesRegex(ValueError, "elementSize"):
            validate_mesh_control(document({**control("Z_POINT"), "elementSize": 5}))

    def test_rejects_non_json_or_non_finite_numbers_and_nonpositive_sizes(self):
        for number in (True, False, "5", None, float("nan"), float("inf"), 10 ** 400):
            for field in ("globalElementSize", "elementSize", "z", "offset"):
                value = document(control("Z_POINT", "absolute"))
                if field == "globalElementSize":
                    value[field] = number
                elif field == "elementSize":
                    value["controls"] = [{**control(mode="absolute"), "elementSize": number}]
                elif field == "z":
                    value["controls"][0][field] = number
                else:
                    entry = control()
                    entry["startZ"]["offset"] = number
                    value["controls"] = [entry]
                with self.subTest(number=number, field=field), self.assertRaisesRegex(ValueError, field):
                    validate_mesh_control(value)
        for size in (0, -1):
            with self.assertRaisesRegex(ValueError, "greater than 0"):
                validate_mesh_control(document({**control(), "elementSize": size}))

    def test_rejects_mixed_coordinates_and_old_per_coordinate_wrappers(self):
        for mode in ("reference", "absolute"):
            invalid_locations = (100, {"mode": "relative", "anchor": "z_min", "offset": 0}) if mode == "reference" else (
                {"anchor": "z_min", "offset": 0}, {"mode": "absolute", "value": 100})
            for value in invalid_locations:
                entry = control(mode=mode)
                entry["startZ"] = value
                with self.subTest(mode=mode, value=value), self.assertRaisesRegex(ValueError, "startZ"):
                    validate_mesh_control(document(entry))
        entry = control("3D_REGION")
        entry["region"]["min"]["x"] = 0
        with self.assertRaisesRegex(ValueError, r"region.min.x"):
            validate_mesh_control(document(entry))

    def test_reference_anchors_must_match_the_coordinate_axis(self):
        for shape in ("box", "polygon"):
            entry = control("3D_REGION", shape=shape)
            point = entry["region"]["min"] if shape == "box" else entry["region"]["loops"][0][0]
            point["x"]["anchor"] = "y_min"
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, r"\.x.anchor"):
                validate_mesh_control(document(entry))
        entry = control()
        entry["startZ"]["anchor"] = "x_min"
        with self.assertRaisesRegex(ValueError, "startZ.anchor"):
            validate_mesh_control(document(entry))

    def test_region_requires_supported_shape_and_exact_point_fields(self):
        for region in (None, [], {}, {"type": "cone"}, {"type": "box", "min": {}, "max": {}},
                       {"type": "box", "min": {"x": 0, "y": 0}, "max": {"x": 1, "y": 1, "z": 1}}):
            entry = control("3D_REGION", "absolute")
            entry["region"] = region
            with self.subTest(region=region), self.assertRaisesRegex(ValueError, "region"):
                validate_mesh_control(document(entry))
        entry = control("3D_REGION", "absolute", "polygon")
        entry["region"]["loops"][0][0]["z"] = 0
        with self.assertRaisesRegex(ValueError, "unsupported field.*z"):
            validate_mesh_control(document(entry))

    def test_polygon_loop_shape_and_each_z_coordinate_are_checked(self):
        for loops in (None, {}, [], [[]], [[{"x": 0, "y": 0}]], [[None, None, None]]):
            entry = control("3D_REGION", "absolute", "polygon")
            entry["region"]["loops"] = loops
            with self.subTest(loops=loops), self.assertRaisesRegex(ValueError, "loops"):
                validate_mesh_control(document(entry))
        entry = control("3D_REGION", "absolute", "polygon")
        entry["region"]["endZ"] = {"anchor": "z_max", "offset": 0}
        with self.assertRaisesRegex(ValueError, "endZ"):
            validate_mesh_control(document(entry))

    def test_multiple_polygon_loops_are_preserved(self):
        entry = control("3D_REGION", "absolute", "polygon")
        entry["region"]["loops"].append([{"x": 10, "y": 10}, {"x": 20, "y": 10}, {"x": 10, "y": 20}])
        before = copy.deepcopy(entry)
        validate_mesh_control(document(entry))
        self.assertEqual(entry, before)

    def test_envelope_and_control_types_are_strict(self):
        for value in (None, [], 12, "json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_mesh_control(value)
        for key, value in (("schemaVersion", "2"), ("unitSystem", "mm"), ("mesher", "unknown"),
                           ("symmetry", "quarter"), ("controls", {}), ("extra", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mesh_control({**document(), key: value})
        with self.assertRaisesRegex(ValueError, r"controls\[0\]"):
            validate_mesh_control(document(None))
