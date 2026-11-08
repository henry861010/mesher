"""Public process-flow behavior with the experimental region adapter enabled."""

import copy
import unittest

import numpy as np

from mesher.mesh25d import build_mesh


def _geometry(bounds=(-20, -20, 20, 20), thickness=4):
    x0, y0, x1, y1 = bounds
    return {"type": "BoxGeometry", "bottom_left": [x0, y0, 0],
            "top_right": [x1, y1, 0], "thk": thickness}


def _structure(bounds=(-20, -20, 20, 20), thickness=4):
    return {"root": {
        "bodies": [{"id": "body", "geometry": _geometry(bounds, thickness), "material": "Si"}],
        "vias": [], "circuits": [], "bumps": [], "children": [],
    }}


def _region(bounds=(-2, -2, 2, 2), size=0.5):
    x0, y0, x1, y1 = bounds
    return {"method": "3D_REGION", "coordinateMode": "absolute", "label": "Local region",
            "elementSize": size, "region": {"type": "box",
            "min": {"x": x0, "y": y0, "z": 100},
            "max": {"x": x1, "y": y1, "z": 200}}}


def _settings(controls=(), *, size=2, symmetry="full", strategy="adapter"):
    settings = {"schemaVersion": "1.0.0", "unitSystem": "um", "mesher": "process_flow_2_5d",
                "globalElementSize": size, "symmetry": symmetry, "controls": list(controls)}
    if strategy is not None:
        settings["regionMeshing"] = strategy
    return settings


class RegionAdapterIntegrationTests(unittest.TestCase):
    def test_default_grid_is_identical_to_explicit_grid(self):
        structure = _structure()
        implicit = build_mesh(structure, _settings([_region()], strategy=None))
        explicit = build_mesh(structure, _settings([_region()], strategy="grid"))
        np.testing.assert_array_equal(implicit.nodes, explicit.nodes)
        np.testing.assert_array_equal(implicit.elements, explicit.elements)
        np.testing.assert_array_equal(implicit.element_comps, explicit.element_comps)

    def test_reference_region_and_symmetry_preserve_z_control_and_inputs(self):
        region = _region((6, 6, 10, 10))
        region["coordinateMode"] = "reference"
        region["reference"] = {"kind": "body", "id": "body"}
        region["region"]["min"] = {
            "x": {"anchor": "x_min", "offset": 26},
            "y": {"anchor": "y_min", "offset": 26},
            "z": {"anchor": "z_min", "offset": 100},
        }
        region["region"]["max"] = {
            "x": {"anchor": "x_max", "offset": -10},
            "y": {"anchor": "y_max", "offset": -10},
            "z": {"anchor": "z_max", "offset": 100},
        }
        point = {"method": "Z_POINT", "coordinateMode": "absolute", "label": "Layer cut", "z": 1}
        for symmetry in ("full", "right_half", "upper_half", "upper_right_quarter"):
            with self.subTest(symmetry=symmetry):
                structure = _structure()
                settings = _settings([point, region], symmetry=symmetry)
                before = copy.deepcopy((structure, settings))
                events = []
                mesh = build_mesh(structure, settings, progress=events.append)
                np.testing.assert_array_equal(np.unique(mesh.nodes[:, 2]), [0, 1, 4])
                if symmetry in ("right_half", "upper_right_quarter"):
                    self.assertTrue(np.all(mesh.nodes[:, 0] >= 0))
                if symmetry in ("upper_half", "upper_right_quarter"):
                    self.assertTrue(np.all(mesh.nodes[:, 1] >= 0))
                self.assertEqual((structure, settings), before)
                analysis = next(e["data"] for e in events if e["event"] == "stage.completed"
                                and e["stage"] == "analyzing_geometry")
                self.assertEqual(analysis["regionMeshing"], "adapter")
                self.assertEqual(len(analysis["adapterRegions"]), 1)
                self.assertEqual(analysis["adapterRegions"][0]["seedSize"], 0.5)

    def test_narrow_koz_requires_explicit_precision_and_survives_adapter_assembly(self):
        structure = _structure()
        structure["root"]["vias"] = [{
            "id": "metal", "geometry": _geometry((-2, -2, 2, 2)),
            "material": "Cu", "density": 100, "koz": 0.0001,
        }]
        settings = _settings([_region((-4, -4, 4, 4), 1)])
        with self.assertRaisesRegex(ValueError, "KOZ width must exceed geometryTolerance"):
            build_mesh(structure, settings)
        settings["geometryTolerance"] = 1e-6
        mesh = build_mesh(structure, settings)
        metal = mesh.elements[mesh.element_comps == mesh.comps["Cu"], :8]
        self.assertGreater(len(metal), 0)
        corners = mesh.nodes[metal, :2]
        self.assertTrue(np.all(np.abs(corners) <= 1.9999 + 1e-10))
        for coordinate in (-2, -1.9999, 1.9999, 2):
            self.assertTrue(np.any(np.isclose(mesh.nodes[:, 0], coordinate, atol=1e-10, rtol=0)))
        np.testing.assert_array_equal(np.unique(mesh.nodes[:, 2]), [0, 2, 4])

    def test_material_boundary_crossing_transition_preserves_exact_metal_volume(self):
        structure = _structure()
        structure["root"]["vias"] = [{
            "id": "metal", "geometry": _geometry((-3, -3, 3, 3)),
            "material": "Cu", "density": 100, "koz": 0.25,
        }]
        mesh = build_mesh(structure, _settings([_region()]))
        metal = mesh.elements[mesh.element_comps == mesh.comps["Cu"], :8]
        self.assertGreater(len(metal), 0)
        corners = mesh.nodes[metal]
        self.assertTrue(np.all(np.abs(corners[:, :, :2]) <= 2.75 + 1e-10))
        bottom = corners[:, :4, :2]
        areas = 0.5 * np.abs(np.sum(
            bottom[..., 0] * np.roll(bottom[..., 1], -1, axis=1)
            - bottom[..., 1] * np.roll(bottom[..., 0], -1, axis=1), axis=1))
        volumes = areas * np.ptp(corners[:, :, 2], axis=1)
        self.assertAlmostEqual(float(volumes.sum()), 5.5 ** 2 * 4, places=9)

    def test_circular_feature_is_imprinted_after_adapter_completion(self):
        structure = _structure((-60, -60, 60, 60), thickness=6)
        structure["root"]["vias"] = [{
            "id": "round-via", "geometry": {"type": "CylinderGeometry", "center": [20, 0, 0],
            "bottom_radius": 9, "thk": 6}, "material": "Cu", "density": 100, "koz": 0,
        }]
        events = []
        mesh = build_mesh(structure,
                          _settings([_region((-24, -4, -16, 4), 1)], size=3),
                          progress=events.append)
        radii = np.linalg.norm(mesh.nodes[:, :2] - [20, 0], axis=1)
        self.assertGreater(np.count_nonzero(np.isclose(radii, 9, atol=1e-9, rtol=0)), 8)
        self.assertTrue(np.any(mesh.element_comps == mesh.comps["Cu"]))
        operations = [e["data"] for e in events if e["event"] == "item.completed"
                      and e["stage"] == "building_2d_mesh" and e["data"].get("operation") == "imprint"]
        self.assertEqual(len(operations), 1)
        self.assertIn("round-via", operations[0]["sourceRefs"])
        np.testing.assert_array_equal(np.unique(mesh.nodes[:, 2]), [0, 3, 6])

    def test_reference_removed_by_symmetry_does_not_activate_adapter(self):
        structure = _structure()
        structure["root"]["bodies"].append({
            "id": "removed", "geometry": _geometry((-18, -2, -14, 2)), "material": "Si",
        })
        region = _region()
        region.update(coordinateMode="reference", reference={"kind": "body", "id": "removed"})
        region["region"]["min"] = {axis: {"anchor": f"{axis}_min", "offset": 0} for axis in "xyz"}
        region["region"]["max"] = {axis: {"anchor": f"{axis}_max", "offset": 0} for axis in "xyz"}
        expected = build_mesh(structure, _settings(symmetry="right_half", strategy="grid"))
        actual = build_mesh(structure, _settings([region], symmetry="right_half"))
        np.testing.assert_array_equal(actual.nodes, expected.nodes)
        np.testing.assert_array_equal(actual.elements, expected.elements)


if __name__ == "__main__":
    unittest.main()
