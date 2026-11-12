"""Generated UF survives normalization, meshing and the actual CDB worker."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from shapely import union_all
from shapely.geometry import Polygon, box

from mesher.mesh25d import build_mesh
from tests.integration.mesh25d.uf_shared_vertex_fixture import mesh_settings, uf_structure


def expected_footprints(model, symmetry):
    base = box(-2, -2, 30, 30)
    dies = union_all([box(x, y, x + 6, y + 6)
                      for x, y in ((0, 0), (0, 10), (10, 10), (10, 20), (20, 0), (20, 20))])
    loops = model["root"]["children"][-1]["bodies"][0]["geometry"]["polys"]
    rings = sorted((Polygon(np.asarray(loop)[:, :2]) for loop in loops), key=lambda p: -p.area)
    uf = Polygon(rings[0].exterior, [ring.exterior for ring in rings[1:]])
    clip = box(14 if symmetry in ("right_half", "upper_right_quarter") else -2,
               14 if symmetry in ("upper_half", "upper_right_quarter") else -2, 30, 30)
    return {name: (shape.intersection(clip), height)
            for name, shape, height in (("base", base, 4), ("Si", dies, 6), ("UF", uf, 6))}


class UfSharedVertexExportTests(unittest.TestCase):
    def assert_material_geometry(self, mesh, expected):
        corners = mesh.nodes[mesh.elements[:, :8]]
        np.testing.assert_array_equal(corners[:, :4, :2], corners[:, 4:8, :2])
        heights = corners[:, 4, 2] - corners[:, 0, 2]
        self.assertTrue(np.all(heights > 0))
        cells = [Polygon(points) for points in corners[:, :4, :2]]
        areas = np.array([cell.area for cell in cells])
        self.assertTrue(np.all(areas > 0))
        self.assertTrue(all(cell.is_valid for cell in cells))
        for name, (footprint, thickness) in expected.items():
            selected = mesh.element_comps == mesh.comps[name]
            self.assertAlmostEqual(float((areas[selected] * heights[selected]).sum()),
                                   footprint.area * thickness, delta=1e-8)
            coverage = union_all([cell for cell, keep in zip(cells, selected) if keep])
            self.assertLess(coverage.symmetric_difference(footprint).area, 1e-8)
        # The UF hole must be occupied by Si, with no UF cells crossing it.
        self.assertFalse(any(cell.intersection(box(10, 10, 16, 16)).area > 1e-8
                             for cell, comp in zip(cells, mesh.element_comps)
                             if comp == mesh.comps["UF"]))

    def test_generated_uf_grid_symmetry_and_real_adapter_preserve_materials(self):
        cases = [(mode, False) for mode in ("full", "upper_half", "right_half", "upper_right_quarter")]
        cases.append(("full", True))
        for symmetry, adapter in cases:
            with self.subTest(symmetry=symmetry, adapter=adapter):
                model = uf_structure()
                settings = mesh_settings(symmetry=symmetry, adapter=adapter)
                original = deepcopy((model, settings))
                events = []
                mesh = build_mesh(model, settings, progress=events.append)
                expected = expected_footprints(model, symmetry)
                self.assert_material_geometry(mesh, expected)
                if symmetry == "full":
                    self.assertEqual(expected["UF"][0].area * 6, 2160)
                if adapter:
                    self.assertTrue(any(event.get("data", {}).get("adapterRegions") for event in events))
                self.assertEqual((model, settings), original)
                self.assertEqual(len(model["root"]["children"][-1]["bodies"]), 1)

    def test_worker_cdb_contains_correct_uf_and_preserves_input_files(self):
        for adapter in (False, True):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                model = uf_structure()
                geometry_path, control_path, output_path = [folder / name for name in (
                    "geometry.json", "control.json", "mesh.cdb")]
                geometry_path.write_text(json.dumps(model), encoding="utf-8")
                control_path.write_text(json.dumps(mesh_settings(adapter=adapter)), encoding="utf-8")
                original = (geometry_path.read_bytes(), control_path.read_bytes())
                env = os.environ.copy()
                source = Path(__file__).resolve().parents[3] / "src"
                env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
                env.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
                result = subprocess.run(
                    [sys.executable, "-m", "mesher.process_flow.worker",
                     str(geometry_path), str(control_path), str(output_path)],
                    capture_output=True, text=True, env=env, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads(result.stdout)
                sections = {}
                for block in output_path.read_text(encoding="utf-8").split("\n*")[1:]:
                    header, *rows = block.strip().splitlines()
                    sections[header.split(",")[0]] = rows
                nodes = np.array([[float(value) for value in row.split(",")[1:]]
                                  for row in sections["NODES"]])
                elements = np.array([[int(value) for value in row.split(",")[1:]]
                                     for row in sections["ELEMENTS"]])
                comps = {json.loads(name): int(index) for index, name in
                         (row.split(",", 1) for row in sections["COMPS"])}
                element_comps = np.array([int(row.split(",")[1]) for row in sections["ELEMENT_COMP"]])
                self.assertEqual(len(nodes), metadata["nodeCount"])
                self.assertEqual(len(elements), metadata["elementCount"])
                self.assertEqual(metadata["geometryTolerance"], .001)
                self.assert_material_geometry(
                    SimpleNamespace(nodes=nodes, elements=elements, comps=comps, element_comps=element_comps),
                    expected_footprints(model, "full"),
                )
                self.assertEqual((geometry_path.read_bytes(), control_path.read_bytes()), original)
                self.assertEqual(list(folder.glob(".mesh.cdb.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
