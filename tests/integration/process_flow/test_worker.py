import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _box_structure():
    return {
        "root": {
            "bodies": [
                {
                    "geometry": {
                        "type": "BoxGeometry",
                        "bottom_left": [0.0, 0.0, 0.0],
                        "top_right": [1.0, 1.0, 0.0],
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
    }


def _multi_circle_structure():
    return {
        "root": {
            "bodies": [
                {
                    "geometry": {
                        "type": "CylinderGeometry",
                        "center": [-3.0, 0.0, 0.0],
                        "bottom_radius": 2.0,
                        "thk": 1.0,
                    },
                    "material": "Si",
                },
                {
                    "geometry": {
                        "type": "CylinderGeometry",
                        "center": [3.0, 0.0, 0.0],
                        "bottom_radius": 2.0,
                        "thk": 1.0,
                    },
                    "material": "Si",
                },
            ],
            "vias": [],
            "circuits": [],
            "bumps": [],
            "children": [],
        }
    }


def _mesh_control(*, element_size=1.0, symmetry="full", controls=None, region_meshing=None):
    document = {
        "schemaVersion": "1.0.0",
        "unitSystem": "um",
        "mesher": "process_flow_2_5d",
        "globalElementSize": element_size,
        "symmetry": symmetry,
        "controls": [] if controls is None else controls,
    }
    if region_meshing is not None:
        document["regionMeshing"] = region_meshing
    return document


def _adapter_fixture(*, touches_boundary=False):
    structure = _box_structure()
    structure["root"]["bodies"][0]["geometry"]["top_right"] = [12.0, 12.0, 0.0]
    low, high = (0, 4) if touches_boundary else (4, 8)
    control = {
        "method": "3D_REGION", "coordinateMode": "absolute", "label": "Fine focus",
        "elementSize": 0.2,
        "region": {"type": "box",
                   "min": {"x": low, "y": low, "z": 20},
                   "max": {"x": high, "y": high, "z": 30}},
    }
    return structure, [control]


class WorkerIntegrationTests(unittest.TestCase):
    def _run_worker(
        self,
        input_path: Path,
        output_path: Path,
        *,
        symmetry: str | None = None,
        controls=None,
        region_meshing=None,
    ):
        env = os.environ.copy()
        source_dir = Path(__file__).resolve().parents[3] / "src"
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(source_dir), existing_pythonpath) if part
        )
        env.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        mesh_control_path = input_path.with_name("mesh_control.json")
        mesh_control_path.write_text(
            json.dumps(
                _mesh_control(
                    symmetry="full" if symmetry is None else symmetry,
                    controls=controls,
                    region_meshing=region_meshing,
                )
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "mesher.process_flow.worker",
            str(input_path),
            str(mesh_control_path),
            str(output_path),
        ]
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_worker_writes_cdb_and_json_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(_box_structure()), encoding="utf-8")

            result = self._run_worker(input_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "outputPath": str(output_path),
                    "nodeCount": 18,
                    "elementCount": 4,
                    "componentCount": 2,
                    "typeCount": 1,
                    "realCount": 0,
                    "sectionCount": 0,
                    "setCount": 0,
                    "geometryTolerance": 0.001,
                    "unitSystem": "um",
                    "tolerancePolicyVersion": "1",
                },
            )
            self.assertIn("node_count=18", output_path.read_text(encoding="utf-8"))

    def test_worker_exports_multiple_circles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(_multi_circle_structure()), encoding="utf-8")

            result = self._run_worker(input_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            count = json.loads(result.stdout)["elementCount"]
            self.assertGreater(count, 0)
            self.assertIn(f"element_count={count}", output_path.read_text(encoding="utf-8"))

    def test_worker_accepts_an_optional_quarter_model(self):
        structure = _box_structure()
        geometry = structure["root"]["bodies"][0]["geometry"]
        geometry["bottom_left"] = [10.0, 20.0, 0.0]
        geometry["top_right"] = [14.0, 24.0, 0.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "quarter.cdb"
            input_path.write_text(json.dumps(structure), encoding="utf-8")

            result = self._run_worker(
                input_path,
                output_path,
                symmetry="upper_right_quarter",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)
            self.assertEqual(metadata["elementCount"], 4)
            node_lines = output_path.read_text(encoding="utf-8").split(
                "\n*ELEMENTS",
                maxsplit=1,
            )[0].split("\n*NODES,index,x,y,z\n", maxsplit=1)[1]
            coordinates = [
                [float(value) for value in line.split(",")[1:3]]
                for line in node_lines.splitlines()
                if line
            ]
            self.assertTrue(all(x >= 12.0 and y >= 22.0 for x, y in coordinates))

    def test_worker_rejects_an_unknown_symmetry_without_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(_box_structure()), encoding="utf-8")

            result = self._run_worker(
                input_path,
                output_path,
                symmetry="Upper_Model",
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("symmetry must be one of", result.stderr)
            self.assertFalse(output_path.exists())

    def test_worker_applies_mesh_controls_without_a_compatibility_warning(self):
        controls = [
            {
                "method": "Z_SECTION_AVG",
                "coordinateMode": "reference",
                "label": "Base die",
                "reference": {"kind": "container", "key": "hbm"},
                "elementSize": 0.25,
                "startZ": {"anchor": "z_min", "offset": 0},
                "endZ": {"anchor": "z_max", "offset": 0},
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            structure = _box_structure()
            structure["root"]["key"] = "hbm"
            input_path.write_text(json.dumps(structure), encoding="utf-8")

            result = self._run_worker(input_path, output_path, controls=controls)

            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)
            self.assertNotIn("warnings", metadata)
            self.assertGreater(metadata["elementCount"], 1)
            self.assertTrue(output_path.exists())

    def test_worker_applies_xy_region_cuts_and_writes_refined_cdb(self):
        controls = [{
            "method": "3D_REGION", "coordinateMode": "absolute", "label": "XY grid",
            "elementSize": 0.25, "region": {"type": "box",
                "min": {"x": 0, "y": 0, "z": -10},
                "max": {"x": 1, "y": 1, "z": 10},
            },
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(_box_structure()), encoding="utf-8")

            result = self._run_worker(input_path, output_path, controls=controls)

            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)
            self.assertEqual(metadata["elementCount"], 16)
            self.assertEqual(metadata["nodeCount"], 50)
            self.assertTrue(output_path.exists())

    def test_explicit_grid_matches_default_worker_output(self):
        structure, controls = _adapter_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(structure), encoding="utf-8")
            baseline = self._run_worker(input_path, output_path, controls=controls)
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            baseline_cdb = output_path.read_bytes()

            result = self._run_worker(
                input_path, output_path, controls=controls, region_meshing="grid",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), json.loads(baseline.stdout))
            self.assertEqual(output_path.read_bytes(), baseline_cdb)

    def test_worker_exports_adapter_mesh_with_unchanged_metadata_and_z_layers(self):
        structure, controls = _adapter_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(structure), encoding="utf-8")

            result = self._run_worker(
                input_path, output_path, controls=controls, region_meshing="adapter",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)
            self.assertEqual(set(metadata), {
                "outputPath", "nodeCount", "elementCount", "componentCount",
                "typeCount", "realCount", "sectionCount", "setCount",
                "geometryTolerance", "unitSystem", "tolerancePolicyVersion",
            })
            self.assertEqual(metadata["outputPath"], str(output_path))
            self.assertEqual(metadata["componentCount"], 2)
            self.assertGreater(metadata["elementCount"], 144)
            # The equivalent full-axis grid has 28 * 28 XY elements.
            self.assertLess(metadata["elementCount"], 784)
            cdb = output_path.read_text(encoding="utf-8")
            self.assertIn(f"node_count={metadata['nodeCount']}", cdb)
            self.assertIn(f"element_count={metadata['elementCount']}", cdb)
            node_lines = cdb.split("\n*ELEMENTS", maxsplit=1)[0].split(
                "\n*NODES,index,x,y,z\n", maxsplit=1,
            )[1]
            coordinates = [tuple(map(float, line.split(",")[1:]))
                           for line in node_lines.splitlines() if line]
            # The region's Z span lies outside the body and must remain XY-only.
            self.assertEqual({point[2] for point in coordinates}, {0.0, 1.0})
            self.assertTrue(any(abs(x - 4.2) < 1e-9 and abs(y - 4.2) < 1e-9
                                for x, y, _ in coordinates))
            self.assertEqual(list(Path(temp_dir).glob(".mesh.cdb.*.tmp")), [])

    def test_adapter_failure_does_not_publish_or_replace_worker_output(self):
        structure, controls = _adapter_fixture(touches_boundary=True)
        for previous in (None, b"previous-complete-output"):
            with self.subTest(existing_output=previous is not None), tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / "geometry.json"
                output_path = Path(temp_dir) / "mesh.cdb"
                input_path.write_text(json.dumps(structure), encoding="utf-8")
                if previous is not None:
                    output_path.write_bytes(previous)

                result = self._run_worker(
                    input_path, output_path, controls=controls, region_meshing="adapter",
                )

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("adapter", result.stderr.lower())
                self.assertNotIn('"stage":"writing_output"', result.stderr)
                if previous is None:
                    self.assertFalse(output_path.exists())
                else:
                    self.assertEqual(output_path.read_bytes(), previous)
                self.assertEqual(list(Path(temp_dir).glob(".mesh.cdb.*.tmp")), [])

    def test_worker_rejects_invalid_region_meshing_before_analyzing_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text("{}", encoding="utf-8")
            output_path.write_bytes(b"previous-complete-output")

            result = self._run_worker(input_path, output_path, region_meshing="automatic")

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("meshControl.regionMeshing must be one of: grid, adapter", result.stderr)
            self.assertNotIn('"stage":"analyzing_geometry"', result.stderr)
            self.assertEqual(output_path.read_bytes(), b"previous-complete-output")
            self.assertEqual(list(Path(temp_dir).glob(".mesh.cdb.*.tmp")), [])

    def test_adapter_runtime_failure_preserves_output_and_reports_control_context(self):
        structure, controls = _adapter_fixture()
        # The seed is strictly interior, but its one-unit transition reaches
        # the domain boundary and fails inside the real adapter backend.
        controls[0]["region"]["min"]["x"] = 0.5
        controls[0]["region"]["max"]["x"] = 4
        for previous in (None, b"previous-complete-output"):
            with self.subTest(existing_output=previous is not None), tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / "geometry.json"
                output_path = Path(temp_dir) / "mesh.cdb"
                input_path.write_text(json.dumps(structure), encoding="utf-8")
                if previous is not None:
                    output_path.write_bytes(previous)

                result = self._run_worker(
                    input_path, output_path, controls=controls, region_meshing="adapter",
                )

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                prefix = "PROCESS_FLOW_PROGRESS "
                events = [json.loads(line[len(prefix):]) for line in result.stderr.splitlines()
                          if line.startswith(prefix)]
                self.assertIn(("stage.started", "building_2d_mesh"),
                              [(event["event"], event["stage"]) for event in events])
                self.assertFalse(any(event["stage"] == "writing_output" for event in events))
                diagnostic = result.stderr.rstrip().splitlines()[-1]
                self.assertIn("3D_REGION adapter (controls[0] ('Fine focus'))", diagnostic)
                self.assertIn("code=topology_change", diagnostic)
                self.assertIn("context=", diagnostic)
                self.assertIn("'stage': 'final_front_validation'", diagnostic)
                self.assertIn("'region_sources'", diagnostic)
                if previous is None:
                    self.assertFalse(output_path.exists())
                else:
                    self.assertEqual(output_path.read_bytes(), previous)
                self.assertEqual(list(Path(temp_dir).glob(".mesh.cdb.*.tmp")), [])

    def test_worker_does_not_write_partial_output_for_incompatible_circles(self):
        structure = _multi_circle_structure()
        structure["root"]["bodies"][1]["geometry"]["center"][0] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(structure), encoding="utf-8")

            result = self._run_worker(input_path, output_path)

            self.assertEqual(result.returncode, 1)
            self.assertIn("overlapping imprint bands", result.stderr)
            self.assertFalse(output_path.exists())

    def test_worker_returns_nonzero_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text("{}", encoding="utf-8")

            result = self._run_worker(input_path, output_path)

            self.assertEqual(result.returncode, 1)
            self.assertIn("geometryStructure.root must be an object", result.stderr)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
