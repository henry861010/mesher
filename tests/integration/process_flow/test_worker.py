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


class WorkerIntegrationTests(unittest.TestCase):
    def _run_worker(
        self,
        input_path: Path,
        output_path: Path,
        *,
        symmetry: str | None = None,
    ):
        env = os.environ.copy()
        source_dir = Path(__file__).resolve().parents[3] / "src"
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(source_dir), existing_pythonpath) if part
        )
        env.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        command = [
            sys.executable,
            "-m",
            "mesher.process_flow.worker",
            str(input_path),
            "1.0",
            str(output_path),
        ]
        if symmetry is not None:
            command.append(symmetry)
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
                    "nodeCount": 8,
                    "elementCount": 1,
                    "componentCount": 2,
                },
            )
            self.assertIn("node_count=8", output_path.read_text(encoding="utf-8"))

    def test_worker_exports_multiple_circles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "geometry.json"
            output_path = Path(temp_dir) / "mesh.cdb"
            input_path.write_text(json.dumps(_multi_circle_structure()), encoding="utf-8")

            result = self._run_worker(input_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["elementCount"], 66)
            self.assertIn("element_count=66", output_path.read_text(encoding="utf-8"))

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
