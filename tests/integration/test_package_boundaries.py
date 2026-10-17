import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from mesher import ElementType2D, Mesh2D, Mesh3D


class PackageBoundaryTests(unittest.TestCase):
    def test_root_exports_only_core_mesh_types(self):
        import mesher

        self.assertEqual(
            set(mesher.__all__),
            {"ElementType2D", "Mesh2D", "Mesh3D"},
        )
        self.assertIs(mesher.Mesh2D, Mesh2D)
        self.assertIs(mesher.Mesh3D, Mesh3D)
        self.assertIs(mesher.ElementType2D, ElementType2D)
        self.assertFalse(hasattr(mesher, "ElementType3D"))

        import mesher.mesh3d as mesh3d

        self.assertFalse(hasattr(mesh3d, "ElementType3D"))

    def test_removed_top_level_domain_modules_are_not_shimmed(self):
        for module_name in (
            "mesher.circular",
            "mesher.generators",
            "mesher.quality",
            "mesher.visualization",
            "mesher.process_flow.pipeline",
            "mesher.process_flow.domain",
            "mesher.process_flow.circle_planning",
            "mesher.process_flow.translation.standard_v1",
            "mesher.process_flow.translation.parse_meshcontrol",
        ):
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_core_import_does_not_load_optional_dependencies(self):
        repository_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository_root / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mesher; "
                    "optional = {'matplotlib', 'pyvista', 'PySide6', "
                    "'process_flow_kernel', 'shapely'}; "
                    "print(','.join(sorted(optional.intersection(sys.modules))))"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_mesh_control_contract_import_does_not_load_meshing_runtime(self):
        repository_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository_root / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from mesher.contracts.process_flow_2_5d import "
                    "validate_mesh_control; "
                    "runtime = {'matplotlib', 'pyvista', 'PySide6', "
                    "'process_flow_kernel', 'shapely'}; "
                    "print(','.join(sorted(runtime.intersection(sys.modules))))"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


    def test_control_methods_and_interfaces_import_without_meshing_runtime(self):
        repository_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository_root / "src")
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; from mesher.meshcontrol import get_control_obj; "
                "from mesher.mesh2d.controls import XYRegionControl; "
                "from mesher.mesh3d.controls import ZControl, LayerInfo; "
                "from mesher.mesh25d import build_mesh, SymmetryMode; "
                "runtime = {'matplotlib', 'pyvista', 'PySide6', "
                "'process_flow_kernel', 'shapely', 'mesher.mesh25d.pipeline', "
                "'mesher.mesh3d.extrusion'}; "
                "print(','.join(sorted(runtime.intersection(sys.modules))))"
            )],
            check=False, capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_dimension_algorithms_do_not_import_orchestration(self):
        repository_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository_root / "src")
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; import mesher.mesh2d.circular; "
                "from mesher.mesh3d.extrusion import Dragger; "
                "print(','.join(sorted(name for name in sys.modules if "
                "name.startswith(('mesher.mesh25d', 'mesher.meshcontrol', "
                "'mesher.process_flow')))))"
            )],
            check=False, capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
