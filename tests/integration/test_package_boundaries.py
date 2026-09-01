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
                    "'process_flow_kernel'}; "
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


if __name__ == "__main__":
    unittest.main()
