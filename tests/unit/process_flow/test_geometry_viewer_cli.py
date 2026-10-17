import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


VIEWER = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "script" / "geometry_viewer.py")
)


class GeometryViewerCliTests(unittest.TestCase):
    def test_global_size_precedence_and_local_control_preservation(self):
        structure = {"container": {}}
        local_controls = [{"method": "Z_SECTION_AVG", "elementSize": 10.0}]
        with tempfile.TemporaryDirectory() as temp_dir:
            geometry_path = Path(temp_dir) / "geometry.json"
            control_path = Path(temp_dir) / "control.json"
            for source, original_size in (
                ("default", 1000.0), ("embedded", 400.0), ("external", 600.0)
            ):
                document = {"structure": structure}
                if source != "default":
                    document["meshControl"] = {
                        "globalElementSize": 400.0,
                        "controls": local_controls,
                    }
                geometry_path.write_text(json.dumps(document), encoding="utf-8")
                control_path.write_text(
                    json.dumps({"globalElementSize": 600.0, "controls": local_controls}),
                    encoding="utf-8",
                )
                for override in (None, "200.5"):
                    with self.subTest(source=source, override=override):
                        argv = ["--json", str(geometry_path)]
                        if source == "external":
                            argv += ["--mesh-control", str(control_path)]
                        if override is not None:
                            argv += ["--element_size", override]
                        mesh25d = ModuleType("mesher.mesh25d")
                        mesh25d.build_mesh = Mock(return_value=object())
                        visualization = ModuleType("mesher.mesh3d.visualization")
                        visualization.MeshViewer = Mock()
                        with patch.dict(sys.modules, {
                            "mesher.mesh25d": mesh25d,
                            "mesher.mesh3d.visualization": visualization,
                        }):
                            self.assertEqual(VIEWER["main"](argv), 0)
                        builder = mesh25d.build_mesh
                        builder.assert_called_once()
                        actual_structure, control = builder.call_args.args
                        self.assertEqual(actual_structure, structure)
                        self.assertEqual(
                            control["globalElementSize"],
                            float(override) if override is not None else original_size,
                        )
                        self.assertEqual(
                            control["controls"], [] if source == "default" else local_controls
                        )
                        visualization.MeshViewer.assert_called_once_with(builder.return_value)
                        visualization.MeshViewer.return_value.show.assert_called_once_with(
                            randomize_colors=False
                        )

    def test_invalid_element_sizes_report_argparse_errors(self):
        for value in ("invalid", "0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    VIEWER["main"](["--json", "unused.json", f"--element_size={value}"])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--element_size", stderr.getvalue())
                self.assertIn("positive finite number", stderr.getvalue())

    def test_element_size_defaults_to_none(self):
        args = VIEWER["_parser"]().parse_args(["--json", "unused.json"])
        self.assertIsNone(args.element_size)

    def test_help_explains_element_size_override(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit) as error:
                VIEWER["main"](["--help"])
        self.assertEqual(error.exception.code, 0)
        help_text = " ".join(stdout.getvalue().split())
        self.assertIn("--element_size SIZE", help_text)
        self.assertIn("global element size in um", help_text)
        self.assertIn("overrides --mesh-control, embedded meshControl", help_text)
        self.assertIn("preserves local elementSize controls", help_text)


if __name__ == "__main__":
    unittest.main()
