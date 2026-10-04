"""Build and display a 3D mesh from a process-flow geometry JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a process-flow geometry mesh.")
    parser.add_argument(
        "-json",
        "--json",
        required=True,
        type=Path,
        metavar="PATH",
        help="geometry JSON file (bare geometry or an object containing 'structure')",
    )
    parser.add_argument(
        "--mesh-control",
        type=Path,
        metavar="PATH",
        help=(
            "mesh-control JSON file; when omitted, use the input's 'meshControl' "
            "or the default mesh-control configuration"
        ),
    )
    parser.add_argument(
        "--randomize-colors",
        action="store_true",
        help="randomize component colors in the viewer",
    )
    return parser


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {label} file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in {label} file {path} at "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return value


def _geometry_structure(document: dict[str, Any]) -> dict[str, Any]:
    structure = document.get("structure", document.get("geometryStructure", document))
    if not isinstance(structure, dict):
        raise ValueError("geometry structure must be a JSON object")
    return structure


def _mesh_control(
    document: dict[str, Any],
    mesh_control_path: Path | None,
) -> dict[str, Any]:
    if mesh_control_path is not None:
        mesh_control = _read_json(mesh_control_path, label="mesh-control")
    else:
        embedded = document.get("meshControl")
        if embedded is not None and not isinstance(embedded, dict):
            raise ValueError("meshControl must be a JSON object")
        mesh_control = dict(embedded) if embedded is not None else {
            "schemaVersion": "1.0.0",
            "unitSystem": "um",
            "mesher": "process_flow_2_5d",
            "globalElementSize": 1000.0,
            "symmetry": "full",
            "controls": [],
        }

    return mesh_control


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        document = _read_json(args.json, label="geometry")
        structure = _geometry_structure(document)
        mesh_control = _mesh_control(document, args.mesh_control)
    except ValueError as error:
        parser.error(str(error))

    # This repository uses a src layout. Make a direct checkout invocation work
    # without requiring an editable installation first.
    source_root = str(SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from mesher.mesh3d.visualization import MeshViewer
    from mesher.process_flow import build_mesh_from_structure

    mesh = build_mesh_from_structure(structure, mesh_control)
    MeshViewer(mesh).show(randomize_colors=args.randomize_colors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
