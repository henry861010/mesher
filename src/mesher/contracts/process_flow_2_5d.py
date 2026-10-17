"""Uniform-coordinate JSON contract for the ``process_flow_2_5d`` mesher.

Validation is syntax-only and does not import the geometry/meshing runtime.
Reference resolution, region bounds and polygon topology are pipeline concerns.
"""

from __future__ import annotations

import math
from typing import Any

JsonObject = dict[str, Any]

SCHEMA_VERSION = "1.0.0"
UNIT_SYSTEM = "um"
MESHER = "process_flow_2_5d"
SYMMETRY_MODES = (
    "full",
    "upper_half",
    "right_half",
    "upper_right_quarter",
)
Z_SECTION_METHODS = (
    "Z_SECTION_AVG",
    "Z_SECTION_TOP",
    "Z_SECTION_BOT",
    "Z_SECTION_CENTER",
)
COORDINATE_MODES = ("reference", "absolute")
REFERENCE_KINDS = ("root", "container", "body", "via", "circuit", "bump")


def validate_mesh_control(mesh_control: JsonObject) -> None:
    """Validate a mesh-control document without resolving geometry references."""
    mesh_control_settings(mesh_control)


def mesh_control_settings(mesh_control: JsonObject) -> tuple[float, str, int]:
    """Validate the document and return settings consumed by the current pipeline."""
    if not isinstance(mesh_control, dict):
        raise ValueError("meshControl must be an object.")
    _require_object_keys(
        mesh_control,
        "meshControl",
        required={
            "schemaVersion",
            "unitSystem",
            "mesher",
            "globalElementSize",
            "symmetry",
            "controls",
        },
    )
    if mesh_control["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError(f"meshControl.schemaVersion must be {SCHEMA_VERSION!r}.")
    if mesh_control["unitSystem"] != UNIT_SYSTEM:
        raise ValueError(f"meshControl.unitSystem must be {UNIT_SYSTEM!r}.")
    if mesh_control["mesher"] != MESHER:
        raise ValueError(f"meshControl.mesher must be {MESHER!r}.")

    element_size = _positive_finite_number(
        mesh_control["globalElementSize"],
        "meshControl.globalElementSize",
    )
    symmetry = mesh_control["symmetry"]
    if not isinstance(symmetry, str) or symmetry not in SYMMETRY_MODES:
        allowed = ", ".join(SYMMETRY_MODES)
        raise ValueError(f"meshControl.symmetry must be one of: {allowed}.")

    controls = mesh_control["controls"]
    if not isinstance(controls, list):
        raise ValueError("meshControl.controls must be a list.")
    for index, control in enumerate(controls):
        _validate_mesh_control_entry(control, f"meshControl.controls[{index}]")
    return element_size, symmetry, len(controls)


def _validate_mesh_control_entry(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    method = value.get("method")
    if isinstance(method, str) and method in Z_SECTION_METHODS:
        fields = {"elementSize", "startZ", "endZ"}
    elif method == "Z_POINT":
        fields = {"z"}
    elif method == "3D_REGION":
        fields = {"elementSize", "region"}
    else:
        allowed = ", ".join([*Z_SECTION_METHODS, "Z_POINT", "3D_REGION"])
        raise ValueError(f"{path}.method must be one of: {allowed}.")

    _require_object_keys(
        value,
        path,
        required={"method", "coordinateMode", "label", *fields},
        optional={"reference"},
    )
    _non_blank_string(value["label"], f"{path}.label")
    mode = value["coordinateMode"]
    if not isinstance(mode, str) or mode not in COORDINATE_MODES:
        raise ValueError(f"{path}.coordinateMode must be 'reference' or 'absolute'.")
    if mode == "reference":
        if "reference" not in value:
            raise ValueError(f"{path}.reference is required in Reference mode.")
        _validate_mesh_control_reference(value["reference"], f"{path}.reference")
    elif "reference" in value:
        raise ValueError(f"{path}.reference is not allowed in Absolute mode.")

    if method != "Z_POINT":
        _positive_finite_number(value["elementSize"], f"{path}.elementSize")
    if method == "3D_REGION":
        _validate_region(value["region"], mode, f"{path}.region")
    else:
        for field in (("z",) if method == "Z_POINT" else ("startZ", "endZ")):
            _validate_coordinate(value[field], "z", mode, f"{path}.{field}")


def _validate_mesh_control_reference(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in REFERENCE_KINDS:
        allowed = ", ".join(REFERENCE_KINDS)
        raise ValueError(f"{path}.kind must be one of: {allowed}.")
    if kind == "root":
        _require_object_keys(value, path, required={"kind"})
        return
    if kind in {"container", "body"}:
        _require_object_keys(
            value,
            path,
            required={"kind"},
            optional={"key", "id"},
        )
    else:
        _require_object_keys(
            value,
            path,
            required={"kind"},
            optional={"id"},
        )
    key = value.get("key")
    item_id = value.get("id")
    if "key" not in value and "id" not in value:
        raise ValueError(f"{path} requires key or id.")
    if "key" in value:
        _non_blank_string(key, f"{path}.key")
    if "id" in value:
        _non_blank_string(item_id, f"{path}.id")


def _validate_coordinate(value: Any, axis: str, mode: str, path: str) -> None:
    if mode == "absolute":
        _finite_number(value, path)
        return
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an anchor/offset object in Reference mode.")
    _require_object_keys(value, path, required={"anchor", "offset"})
    if not isinstance(value["anchor"], str) or value["anchor"] not in {
        f"{axis}_min", f"{axis}_max",
    }:
        raise ValueError(f"{path}.anchor must be '{axis}_min' or '{axis}_max'.")
    _finite_number(value["offset"], f"{path}.offset")


def _validate_region_point(value: Any, axes: str, mode: str, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    _require_object_keys(value, path, required=set(axes))
    for axis in axes:
        _validate_coordinate(value[axis], axis, mode, f"{path}.{axis}")


def _validate_region(value: Any, mode: str, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    shape = value.get("type")
    if shape == "box":
        _require_object_keys(value, path, required={"type", "min", "max"})
        for bound in ("min", "max"):
            _validate_region_point(value[bound], "xyz", mode, f"{path}.{bound}")
    elif shape == "polygon":
        _require_object_keys(value, path, required={"type", "loops", "startZ", "endZ"})
        loops = value["loops"]
        if not isinstance(loops, list) or not loops:
            raise ValueError(f"{path}.loops must be a non-empty list.")
        for loop_index, loop in enumerate(loops):
            loop_path = f"{path}.loops[{loop_index}]"
            if not isinstance(loop, list) or len(loop) < 3:
                raise ValueError(f"{loop_path} must contain at least three vertices.")
            for point_index, point in enumerate(loop):
                _validate_region_point(point, "xy", mode, f"{loop_path}[{point_index}]")
        for field in ("startZ", "endZ"):
            _validate_coordinate(value[field], "z", mode, f"{path}.{field}")
    else:
        raise ValueError(f"{path}.type must be 'box' or 'polygon'.")


def _require_object_keys(
    value: JsonObject,
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}.")
    extra = sorted(set(value) - allowed)
    if extra:
        raise ValueError(f"{path} has unsupported field(s): {', '.join(extra)}.")


def _non_blank_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _positive_finite_number(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return number


def _finite_number(value: Any, name: str) -> float:
    # JSON numbers only: bool is an int subclass, and numeric strings must not
    # silently pass as Absolute coordinates or element sizes.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number
