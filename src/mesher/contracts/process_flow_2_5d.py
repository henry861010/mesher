"""Contract validation for the ``process_flow_2_5d`` mesher."""

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
        _require_object_keys(
            value,
            path,
            required={"method", "elementSize", "startZ", "endZ"},
            optional={"reference"},
        )
        _positive_finite_number(value["elementSize"], f"{path}.elementSize")
        locations = (
            _validate_z_location(value["startZ"], f"{path}.startZ"),
            _validate_z_location(value["endZ"], f"{path}.endZ"),
        )
    elif method == "Z_POINT":
        _require_object_keys(
            value,
            path,
            required={"method", "z"},
            optional={"reference"},
        )
        locations = (_validate_z_location(value["z"], f"{path}.z"),)
    else:
        allowed = ", ".join([*Z_SECTION_METHODS, "Z_POINT"])
        raise ValueError(f"{path}.method must be one of: {allowed}.")

    reference = value.get("reference")
    if reference is not None:
        _validate_mesh_control_reference(reference, f"{path}.reference")
    if any(mode == "relative" for mode in locations) and reference is None:
        raise ValueError(f"{path} relative Z locations require reference.")


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
    if key is None and item_id is None:
        raise ValueError(f"{path} requires key or id.")
    if key is not None:
        _non_blank_string(key, f"{path}.key")
    if item_id is not None:
        _non_blank_string(item_id, f"{path}.id")


def _validate_z_location(value: Any, path: str) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    mode = value.get("mode")
    if mode == "relative":
        _require_object_keys(
            value,
            path,
            required={"mode", "anchor", "offset"},
        )
        if not isinstance(value["anchor"], str) or value["anchor"] not in {
            "z_min",
            "z_max",
        }:
            raise ValueError(f"{path}.anchor must be 'z_min' or 'z_max'.")
        _finite_number(value["offset"], f"{path}.offset")
        return mode
    if mode == "absolute":
        _require_object_keys(value, path, required={"mode", "value"})
        _finite_number(value["value"], f"{path}.value")
        return mode
    raise ValueError(f"{path}.mode must be 'relative' or 'absolute'.")


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
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number
