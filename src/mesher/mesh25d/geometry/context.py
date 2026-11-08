"""Prepare an owned geometry tree and its symmetry-reduced footprints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..domain import ModelDomain, SymmetryMode, filter_container_to_domain, model_domain
from .patterns import Pattern2D, extract_pattern
from .primitives import _geometry_to_face
from .normalization import normalize_geometry, _features
from ...tolerance import GeometryTolerance, GeometryToleranceError, resolve_tolerance, within_distance
import numpy as np


@dataclass(frozen=True)
class GeometryContext:
    container: dict[str, Any]
    pattern: Pattern2D
    domain: ModelDomain
    original: dict[str, Any]
    tolerance: GeometryTolerance


def prepare_geometry(root: dict[str, Any], symmetry: SymmetryMode, *, tolerance=None) -> GeometryContext:
    policy = resolve_tolerance(tolerance)
    container = deepcopy(root)
    _normalize_lists(container)
    normalize_geometry(container, policy)
    pattern = extract_pattern(container, policy)
    if pattern.base_face is None:
        raise ValueError("CDB export requires at least one geometry body or feature.")
    root_faces = [_geometry_to_face(body["geometry"]) for body in container["bodies"]]
    domain = model_domain(symmetry, root_faces or pattern.all_faces)
    if symmetry is not SymmetryMode.FULL:
        for (feature, source, _), (raw, _, _) in zip(_features(container), _features(root)):
            geometry = feature["geometry"]
            if geometry["type"] != "CylinderGeometry":
                continue
            center = np.asarray(geometry["center"], dtype=float)
            for axis, restricted, boundary in ((0, domain.restrict_x, domain.center_x),
                                              (1, domain.restrict_y, domain.center_y)):
                if restricted and abs(center[axis] - boundary) <= policy.absolute:
                    center[axis] = boundary
            before = np.asarray(raw["geometry"]["center"], dtype=float)
            radial_motion = (np.linalg.norm(center[:2] - before[:2])
                             + abs(geometry["bottom_radius"] - raw["geometry"]["bottom_radius"]))
            z_motion = max(abs(center[2] - before[2]),
                           abs(center[2] + geometry["thk"] - before[2] - raw["geometry"]["thk"]))
            if not within_distance([radial_motion, z_motion], [0.0, 0.0], policy):
                raise GeometryToleranceError(
                    "symmetry normalization exceeds the displacement budget",
                    code="ambiguous_geometry", stage="symmetry_normalization", source=source,
                    tolerance=policy.absolute, displacement=float(np.hypot(radial_motion, z_motion)),
                )
            geometry["center"] = center.tolist()
    original = deepcopy(container)
    if symmetry is not SymmetryMode.FULL:
        filter_container_to_domain(container, domain, tolerance=policy)
        pattern = extract_pattern(container, policy)
        if pattern.base_face is None:
            raise ValueError(
                "CDB export has no geometry with positive XY area in "
                f"{symmetry.value}."
            )
    return GeometryContext(container, pattern, domain, original, policy)


def _normalize_lists(container: dict[str, Any], path: str = "root") -> None:
    # The preserved assignment code consumes complete container lists.
    for field in ("bodies", "vias", "circuits", "bumps", "children"):
        items = container.get(field)
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ValueError(f"{path}.{field} must be a list")
        container[field] = items
    for index, child in enumerate(container["children"]):
        if not isinstance(child, dict):
            raise ValueError(f"{path}.children[{index}] must be a dictionary")
        _normalize_lists(child, f"{path}.children[{index}]")
