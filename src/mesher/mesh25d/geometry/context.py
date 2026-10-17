"""Prepare an owned geometry tree and its symmetry-reduced footprints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..domain import ModelDomain, SymmetryMode, filter_container_to_domain, model_domain
from .patterns import Pattern2D, extract_pattern
from .primitives import _geometry_to_face


@dataclass(frozen=True)
class GeometryContext:
    container: dict[str, Any]
    pattern: Pattern2D
    domain: ModelDomain


def prepare_geometry(root: dict[str, Any], symmetry: SymmetryMode) -> GeometryContext:
    container = deepcopy(root)
    _normalize_lists(container)
    pattern = extract_pattern(container)
    if pattern.base_face is None:
        raise ValueError("CDB export requires at least one geometry body or feature.")
    root_faces = [_geometry_to_face(body["geometry"]) for body in container["bodies"]]
    domain = model_domain(symmetry, root_faces or pattern.all_faces)
    if symmetry is not SymmetryMode.FULL:
        filter_container_to_domain(container, domain)
        pattern = extract_pattern(container)
        if pattern.base_face is None:
            raise ValueError(
                "CDB export has no geometry with positive XY area in "
                f"{symmetry.value}."
            )
    return GeometryContext(container, pattern, domain)


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
