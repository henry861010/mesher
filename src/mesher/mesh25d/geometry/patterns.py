"""Geometry-only footprint patterns and source provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .koz import inset_face, koz_tolerance, koz_value
from .primitives import _geometry_to_face, _required_field, _finite_number, _positive_number

JsonObject = dict[str, Any]
DEFAULT_TOLERANCE = 1e-6
CONTAINER_ITEM_FIELDS = ("bodies", "vias", "circuits", "bumps")


@dataclass(frozen=True)
class Pattern2D:
    base_face: JsonObject | None
    faces: list[JsonObject]
    circle_sources: dict[tuple[float, float, float], list[str]]
    koz_circle_sources: dict[tuple[float, float, float], list[str]]
    coordinate_tolerance: float
    clearance_tolerance: float

    @property
    def all_faces(self) -> list[JsonObject]:
        return ([] if self.base_face is None else [self.base_face]) + self.faces


def extract_pattern(container, tolerance=DEFAULT_TOLERANCE) -> Pattern2D:
    """Extract original/KOZ footprints without consulting mesh controls."""
    normalized_tolerance = _normalize_tolerance(tolerance)
    all_faces = _collect_faces(container)
    normalized_tolerance = koz_tolerance(container, normalized_tolerance)
    unique_faces = _dedupe_faces(all_faces, normalized_tolerance)
    # KOZ faces are constraints inside physical footprints, never the
    # model's base. Polygon source winding need not match buffer winding.
    original_faces = _dedupe_faces(
        _collect_faces(container, include_koz=False), normalized_tolerance
    )
    base_face = _select_base_face(original_faces)
    faces = _remove_base_face(unique_faces, base_face, normalized_tolerance)

    return Pattern2D(
        base_face, faces, _circle_sources(container),
        _circle_sources(container, koz_only=True),
        koz_tolerance(container, 1e-3), koz_tolerance(container, 1e-6),
    )


def _circle_sources(
    container: JsonObject,
    *,
    koz_only: bool = False,
) -> dict[tuple[float, float, float], list[str]]:
    """Map original/KOZ circles to feature refs, optionally collecting only KOZ."""
    refs: dict[tuple[float, float, float], list[str]] = {}

    def visit(current: JsonObject, path: str) -> None:
        for field in ("bodies", "vias", "circuits", "bumps"):
            items = current.get(field, [])
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                if not isinstance(item, dict) or not isinstance(item.get("geometry"), dict):
                    continue
                face = _geometry_to_face(item["geometry"])
                if face.get("type") != "CIRCLE":
                    continue
                item_faces = [] if koz_only else [face]
                if field != "bodies" and koz_value(item) > 0:
                    inner = inset_face(face, koz_value(item))
                    if inner is not None:
                        item_faces.append(inner)
                source_ref = str(item.get("id") or f"{path}.{field}[{index}]")
                for item_face in item_faces:
                    pattern = tuple(float(value) for value in item_face["dim"])
                    pattern_refs = refs.setdefault(pattern, [])
                    if source_ref not in pattern_refs:
                        pattern_refs.append(source_ref)
        children = current.get("children", [])
        if isinstance(children, list):
            for child_index, child in enumerate(children):
                if isinstance(child, dict):
                    visit(child, f"{path}.children[{child_index}]")

    visit(container, "root")
    return refs


def _collect_faces(container, *, include_koz=True):
    """Collects all 2D faces from a standard container subtree.

    Args:
        container (dict): The root or subtree container payload.

    Returns:
        list: All 2D face payloads found in the container subtree.

    Raises:
        ValueError: If the container or one of its geometry items is malformed.
    """
    if not isinstance(container, dict):
        raise ValueError("container must be a dictionary")

    faces = []
    for item_type in CONTAINER_ITEM_FIELDS:
        for item_index, item in enumerate(_collect_items(container, item_type)):
            geometry = _required_field(
                item,
                "geometry",
                f"container.{item_type}[{item_index}]",
            )
            faces.append(_geometry_to_face(geometry))
            if include_koz and item_type != "bodies":
                koz = koz_value(item)
                if koz > 0:
                    inner = inset_face(faces[-1], koz)
                    if inner is not None:
                        faces.append(inner)

    children = container.get("children", [])
    if children is None:
        children = []
    if not isinstance(children, list):
        raise ValueError("container.children must be a list")

    for child in children:
        faces.extend(_collect_faces(child, include_koz=include_koz))

    return faces


def _collect_items(container, item_type):
    """Reads an item list from a container payload.

    Args:
        container (dict): The container payload to read from.
        item_type (str): One of ``bodies``, ``vias``, ``circuits``, or
            ``bumps``.

    Returns:
        list: The item payload list.

    Raises:
        ValueError: If the item field is present but is not a list.
    """
    items = container.get(item_type, [])
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError(f"container.{item_type} must be a list")
    return items


def _select_base_face(faces):
    """Selects the largest face as the base face.

    Args:
        faces (list): Candidate 2D face payloads.

    Returns:
        dict | None: The largest face, or ``None`` when no face exists.
    """
    base_face = None
    for face in faces:
        if base_face is None or _face_area(base_face) < _face_area(face):
            base_face = face
    return base_face


def _dedupe_faces(faces, tolerance):
    """Removes completely overlapping faces with tolerance.

    Args:
        faces (list): Candidate 2D face payloads.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        list: Deduplicated faces, preserving the first occurrence.
    """
    unique_faces = []
    for face in faces:
        if any(_same_face(face, unique_face, tolerance) for unique_face in unique_faces):
            continue
        unique_faces.append(face)
    return unique_faces


def _remove_base_face(faces, base_face, tolerance):
    """Removes the selected base face from a face list.

    Args:
        faces (list): Deduplicated 2D face payloads.
        base_face (dict | None): The selected base face.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        list: Faces that are not equivalent to ``base_face``.
    """
    if base_face is None:
        return faces
    return [face for face in faces if not _same_face(face, base_face, tolerance)]


def _same_face(left, right, tolerance):
    """Checks whether two faces have the same footprint.

    Args:
        left (dict): The first 2D face payload.
        right (dict): The second 2D face payload.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when both faces have the same type and equivalent
        footprint within tolerance.
    """
    if left["type"] != right["type"]:
        return False

    if left["type"] == "BOX":
        return _same_number_list(
            _normalized_box_dim(left["dim"]),
            _normalized_box_dim(right["dim"]),
            tolerance,
        )

    if left["type"] == "CIRCLE":
        return _same_number_list(left["dim"], right["dim"], tolerance)

    if left["type"] == "POLYGON":
        return _same_polygon_dim(left["dim"], right["dim"], tolerance)

    raise ValueError(f'Face type {left["type"]} is not supported')


def _same_polygon_dim(left_polygons, right_polygons, tolerance):
    """Checks whether two polygon dimensions contain the same loops.

    Args:
        left_polygons (list): Polygon loops from the first face.
        right_polygons (list): Polygon loops from the second face.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when all loops match, ignoring loop order, loop starting
        point, and clockwise/counterclockwise direction.
    """
    if len(left_polygons) != len(right_polygons):
        return False

    unmatched_right_indices = set(range(len(right_polygons)))
    for left_loop in left_polygons:
        matched_index = None
        for right_index in unmatched_right_indices:
            if _same_polygon_loop(left_loop, right_polygons[right_index], tolerance):
                matched_index = right_index
                break
        if matched_index is None:
            return False
        unmatched_right_indices.remove(matched_index)

    return True


def _same_polygon_loop(left_loop, right_loop, tolerance):
    """Checks whether two polygon loops are equivalent.

    Args:
        left_loop (list): The first polygon loop.
        right_loop (list): The second polygon loop.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when the loops match after allowing rotation and
        reversed direction.
    """
    left_open = _open_polygon_loop(left_loop, tolerance)
    right_open = _open_polygon_loop(right_loop, tolerance)
    if len(left_open) != len(right_open):
        return False

    return _same_loop_with_rotation(
        left_open,
        right_open,
        tolerance,
    ) or _same_loop_with_rotation(
        left_open,
        list(reversed(right_open)),
        tolerance,
    )


def _same_loop_with_rotation(left_loop, right_loop, tolerance):
    """Checks whether two polygon loops match under any start index.

    Args:
        left_loop (list): The first open polygon loop.
        right_loop (list): The second open polygon loop.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when a rotation of ``right_loop`` matches ``left_loop``.
    """
    if len(left_loop) != len(right_loop):
        return False
    if not left_loop:
        return True

    for start_index in range(len(right_loop)):
        matches = True
        for left_index, left_point in enumerate(left_loop):
            right_index = (start_index + left_index) % len(right_loop)
            if not _same_point(left_point, right_loop[right_index], tolerance):
                matches = False
                break
        if matches:
            return True

    return False


def _open_polygon_loop(loop, tolerance):
    """Removes a duplicate closing point from a polygon loop.

    Args:
        loop (list): A polygon loop represented as ``[[x, y], ...]``.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        list: The loop without a final point equal to the first point.
    """
    if len(loop) > 1 and _same_point(loop[0], loop[-1], tolerance):
        return loop[:-1]
    return loop


def _same_point(left, right, tolerance):
    """Checks whether two 2D points are equivalent within tolerance.

    Args:
        left (list): The first point represented as ``[x, y]``.
        right (list): The second point represented as ``[x, y]``.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when both coordinates match within tolerance.
    """
    return _same_number_list(left, right, tolerance)


def _same_number_list(left, right, tolerance):
    """Checks whether two numeric lists match within tolerance.

    Args:
        left (list): The first numeric list.
        right (list): The second numeric list.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when both lists have the same length and each pair of
        numbers differs by no more than ``tolerance``.
    """
    if len(left) != len(right):
        return False
    return all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left, right)
    )


def _normalized_box_dim(dim):
    """Normalizes a box dimension into ordered bounds.

    Args:
        dim (list): A box dimension represented as ``[x1, y1, x2, y2]``.

    Returns:
        list: Ordered box bounds as ``[x_min, y_min, x_max, y_max]``.
    """
    x1, y1, x2, y2 = dim
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _face_area(face):
    """Calculates the XY area of a 2D face.

    Args:
        face (dict): A 2D face payload.

    Returns:
        float: The absolute XY area of the face.

    Raises:
        ValueError: If the face type is unsupported.
    """
    if face["type"] == "BOX":
        x1, y1, x2, y2 = _normalized_box_dim(face["dim"])
        return abs((x2 - x1) * (y2 - y1))

    if face["type"] == "CIRCLE":
        _, _, radius = face["dim"]
        return math.pi * (radius ** 2)

    if face["type"] == "POLYGON":
        total_area = 0.0
        for polygon in face["dim"]:
            total_area += _signed_polygon_area(polygon)
        return abs(total_area)

    raise ValueError(f'Face type {face["type"]} is not supported')


def _signed_polygon_area(points):
    """Calculates the signed XY area for one polygon loop.

    Args:
        points (list): A polygon loop represented as ``[[x, y], ...]``.

    Returns:
        float: The signed polygon area.
    """
    area_value = 0.0
    for index in range(len(points)):
        next_index = (index + 1) % len(points)
        area_value += points[index][0] * points[next_index][1]
        area_value -= points[next_index][0] * points[index][1]
    return area_value / 2.0


def _normalize_tolerance(tolerance):
    """Converts and validates the face deduplication tolerance.

    Args:
        tolerance (object): The candidate tolerance value.

    Returns:
        float: The validated positive tolerance.

    Raises:
        ValueError: If tolerance is not a positive finite number.
    """
    return _positive_number(tolerance, "tolerance")
