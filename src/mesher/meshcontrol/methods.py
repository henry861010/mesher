"""Resolve mesh controls and expose each control's mesher-specific behavior.

The JSON contract validates field shapes before this module is called. Resolution
checks geometry references, finite coordinates, sizes, and Z interval semantics.
This module does not combine overlapping controls or operate a meshing engine.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import ClassVar

from ..mesh2d.controls import XYRegionControl
from ..mesh3d.controls import ZControl


FEATURE_FIELDS = {"body": "bodies", "via": "vias", "circuit": "circuits", "bump": "bumps"}


def _required(payload, field, path):
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a dictionary")
    if field not in payload:
        raise ValueError(f"{path} missing field {field}")
    return payload[field]


def _number(value, path):
    try:
        if isinstance(value, bool):
            raise ValueError
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _positive_number(value, path):
    result = _number(value, path)
    if result <= 0:
        raise ValueError(f"{path} must be greater than 0")
    return result


def _min_distance(value):
    result = _number(value, "min_dis")
    if result < 0:
        raise ValueError("min_dis must be non-negative")
    return result


def _geometry_bounding(geometry, *, path="geometry"):
    """Return [xmin, ymin, zmin, xmax, ymax, zmax] for one primitive.

    Boxes use bottom_left Z plus thk (top_right Z is validated but does not
    determine height). Cylinders use base-face center and positive radius.
    Polygons include all loops and each vertex extruded by thk. Reversed box
    corners and negative thicknesses are normalized; zero thickness is valid.
    Coordinates are finite floats, without rounding or mutation of the input.
    Unsupported types and malformed geometry raise ValueError with its path.
    """
    def point(value, point_path):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{point_path} must be a point [x, y, z]")
        return [_number(coord, f"{point_path}[{axis}]") for axis, coord in enumerate(value)]

    lower = [math.inf] * 3
    upper = [-math.inf] * 3

    def include_point(value, point_path):
        for axis, coord in enumerate(point(value, point_path)):
            lower[axis] = min(lower[axis], coord)
            upper[axis] = max(upper[axis], coord)

    kind = _required(geometry, "type", path)
    if kind not in ("BoxGeometry", "CylinderGeometry", "PolygonGeometry"):
        raise ValueError(f"{path}: geometry type {kind} is not supported")
    thickness = _number(_required(geometry, "thk", path), f"{path}.thk")
    if kind == "BoxGeometry":
        bottom = point(_required(geometry, "bottom_left", path), f"{path}.bottom_left")
        top = point(_required(geometry, "top_right", path), f"{path}.top_right")
        include_point(bottom, f"{path}.bottom_left")
        include_point([top[0], top[1], bottom[2] + thickness], f"{path}.top_face")
    elif kind == "CylinderGeometry":
        x, y, z = point(_required(geometry, "center", path), f"{path}.center")
        radius = _positive_number(_required(geometry, "bottom_radius", path), f"{path}.bottom_radius")
        include_point([x - radius, y - radius, z], f"{path}.lower_extent")
        include_point([x + radius, y + radius, z + thickness], f"{path}.upper_extent")
    else:
        loops = _required(geometry, "polys", path)
        if not isinstance(loops, list) or not loops:
            raise ValueError(f"{path}.polys must be a non-empty list")
        for loop_index, loop in enumerate(loops):
            loop_path = f"{path}.polys[{loop_index}]"
            if not isinstance(loop, list) or len(loop) < 3:
                raise ValueError(f"{loop_path} must contain at least three points")
            for point_index, vertex in enumerate(loop):
                vertex_path = f"{loop_path}[{point_index}]"
                x, y, z = point(vertex, vertex_path)
                include_point([x, y, z], vertex_path)
                include_point([x, y, z + thickness], f"{vertex_path}.extruded")
    return lower + upper


def _items(container, field, path):
    items = container.get(field)
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError(f"{path}.{field} must be a list")
    return items


def _walk_containers(container, path="container"):
    if not isinstance(container, dict):
        raise ValueError(f"{path} must be a dictionary")
    yield container, path
    for index, child in enumerate(_items(container, "children", path)):
        yield from _walk_containers(child, f"{path}.children[{index}]")


def _container_bounding_geometry(container, *, path="container"):
    """Return a BoxGeometry enclosing all features and descendants.

    Missing/None lists are empty. Primitive bounds share _geometry_bounding's
    validation and normalization. Both returned corners lie on the bottom face;
    thk locates the top face. Empty subtrees raise ValueError. Input is unchanged.
    """
    lower = [math.inf] * 3
    upper = [-math.inf] * 3
    for current, current_path in _walk_containers(container, path):
        for field in FEATURE_FIELDS.values():
            for index, item in enumerate(_items(current, field, current_path)):
                item_path = f"{current_path}.{field}[{index}]"
                geometry = _required(item, "geometry", item_path)
                bounds = _geometry_bounding(geometry, path=f"{item_path}.geometry")
                for axis in range(3):
                    lower[axis] = min(lower[axis], bounds[axis])
                    upper[axis] = max(upper[axis], bounds[axis + 3])
    if lower[0] == math.inf:
        raise ValueError(f"{path} does not contain any geometry")
    return {
        "type": "BoxGeometry",
        "bottom_left": lower,
        "top_right": [upper[0], upper[1], lower[2]],
        "thk": _number(upper[2] - lower[2], f"{path} bounding box thickness"),
    }


def _resolve_coordinate(value, axis, coordinate_mode, bounds, path):
    if coordinate_mode == "absolute":
        return _number(value, path)
    anchor = value["anchor"]
    if anchor not in (f"{axis}_min", f"{axis}_max"):
        raise ValueError(f"{path}: Invalid {axis} anchor {anchor}")
    index = "xyz".index(axis) + (3 if anchor.endswith("_max") else 0)
    return _number(bounds[index] + _number(value["offset"], f"{path}.offset"), path)


class MeshControl(ABC):
    """Explicit XY/Z interface; no default silently ignores a control."""

    METHOD: ClassVar[str]

    def __init__(self, control: dict, geometry: dict | None = None):
        if control["method"] != self.METHOD:
            raise ValueError(f"{type(self).__name__} is not for control method {control['method']}")
        self.source_index: int | None = None
        self.method = self.METHOD
        self.label = control.get("label")
        self.coordinate_mode = control["coordinateMode"]
        if self.coordinate_mode not in ("absolute", "reference"):
            raise ValueError(f"Unknown coordinateMode {self.coordinate_mode}")
        self._bounds = None
        if self.coordinate_mode == "reference":
            if geometry is None:
                raise ValueError("coordinateMode of reference requires reference geometry")
            self._bounds = _geometry_bounding(geometry)

    def _coordinate(self, value, axis, path):
        return _resolve_coordinate(value, axis, self.coordinate_mode, self._bounds, path)

    @abstractmethod
    def get_25D_z_controls(
        self, min_dis: float = 0.01, *, global_element_size: float | None = None,
    ) -> list[ZControl]:
        """Return Z constraints; Z_POINT requires global_element_size.

        min_dis is a finite non-negative distance in input length units. An
        empty list explicitly means this control imposes no Z constraint.
        """
        raise NotImplementedError

    @abstractmethod
    def get_25D_xy_controls(self) -> list[XYRegionControl]:
        """Return XY constraints; [] explicitly means no XY constraint."""
        raise NotImplementedError


class ZSectionControl(MeshControl):
    def __init__(self, control: dict, geometry: dict | None = None):
        super().__init__(control, geometry)
        self.element_size = _positive_number(control["elementSize"], "elementSize")
        self.start_z = self._coordinate(control["startZ"], "z", "startZ")
        self.end_z = self._coordinate(control["endZ"], "z", "endZ")
        if self.start_z >= self.end_z:
            raise ValueError("Z section requires startZ < endZ; use Z_POINT for a single position")
        self._length = _positive_number(self.end_z - self.start_z, "Z section length")

    def get_25D_xy_controls(self) -> list[XYRegionControl]:
        return []

    def _sections(self, *cuts) -> list[ZControl]:
        # Floating-point addition may collapse an interior cut onto an endpoint.
        points = [self.start_z]
        for cut in cuts:
            if points[-1] < cut < self.end_z:
                points.append(cut)
        points.append(self.end_z)
        return [ZControl(element_size=self.element_size, method=self.method, start=start, end=end)
                for start, end in zip(points, points[1:])]

    def _division(self, min_dis):
        tolerance = _min_distance(min_dis)
        if self._length < self.element_size:
            return 0, 0.0
        ratio = _number(self._length / self.element_size, "Z section length / elementSize")
        nearest = round(ratio)
        if math.isclose(self._length, nearest * self.element_size, rel_tol=1e-12, abs_tol=0.0):
            return nearest, 0.0
        count = math.floor(ratio)
        remainder = self._length - count * self.element_size
        return count, 0.0 if remainder < tolerance else remainder


class MeshControl_ZsectionAvg(ZSectionControl):
    METHOD = "Z_SECTION_AVG"

    def get_25D_z_controls(self, min_dis=0.01, *, global_element_size=None) -> list[ZControl]:
        _min_distance(min_dis)
        return self._sections()


class MeshControl_ZsectionTop(ZSectionControl):
    METHOD = "Z_SECTION_TOP"

    def get_25D_z_controls(self, min_dis=0.01, *, global_element_size=None) -> list[ZControl]:
        count, remainder = self._division(min_dis)
        if not remainder:
            return self._sections()
        return self._sections(self.start_z + count * self.element_size)


class MeshControl_ZsectionBot(ZSectionControl):
    METHOD = "Z_SECTION_BOT"

    def get_25D_z_controls(self, min_dis=0.01, *, global_element_size=None) -> list[ZControl]:
        _, remainder = self._division(min_dis)
        if not remainder:
            return self._sections()
        return self._sections(self.start_z + remainder)


class MeshControl_ZsectionCenter(ZSectionControl):
    METHOD = "Z_SECTION_CENTER"

    def get_25D_z_controls(self, min_dis=0.01, *, global_element_size=None) -> list[ZControl]:
        count, remainder = self._division(min_dis)
        if not remainder:
            return self._sections()
        lower_end = self.start_z + (count // 2) * self.element_size
        return self._sections(lower_end, lower_end + remainder)


class MeshControl_Zpoint(MeshControl):
    METHOD = "Z_POINT"

    def __init__(self, control: dict, geometry: dict | None = None):
        super().__init__(control, geometry)
        self.z = self._coordinate(control["z"], "z", "z")

    def get_25D_z_controls(self, min_dis=0.01, *, global_element_size=None) -> list[ZControl]:
        _min_distance(min_dis)
        size = _positive_number(global_element_size, "global_element_size")
        return [ZControl(element_size=size, method=self.method, start=self.z, end=self.z)]

    def get_25D_xy_controls(self) -> list[XYRegionControl]:
        return []


class MeshControl_3dRegion(MeshControl):
    METHOD = "3D_REGION"

    def __init__(self, control: dict, geometry: dict | None = None):
        super().__init__(control, geometry)
        self.element_size = _positive_number(control["elementSize"], "elementSize")
        self.region = self._get_region(control)

    def _get_region(self, control):
        """Return a detached region with finite absolute coordinates."""
        region = deepcopy(control["region"])
        if region["type"] == "box":
            for bound in ("min", "max"):
                region[bound] = {
                    axis: self._coordinate(region[bound][axis], axis, f"region.{bound}.{axis}")
                    for axis in "xyz"
                }
        elif region["type"] == "polygon":
            region["loops"] = [
                [{axis: self._coordinate(point[axis], axis, f"region.loops[{i}][{j}].{axis}")
                  for axis in "xy"} for j, point in enumerate(loop)]
                for i, loop in enumerate(region["loops"])
            ]
            for field in ("startZ", "endZ"):
                region[field] = self._coordinate(region[field], "z", f"region.{field}")
        else:
            raise ValueError(f"Unsupported region type {region['type']}")
        return region

    def get_25D_z_controls(self, min_dis=0.01, *, global_element_size=None) -> list[ZControl]:
        """3D_REGION controls only the XY plane in the 25D mesher."""
        _min_distance(min_dis)
        return []

    def get_25D_xy_controls(self) -> list[XYRegionControl]:
        """Project to the shared XY plane; Z extent imposes no constraint."""
        if self.region["type"] == "box":
            lower, upper = self.region["min"], self.region["max"]
            region = {"type": "BOX", "dim": [
                min(lower["x"], upper["x"]), min(lower["y"], upper["y"]),
                max(lower["x"], upper["x"]), max(lower["y"], upper["y"]),
            ]}
        else:
            region = {"type": "POLYGON", "dim": [
                [[point["x"], point["y"]] for point in loop]
                for loop in self.region["loops"]
            ]}
        return [XYRegionControl(region=region, element_size=self.element_size)]


CONTROL_CLASSES: dict[str, type[MeshControl]] = {
    cls.METHOD: cls for cls in (
        MeshControl_ZsectionAvg, MeshControl_ZsectionTop, MeshControl_ZsectionBot,
        MeshControl_ZsectionCenter, MeshControl_Zpoint, MeshControl_3dRegion,
    )
}


def _find_target_geometry(control, container):
    """Resolve all matching targets in preorder; root is the supplied subtree."""
    reference = _required(control, "reference", "control")
    kind = _required(reference, "kind", "control.reference")
    if kind == "root":
        return [_container_bounding_geometry(container)]
    if kind not in ("container", *FEATURE_FIELDS):
        raise ValueError(f"Unknown reference kind {kind}")

    def matches(item, path):
        if not isinstance(item, dict):
            raise ValueError(f"{path} must be a dictionary")
        return all(item.get(field) == reference[field] for field in ("key", "id") if field in reference)

    targets = []
    for current, path in _walk_containers(container):
        if kind == "container":
            if matches(current, path):
                targets.append(_container_bounding_geometry(current, path=path))
        else:
            field = FEATURE_FIELDS[kind]
            for index, item in enumerate(_items(current, field, path)):
                item_path = f"{path}.{field}[{index}]"
                if matches(item, item_path):
                    targets.append(_required(item, "geometry", item_path))
    return targets


def get_control_obj(container, controls) -> list[MeshControl]:
    """Build controls without mutating inputs; unresolved references are errors."""
    control_objs = []
    for index, control in enumerate(controls):
        context = f"controls[{index}]"
        if control.get("label") is not None:
            context += f" ({control['label']!r})"
        first_object = len(control_objs)
        try:
            method = _required(control, "method", context)
            control_class = CONTROL_CLASSES.get(method)
            if control_class is None:
                raise ValueError(f"Unknown mesh control method {method}")
            mode = _required(control, "coordinateMode", context)
            if mode == "absolute":
                control_objs.append(control_class(control))
            elif mode == "reference":
                targets = _find_target_geometry(control, container)
                if not targets:
                    raise ValueError(f"No geometry matches reference {control['reference']!r}")
                control_objs.extend(control_class(control, geometry) for geometry in targets)
            else:
                raise ValueError(f"Unknown coordinateMode {mode}")
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
        for obj in control_objs[first_object:]:
            obj.source_index = index
    return control_objs
