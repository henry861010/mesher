"""Standard V1 primitive conversion helpers."""

import math

def _geometry_to_face(geometry):
    """Converts a standard geometry primitive into a 2D face.

    Args:
        geometry (dict): A standard geometry primitive payload.

    Returns:
        dict: A 2D face payload. Supported face types are ``BOX``,
        ``POLYGON``, and ``CIRCLE``.

    Raises:
        ValueError: If the geometry type is unsupported or malformed.
    """
    geometry_type = _required_field(geometry, "type", "geometry")

    if geometry_type == "BoxGeometry":
        x1, y1 = _point_xy(
            _required_field(geometry, "bottom_left", "BoxGeometry"),
            "BoxGeometry.bottom_left",
        )
        x2, y2 = _point_xy(
            _required_field(geometry, "top_right", "BoxGeometry"),
            "BoxGeometry.top_right",
        )
        return {"type": "BOX", "dim": [x1, y1, x2, y2]}

    if geometry_type == "PolygonGeometry":
        return {"type": "POLYGON", "dim": _polygon_dim(geometry)}

    if geometry_type == "CylinderGeometry":
        x, y = _point_xy(
            _required_field(geometry, "center", "CylinderGeometry"),
            "CylinderGeometry.center",
        )
        radius = _positive_number(
            _required_field(geometry, "bottom_radius", "CylinderGeometry"),
            "CylinderGeometry.bottom_radius",
        )
        return {"type": "CIRCLE", "dim": [x, y, radius]}

    if geometry_type == "ConeGeometry":
        raise ValueError("ConeGeometry is not supported by mesh25d")

    raise ValueError(f"Geometry type {geometry_type} is not supported")


def _geometry_to_z(geometry, isStart=True):
    if geometry["type"] == "BoxGeometry":
        z = geometry["bottom_left"][2] 
        if not isStart:
            z += geometry["thk"]
        return round(z,3)

    if geometry["type"] == "PolygonGeometry":
        z = geometry["polys"][0][0][2]
        if not isStart:
            z += geometry["thk"]
        return round(z,3)
    
    if geometry["type"] == "CylinderGeometry":
        z = geometry["center"][2]
        if not isStart:
            z += geometry["thk"]
        return round(z,3)
    
    if geometry["type"] == "ConeGeometry":
        raise ValueError("ConeGeometry is not supported by mesh25d")


def _polygon_dim(geometry):
    """Builds a 2D polygon dimension payload from ``PolygonGeometry``.

    Args:
        geometry (dict): A ``PolygonGeometry`` payload.

    Returns:
        list: Polygon loops represented as ``[[[x, y], ...], ...]``.

    Raises:
        ValueError: If ``polys`` is missing or malformed.
    """
    polygons = _required_field(geometry, "polys", "PolygonGeometry")
    if not isinstance(polygons, list) or len(polygons) == 0:
        raise ValueError("PolygonGeometry.polys must be a non-empty list")

    polygons_2d = []
    for polygon_index, polygon in enumerate(polygons):
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(
                f"PolygonGeometry.polys[{polygon_index}] must contain at least 3 points"
            )
        polygons_2d.append(
            [
                _point_xy(
                    point,
                    f"PolygonGeometry.polys[{polygon_index}][{point_index}]",
                )
                for point_index, point in enumerate(polygon)
            ]
        )

    return polygons_2d


def _required_field(payload, field_name, context):
    """Reads a required field from a dictionary payload.

    Args:
        payload (dict): The source payload to read from.
        field_name (str): The required field name.
        context (str): Human-readable payload context for error messages.

    Returns:
        object: The value stored under ``field_name``.

    Raises:
        ValueError: If ``payload`` is not a dictionary or the field is missing.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a dictionary")
    if field_name not in payload:
        raise ValueError(f"{context} missing field {field_name}")
    return payload[field_name]


def _point_xy(point, context):
    """Extracts the XY coordinates from a point payload.

    Args:
        point (list | tuple): A point payload with at least ``[x, y]``.
        context (str): Human-readable point context for error messages.

    Returns:
        list: The ``[x, y]`` coordinates.

    Raises:
        ValueError: If the point does not contain finite XY values.
    """
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError(f"{context} must be a point with at least [x, y]")
    return [
        _finite_number(point[0], f"{context}[0]"),
        _finite_number(point[1], f"{context}[1]"),
    ]


def _positive_number(value, context):
    """Converts a value into a positive finite number.

    Args:
        value (object): The candidate numeric value.
        context (str): Human-readable value context for error messages.

    Returns:
        float: The converted positive finite number.

    Raises:
        ValueError: If the value is not a positive finite number.
    """
    number = _finite_number(value, context)
    if number <= 0:
        raise ValueError(f"{context} must be greater than 0")
    return number


def _finite_number(value, context):
    """Converts a value into a finite number.

    Args:
        value (object): The candidate numeric value.
        context (str): Human-readable value context for error messages.

    Returns:
        float: The converted finite number.

    Raises:
        ValueError: If the value cannot be converted to a finite number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be a finite number")
    return number
