import numpy as np
from numpy.typing import ArrayLike
from mesher.tolerance import (GeometryTolerance, GeometryToleranceError,
                             canonicalize_values, resolve_tolerance, subdivision_count)

from ..model import Mesh2D


def _sort(float_list, tolerance=None):
    return np.unique(canonicalize_values(float_list, tolerance, stage="grid_coordinates",
                                        source="rectilinear_coordinates"))


def _densify(target_edge_size, coordinates):
    out = []
    for a, b in zip(coordinates[:-1], coordinates[1:]):
        length = float(b - a)
        if length == 0.0:
            # duplicate line: keep only one point when joining segments
            if not out:
                out.append(a)
            continue
        nseg = subdivision_count(length, target_edge_size)
        seg = np.linspace(a, b, nseg + 1, endpoint=True, dtype=np.float64)
        if np.any(np.diff(seg) <= 0):
            raise GeometryToleranceError(
                "target size cannot be represented at these coordinates",
                code="numeric_resolution_exceeded", stage="grid_sampling",
                source="generated_edges", target_size=target_edge_size,
                measured_distance=float(np.diff(seg).min()))
        if out:
            seg = seg[1:]  # avoid boundary duplicate
        out.extend(seg.tolist())
    return np.asarray(out, dtype=np.float64)


def generate_rectilinear_mesh(
    target_edge_size: float,
    x_coordinates: ArrayLike,
    y_coordinates: ArrayLike,
    *,
    tolerance: GeometryTolerance | float | None = None,
) -> Mesh2D:
    """Generate a planar rectilinear Quad4 mesh.

    The supplied coordinates define mandatory grid lines. Intervals are
    subdivided as needed so no generated edge exceeds ``target_edge_size``.
    ``tolerance`` is a policy or positive absolute distance, defaulting to
    1e-3. It controls bounded merging of nearly coincident input lines;
    ambiguous chains fail. Subdivision counts use numerical roundoff only.
    """
    try:
        target_edge_size = float(target_edge_size)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("target_edge_size must be a real number") from error
    if not np.isfinite(target_edge_size) or target_edge_size <= 0.0:
        raise ValueError("target_edge_size must be positive")

    tolerance = resolve_tolerance(tolerance)
    if target_edge_size <= tolerance.absolute:
        raise GeometryToleranceError("target size must exceed the geometric tolerance",
                                     code="feature_below_resolution", stage="grid_sampling",
                                     source="target_edge_size", measured_distance=target_edge_size,
                                     target_size=target_edge_size, tolerance=tolerance.absolute)
    tolerance.check_resolution(np.r_[x_coordinates, y_coordinates], stage="grid_coordinates",
                               source="rectilinear_coordinates")

    x_coordinates = _sort(x_coordinates, tolerance)
    y_coordinates = _sort(y_coordinates, tolerance)

    x = _densify(target_edge_size, x_coordinates)
    y = _densify(target_edge_size, y_coordinates)

    if x.size < 2 or y.size < 2:
        raise ValueError("After _densify, need at least 2 x-lines and 2 y-lines.")
    minimum = float(min(np.diff(x).min(), np.diff(y).min()))
    if minimum <= tolerance.absolute:
        raise GeometryToleranceError("subdivision creates edges below geometric resolution",
                                     code="feature_below_resolution", stage="grid_sampling",
                                     source="generated_edges", measured_distance=minimum,
                                     tolerance=tolerance.absolute)

    Nx, Ny = int(x.size), int(y.size)

    ### nodes (x varies fastest)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.zeros_like(X)
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
        np.float64, copy=False
    )  # (Ny*Nx, 3)

    ### element node ids
    ix = np.arange(Nx - 1, dtype=np.int32)
    iy = np.arange(Ny - 1, dtype=np.int32)
    GX, GY = np.meshgrid(ix, iy, indexing="xy")

    n00 = (GY    ) * Nx + (GX    )  # BL
    n10 = (GY    ) * Nx + (GX + 1)  # BR
    n11 = (GY + 1) * Nx + (GX + 1)  # TR
    n01 = (GY + 1) * Nx + (GX    )  # TL

    # COUNTER-CLOCKWISE: BL, BR, TR, TL
    elements = np.stack([n00, n10, n11, n01], axis=-1).reshape(-1, 4).astype(np.int32)

    return Mesh2D(nodes=nodes, elements=elements)


__all__ = ["generate_rectilinear_mesh"]
