"""Absolute geometric precision shared by every meshing operation.

This module deliberately does not define an area, angular or solver epsilon.
Those quantities have different units and must retain numerical error budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real

import numpy as np


class GeometryToleranceError(ValueError):
    """An unresolved geometric constraint, with machine-readable provenance."""

    def __init__(self, message, *, code, stage=None, source=None, **context):
        self.message = message
        self.code = code
        self.context = dict(context)
        if stage is not None:
            self.context["stage"] = stage
        if source is not None:
            self.context["source"] = source
        super().__init__(f"{message}; code={code}, context={self.context}")


@dataclass(frozen=True)
class GeometryTolerance:
    """A finite positive absolute distance in the model's coordinate units."""

    absolute: float = 0.001

    def __post_init__(self):
        if isinstance(self.absolute, (bool, np.bool_)) or not isinstance(self.absolute, Real):
            raise ValueError("tolerance.absolute must be a finite positive number")
        try:
            value = float(self.absolute)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("tolerance.absolute must be a finite positive number") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError("tolerance.absolute must be a finite positive number")
        object.__setattr__(self, "absolute", value)

    def normalized(self, scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("normalization scale must be finite and positive")
        value = self.absolute / scale
        if not math.isfinite(value) or value == 0:
            raise GeometryToleranceError(
                "tolerance cannot be represented in the local coordinate frame",
                code="numeric_resolution_exceeded", tolerance=self.absolute, scale=scale)
        return value

    def check_resolution(self, points, *, stage="normalization", source=None):
        values = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("geometry coordinates must be finite")
        if not values.size:
            return
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            uncertainty = 4.0 * float(np.max(np.abs(np.spacing(values))))
        if not math.isfinite(uncertainty) or uncertainty > self.absolute:
            raise GeometryToleranceError(
                "coordinate precision cannot resolve the requested tolerance",
                code="numeric_resolution_exceeded", stage=stage, source=source,
                numerical_resolution=uncertainty, tolerance=self.absolute)


def resolve_tolerance(value=None) -> GeometryTolerance:
    """Resolve one policy at an entry point, retaining identity downstream."""
    if value is None:
        return GeometryTolerance()
    if isinstance(value, GeometryTolerance):
        return value
    return GeometryTolerance(value)


def _scalar_close(first, second, distance):
    gap = abs(float(first) - float(second))
    uncertainty = 8 * np.finfo(float).eps * max(abs(first), abs(second), distance)
    if abs(gap - distance) > uncertainty:
        return gap <= distance
    return abs(Fraction(float(first)) - Fraction(float(second))) <= Fraction(float(distance))


def within_distance(first, second, tolerance=None):
    """Inclusive Euclidean proximity, with exact fallback at the threshold.

    The last dimension contains coordinates; leading dimensions broadcast.
    The fallback compares the actual encoded input floats, without growing T.
    """
    policy = resolve_tolerance(tolerance)
    a, b = np.broadcast_arrays(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("distance operands must be finite")
    if a.ndim == 0:
        return _scalar_close(float(a), float(b), policy.absolute)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        distance = np.hypot.reduce(a - b, axis=-1)
    answer = np.asarray(distance <= policy.absolute).copy()
    scale = np.maximum(np.max(np.maximum(np.abs(a), np.abs(b)), axis=-1), policy.absolute)
    with np.errstate(under="ignore"):
        uncertain = np.abs(distance - policy.absolute) <= 16 * np.finfo(float).eps * scale
    rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    flat = answer.reshape(-1)
    limit = Fraction(policy.absolute) ** 2
    for row in np.flatnonzero(np.asarray(uncertain).reshape(-1)):
        squared = sum((Fraction(float(x)) - Fraction(float(y))) ** 2
                      for x, y in zip(rows_a[row], rows_b[row]))
        flat[row] = squared <= limit
    return bool(answer) if answer.ndim == 0 else answer


def circle_membership(points, center, radius, tolerance=None, *, boundary=False):
    """Inclusive circle/disk membership without rounding ``radius + T``.

    With boundary=True, select the shell whose radial residual is at most T.
    Otherwise select the disk including its outer T-wide contact region.
    Exact squared-distance comparisons settle uncertain boundary cases.
    """
    policy = resolve_tolerance(tolerance)
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("circle radius must be finite and non-negative")
    points, center = np.broadcast_arrays(np.asarray(points, dtype=float),
                                        np.asarray(center, dtype=float))
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError("circle coordinates must have a final XY dimension")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(center)):
        raise ValueError("circle coordinates must be finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        distance = np.hypot.reduce(points - center, axis=-1)
        lower, upper = max(0., radius - policy.absolute), radius + policy.absolute
        answer = np.asarray((distance <= upper) & ((distance >= lower) if boundary else True)).copy()
        scale = np.maximum(np.max(np.maximum(np.abs(points), np.abs(center)), axis=-1),
                           max(radius, policy.absolute))
        error = 32 * np.finfo(float).eps * scale
        uncertain = (np.abs(distance - upper) <= error) | ~np.isfinite(distance) | ~np.isfinite(upper)
        if boundary:
            uncertain |= np.abs(distance - lower) <= error
    outer = (Fraction(radius) + Fraction(policy.absolute)) ** 2
    inner = max(Fraction(0), Fraction(radius) - Fraction(policy.absolute)) ** 2
    first, second = points.reshape(-1, 2), center.reshape(-1, 2)
    flat = answer.reshape(-1)
    for row in np.flatnonzero(np.asarray(uncertain).reshape(-1)):
        squared = sum((Fraction(float(x)) - Fraction(float(y))) ** 2
                      for x, y in zip(first[row], second[row]))
        flat[row] = squared <= outer and (not boundary or squared >= inner)
    return bool(answer) if answer.ndim == 0 else answer


def canonicalize_values(values, tolerance=None, *, fixed=None,
                        stage="normalization", source=None):
    """Map scalar coordinates to deterministic, diameter-bounded components.

    Proximity is not transitive. A connected chain spanning more than T is an
    ambiguity, not permission to collapse an arbitrarily wide interval.
    A fixed coordinate is an exact constraint and wins over movable values.
    This function does not decide whether moving a geometric feature is legal.
    """
    policy = resolve_tolerance(tolerance)
    data = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("coordinates must be finite")
    if not data.size:
        return data.copy()
    protected = np.zeros(data.shape, dtype=bool) if fixed is None else np.broadcast_to(
        np.asarray(fixed, dtype=bool), data.shape)
    unique, inverse = np.unique(data, return_inverse=True)
    inverse = inverse.reshape(-1)
    fixed_unique = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(fixed_unique, inverse, protected.reshape(-1))
    delta = np.diff(unique)
    connected = delta <= policy.absolute
    with np.errstate(under="ignore"):
        uncertainty = 8 * np.finfo(float).eps * np.maximum(
            np.maximum(np.abs(unique[:-1]), np.abs(unique[1:])), policy.absolute)
    for index in np.flatnonzero(np.abs(delta - policy.absolute) <= uncertainty):
        connected[index] = _scalar_close(unique[index], unique[index + 1], policy.absolute)
    boundaries = np.r_[0, np.flatnonzero(~connected) + 1, len(unique)]
    representatives = unique.copy()
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if end - begin == 1:
            continue
        if not _scalar_close(unique[begin], unique[end - 1], policy.absolute):
            raise GeometryToleranceError(
                "a chain of near coordinates exceeds the allowed cluster diameter",
                code="ambiguous_geometry", stage=stage, source=source,
                minimum=float(unique[begin]), maximum=float(unique[end - 1]),
                measured_distance=float(unique[end - 1] - unique[begin]),
                tolerance=policy.absolute)
        anchors = unique[begin:end][fixed_unique[begin:end]]
        if len(anchors) > 1:
            raise GeometryToleranceError(
                "distinct fixed coordinates cannot share one geometric identity",
                code="fixed_constraint_conflict", stage=stage, source=source,
                fixed_coordinates=anchors.tolist(), tolerance=policy.absolute)
        representatives[begin:end] = anchors[0] if len(anchors) else unique[begin]
    return representatives[inverse].reshape(data.shape)


def canonicalize_points(points, tolerance=None, *, fixed=None,
                        stage="normalization", source=None):
    """Canonicalize rectilinear coordinates, bounding total Euclidean motion.

    Callers must check their own topology/material constraints after this step.
    Fixed is a mask over points (or the full coordinate-shaped mask).
    """
    policy = resolve_tolerance(tolerance)
    data = np.asarray(points, dtype=float)
    if data.ndim < 2 or data.shape[-1] not in (2, 3):
        raise ValueError("points must have a final coordinate dimension of 2 or 3")
    mask = np.zeros(data.shape, dtype=bool)
    if fixed is not None:
        protected = np.asarray(fixed, dtype=bool)
        if protected.shape != data.shape:
            protected = np.broadcast_to(protected, data.shape[:-1])[..., None]
        mask = np.broadcast_to(protected, data.shape)
    result = np.empty_like(data)
    for axis in range(data.shape[-1]):
        result[..., axis] = canonicalize_values(
            data[..., axis], policy, fixed=mask[..., axis], stage=stage, source=source)
    accepted = within_distance(data, result, policy)
    if not np.all(accepted):
        row = int(np.flatnonzero(~np.asarray(accepted).reshape(-1))[0])
        before, after = data.reshape(-1, data.shape[-1])[row], result.reshape(-1, data.shape[-1])[row]
        raise GeometryToleranceError(
            "coordinate normalization would exceed the displacement budget",
            code="ambiguous_geometry", stage=stage, source=source, point_index=row,
            original=before.tolist(), proposed=after.tolist(),
            displacement=float(np.linalg.norm(after - before)), tolerance=policy.absolute)
    return result


def subdivision_count(length, target, *, rounding="ceil"):
    """Round a subdivision ratio using arithmetic uncertainty, not distance T."""
    length, target = float(length), float(target)
    if not math.isfinite(length) or length <= 0 or not math.isfinite(target) or target <= 0:
        raise ValueError("subdivision length and target must be finite and positive")
    ratio = length / target
    if not math.isfinite(ratio):
        raise GeometryToleranceError("subdivision count exceeds numeric range",
                                     code="numeric_resolution_exceeded", length=length, target=target)
    nearest = round(ratio)
    if abs(ratio - nearest) <= 32 * np.finfo(float).eps * max(1.0, abs(ratio)):
        ratio = float(nearest)
    if rounding == "ceil":
        return max(1, math.ceil(ratio))
    if rounding == "floor":
        return max(1, math.floor(ratio))
    raise ValueError("rounding must be 'ceil' or 'floor'")


def point_segment_distance(points, first, last):
    """Euclidean distance to a closed segment; zero-length segments are points."""
    points, first, last = (np.asarray(value, dtype=float) for value in (points, first, last))
    delta = last - first
    length = np.hypot.reduce(delta, axis=-1)
    direction = np.divide(delta, np.expand_dims(length, -1), out=np.zeros_like(delta),
                          where=np.expand_dims(length != 0, -1))
    position = np.clip(np.sum((points - first) * direction, axis=-1), 0.0, length)
    residual = (points - first) - np.expand_dims(position, -1) * direction
    return np.hypot.reduce(residual, axis=-1)


def within_segment(points, first, last, tolerance=None):
    """Inclusive distance to a closed segment, exact at uncertain thresholds.

    The exact fallback avoids rounding an intermediate projected coordinate,
    which can otherwise change membership near large coordinate origins.
    """
    policy = resolve_tolerance(tolerance)
    points, first, last = np.broadcast_arrays(*(np.asarray(value, dtype=float)
                                               for value in (points, first, last)))
    if points.ndim == 0 or points.shape[-1] not in (2, 3):
        raise ValueError("segment coordinates must have an XY or XYZ dimension")
    if not all(np.all(np.isfinite(value)) for value in (points, first, last)):
        raise ValueError("segment coordinates must be finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        distance = point_segment_distance(points, first, last)
        answer = np.asarray(distance <= policy.absolute).copy()
        scale = np.maximum.reduce([np.max(np.abs(value), axis=-1) for value in (points, first, last)])
        uncertain = (~np.isfinite(distance) | (np.abs(distance - policy.absolute)
                     <= 32 * np.finfo(float).eps * np.maximum(scale, policy.absolute)))
    dimension = points.shape[-1]
    p, a, b = (value.reshape(-1, dimension) for value in (points, first, last))
    flat = answer.reshape(-1)
    threshold = Fraction(policy.absolute) ** 2
    for row in np.flatnonzero(np.asarray(uncertain).reshape(-1)):
        offset = [Fraction(float(x)) - Fraction(float(y)) for x, y in zip(p[row], a[row])]
        edge = [Fraction(float(x)) - Fraction(float(y)) for x, y in zip(b[row], a[row])]
        squared = sum(value * value for value in edge)
        dot = sum(x * y for x, y in zip(offset, edge))
        if dot <= 0:
            flat[row] = sum(value * value for value in offset) <= threshold
        elif dot >= squared:
            flat[row] = sum((x - y) ** 2 for x, y in zip(offset, edge)) <= threshold
        else:
            flat[row] = (sum(value * value for value in offset) * squared - dot * dot
                         <= threshold * squared)
    return bool(answer) if answer.ndim == 0 else answer


__all__ = ["GeometryTolerance", "GeometryToleranceError"]
