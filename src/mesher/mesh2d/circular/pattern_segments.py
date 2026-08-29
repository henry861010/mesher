"""Shared normalization and circle geometry for pattern guide segments."""

from dataclasses import dataclass

import numpy as np


def _readonly(values, *, dtype=None):
    """Return an owned, read-only NumPy array."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class _PatternSegment:
    """Prepared geometry for one finite axis-aligned pattern segment."""

    coordinates: np.ndarray
    fixed_axis: int
    varying_axis: int
    fixed_value: float
    lower: float
    upper: float
    fixed_tolerance: float
    bound_tolerance: float

    @property
    def vertical(self):
        """Return whether X is the fixed coordinate."""
        return self.fixed_axis == 0

    @property
    def tolerance(self):
        """Return a conservative scalar tolerance for generic edge tests."""
        return max(self.fixed_tolerance, self.bound_tolerance)


@dataclass(frozen=True)
class _PatternGuideSet:
    """Immutable, reusable metadata for pattern guide segments.

    The public circular helpers still accept raw array-like guide segments.
    Preparing them once at the imprint entry point avoids repeated conversion,
    validation, and axis classification in every downstream stage.
    """

    segments: np.ndarray
    fixed_axes: np.ndarray
    varying_axes: np.ndarray
    fixed_values: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    fixed_tolerances: np.ndarray
    bound_tolerances: np.ndarray

    @classmethod
    def from_values(
        cls,
        guide_segments,
        *,
        coordinate_scale,
        minimum_tolerance=0.0,
    ):
        """Normalize, validate, and classify raw pattern segments."""
        try:
            coordinate_scale = float(coordinate_scale)
            minimum_tolerance = float(minimum_tolerance)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "pattern segment tolerances must be real numbers"
            ) from error
        if (
            not np.isfinite(coordinate_scale)
            or coordinate_scale < 0.0
            or not np.isfinite(minimum_tolerance)
            or minimum_tolerance < 0.0
        ):
            raise ValueError(
                "pattern segment tolerances must be finite and non-negative"
            )

        if guide_segments is None:
            values = np.empty((0, 2, 2), dtype=np.float64)
        else:
            try:
                values = np.asarray(guide_segments, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "guide_segments must have shape (L, 2, 2)"
                ) from error
            if values.size == 0:
                if values.shape not in ((0,), (0, 2, 2)):
                    raise ValueError(
                        "guide_segments must have shape (L, 2, 2)"
                    )
                values = np.empty((0, 2, 2), dtype=np.float64)
            elif values.ndim != 3 or values.shape[1:] != (2, 2):
                raise ValueError("guide_segments must have shape (L, 2, 2)")
        if not np.all(np.isfinite(values)):
            raise ValueError("guide_segments must contain finite coordinates")

        values = _readonly(values, dtype=np.float64)
        count = values.shape[0]
        if count == 0:
            empty_float = np.empty(0, dtype=np.float64)
            empty_axis = np.empty(0, dtype=np.int8)
            return cls(
                values,
                _readonly(empty_axis),
                _readonly(empty_axis),
                _readonly(empty_float),
                _readonly(empty_float),
                _readonly(empty_float),
                _readonly(empty_float),
                _readonly(empty_float),
            )

        segment_axis_scales = np.maximum(
            coordinate_scale,
            np.max(np.abs(values), axis=1),
        )
        axis_tolerances = np.maximum(
            minimum_tolerance,
            64.0 * np.finfo(np.float64).eps * segment_axis_scales,
        )
        deltas = values[:, 1] - values[:, 0]
        vertical = (
            (np.abs(deltas[:, 0]) <= axis_tolerances[:, 0])
            & (np.abs(deltas[:, 1]) > axis_tolerances[:, 1])
        )
        horizontal = (
            (np.abs(deltas[:, 1]) <= axis_tolerances[:, 1])
            & (np.abs(deltas[:, 0]) > axis_tolerances[:, 0])
        )
        if np.any(vertical == horizontal):
            raise ValueError(
                "each pattern segment must be non-zero and horizontal or vertical"
            )

        fixed_axes = np.where(vertical, 0, 1).astype(np.int8)
        varying_axes = (1 - fixed_axes).astype(np.int8)
        positions = np.arange(count, dtype=np.intp)
        fixed_values = values[positions, 0, fixed_axes]
        varying_endpoints = values[
            positions[:, None],
            np.arange(2, dtype=np.intp)[None, :],
            varying_axes[:, None],
        ]
        lower_bounds = np.min(varying_endpoints, axis=1)
        upper_bounds = np.max(varying_endpoints, axis=1)
        fixed_tolerances = axis_tolerances[positions, fixed_axes]
        bound_tolerances = axis_tolerances[positions, varying_axes]
        return cls(
            values,
            _readonly(fixed_axes),
            _readonly(varying_axes),
            _readonly(fixed_values),
            _readonly(lower_bounds),
            _readonly(upper_bounds),
            _readonly(fixed_tolerances),
            _readonly(bound_tolerances),
        )

    def __len__(self):
        return int(self.segments.shape[0])

    def __iter__(self):
        for position in range(len(self)):
            yield _PatternSegment(
                coordinates=self.segments[position],
                fixed_axis=int(self.fixed_axes[position]),
                varying_axis=int(self.varying_axes[position]),
                fixed_value=float(self.fixed_values[position]),
                lower=float(self.lower_bounds[position]),
                upper=float(self.upper_bounds[position]),
                fixed_tolerance=float(self.fixed_tolerances[position]),
                bound_tolerance=float(self.bound_tolerances[position]),
            )

    def subset(self, mask):
        """Return the selected segments with order and duplicates preserved."""
        selected = np.asarray(mask)
        if selected.shape != (len(self),) or selected.dtype != np.bool_:
            raise ValueError("pattern guide mask must be a boolean vector")
        return _PatternGuideSet(
            _readonly(self.segments[selected]),
            _readonly(self.fixed_axes[selected]),
            _readonly(self.varying_axes[selected]),
            _readonly(self.fixed_values[selected]),
            _readonly(self.lower_bounds[selected]),
            _readonly(self.upper_bounds[selected]),
            _readonly(self.fixed_tolerances[selected]),
            _readonly(self.bound_tolerances[selected]),
        )

    def intersecting_circle(self, center, radius):
        """Return only finite segments that meet the requested full circle."""
        return self.subset(_circle_intersection_mask(self, center, radius))


def _coerce_pattern_guides(
    guide_segments,
    *,
    coordinate_scale,
    minimum_tolerance=0.0,
):
    """Reuse a prepared guide set or prepare raw array-like input."""
    if isinstance(guide_segments, _PatternGuideSet):
        return guide_segments
    return _PatternGuideSet.from_values(
        guide_segments,
        coordinate_scale=coordinate_scale,
        minimum_tolerance=minimum_tolerance,
    )


def _read_circle(center, radius):
    """Validate shared circle arguments used by intersection helpers."""
    try:
        center = np.asarray(center, dtype=np.float64)
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("circle center and radius must be real numbers") from error
    if (
        center.shape != (2,)
        or not np.all(np.isfinite(center))
        or not np.isfinite(radius)
        or radius <= 0.0
    ):
        raise ValueError("circle center must be finite and radius must be positive")
    return center, radius


def _circle_intersection_mask(pattern_guides, center, radius):
    """Vectorize finite-segment intersection tests for one full circle."""
    center, radius = _read_circle(center, radius)
    if len(pattern_guides) == 0:
        return np.empty(0, dtype=np.bool_)

    fixed_centers = center[pattern_guides.fixed_axes]
    varying_centers = center[pattern_guides.varying_axes]
    with np.errstate(over="ignore", invalid="ignore"):
        fixed_offsets = pattern_guides.fixed_values - fixed_centers
        fixed_distances = np.abs(fixed_offsets)
    line_hits = (
        np.isfinite(fixed_distances)
        & (
            fixed_distances
            <= radius + pattern_guides.fixed_tolerances
        )
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = np.minimum(fixed_distances / radius, 1.0)
        roots = radius * np.sqrt(
            np.maximum(0.0, (1.0 - normalized) * (1.0 + normalized))
        )
        lower_roots = varying_centers - roots
        upper_roots = varying_centers + roots
    bounds_hit = (
        (
            np.isfinite(lower_roots)
            & (
                lower_roots
                >= pattern_guides.lower_bounds
                - pattern_guides.bound_tolerances
            )
            & (
                lower_roots
                <= pattern_guides.upper_bounds
                + pattern_guides.bound_tolerances
            )
        )
        | (
            np.isfinite(upper_roots)
            & (
                upper_roots
                >= pattern_guides.lower_bounds
                - pattern_guides.bound_tolerances
            )
            & (
                upper_roots
                <= pattern_guides.upper_bounds
                + pattern_guides.bound_tolerances
            )
        )
    )
    return line_hits & bounds_hit


def _circle_line_intersections(segment, center, radius, *, tolerance=None):
    """Return intersections between one prepared infinite line and a circle."""
    center, radius = _read_circle(center, radius)
    line_tolerance = segment.fixed_tolerance
    if tolerance is not None:
        try:
            tolerance = float(tolerance)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("circle intersection tolerance must be real") from error
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "circle intersection tolerance must be finite and non-negative"
            )
        line_tolerance = max(line_tolerance, tolerance)
    with np.errstate(over="ignore", invalid="ignore"):
        fixed_offset = segment.fixed_value - center[segment.fixed_axis]
    if not np.isfinite(fixed_offset):
        return np.empty((0, 2), dtype=np.float64)
    fixed_distance = abs(float(fixed_offset))
    if fixed_distance > radius + line_tolerance:
        return np.empty((0, 2), dtype=np.float64)

    normalized = min(fixed_distance / radius, 1.0)
    root = radius * np.sqrt(max(0.0, (1.0 - normalized) * (1.0 + normalized)))
    varying_center = center[segment.varying_axis]
    varying_values = [float(varying_center - root)]
    if root > line_tolerance:
        varying_values.append(float(varying_center + root))

    points = np.empty((len(varying_values), 2), dtype=np.float64)
    points[:, segment.fixed_axis] = segment.fixed_value
    points[:, segment.varying_axis] = varying_values
    return points


def _circle_segment_intersections(
    segment,
    center,
    radius,
    *,
    tolerance=None,
):
    """Return circle intersections lying on one finite prepared segment."""
    points = _circle_line_intersections(
        segment,
        center,
        radius,
        tolerance=tolerance,
    )
    if points.shape[0] == 0:
        return points
    segment_tolerance = segment.bound_tolerance
    if tolerance is not None:
        segment_tolerance = max(segment_tolerance, float(tolerance))
    varying = points[:, segment.varying_axis]
    on_segment = (
        np.isfinite(varying)
        & (varying >= segment.lower - segment_tolerance)
        & (varying <= segment.upper + segment_tolerance)
    )
    return points[on_segment]


def _interval_overlap(
    first_lower,
    first_upper,
    second_lower,
    second_upper,
    tolerance,
):
    """Return a tolerant interval intersection and whether it is nontrivial."""
    lower = max(float(first_lower), float(second_lower))
    upper = min(float(first_upper), float(second_upper))
    return lower, upper, upper - lower > float(tolerance)


__all__ = [
    "_PatternGuideSet",
    "_PatternSegment",
    "_circle_line_intersections",
    "_circle_segment_intersections",
    "_coerce_pattern_guides",
    "_interval_overlap",
]
