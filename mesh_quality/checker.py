"""Finite-element mesh quality checks for planar Tri3 and Quad4 cells."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, StrEnum
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mesh import Mesh


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ElementType(StrEnum):
    """Element types recognized by :class:`MeshQualityChecker`."""

    TRI3 = "TRI3"
    QUAD4 = "QUAD4"


class QualityReason(IntFlag):
    """Per-element reasons for an invalid or failed quality check."""

    NONE = 0
    INVALID_CONNECTIVITY = 1 << 0
    OUT_OF_BOUNDS = 1 << 1
    NONFINITE_COORDINATES = 1 << 2
    DEGENERATE = 1 << 3
    INVERTED = 1 << 4
    FOLDED = 1 << 5
    BELOW_MINIMUM = 1 << 6
    ABOVE_MAXIMUM = 1 << 7
    UNDEFINED_METRIC = 1 << 8


_INVALID_REASONS = (
    QualityReason.INVALID_CONNECTIVITY
    | QualityReason.OUT_OF_BOUNDS
    | QualityReason.NONFINITE_COORDINATES
    | QualityReason.DEGENERATE
    | QualityReason.INVERTED
    | QualityReason.FOLDED
    | QualityReason.UNDEFINED_METRIC
)


@dataclass(frozen=True)
class GeometryTolerance:
    """Scale-aware tolerances used to classify element geometry.

    ``absolute`` has units of length. The Jacobian tolerance is derived from
    it as ``absolute**2 + relative*h**2``, where ``h`` is the longest edge in
    an element. ``planar`` is the absolute Z tolerance for XY meshes.
    """

    absolute: float = 1.0e-12
    relative: float = 1.0e-12
    planar: float = 1.0e-12

    def __post_init__(self) -> None:
        for name in ("absolute", "relative", "planar"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} tolerance must be a real number")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} tolerance must be finite and non-negative")

    @property
    def absolute_length(self) -> float:
        return self.absolute

    @property
    def relative_length(self) -> float:
        return self.relative

    @property
    def planarity(self) -> float:
        return self.planar


@dataclass(frozen=True)
class QualitySummary:
    """Aggregate statistics and status counts for a quality report."""

    count: int
    minimum: float
    maximum: float
    mean: float
    worst_index: int | None
    total_count: int
    passed_count: int
    failed_count: int
    invalid_count: int


@dataclass(frozen=True)
class QualityCalculationSummary:
    """Aggregate statistics for a metric calculation over a subset."""

    count: int
    minimum: float
    maximum: float
    mean: float
    worst_index: int | None
    total_count: int
    invalid_count: int


def _readonly_array(values: ArrayLike, dtype: Any) -> NDArray[Any]:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class QualityReport:
    """Immutable per-element values and indices from one quality check.

    ``values`` is the value compared with the requested criterion. It is the
    minimum sampled value for the Jacobian checks and the edge ratio for the
    aspect-ratio check.
    """

    metric_name: str
    element_indices: IntArray
    values: FloatArray
    minimum_values: FloatArray
    maximum_values: FloatArray
    element_types: tuple[ElementType, ...]
    reason_flags: tuple[QualityReason, ...]
    passed_indices: IntArray
    failed_indices: IntArray
    invalid_indices: IntArray
    summary: QualitySummary

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_indices",
            _readonly_array(self.element_indices, np.int64),
        )
        object.__setattr__(self, "values", _readonly_array(self.values, np.float64))
        object.__setattr__(
            self,
            "minimum_values",
            _readonly_array(self.minimum_values, np.float64),
        )
        object.__setattr__(
            self,
            "maximum_values",
            _readonly_array(self.maximum_values, np.float64),
        )
        object.__setattr__(
            self,
            "passed_indices",
            _readonly_array(self.passed_indices, np.int64),
        )
        object.__setattr__(
            self,
            "failed_indices",
            _readonly_array(self.failed_indices, np.int64),
        )
        object.__setattr__(
            self,
            "invalid_indices",
            _readonly_array(self.invalid_indices, np.int64),
        )
        object.__setattr__(self, "element_types", tuple(self.element_types))
        object.__setattr__(self, "reason_flags", tuple(self.reason_flags))

    @property
    def metrics(self) -> FloatArray:
        return self.values

    @property
    def types(self) -> tuple[ElementType, ...]:
        return self.element_types

    @property
    def flags(self) -> tuple[QualityReason, ...]:
        return self.reason_flags

    @property
    def min_values(self) -> FloatArray:
        return self.minimum_values

    @property
    def max_values(self) -> FloatArray:
        return self.maximum_values


@dataclass(frozen=True)
class QualityResult:
    """Immutable values from calculating one metric over a mesh subset."""

    metric_name: str
    element_indices: IntArray
    values: FloatArray
    minimum_values: FloatArray
    maximum_values: FloatArray
    element_types: tuple[ElementType, ...]
    reason_flags: tuple[QualityReason, ...]
    invalid_indices: IntArray
    summary: QualityCalculationSummary

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_indices",
            _readonly_array(self.element_indices, np.int64),
        )
        object.__setattr__(self, "values", _readonly_array(self.values, np.float64))
        object.__setattr__(
            self,
            "minimum_values",
            _readonly_array(self.minimum_values, np.float64),
        )
        object.__setattr__(
            self,
            "maximum_values",
            _readonly_array(self.maximum_values, np.float64),
        )
        object.__setattr__(
            self,
            "invalid_indices",
            _readonly_array(self.invalid_indices, np.int64),
        )
        object.__setattr__(self, "element_types", tuple(self.element_types))
        object.__setattr__(self, "reason_flags", tuple(self.reason_flags))

    @property
    def metrics(self) -> FloatArray:
        return self.values

    @property
    def types(self) -> tuple[ElementType, ...]:
        return self.element_types

    @property
    def flags(self) -> tuple[QualityReason, ...]:
        return self.reason_flags

    @property
    def min_values(self) -> FloatArray:
        return self.minimum_values

    @property
    def max_values(self) -> FloatArray:
        return self.maximum_values


@dataclass(frozen=True)
class _GeometryMetrics:
    element_types: tuple[ElementType, ...]
    reasons: NDArray[np.uint32]
    jacobian_minimum: FloatArray
    jacobian_maximum: FloatArray
    scaled_jacobian_minimum: FloatArray
    scaled_jacobian_maximum: FloatArray
    aspect_ratio: FloatArray


def _cross_2d(left: FloatArray, right: FloatArray) -> FloatArray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _validate_threshold(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class MeshQualityChecker:
    """Check the quality of a mixed planar Tri3/Quad4 mesh.

    Connectivity always has shape ``(M, 4)``. A row whose last two entries
    are equal is a padded Tri3 row; every other row is interpreted as Quad4.
    The unique perimeter nodes must be counter-clockwise in the XY plane.
    """

    _REFERENCE_CORNERS = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )

    def __init__(
        self,
        mesh: Mesh,
        *,
        tolerance: GeometryTolerance | None = None,
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise ValueError("mesh must be a Mesh instance")
        self.tolerance = tolerance if tolerance is not None else GeometryTolerance()
        if not isinstance(self.tolerance, GeometryTolerance):
            raise ValueError("tolerance must be a GeometryTolerance instance")

        node_array = np.asarray(mesh.nodes)
        if node_array.ndim != 2 or node_array.shape[1] not in (2, 3):
            raise ValueError("nodes must have shape (N, 2) or (N, 3)")
        if (
            not np.issubdtype(node_array.dtype, np.number)
            or np.issubdtype(node_array.dtype, np.complexfloating)
            or np.issubdtype(node_array.dtype, np.bool_)
        ):
            raise ValueError("nodes must have a real numeric dtype")

        element_array = np.asarray(mesh.elements)
        if element_array.ndim != 2 or element_array.shape[1] != 4:
            raise ValueError("elements must have shape (M, 4)")
        if not np.issubdtype(element_array.dtype, np.integer) or np.issubdtype(
            element_array.dtype,
            np.bool_,
        ):
            raise ValueError("elements must have an integer dtype")

        self._nodes = np.array(node_array, dtype=np.float64, copy=True)
        self._elements = np.array(element_array, dtype=np.int64, copy=True)
        if self._nodes.shape[1] == 3:
            finite_z = np.isfinite(self._nodes[:, 2])
            if np.any(np.abs(self._nodes[finite_z, 2]) > self.tolerance.planar):
                raise ValueError(
                    "3D nodes must lie in the XY plane within the planar tolerance"
                )
        self._nodes.setflags(write=False)
        self._elements.setflags(write=False)

    def calculate_jacobian(
        self,
        indices: list[int] | IntArray | None = None,
    ) -> QualityResult:
        """Calculate the minimum and maximum raw Jacobian on a subset."""

        element_indices = self._validate_indices(indices)
        geometry = self._calculate_geometry(element_indices)
        return self._build_result(
            metric_name="jacobian",
            element_indices=element_indices,
            values=geometry.jacobian_minimum,
            minimum_values=geometry.jacobian_minimum,
            maximum_values=geometry.jacobian_maximum,
            geometry=geometry,
            worst_is_minimum=True,
        )

    def calculate_scaled_jacobian(
        self,
        indices: list[int] | IntArray | None = None,
    ) -> QualityResult:
        """Calculate the minimum and maximum scaled Jacobian on a subset."""

        element_indices = self._validate_indices(indices)
        geometry = self._calculate_geometry(element_indices)
        return self._build_result(
            metric_name="scaled_jacobian",
            element_indices=element_indices,
            values=geometry.scaled_jacobian_minimum,
            minimum_values=geometry.scaled_jacobian_minimum,
            maximum_values=geometry.scaled_jacobian_maximum,
            geometry=geometry,
            worst_is_minimum=True,
        )

    def calculate_aspect_ratio(
        self,
        indices: list[int] | IntArray | None = None,
    ) -> QualityResult:
        """Calculate longest-edge / shortest-edge on a subset."""

        element_indices = self._validate_indices(indices)
        geometry = self._calculate_geometry(element_indices)
        return self._build_result(
            metric_name="aspect_ratio",
            element_indices=element_indices,
            values=geometry.aspect_ratio,
            minimum_values=geometry.aspect_ratio,
            maximum_values=geometry.aspect_ratio,
            geometry=geometry,
            worst_is_minimum=False,
        )

    def check_jacobian(
        self,
        minimum: float = 0.0,
        indices: list[int] | IntArray | None = None,
    ) -> QualityReport:
        """Check the minimum raw FEM Jacobian determinant per element."""

        criterion = _validate_threshold(minimum, "minimum")
        result = self.calculate_jacobian(indices)
        return self._build_report(
            result=result,
            criterion=criterion,
            criterion_is_minimum=True,
        )

    def check_scaled_jacobian(
        self,
        minimum: float,
        indices: list[int] | IntArray | None = None,
    ) -> QualityReport:
        """Check the minimum dimensionless scaled Jacobian per element."""

        criterion = _validate_threshold(minimum, "minimum")
        result = self.calculate_scaled_jacobian(indices)
        return self._build_report(
            result=result,
            criterion=criterion,
            criterion_is_minimum=True,
        )

    def check_aspect_ratio(
        self,
        maximum: float,
        indices: list[int] | IntArray | None = None,
    ) -> QualityReport:
        """Check longest-perimeter-edge / shortest-perimeter-edge."""

        criterion = _validate_threshold(maximum, "maximum")
        result = self.calculate_aspect_ratio(indices)
        return self._build_report(
            result=result,
            criterion=criterion,
            criterion_is_minimum=False,
        )

    def _validate_indices(
        self,
        indices: list[int] | IntArray | None,
    ) -> IntArray:
        element_count = self._elements.shape[0]
        if indices is None:
            return np.arange(element_count, dtype=np.int64)

        if isinstance(indices, list):
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in indices
            ):
                raise ValueError("indices must contain only integers")
            normalized = [int(value) for value in indices]
            if any(value < 0 for value in normalized):
                raise ValueError("indices must not contain negative values")
            if any(value >= element_count for value in normalized):
                raise ValueError("indices contains an out-of-range element index")
            result = np.array(normalized, dtype=np.int64)
        elif isinstance(indices, np.ndarray):
            if indices.ndim != 1:
                raise ValueError("indices must be one-dimensional")
            if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(
                indices.dtype,
                np.bool_,
            ):
                raise ValueError("indices must have an integer dtype")
            if np.any(indices < 0):
                raise ValueError("indices must not contain negative values")
            if np.any(indices >= element_count):
                raise ValueError("indices contains an out-of-range element index")
            result = np.array(indices, dtype=np.int64, copy=True)
        else:
            raise ValueError("indices must be None, a list, or a NumPy array")

        if np.unique(result).size != result.size:
            raise ValueError("indices must not contain duplicates")
        return result

    def _calculate_geometry(self, element_indices: IntArray) -> _GeometryMetrics:
        elements = self._elements[element_indices]
        element_count = elements.shape[0]
        is_triangle = elements[:, 2] == elements[:, 3]
        element_types = tuple(
            ElementType.TRI3 if triangle else ElementType.QUAD4
            for triangle in is_triangle
        )
        reasons = np.zeros(element_count, dtype=np.uint32)
        jacobian_minimum = np.full(element_count, np.nan, dtype=np.float64)
        jacobian_maximum = np.full(element_count, np.nan, dtype=np.float64)
        scaled_minimum = np.full(element_count, np.nan, dtype=np.float64)
        scaled_maximum = np.full(element_count, np.nan, dtype=np.float64)
        aspect_ratio = np.full(element_count, np.nan, dtype=np.float64)

        if element_count == 0:
            return _GeometryMetrics(
                element_types,
                reasons,
                jacobian_minimum,
                jacobian_maximum,
                scaled_minimum,
                scaled_maximum,
                aspect_ratio,
            )

        node_count = self._nodes.shape[0]
        in_bounds = np.all(
            (elements >= 0) & (elements < node_count),
            axis=1,
        )
        out_of_bounds = ~in_bounds
        reasons[out_of_bounds] |= int(
            QualityReason.INVALID_CONNECTIVITY | QualityReason.OUT_OF_BOUNDS
        )

        duplicate_triangle = is_triangle & (
            (elements[:, 0] == elements[:, 1])
            | (elements[:, 0] == elements[:, 2])
            | (elements[:, 1] == elements[:, 2])
        )
        quad = ~is_triangle
        duplicate_quad = np.zeros(element_count, dtype=bool)
        for left in range(4):
            for right in range(left + 1, 4):
                duplicate_quad |= quad & (
                    elements[:, left] == elements[:, right]
                )
        duplicate_connectivity = duplicate_triangle | duplicate_quad
        reasons[duplicate_connectivity] |= int(
            QualityReason.INVALID_CONNECTIVITY | QualityReason.DEGENERATE
        )

        coordinates = np.full(
            (element_count, 4, self._nodes.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        if np.any(in_bounds):
            coordinates[in_bounds] = self._nodes[elements[in_bounds]]

        finite_coordinates = in_bounds & np.all(
            np.isfinite(coordinates),
            axis=(1, 2),
        )
        nonfinite = in_bounds & ~finite_coordinates
        reasons[nonfinite] |= int(QualityReason.NONFINITE_COORDINATES)

        edge_lengths = np.full((element_count, 4), np.nan, dtype=np.float64)
        finite_triangles = np.flatnonzero(finite_coordinates & is_triangle)
        finite_quads = np.flatnonzero(finite_coordinates & quad)
        with np.errstate(over="ignore", invalid="ignore"):
            if finite_triangles.size:
                points = coordinates[finite_triangles, :3, :2]
                edges = np.roll(points, -1, axis=1) - points
                edge_lengths[finite_triangles, :3] = np.linalg.norm(edges, axis=2)
            if finite_quads.size:
                points = coordinates[finite_quads, :, :2]
                edges = np.roll(points, -1, axis=1) - points
                edge_lengths[finite_quads] = np.linalg.norm(edges, axis=2)

        element_scale = np.full(element_count, np.nan, dtype=np.float64)
        if finite_triangles.size:
            element_scale[finite_triangles] = np.max(
                edge_lengths[finite_triangles, :3],
                axis=1,
            )
        if finite_quads.size:
            element_scale[finite_quads] = np.max(
                edge_lengths[finite_quads],
                axis=1,
            )

        calculable = finite_coordinates
        length_tolerance = (
            self.tolerance.absolute + self.tolerance.relative * element_scale
        )
        jacobian_tolerance = (
            self.tolerance.absolute**2
            + self.tolerance.relative * element_scale**2
        )

        calculable_triangles = np.flatnonzero(calculable & is_triangle)
        calculable_quads = np.flatnonzero(calculable & quad)
        if calculable_triangles.size:
            self._calculate_triangles(
                calculable_triangles,
                coordinates,
                edge_lengths,
                length_tolerance,
                jacobian_tolerance,
                reasons,
                jacobian_minimum,
                jacobian_maximum,
                scaled_minimum,
                scaled_maximum,
                aspect_ratio,
            )
        if calculable_quads.size:
            self._calculate_quads(
                calculable_quads,
                coordinates,
                edge_lengths,
                length_tolerance,
                jacobian_tolerance,
                reasons,
                jacobian_minimum,
                jacobian_maximum,
                scaled_minimum,
                scaled_maximum,
                aspect_ratio,
            )

        return _GeometryMetrics(
            element_types,
            reasons,
            jacobian_minimum,
            jacobian_maximum,
            scaled_minimum,
            scaled_maximum,
            aspect_ratio,
        )

    def _calculate_triangles(
        self,
        rows: IntArray,
        coordinates: FloatArray,
        edge_lengths: FloatArray,
        length_tolerance: FloatArray,
        jacobian_tolerance: FloatArray,
        reasons: NDArray[np.uint32],
        jacobian_minimum: FloatArray,
        jacobian_maximum: FloatArray,
        scaled_minimum: FloatArray,
        scaled_maximum: FloatArray,
        aspect_ratio: FloatArray,
    ) -> None:
        points = coordinates[rows, :3, :2]
        p0, p1, p2 = points[:, 0], points[:, 1], points[:, 2]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw_jacobian = _cross_2d(p1 - p0, p2 - p0)
            jacobian_minimum[rows] = raw_jacobian
            jacobian_maximum[rows] = raw_jacobian

            first_vectors = np.stack((p1 - p0, p2 - p1, p0 - p2), axis=1)
            second_vectors = np.stack((p2 - p0, p0 - p1, p1 - p2), axis=1)
            corner_cross = _cross_2d(first_vectors, second_vectors)
            denominator = np.linalg.norm(first_vectors, axis=2) * np.linalg.norm(
                second_vectors,
                axis=2,
            )
            scaled_samples = (2.0 / np.sqrt(3.0)) * corner_cross / denominator
            valid_scaled = np.all(
                denominator > length_tolerance[rows, None] ** 2,
                axis=1,
            )
            scaled_minimum[rows[valid_scaled]] = np.min(
                scaled_samples[valid_scaled],
                axis=1,
            )
            scaled_maximum[rows[valid_scaled]] = np.max(
                scaled_samples[valid_scaled],
                axis=1,
            )

            lengths = edge_lengths[rows, :3]
            shortest = np.min(lengths, axis=1)
            longest = np.max(lengths, axis=1)
            valid_aspect = shortest > length_tolerance[rows]
            aspect_ratio[rows[valid_aspect]] = (
                longest[valid_aspect] / shortest[valid_aspect]
            )
            aspect_ratio[rows[~valid_aspect]] = np.inf

        zero_edge = np.any(
            edge_lengths[rows, :3] <= length_tolerance[rows, None],
            axis=1,
        )
        near_zero = np.abs(raw_jacobian) <= jacobian_tolerance[rows]
        inverted = raw_jacobian < -jacobian_tolerance[rows]
        reasons[rows[zero_edge | near_zero]] |= int(QualityReason.DEGENERATE)
        reasons[rows[inverted]] |= int(QualityReason.INVERTED)

    def _calculate_quads(
        self,
        rows: IntArray,
        coordinates: FloatArray,
        edge_lengths: FloatArray,
        length_tolerance: FloatArray,
        jacobian_tolerance: FloatArray,
        reasons: NDArray[np.uint32],
        jacobian_minimum: FloatArray,
        jacobian_maximum: FloatArray,
        scaled_minimum: FloatArray,
        scaled_maximum: FloatArray,
        aspect_ratio: FloatArray,
    ) -> None:
        points = coordinates[rows, :, :2]
        reference = self._REFERENCE_CORNERS
        d_shape_d_xi = np.empty((4, 4), dtype=np.float64)
        d_shape_d_eta = np.empty((4, 4), dtype=np.float64)
        for sample, (xi, eta) in enumerate(reference):
            d_shape_d_xi[sample] = 0.25 * reference[:, 0] * (
                1.0 + reference[:, 1] * eta
            )
            d_shape_d_eta[sample] = 0.25 * reference[:, 1] * (
                1.0 + reference[:, 0] * xi
            )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            tangent_xi = np.einsum("si,mid->msd", d_shape_d_xi, points)
            tangent_eta = np.einsum("si,mid->msd", d_shape_d_eta, points)
            jacobian_samples = _cross_2d(tangent_xi, tangent_eta)
            jacobian_minimum[rows] = np.min(jacobian_samples, axis=1)
            jacobian_maximum[rows] = np.max(jacobian_samples, axis=1)

            denominator = np.linalg.norm(tangent_xi, axis=2) * np.linalg.norm(
                tangent_eta,
                axis=2,
            )
            scaled_samples = jacobian_samples / denominator
            valid_scaled = np.all(
                denominator > length_tolerance[rows, None] ** 2,
                axis=1,
            )
            scaled_minimum[rows[valid_scaled]] = np.min(
                scaled_samples[valid_scaled],
                axis=1,
            )
            scaled_maximum[rows[valid_scaled]] = np.max(
                scaled_samples[valid_scaled],
                axis=1,
            )

            lengths = edge_lengths[rows]
            shortest = np.min(lengths, axis=1)
            longest = np.max(lengths, axis=1)
            valid_aspect = shortest > length_tolerance[rows]
            aspect_ratio[rows[valid_aspect]] = (
                longest[valid_aspect] / shortest[valid_aspect]
            )
            aspect_ratio[rows[~valid_aspect]] = np.inf

        zero_edge = np.any(
            edge_lengths[rows] <= length_tolerance[rows, None],
            axis=1,
        )
        near_zero = np.any(
            np.abs(jacobian_samples) <= jacobian_tolerance[rows, None],
            axis=1,
        )
        positive = np.any(
            jacobian_samples > jacobian_tolerance[rows, None],
            axis=1,
        )
        negative = np.any(
            jacobian_samples < -jacobian_tolerance[rows, None],
            axis=1,
        )
        folded = positive & negative
        reasons[rows[zero_edge | near_zero | folded]] |= int(QualityReason.DEGENERATE)
        reasons[rows[negative]] |= int(QualityReason.INVERTED)
        reasons[rows[folded]] |= int(QualityReason.FOLDED)

    def _build_result(
        self,
        *,
        metric_name: str,
        element_indices: IntArray,
        values: FloatArray,
        minimum_values: FloatArray,
        maximum_values: FloatArray,
        geometry: _GeometryMetrics,
        worst_is_minimum: bool,
    ) -> QualityResult:
        reasons = np.array(geometry.reasons, dtype=np.uint32, copy=True)
        finite = np.isfinite(values)
        reasons[~finite] |= int(QualityReason.UNDEFINED_METRIC)
        invalid = (reasons & int(_INVALID_REASONS)) != 0
        finite_positions = np.flatnonzero(finite & ~invalid)
        if finite_positions.size:
            finite_values = values[finite_positions]
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
            mean = float(np.mean(finite_values))
            local_worst = (
                int(np.argmin(finite_values))
                if worst_is_minimum
                else int(np.argmax(finite_values))
            )
            worst_index = int(element_indices[finite_positions[local_worst]])
        else:
            minimum = np.nan
            maximum = np.nan
            mean = np.nan
            worst_index = None

        summary = QualityCalculationSummary(
            count=int(finite_positions.size),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            worst_index=worst_index,
            total_count=int(values.size),
            invalid_count=int(np.count_nonzero(invalid)),
        )
        return QualityResult(
            metric_name=metric_name,
            element_indices=element_indices,
            values=values,
            minimum_values=minimum_values,
            maximum_values=maximum_values,
            element_types=geometry.element_types,
            reason_flags=tuple(QualityReason(int(value)) for value in reasons),
            invalid_indices=element_indices[invalid],
            summary=summary,
        )

    def _build_report(
        self,
        *,
        result: QualityResult,
        criterion: float,
        criterion_is_minimum: bool,
    ) -> QualityReport:
        values = result.values
        reasons = np.array(result.reason_flags, dtype=np.uint32)
        finite = np.isfinite(values)
        # Treat round-off around an exactly equal criterion as equality without
        # weakening the user-supplied quality threshold by a geometry tolerance.
        comparison_margin = 32.0 * np.finfo(np.float64).eps * np.maximum(
            np.abs(values),
            abs(criterion),
        )
        if criterion_is_minimum:
            criterion_failed = finite & (values < criterion - comparison_margin)
            reasons[criterion_failed] |= int(QualityReason.BELOW_MINIMUM)
        else:
            criterion_failed = finite & (values > criterion + comparison_margin)
            reasons[criterion_failed] |= int(QualityReason.ABOVE_MAXIMUM)

        invalid = (reasons & int(_INVALID_REASONS)) != 0
        failed = invalid | criterion_failed
        passed = ~failed
        passed_indices = result.element_indices[passed]
        failed_indices = result.element_indices[failed]
        invalid_indices = result.element_indices[invalid]

        finite_positions = np.flatnonzero(finite & ~invalid)
        if finite_positions.size:
            finite_values = values[finite_positions]
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
            mean = float(np.mean(finite_values))
            local_worst = (
                int(np.argmin(finite_values))
                if criterion_is_minimum
                else int(np.argmax(finite_values))
            )
            worst_index = int(result.element_indices[finite_positions[local_worst]])
        else:
            minimum = np.nan
            maximum = np.nan
            mean = np.nan
            worst_index = None

        summary = QualitySummary(
            count=int(finite_positions.size),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            worst_index=worst_index,
            total_count=int(values.size),
            passed_count=int(passed_indices.size),
            failed_count=int(failed_indices.size),
            invalid_count=int(invalid_indices.size),
        )
        reason_flags = tuple(QualityReason(int(value)) for value in reasons)
        return QualityReport(
            metric_name=result.metric_name,
            element_indices=result.element_indices,
            values=result.values,
            minimum_values=result.minimum_values,
            maximum_values=result.maximum_values,
            element_types=result.element_types,
            reason_flags=reason_flags,
            passed_indices=passed_indices,
            failed_indices=failed_indices,
            invalid_indices=invalid_indices,
            summary=summary,
        )
