"""UI-independent state and calculations for the element quality explorer."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..mesh import Mesh2D
from .checker import ElementType, MeshQualityChecker, QualityReason


FloatArray = NDArray[np.float64]


TRIANGLE_PRESET = np.array(
    [[-0.70, -0.55], [0.70, -0.55], [0.00, 0.66]],
    dtype=np.float64,
)
QUADRILATERAL_PRESET = np.array(
    [[-0.65, -0.55], [0.65, -0.55], [0.65, 0.55], [-0.65, 0.55]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ElementQuality:
    """Quality values for one interactive Tri3 or Quad4 element."""

    element_type: ElementType
    jacobian_minimum: float
    jacobian_maximum: float
    scaled_jacobian_minimum: float
    scaled_jacobian_maximum: float
    aspect_ratio: float
    reasons: QualityReason

    @property
    def is_valid(self) -> bool:
        """Whether the checker considers the element geometry valid."""

        return self.reasons == QualityReason.NONE


@dataclass(frozen=True)
class QualityThresholds:
    """User-facing limits used to turn quality metrics into pass/fail status."""

    minimum_scaled_jacobian: float = 0.30
    maximum_aspect_ratio: float = 4.00

    def __post_init__(self) -> None:
        for name in ("minimum_scaled_jacobian", "maximum_aspect_ratio"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not -1.0 <= self.minimum_scaled_jacobian <= 1.0:
            raise ValueError("minimum_scaled_jacobian must be between -1 and 1")
        if self.maximum_aspect_ratio < 1.0:
            raise ValueError("maximum_aspect_ratio must be at least 1")


@dataclass(frozen=True)
class QualityStatus:
    """Display status after applying user thresholds to an element."""

    key: str
    label: str
    detail: str


def preset_nodes(element_type: ElementType | str) -> FloatArray:
    """Return an independent copy of the default nodes for an element type."""

    normalized = ElementType(element_type)
    preset = TRIANGLE_PRESET if normalized is ElementType.TRI3 else QUADRILATERAL_PRESET
    return preset.copy()


def evaluate_element(nodes: ArrayLike) -> ElementQuality:
    """Calculate all public quality metrics for one three- or four-node element."""

    node_array = np.asarray(nodes, dtype=np.float64)
    if node_array.ndim != 2 or node_array.shape not in ((3, 2), (4, 2)):
        raise ValueError("nodes must have shape (3, 2) or (4, 2)")

    if node_array.shape[0] == 3:
        element = np.array([[0, 1, 2, 2]], dtype=np.int64)
        element_type = ElementType.TRI3
    else:
        element = np.array([[0, 1, 2, 3]], dtype=np.int64)
        element_type = ElementType.QUAD4

    checker = MeshQualityChecker(Mesh2D(node_array, element))
    jacobian = checker.calculate_jacobian()
    scaled = checker.calculate_scaled_jacobian()
    aspect = checker.calculate_aspect_ratio()
    reasons = QualityReason(
        int(jacobian.reason_flags[0])
        | int(scaled.reason_flags[0])
        | int(aspect.reason_flags[0])
    )
    return ElementQuality(
        element_type=element_type,
        jacobian_minimum=float(jacobian.minimum_values[0]),
        jacobian_maximum=float(jacobian.maximum_values[0]),
        scaled_jacobian_minimum=float(scaled.minimum_values[0]),
        scaled_jacobian_maximum=float(scaled.maximum_values[0]),
        aspect_ratio=float(aspect.values[0]),
        reasons=reasons,
    )


def classify_quality(
    quality: ElementQuality,
    thresholds: QualityThresholds,
) -> QualityStatus:
    """Classify quality without changing the checker's geometry semantics."""

    if not quality.is_valid:
        return QualityStatus(
            key="invalid",
            label="INVALID",
            detail=format_reasons(quality.reasons),
        )

    failures = []
    if quality.scaled_jacobian_minimum < thresholds.minimum_scaled_jacobian:
        failures.append("scaled Jacobian is below the minimum")
    if quality.aspect_ratio > thresholds.maximum_aspect_ratio:
        failures.append("aspect ratio is above the maximum")
    if failures:
        return QualityStatus(
            key="warning",
            label="BELOW THRESHOLD",
            detail="; ".join(failures),
        )
    return QualityStatus(
        key="pass",
        label="PASS",
        detail="Element meets both quality thresholds",
    )


_REASON_LABELS = {
    QualityReason.INVALID_CONNECTIVITY: "invalid connectivity",
    QualityReason.OUT_OF_BOUNDS: "node index out of bounds",
    QualityReason.NONFINITE_COORDINATES: "non-finite coordinates",
    QualityReason.DEGENERATE: "degenerate geometry",
    QualityReason.INVERTED: "inverted orientation",
    QualityReason.FOLDED: "folded geometry",
    QualityReason.BELOW_MINIMUM: "below minimum",
    QualityReason.ABOVE_MAXIMUM: "above maximum",
    QualityReason.UNDEFINED_METRIC: "undefined metric",
}


def format_reasons(reasons: QualityReason) -> str:
    """Convert checker flags into a concise human-readable message."""

    if reasons == QualityReason.NONE:
        return "No geometry issues"
    return ", ".join(
        label for reason, label in _REASON_LABELS.items() if reasons & reason
    )


def format_metric(value: float) -> str:
    """Format a metric consistently for the compact GUI cards."""

    if np.isnan(value):
        return "undefined"
    if np.isposinf(value):
        return "+infinity"
    if np.isneginf(value):
        return "-infinity"
    return f"{value:.6f}"


__all__ = [
    "ElementQuality",
    "QualityStatus",
    "QualityThresholds",
    "classify_quality",
    "evaluate_element",
    "format_metric",
    "format_reasons",
    "preset_nodes",
]
