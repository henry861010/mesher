"""Input contract shared by single-layer and full adaptive LINE meshing."""

from __future__ import annotations

import math

import numpy as np

from ._types import QuadLayerError

DEFAULT_MAX_TRANSITION_COUNT = 20
DEFAULT_DENSE_MIN_GAP_RATIO = 0.5


def _finite_dense_number(name, value):
    if (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))):
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except (ValueError, OverflowError):
            pass
    raise QuadLayerError(f"{name} must be a finite real number", code="invalid_input",
                         stage="input_validation", parameter=name)


def validate_dense_parameters(small_element_size, small_element_group_count,
                              dense_mapping_angle, dense_min_gap_ratio):
    """Validate the dense-run detection and endpoint mapping controls."""
    if (isinstance(small_element_group_count, (bool, np.bool_))
            or not isinstance(small_element_group_count, (int, np.integer))
            or small_element_group_count < 0):
        raise QuadLayerError("small_element_group_count must be a nonnegative integer",
                             code="invalid_input", stage="input_validation",
                             parameter="small_element_group_count")
    size = small_element_size
    if size is not None:
        size = _finite_dense_number("small_element_size", size)
        if size <= 0:
            raise QuadLayerError("small_element_size must be positive or None", code="invalid_input",
                                 stage="input_validation", parameter="small_element_size")
    angle = _finite_dense_number("dense_mapping_angle", dense_mapping_angle)
    if not 0 <= angle < 90:
        raise QuadLayerError("dense_mapping_angle must be in [0, 90) degrees", code="invalid_input",
                             stage="input_validation", parameter="dense_mapping_angle")
    ratio = _finite_dense_number("dense_min_gap_ratio", dense_min_gap_ratio)
    if not 0 <= ratio <= 1:
        raise QuadLayerError("dense_min_gap_ratio must be in [0, 1]", code="invalid_input",
                             stage="input_validation", parameter="dense_min_gap_ratio")
    return size, int(small_element_group_count), angle, ratio


def validate_line_parameters(max_transition_count, min_size_ratio=None, max_size_ratio=None):
    """Return normalized policy values, reporting typed input errors early."""
    if (isinstance(max_transition_count, (bool, np.bool_))
            or not isinstance(max_transition_count, (int, np.integer))
            or max_transition_count <= 0):
        raise QuadLayerError("max_transition_count must be a positive integer",
                             code="invalid_input", stage="input_validation",
                             parameter="max_transition_count")
    ratios = []
    for name, value, domain in (("min_size_ratio", min_size_ratio, "(0, 1]"),
                                ("max_size_ratio", max_size_ratio, "[1, infinity)")):
        if value is None:
            ratios.append(None)
            continue
        message = f"{name} must be None or a finite real number in {domain}"
        valid_type = (not isinstance(value, (bool, np.bool_))
                      and isinstance(value, (int, float, np.integer, np.floating)))
        try:
            ratio = float(value) if valid_type else float("nan")
        except (ValueError, TypeError, OverflowError) as error:
            raise QuadLayerError(message, code="invalid_input", stage="input_validation",
                                 parameter=name) from error
        in_range = 0 < ratio <= 1 if name == "min_size_ratio" else ratio >= 1
        if not math.isfinite(ratio) or not in_range:
            raise QuadLayerError(message, code="invalid_input", stage="input_validation",
                                 parameter=name)
        ratios.append(ratio)
    return int(max_transition_count), ratios[0], ratios[1]
