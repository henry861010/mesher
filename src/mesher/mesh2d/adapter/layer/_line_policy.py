"""Input contract shared by single-layer and full adaptive LINE meshing."""

from __future__ import annotations

import math

import numpy as np

from ._types import QuadLayerError

DEFAULT_MAX_TRANSITION_COUNT = 20
DEFAULT_MAX_SIZE_RATIO = 1.25


def validate_dense_parameters(small_element_size, dense_mapping_angle):
    """World-length threshold and angular mapping limit in degrees."""
    values = []
    for name, value, optional in (("small_element_size", small_element_size, True),
                                   ("dense_mapping_angle", dense_mapping_angle, False)):
        if optional and value is None:
            values.append(None)
            continue
        valid = (not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, float, np.integer, np.floating)))
        try:
            number = float(value) if valid else float("nan")
        except (ValueError, OverflowError):
            number = float("nan")
        if not math.isfinite(number) or (number <= 0 if optional else not 0 <= number < 90):
            requirement = "positive or None" if optional else "in [0, 90) degrees"
            raise QuadLayerError(f"{name} must be finite and {requirement}",
                                 code="invalid_input", stage="input_validation", parameter=name)
        values.append(number)
    return tuple(values)


def validate_line_parameters(max_transition_count, min_size_ratio, max_size_ratio=DEFAULT_MAX_SIZE_RATIO):
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
        message = f"{name} must be a finite real number in {domain}"
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
