"""Input contract shared by single-layer and full adaptive LINE meshing."""

from __future__ import annotations

import math

import numpy as np

from ._types import QuadLayerError

DEFAULT_MAX_TRANSITION_COUNT = 20


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


def validate_line_parameters(max_transition_count, min_size_ratio):
    """Return normalized policy values, reporting typed input errors early."""
    if (isinstance(max_transition_count, (bool, np.bool_))
            or not isinstance(max_transition_count, (int, np.integer))
            or max_transition_count <= 0):
        raise QuadLayerError("max_transition_count must be a positive integer",
                             code="invalid_input", stage="input_validation",
                             parameter="max_transition_count")
    if (isinstance(min_size_ratio, (bool, np.bool_))
            or not isinstance(min_size_ratio, (int, float, np.integer, np.floating))):
        raise QuadLayerError("min_size_ratio must be a finite real number in (0, 1]",
                             code="invalid_input", stage="input_validation",
                             parameter="min_size_ratio")
    try:
        ratio = float(min_size_ratio)
    except (ValueError, TypeError, OverflowError) as error:
        raise QuadLayerError("min_size_ratio must be a finite real number in (0, 1]",
                             code="invalid_input", stage="input_validation",
                             parameter="min_size_ratio") from error
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise QuadLayerError("min_size_ratio must be a finite real number in (0, 1]",
                             code="invalid_input", stage="input_validation",
                             parameter="min_size_ratio")
    return int(max_transition_count), ratio
