"""Input contract shared by single-layer and full adaptive LINE meshing."""

from __future__ import annotations

import math

import numpy as np

from ._types import QuadLayerError


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
