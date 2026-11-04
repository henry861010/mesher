"""Count-only guards for the number of quadrilaterals added by one layer."""

from __future__ import annotations

from math import ceil, floor

import numpy as np

from ._types import QuadLayerError


class ElementLimitExceeded(QuadLayerError):
    """A candidate or selected layer exceeds the caller's element limit."""

    def __init__(self, count: int, limit: int, *, stage: str,
                 count_kind: str = "actual", **context):
        super().__init__(
            "the layer exceeds max_added_elements",
            code="element_limit_exceeded", stage=stage,
            limit=int(limit), required_elements=int(count), count_kind=count_kind,
            **context,
        )


def check_element_limit(count, limit, *, stage, count_kind="actual", **context):
    if limit is not None and count > limit:
        raise ElementLimitExceeded(count, limit, stage=stage,
                                   count_kind=count_kind, **context)


def outer_segment_counts(source_segments, left_nodes, right_nodes,
                         length, h, mandatory_intervals, *, fixed_outer_boundary=False):
    """Nearest parity-compatible counts, including an empty INNER front."""
    parity = (source_segments + left_nodes + right_nodes) % 2
    if fixed_outer_boundary:
        return [mandatory_intervals] if mandatory_intervals % 2 == parity else []
    if mandatory_intervals == 0:
        return [0] if parity == 0 else []
    target = float(length) / h
    lower = floor(target)
    lower -= (lower - parity) % 2
    upper = ceil(max(target, mandatory_intervals))
    upper += (parity - upper) % 2
    return sorted({n for n in (lower, upper) if n >= mandatory_intervals})


def section_element_lower_bound(source_segments, outer_segments,
                                left_nodes, right_nodes):
    # Euler's formula for a quadrangulated disk: Q = B/2 + I - 1.
    boundary_segments = source_segments + outer_segments + left_nodes + right_nodes - 2
    return boundary_segments // 2 - 1


def bounded_outer_counts(counts, source_segments, left_nodes, right_nodes, limit):
    """Filter before sampling; retain a typed reason if other counts fail."""
    retained, rejected = [], None
    for count in counts:
        lower = section_element_lower_bound(source_segments, count, left_nodes, right_nodes)
        if limit is not None and lower > limit:
            if rejected is None:
                rejected = ElementLimitExceeded(
                    lower, limit, stage="section_count_preflight", count_kind="lower_bound",
                    outer_segments=int(count),
                )
        else:
            retained.append(count)
    return retained, rejected


def preflight_element_limit(plan, data):
    """Reject a proven whole-layer excess before any sampling or unit build.

    One minimum is counted per section. Seam alternatives and trials are not
    additive output, and independently minimized seams only weaken this bound.
    """
    limit = data.get("max_added_elements")
    if limit is None:
        return
    lower_bound = 0
    for section in plan.sections:
        source_segments = sum(len(chain) - 1 for chain in section.source_chains)
        outer = section.outer_boundary
        if section.kind == "LINE":
            unique = [outer[0]]
            for point in outer[1:]:
                if np.linalg.norm(point - unique[-1]) > data["tol"]:
                    unique.append(point)
            outer = np.asarray(unique)
        lengths = np.linalg.norm(np.diff(outer, axis=0), axis=1)
        line_bounds = None
        if section.kind == "LINE" and not data.get("fixed_outer_boundary", False):
            from ._construction.sampling import line_count_bounds
            try:
                line_bounds = line_count_bounds(
                    lengths, source_segments, data["h"], data["tol"],
                    data.get("max_transition_count", 10), data.get("min_size_ratio", .5))
            except QuadLayerError as error:
                error.context.setdefault("section_id", section.id)
                raise
        minima = []
        left_base = len(plan.seams[section.left_seam].points)
        right_base = len(plan.seams[section.right_seam].points)
        for left in (left_base, left_base + 1):
            for right in (right_base, right_base + 1):
                if line_bounds is None:
                    counts = outer_segment_counts(source_segments, left, right,
                                                  float(lengths.sum()), data["h"], len(lengths),
                                                  fixed_outer_boundary=data.get("fixed_outer_boundary", False))
                else:
                    lower, upper = line_bounds
                    count = lower + ((source_segments + left + right - lower) % 2)
                    counts = [count] if count <= upper else []
                minima.extend(section_element_lower_bound(source_segments, count, left, right)
                              for count in counts)
        if minima:
            lower_bound += min(minima)
    check_element_limit(lower_bound, limit, stage="element_count_preflight",
                        count_kind="lower_bound")
