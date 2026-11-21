"""Build a LINE from ordered source/front correspondences and full-height cuts."""

from __future__ import annotations

import numpy as np

from .._limits import ElementLimitExceeded, check_element_limit, section_element_lower_bound
from .._line_policy import DEFAULT_MAX_SIZE_RATIO, DEFAULT_MAX_TRANSITION_COUNT, validate_line_parameters
from .aligned import plan_resampled
from .strip import build_strip, fixed_cut_plans
from .units import SectionPatch, xyz_chain


def plan_line(source: np.ndarray, outer: np.ndarray, left: np.ndarray,
              right: np.ndarray, h: float, tol: float, *,
              best_only: bool = False,
              _diagnostics: dict | None = None,
              max_added_elements: int | None = None,
              fixed_outer_boundary: bool = False,
              max_transition_count: int = DEFAULT_MAX_TRANSITION_COUNT,
              min_size_ratio: float = 0.5,
              max_size_ratio: float = DEFAULT_MAX_SIZE_RATIO,
              _sampling_cache: dict | None = None) -> list[SectionPatch]:
    """Build bounded alternatives for one pair of prescribed section seams.

    Fixed fronts supply exact vertices. Resampleable fronts supply mandatory
    anchors and let the position of each source cut guide its outer partner.
    The section cache is shared only between seam states of this same LINE.
    """
    source, outer, left, right = map(xyz_chain, (source, outer, left, right))
    cap, min_ratio, max_ratio = validate_line_parameters(
        max_transition_count, min_size_ratio, max_size_ratio)
    if not np.isfinite(h) or h <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("element size and tolerance must be positive and finite")
    # Several contacts can refer to the same anchor. Preserve the surviving
    # coordinate exactly, comparing against the last retained contact.
    if len(outer) > 2:
        unique = [outer[0]]
        for point in outer[1:]:
            if np.linalg.norm(point[:2] - unique[-1][:2]) > tol:
                unique.append(point)
        outer = np.asarray(unique)
    if len(outer) < 2:
        raise ValueError("LINE outer boundary has no usable length")
    for actual, expected in ((left[0], source[0]), (left[-1], outer[0]),
                             (right[0], source[-1]), (right[-1], outer[-1])):
        if np.linalg.norm(actual[:2] - expected[:2]) > tol:
            raise ValueError("LINE source, outer and seam endpoints do not match")
    if any(np.any(np.linalg.norm(np.diff(chain[:, :2], axis=0), axis=1) <= tol)
           for chain in (source, outer, left, right)):
        raise ValueError("LINE boundary contains a zero-length segment")

    cache = {} if _sampling_cache is None else _sampling_cache
    if fixed_outer_boundary:
        if (len(source) + len(outer) + len(left) + len(right)) % 2:
            return []
        lower = section_element_lower_bound(len(source) - 1, len(outer) - 1, len(left), len(right))
        check_element_limit(lower, max_added_elements, stage="section_count_preflight",
                            count_kind="lower_bound")
        plans = ((outer, cuts) for cuts in fixed_cut_plans(source, outer, left, right, tol))
    else:
        plans = plan_resampled(source, outer, left, right, h, tol, cap, min_ratio, max_ratio,
                               cache=cache, max_added_elements=max_added_elements)

    stats = {} if _diagnostics is None else _diagnostics
    stats.update(line_planner="projection", unit_trial_count=0, local_repair_count=0,
                 outer_count_attempts=0, direct_quad_count=0)
    build_cache = cache.setdefault("unit_geometry", {})
    candidates = []
    limit_error = None
    for front, cuts in plans:
        stats["outer_count_attempts"] += 1
        gaps = np.linalg.norm(np.diff(front[:, :2], axis=0), axis=1)
        size_error = float(np.mean((gaps / h - 1) ** 2))
        try:
            patch = build_strip(source, front, left, right, cuts, size_error, tol,
                                cache=build_cache, stats=stats, max_added_elements=max_added_elements,
                                cap=None if fixed_outer_boundary else cap)
        except ElementLimitExceeded as error:
            limit_error = error
            continue
        if patch is None:
            continue
        if not fixed_outer_boundary:
            sampling = cache["sampling"]
            sampling.validate_sample(patch.nodes[patch.outer_ids])
            patch.diagnostics.update(sampling.diagnostics, line_sampling="projection")
        candidates.append(patch)
        if best_only:
            break
    for patch in candidates:
        patch.diagnostics.update(stats)
    if not candidates and limit_error is not None:
        raise limit_error
    return candidates
