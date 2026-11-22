"""Build a LINE from ordered source/front correspondences and full-height cuts."""

from __future__ import annotations

from functools import partial

import numpy as np

from .._limits import ElementLimitExceeded, check_element_limit, section_element_lower_bound
from .._line_policy import DEFAULT_MAX_SIZE_RATIO, DEFAULT_MAX_TRANSITION_COUNT, validate_line_parameters
from .aligned import plan_resampled
from .sampling import LineSampling
from .strip import build_strip, fixed_cut_plans
from .sublayers import build_sublayers
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
              size_band_active: bool = True,
              dense_sublayers: bool = False,
              allow_parity_split: bool = False,
              _sampling_cache: dict | None = None) -> list[SectionPatch]:
    """Build bounded alternatives for one pair of prescribed section seams.

    Fixed fronts supply exact vertices. Resampleable fronts supply mandatory
    anchors and let the position of each source cut guide its outer partner.
    Fully dense sections retain the target size and cap without a size band.
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

    stats = {} if _diagnostics is None else _diagnostics
    stats.update(line_planner="projection", unit_trial_count=0, local_repair_count=0,
                 outer_count_attempts=0, direct_quad_count=0)
    strips = partial(_strip_candidates, h=h, tol=tol, cap=cap, min_ratio=min_ratio,
                     max_ratio=max_ratio, size_band_active=size_band_active,
                     fixed_outer_boundary=fixed_outer_boundary, allow_parity_split=allow_parity_split,
                     stats=stats)
    if dense_sublayers:
        options = build_sublayers(source, outer, left, right, strips, max_added_elements)
    else:
        cache = {} if _sampling_cache is None else _sampling_cache
        options = strips(source, outer, left, right, cache=cache, max_added_elements=max_added_elements)
    candidates = []
    for patch in options:
        candidates.append(patch)
        if best_only:
            break
    for patch in candidates:
        patch.diagnostics.update(stats)
    return candidates


def _strip_candidates(source, outer, left, right, *, h, tol, cap, min_ratio, max_ratio,
                      size_band_active, fixed_outer_boundary, allow_parity_split,
                      stats, cache, max_added_elements):
    """Yield valid single-strip patches in the projection planner's preferred order."""
    parity_split = None
    if allow_parity_split and size_band_active and not fixed_outer_boundary:
        if "sampling" not in cache:
            cache["sampling"] = LineSampling(source, outer, h, tol, cap, min_ratio, max_ratio)
        parity_split = cache["sampling"].split_for_parity(len(left), len(right), max_added_elements)

    if parity_split is not None:
        _, split_sampling = parity_split
        front = split_sampling.anchors
        plans = ((front, cuts) for cuts in fixed_cut_plans(source, front, left, right, tol, cap))
    elif fixed_outer_boundary:
        if (len(source) + len(outer) + len(left) + len(right)) % 2:
            return
        lower = section_element_lower_bound(len(source) - 1, len(outer) - 1, len(left), len(right))
        check_element_limit(lower, max_added_elements, stage="section_count_preflight",
                            count_kind="lower_bound")
        plans = ((outer, cuts) for cuts in fixed_cut_plans(source, outer, left, right, tol))
    else:
        plans = plan_resampled(source, outer, left, right, h, tol, cap, min_ratio, max_ratio,
                               size_band_active=size_band_active, cache=cache,
                               max_added_elements=max_added_elements)

    build_cache = cache.setdefault("unit_geometry", {})
    yielded = False
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
        if fixed_outer_boundary:
            patch.diagnostics["size_band_active"] = False
        else:
            sampling = split_sampling if parity_split is not None else cache["sampling"]
            sampling.validate_sample(patch.nodes[patch.outer_ids])
            patch.diagnostics.update(sampling.diagnostics, line_sampling="projection")
            if parity_split is not None:
                patch.diagnostics["parity_split_index"] = parity_split[0]
        yielded = True
        yield patch
    if not yielded and limit_error is not None:
        raise limit_error
