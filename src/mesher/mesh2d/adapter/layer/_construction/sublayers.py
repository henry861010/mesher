"""Two conforming half-height strips for a fully dense LINE section."""
from __future__ import annotations

import numpy as np

from .._limits import ElementLimitExceeded
from .._types import QuadLayerError
from .units import SectionPatch


def _compose_loads(inner, outer):
    """Count original source segments reaching each final edge through the middle."""
    original = np.cumsum([0, *(n for unit in inner.units for n in unit["source_segments_per_outer"])])
    middle = np.cumsum([0, *(n for unit in outer.units for n in unit["source_segments_per_outer"])])
    return np.diff(original[middle])


def _join(inner, outer):
    loads = _compose_loads(inner, outer)
    # Only the common chain can coincide. Map it by identity, leaving private
    # unit nodes and the two lips of the section untouched.
    unique = np.ones(len(outer.nodes), dtype=bool)
    unique[outer.source_ids] = False
    remap = np.empty(len(outer.nodes), dtype=int)
    remap[outer.source_ids] = inner.outer_ids
    remap[unique] = len(inner.nodes) + np.arange(unique.sum())
    nodes = np.vstack((inner.nodes, outer.nodes[unique]))
    quads = np.vstack((inner.quads, remap[outer.quads]))
    records = []
    for unit in inner.units:
        records.append({**unit, "sublayer": 1, "source_segments_per_outer": [],
                        "sublayer_segments_per_outer": unit["source_segments_per_outer"]})
    start = 0
    for unit in outer.units:
        steps = unit["source_segments_per_outer"]
        record = {**unit, "sublayer": 2, "sublayer_segments_per_outer": steps,
                  "source_segments_per_outer": loads[start:start + len(steps)].tolist(),
                  "element_start": len(inner.quads) + unit["element_start"],
                  "element_end": len(inner.quads) + unit["element_end"],
                  "_polygon_ids": remap[unit["_polygon_ids"]].tolist()}
        if "_fixed_interior" in record:
            record["_fixed_interior"] = remap[record["_fixed_interior"]].tolist()
        records.append(record)
        start += len(steps)
    peak = max(p.diagnostics["max_connector_shear"] for p in (inner, outer))
    n_cuts = [len(p.diagnostics["internal_cuts"]) for p in (inner, outer)]
    alignment = sum(n * p.alignment_error for n, p in zip(n_cuts, (inner, outer))) / max(1, sum(n_cuts))
    # Sampling diagnostics describe the final front. The outer strip's cut
    # indices refer to the midline and cannot index the original source.
    diagnostics = {k: v for k, v in outer.diagnostics.items() if k != "internal_cuts"}
    diagnostics.update(midline_ids=inner.outer_ids.tolist(), alignment_error=alignment,
                       max_connector_shear=peak, max_alignment_angle=float(np.degrees(np.arctan(peak))),
                       sublayer_max_transition_counts=[inner.diagnostics["max_actual_transition_count"],
                                                       outer.diagnostics["max_actual_transition_count"]],
                       max_actual_transition_count=int(loads.max()))
    return SectionPatch(nodes, quads, np.r_[inner.left_ids, remap[outer.left_ids[-1]]],
                        np.r_[inner.right_ids, remap[outer.right_ids[-1]]], inner.source_ids,
                        remap[outer.outer_ids], records, outer.size_error, max(inner.quality, outer.quality),
                        diagnostics, alignment)


def build_sublayers(source, outer, left, right, strips, max_added_elements):
    """Coarsen to the midline first, then apply the same cap to its own edges."""
    middle_anchors = np.stack((left[1], right[1]))
    inner_options = strips(source, middle_anchors, left[:2], right[:2],
                           cache={}, max_added_elements=max_added_elements)
    yielded = False
    rejection = None
    for inner in inner_options:
        used = len(inner.quads)
        remaining = None if max_added_elements is None else max_added_elements - used
        midline = inner.nodes[inner.outer_ids]
        try:
            # Geometry and sampling depend on this particular midline.
            for upper in strips(midline, outer, left[1:], right[1:],
                                cache={}, max_added_elements=remaining):
                yielded = True
                yield _join(inner, upper)
        except ElementLimitExceeded as error:
            rejection = ElementLimitExceeded(used + error.context["required_elements"],
                                             max_added_elements, stage="dense_sublayers",
                                             count_kind=error.context["count_kind"])
        except QuadLayerError as error:
            if error.code != "line_constraints_infeasible":
                raise
            if rejection is None:
                rejection = error
    if not yielded and rejection is not None:
        raise rejection
