"""Optional dense-run boundaries, shared by preview and layer construction."""

from __future__ import annotations

import numpy as np

from .._line_policy import DEFAULT_DENSE_MIN_GAP_RATIO
from .._types import QuadLayerError
from .dense_contacts import DenseLineContacts
from .mapping import target_node_id


class DenseLinePartition:
    def __init__(self, data, interfaces=()):
        self.data = data
        self.sections = []
        self.seams = []
        self.mappings = []
        self.interfaces = tuple(interfaces)
        self.targets = data.get("_dense_seam_targets")
        self.used_targets = set()

    def fail(self, message, edge, indices):
        source = self.data["ids"][edge]
        raise QuadLayerError(message, code="unmeshable_section_layout",
                             stage="section_partition", source_edge=edge,
                             source_indices=list(map(int, indices)),
                             source_nodes=[int(source[i]) for i in indices])

    def cuts(self, edge, start, stop, left, right, registry, outer):
        """Find maximal qualifying runs and register their selected cuts."""
        data = self.data
        size = data.get("small_element_size")
        if size is None:
            return [(start, left), (stop, right)], []
        ids = data["ids"][edge]
        if "_source_mesh" in data:
            points = data["_source_mesh"].nodes[ids[start:stop + 1], :2]
        else:
            points = data["xy"][ids[start:stop + 1]]
            size /= data["scale"]
        delta = np.abs(np.diff(points, axis=0))
        # Allow for rounding at the two endpoints of a varying coordinate.
        # A common coordinate cancels exactly, even far from the origin.
        ulps = np.abs(np.spacing(points))
        uncertainty = np.where(delta == 0, 0., .5 * (ulps[:-1] + ulps[1:]))
        upper_length = np.hypot.reduce(delta + uncertainty, axis=1)
        short = upper_length < size - 8 * np.finfo(float).eps * size
        changes = np.diff(np.r_[False, short, False].astype(np.int8))
        begins = (np.flatnonzero(changes == 1) + start).tolist()
        ends = (np.flatnonzero(changes == -1) + start).tolist()
        group_count = data.get("small_element_group_count", 0)
        runs = [(begin, end) for begin, end in zip(begins, ends) if end - begin > group_count]
        endpoints = sorted((index, role) for run in runs for index, role in zip(run, ("begin", "end"))
                           if start < index < stop)
        if not endpoints:
            return [(start, left), (stop, right)], runs
        interfaces = [seam.points[[0, -1]] for seam in registry.items] + list(self.interfaces)
        mapper = DenseLineContacts(data, edge, endpoints, start, stop, outer, interfaces)
        gaps = [(left[1], right[0]) for left, right in zip(runs, runs[1:])]
        ratio = data.get("dense_min_gap_ratio", DEFAULT_DENSE_MIN_GAP_RATIO)
        if self.targets is None:
            contacts, reasons, limited = mapper.select(gaps, ratio)
        else:
            contacts, reasons, limited = self._restore_contacts(edge, mapper)
            conflicts = mapper.narrow_gaps(gaps, contacts, ratio)
            if conflicts:
                self.fail("planned dense seams no longer preserve their intervening gap",
                          edge, sorted(conflicts))
        cuts = [(start, left)]
        tangent, normal = data["directions"][edge], data["normals"][edge]
        for index, role in endpoints:
            source = data["xy"][ids[index]]
            record = {"source_edge": edge, "source_index": index,
                      "source_node": int(ids[index]), "endpoint": role, "gap_limited": index in limited}
            target, reason = contacts.get(index), reasons[index]
            if target is None:
                self.mappings.append({**record, "status": "skipped", "reason": reason})
                continue
            seam = registry.add((source, target))
            cuts.append((index, seam))
            target_id = target_node_id(target, data)
            world_target = (data["_source_mesh"].nodes[target_id, :2]
                            if target_id is not None and "_source_mesh" in data
                            else target * data["scale"] + data["origin"])
            delta = (world_target - data["origin"]) / data["scale"] - source
            angle = float(np.degrees(np.arctan2(delta @ tangent, delta @ normal)))
            record.update(seam_id=seam, target_node=target_id,
                          target=world_target.tolist(), angle=angle)
            self.seams.append(record)
            self.mappings.append({**record, "status": "mapped", "reason": reason})
        cuts.append((stop, right))
        return cuts, runs

    def _restore_contacts(self, edge, mapper):
        """Reuse the exact contacts sent to the global grid, including protected normals."""
        contacts = mapper.bounds.copy()
        reasons = {}
        limited = set()
        saved_limits = self.data.get("_dense_gap_limited", ())
        for index in mapper.roles:
            key = (edge, index)
            if key in saved_limits:
                limited.add(index)
            if key not in self.targets:
                reasons[index] = "not_planned"
                continue
            point = mapper.match_saved(index, self.targets[key], contacts, limited)
            if point is None:
                self.fail("planned dense seam is absent or incompatible with the fixed interface", edge, [index])
            contacts[index], reasons[index] = point, "canonical_contact"
            self.used_targets.add(key)
        return contacts, reasons, limited

    def record(self, section, start, stop, runs):
        if any(a <= start and stop <= b for a, b in runs):
            self.sections.append({"section_id": section.id, "source_edge": section.source_edge,
                                  "start_node": section.source_chains[0][0],
                                  "end_node": section.source_chains[0][-1],
                                  "left_seam": section.left_seam, "right_seam": section.right_seam})

    def diagnostics(self):
        if self.targets is not None and self.used_targets != set(self.targets):
            edge, index = next(iter(set(self.targets) - self.used_targets))
            self.fail("a planned dense seam disappeared during fixed construction", edge, [index])
        return {"small_element_size": self.data.get("small_element_size"),
                "small_element_group_count": self.data.get("small_element_group_count", 0),
                "dense_mapping_angle": self.data.get("dense_mapping_angle", 30.0),
                "dense_min_gap_ratio": self.data.get("dense_min_gap_ratio", DEFAULT_DENSE_MIN_GAP_RATIO),
                "dense_sections": self.sections, "dense_seams": self.seams,
                "dense_mappings": self.mappings}
