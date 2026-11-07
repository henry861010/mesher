"""Optional dense-run boundaries, shared by preview and layer construction."""

from __future__ import annotations

import numpy as np

from .._types import QuadLayerError
from .mapping import target_node_id, valid_interface


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
        """Map each run endpoint independently, retaining only usable cuts."""
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
        short = np.linalg.norm(np.diff(points, axis=0), axis=1) < size
        changes = np.diff(np.r_[False, short, False].astype(np.int8))
        runs = list(zip((np.flatnonzero(changes == 1) + start).tolist(),
                        (np.flatnonzero(changes == -1) + start).tolist()))
        endpoints = sorted((index, role) for run in runs for index, role in zip(run, ("begin", "end"))
                           if start < index < stop)
        cuts = [(start, left)]
        tangent, normal, tol = data["directions"][edge], data["normals"][edge], data["tol"]
        angle_limit = data.get("dense_mapping_angle", 30.0)
        slope_limit = float(np.tan(np.radians(angle_limit)))
        fixed = data.get("fixed_outer_boundary", False)
        # Candidates are the original contacts of this LINE. A newly projected
        # dense endpoint must not bias the next endpoint's mapping.
        ordered = np.asarray(sorted(outer, key=lambda p: float(p @ tangent)))
        for index, role in endpoints:
            source = data["xy"][ids[index]]
            projected = source + normal * float((outer[0] - source) @ normal)
            previous = registry.items[cuts[-1][1]].points[-1]
            following = registry.items[right].points[-1]
            interfaces = [seam.points[[0, -1]] for seam in registry.items]
            interfaces.extend(self.interfaces)

            def usable(point):
                return (float((point - previous) @ tangent) > tol
                        and float((following - point) @ tangent) > tol
                        and valid_interface(source, point, data, interfaces))

            record = {"source_edge": edge, "source_index": index,
                      "source_node": int(ids[index]), "endpoint": role}
            target, reason = None, "no_eligible_node"
            key = (edge, index)
            if self.targets is not None:
                # Global meshing may add a nearer node: consume only the saved
                # canonical contact, including the absence of an optional cut.
                if key in self.targets:
                    distances = np.linalg.norm(ordered - self.targets[key], axis=1)
                    nearest = int(np.argmin(distances))
                    if distances[nearest] > tol or not usable(ordered[nearest]):
                        self.fail("planned dense seam is absent or incompatible with the fixed interface",
                                  edge, [index])
                    target, reason = ordered[nearest].copy(), "canonical_contact"
                    self.used_targets.add(key)
                else:
                    reason = "not_planned"
            else:
                delta = ordered - source
                along, outward = delta @ tangent, delta @ normal
                eligible = ((outward > tol)
                            & (np.abs(along) <= outward * slope_limit + tol)
                            & ((along <= tol) if role == "begin" else (along >= -tol)))
                angles = np.degrees(np.arctan2(along, outward))
                candidates = sorted(np.flatnonzero(eligible), key=lambda i: (abs(float(angles[i])), int(i)))
                for candidate in candidates:
                    if usable(ordered[candidate]):
                        target, reason = ordered[candidate].copy(), "existing_contact"
                        break
                if target is None and not fixed:
                    if usable(projected):
                        target, reason = projected, "projected_contact"
                    else:
                        reason = "no_valid_projection"
            if target is None:
                self.mappings.append({**record, "status": "skipped", "reason": reason})
                continue
            seam = registry.add((source, target))
            cuts.append((index, seam))
            delta = target - source
            angle = float(np.degrees(np.arctan2(delta @ tangent, delta @ normal)))
            record.update(seam_id=seam, target_node=target_node_id(target, data),
                          target=(target * data["scale"] + data["origin"]).tolist(), angle=angle)
            self.seams.append(record)
            self.mappings.append({**record, "status": "mapped", "reason": reason})
        cuts.append((stop, right))
        return cuts, runs

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
                "dense_mapping_angle": self.data.get("dense_mapping_angle", 30.0),
                "dense_sections": self.sections, "dense_seams": self.seams,
                "dense_mappings": self.mappings}
