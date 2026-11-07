"""Optional dense-run boundaries, shared by preview and layer construction."""

from __future__ import annotations

import numpy as np

from .._types import QuadLayerError


class DenseLinePartition:
    def __init__(self, data):
        self.data = data
        self.sections = []
        self.seams = []
        self.targets = data.get("_dense_seam_targets")
        self.used_targets = set()

    def fail(self, message, edge, indices):
        source = self.data["ids"][edge]
        raise QuadLayerError(message, code="unmeshable_section_layout",
                             stage="section_partition", source_edge=edge,
                             source_indices=list(map(int, indices)),
                             source_nodes=[int(source[i]) for i in indices])

    def cuts(self, edge, start, stop, left, right, registry, outer):
        """Return ordered source/seam cuts and maximal dense source intervals."""
        data = self.data
        size = data.get("small_element_size")
        if size is None:
            return [(start, left), (stop, right)], []
        ids = data["ids"][edge]
        # Compare original lengths, avoiding normalization roundoff at an exact
        # threshold. The fallback supports geometry-only internal callers.
        if "_source_mesh" in data:
            points = data["_source_mesh"].nodes[ids[start:stop + 1], :2]
        else:
            points = data["xy"][ids[start:stop + 1]]
            size /= data["scale"]
        short = np.linalg.norm(np.diff(points, axis=0), axis=1) < size
        changes = np.diff(np.r_[False, short, False].astype(np.int8))
        runs = list(zip((np.flatnonzero(changes == 1) + start).tolist(),
                        (np.flatnonzero(changes == -1) + start).tolist()))
        endpoints = sorted({i for run in runs for i in run if start < i < stop})
        cuts = [(start, left)]
        tangent, tol = data["directions"][edge], data["tol"]
        axis = int(np.argmax(np.abs(tangent)))
        candidates = list(outer)
        for index in endpoints:
            source = data["xy"][ids[index]]
            projected = source.copy()
            projected[1 - axis] = outer[0, 1 - axis]
            if ((projected - outer[0]) @ tangent < -tol
                    or (outer[-1] - projected) @ tangent < -tol):
                self.fail("dense endpoint projection lies outside its LINE front", edge, [index])
            key = (edge, index)
            if self.targets is not None:
                if key not in self.targets:
                    self.fail("fixed interface is missing a planned dense seam", edge, [index])
                # The full orchestrator freezes these before global meshing.
                # Only match that exact planned contact, never a nearer new cut.
                target = self.targets[key]
                distances = np.linalg.norm(outer - target, axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] > tol:
                    self.fail("planned dense seam is absent from the fixed interface", edge, [index])
                target = outer[nearest].copy()
                self.used_targets.add(key)
            else:
                ordered = np.asarray(sorted(candidates, key=lambda p: float(p @ tangent)))
                # Compare the user's absolute tolerance in world coordinates:
                # subtracting normalized coordinates can turn an exact equality
                # into a slightly smaller distance after scaling.
                world = ordered * data["scale"] + data["origin"]
                projected_world = projected * data["scale"] + data["origin"]
                if "_source_mesh" in data:
                    projected_world[axis] = data["_source_mesh"].nodes[ids[index], axis]
                distances = np.linalg.norm(world - projected_world, axis=1)
                nearest = int(np.argmin(distances))
                target = (ordered[nearest].copy()
                          if distances[nearest] <= tol * data["scale"]
                          or distances[nearest] < data["dense_region_tolerance"]
                          else projected)
            candidates.append(target)
            seam = registry.add((source, target))
            cuts.append((index, seam))
            self.seams.append({"source_edge": edge, "source_index": index,
                               "source_node": int(ids[index]), "seam_id": seam})
        cuts.append((stop, right))
        for (a, first), (b, second) in zip(cuts, cuts[1:]):
            delta = registry.items[second].points[-1] - registry.items[first].points[-1]
            if float(delta @ tangent) <= tol:
                self.fail("dense seams create a zero-width or reversed LINE section", edge, [a, b])
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
                "dense_region_tolerance": self.data.get("dense_region_tolerance", 1e-3),
                "dense_sections": self.sections, "dense_seams": self.seams}
