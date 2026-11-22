"""Select dense-run contacts within one corner/pattern-delimited LINE."""

from __future__ import annotations

import numpy as np

from .mapping import canonical_target, dense_angle, valid_interface


def narrow_gap_endpoints(gaps, sources, targets, tangent, ratio):
    """Return both ends of each gap whose proposed tangent width is too small."""
    limited = set()
    if ratio == 0:
        return limited
    for left, right in gaps:
        source_gap = (sources[right] - sources[left]) @ tangent
        # An unmatched optional cut occupies no tangent width beyond its source.
        a = targets.get(left, sources[left])
        b = targets.get(right, sources[right])
        outer_gap = (b - a) @ tangent
        # These widths use the same normalized tangent coordinates. A large
        # world origin in the normal direction cannot affect this comparison.
        scale = max(abs(p @ tangent) for p in (sources[left], sources[right], a, b))
        roundoff = 32 * np.finfo(float).eps * scale
        if outer_gap <= 0 or outer_gap + roundoff < ratio * source_gap:
            limited.update((left, right))
    return limited


class DenseLineContacts:
    def __init__(self, data, edge, endpoints, start, stop, outer, interfaces):
        self.data = data
        self.tangent = data["directions"][edge]
        self.normal = data["normals"][edge]
        self.roles = dict(endpoints)
        self.sources = {i: data["xy"][data["ids"][edge][i]] for i in self.roles}
        self.front_origin = outer[0]
        self.outer = np.asarray(sorted((canonical_target(p, data) for p in outer),
                                       key=lambda p: float(p @ self.tangent)))
        self.bounds = {start: self.outer[0], stop: self.outer[-1]}
        self.interfaces = interfaces
        self.angle = data.get("dense_mapping_angle", 30.0)

    def select(self, gaps, ratio):
        if not gaps or not ratio or not self.angle:
            contacts, reasons = self._resolve(self.roles, ())
            return contacts, reasons, set()

        # Evaluate each facing endpoint against the same unoccupied front. A
        # conflicting snap must not hide the other end's preferred position.
        preferred = {}
        for gap in gaps:
            for index in gap:
                contacts, _ = self._resolve((index,), ())
                if index in contacts:
                    preferred[index] = contacts[index]
        limited = self.narrow_gaps(gaps, preferred, ratio)
        while True:
            contacts, reasons = self._resolve(self.roles, limited)
            # Another seam can force an alternate anchor. Recheck the actual
            # choices before registering any cuts. Each retry locks a new pair.
            newly_limited = self.narrow_gaps(gaps, contacts, ratio) - limited
            if not newly_limited:
                return contacts, reasons, limited
            limited.update(newly_limited)

    def narrow_gaps(self, gaps, targets, ratio):
        return narrow_gap_endpoints(gaps, self.sources, targets, self.tangent, ratio)

    def match_saved(self, index, target, contacts, limited):
        distances = np.linalg.norm(self.outer - target, axis=1)
        nearest = int(np.argmin(distances))
        point = self.outer[nearest]
        limit = 0. if index in limited else self.angle
        if (distances[nearest] <= self.data["tol"] and self._angle_at(index, point, limit)[0]
                and self._usable(index, point, contacts)):
            return point
        return None

    def _usable(self, index, point, contacts):
        other_seams = []
        for other, target in contacts.items():
            if index == other:
                continue
            gap = (point - target if other < index else target - point) @ self.tangent
            if gap <= self.data["tol"]:
                return False
            if other in self.sources:
                other_seams.append((self.sources[other], target))
        return valid_interface(self.sources[index], point, self.data, [*self.interfaces, *other_seams])

    def _angle_at(self, index, point, limit):
        return dense_angle(self.sources[index], point, self.tangent, self.normal, self.roles[index], limit)

    def _first_anchor(self, index, limit, contacts):
        candidates = []
        for order, point in enumerate(self.outer):
            eligible, angle = self._angle_at(index, point, limit)
            if eligible:
                candidates.append((abs(angle), order))
        for _, order in sorted(candidates):
            if self._usable(index, self.outer[order], contacts):
                return self.outer[order]
        return None

    def _representable_expansion(self, index, projected, offset, limit):
        point = projected + offset * self.tangent
        origin, scale = self.data["origin"], self.data["scale"]
        world = point * scale + origin
        represented = (world - origin) / scale
        if self._angle_at(index, represented, limit)[0]:
            return point

        # The normal depth can also shrink when restored to world coordinates.
        # Recompute the tangent bound at that depth, then round toward the
        # projection so the committed point stays inside the same angular limit.
        source = self.sources[index]
        depth = (represented - source) @ self.normal
        axis = int(np.argmax(np.abs(self.tangent)))
        along = np.copysign(depth * np.tan(np.radians(limit)), offset * self.tangent[axis])
        base = source[axis] * scale + origin[axis]
        bound = (source[axis] + along) * scale + origin[axis]
        world[axis] = np.nextafter(bound, base)
        point[axis] = (world[axis] - origin[axis]) / scale
        world[axis] = point[axis] * scale + origin[axis]
        represented = (world - origin) / scale
        return point if self._angle_at(index, represented, limit)[0] else projected

    def _free_targets(self, index, limit):
        source = self.sources[index]
        depth = (self.front_origin - source) @ self.normal
        projected = source + depth * self.normal
        projection = canonical_target(projected, self.data)
        expansion = None
        if limit:
            offset = depth * np.tan(np.radians(limit))
            if self.roles[index] == "begin":
                offset = -offset
            point = canonical_target(self._representable_expansion(index, projected, offset, limit),
                                     self.data)
            if not np.array_equal(point, projection) and self._angle_at(index, point, limit)[0]:
                expansion = point
        if not self._angle_at(index, projection, limit)[0]:
            projection = None
        return projection, expansion

    def _resolve(self, indices, limited):
        """Reserve existing nodes first, then normal projections, then expansions."""
        contacts = self.bounds.copy()
        reasons = {}
        for index in indices:
            limit = 0. if index in limited else self.angle
            point = self._first_anchor(index, limit, contacts)
            if point is None:
                reasons[index] = "no_eligible_node"
            else:
                contacts[index], reasons[index] = point, "existing_contact"
        if self.data.get("fixed_outer_boundary", False):
            return contacts, reasons

        expansions = {}
        for index in indices:
            if index in contacts:
                continue
            limit = 0. if index in limited else self.angle
            projection, expansion = self._free_targets(index, limit)
            if projection is not None and self._usable(index, projection, contacts):
                contacts[index], reasons[index] = projection, "projected_contact"
            else:
                reasons[index] = "no_valid_projection"
            if expansion is not None:
                expansions[index] = expansion
        for index, point in expansions.items():
            if self._usable(index, point, contacts):
                contacts[index], reasons[index] = point, "expanded_contact"
        return contacts, reasons
