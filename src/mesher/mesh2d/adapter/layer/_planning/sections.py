"""Corner-first semantic sections for a single orthogonal expansion layer.

This module creates geometry only. In particular, source node counts and target
unit sizes never split a LINE section. Finite patterns are deliberately limited
to boundary portions and complete source-to-front transverse interfaces.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .._geometry import BoxIndex, area, cross, on_segment, point_location, segment_contacts
from .._patterns import AxisPattern, covered_intervals, pattern_contacts
from .._types import QuadLayerError, QuadSectionPlan, Seam, Section


def _fail(message, *, patterns=False, **context):
    context.setdefault("stage", "pattern_partition" if patterns else "section_partition")
    raise QuadLayerError(message, code=("unsupported_pattern_layout" if patterns
                                       else "unmeshable_section_layout"), **context)


def _same(a, b, tol):
    return bool(np.linalg.norm(np.asarray(a) - np.asarray(b)) <= tol)


def _clean(points, tol):
    result = []
    for point in points:
        if not result or not _same(point, result[-1], tol):
            result.append(np.asarray(point, dtype=float).copy())
    return np.asarray(result).reshape((-1, 2))


def _readonly(points):
    result = np.asarray(points, dtype=float).copy()
    result.setflags(write=False)
    return result


def _segments(points):
    return zip(points[:-1], points[1:])


def _loop_segments(points):
    return zip(points, np.roll(points, -1, axis=0))


def _band_location(point, data):
    outer, inner = data["outer"], data["inner"]
    outside, inside = point_location(point, outer, data["tol"]), point_location(point, inner, data["tol"])
    if outside < 0 or inside > 0:
        return -1
    return 0 if outside == 0 or inside == 0 else 1


def _pattern_ids(points, patterns, tol):
    found = set()
    for a, b in _segments(points):
        for pattern in patterns:
            intervals = covered_intervals(pattern, np.asarray(((a, b),)), tol)
            if any(hi - lo > tol for lo, hi in intervals):
                found.update(pattern.ids)
    return tuple(sorted(found))


def _with_anchors(points, anchors, tol):
    points = _clean(points, tol)
    if len(points) < 2:
        return points
    result = [points[0]]
    for a, b in _segments(points):
        delta = b - a
        length = float(np.linalg.norm(delta))
        interior = [point for point in anchors if on_segment(point, a, b, tol)
                    and not _same(point, a, tol) and not _same(point, b, tol)]
        interior.sort(key=lambda point: float((point - a) @ delta) / length)
        result.extend(interior)
        result.append(b)
    return _clean(result, tol)


@dataclass
class _Chord:
    edge: int
    index: int
    points: np.ndarray
    pattern_ids: tuple[int, ...]


def _union_patterns(patterns, tol):
    """Merge finite collinear intervals, retaining their original identities."""
    records = []
    for pattern in patterns:
        records.append((pattern.axis, pattern.support, pattern.lo, pattern.hi, pattern.ids))
    records.sort()
    groups = []
    for axis, support, low, high, indices in records:
        if groups and axis == groups[-1][0] and abs(support - groups[-1][1]) <= tol:
            groups[-1][2].append((low, high, indices))
        else:
            groups.append((axis, support, [(low, high, indices)]))
    result = []
    for axis, support, intervals in groups:
        merged = []
        for low, high, indices in sorted(intervals):
            if merged and low <= merged[-1][1] + tol:
                merged[-1][1] = max(high, merged[-1][1])
                merged[-1][2].update(indices)
            else:
                merged.append([low, high, set(indices)])
        for low, high, indices in merged:
            result.append(AxisPattern(axis, support, low, high, tuple(sorted(indices))))
    return result


def _admit_patterns(data):
    """Return complete local transverse chords and mandatory pattern anchors."""
    tol, patterns = data["tol"], data["pattern_records"]
    source, front, ids, chains = data["source"], data["front"], data["ids"], data["chains"]
    boundaries = list(_loop_segments(source)) + list(_loop_segments(front))
    boundary_points = np.asarray(boundaries)
    boundary_index = BoxIndex(boundary_points.min(axis=1), boundary_points.max(axis=1), tol)
    chords, anchors = [], list(data["patterns"].reshape((-1, 2)))
    for pattern in _union_patterns(patterns, tol):
        pattern_ids = pattern.ids
        events = [(pattern.lo, pattern.point(pattern.lo), 0),
                  (pattern.hi, pattern.point(pattern.hi), 0)]
        for j in boundary_index.query(*pattern.bounds()):
            for contact in pattern_contacts(pattern, *boundaries[j], tol):
                # Keep exact boundary coordinates, preferring frozen source
                # contacts when a finite endpoint represents the same point.
                priority = 2 if j < len(source) else 1
                events.append((float(contact[pattern.axis]), contact, priority))
                anchors.append(contact)
        ordered = []
        for event in sorted(events, key=lambda event: event[0]):
            if not ordered or event[0] - ordered[-1][0] > tol:
                ordered.append(event)
            elif event[2] > ordered[-1][2]:
                ordered[-1] = event
        for first, last in zip(ordered, ordered[1:]):
            low, high = first[0], last[0]
            if (high - low <= tol
                    or _band_location(pattern.point(low + (high - low) * .5), data) != 1):
                continue
            endpoints = np.asarray((first[1], last[1])).copy()
            source_ends = [point_location(point, source, tol) == 0 for point in endpoints]
            front_ends = [point_location(point, front, tol) == 0 for point in endpoints]
            if source_ends == [False, True] and front_ends == [True, False]:
                endpoints = endpoints[::-1].copy()
            elif source_ends != [True, False] or front_ends != [False, True]:
                _fail("finite patterns must form complete source-to-front chords; interior endpoints and loops are unsupported",
                      patterns=True, pattern_ids=pattern_ids)
            candidates = []
            for edge, chain in enumerate(chains):
                matches = np.flatnonzero(np.linalg.norm(chain - endpoints[0], axis=1) <= tol)
                for node_index in matches:
                    normal = data["normals"][edge]
                    chord_delta = endpoints[1] - endpoints[0]
                    if (abs(float(cross(normal, chord_delta))) <= tol
                            and float(chord_delta @ normal) > tol
                            and on_segment(endpoints[1], front[edge], front[(edge + 1) % len(front)], tol)):
                        candidates.append((edge, int(node_index)))
            if not candidates:
                _fail("a pattern must meet an existing source node and that edge's own front segment",
                      patterns=True, pattern_ids=pattern_ids)
            edge, node_index = min(candidates)
            # Use the exact frozen coordinate, rather than a recomputed intersection.
            endpoints[0] = chains[edge][node_index]
            duplicate = next((chord for chord in chords if _same(chord.points[0], endpoints[0], tol)
                              and _same(chord.points[1], endpoints[1], tol)), None)
            if duplicate is not None:
                duplicate.pattern_ids = tuple(sorted(set(duplicate.pattern_ids) | set(pattern_ids)))
            else:
                chords.append(_Chord(edge, node_index, endpoints, pattern_ids))
    for index, first in enumerate(chords):
        for second in chords[index + 1:]:
            contacts = segment_contacts(*first.points, *second.points, tol)
            if any(_band_location(point, data) == 1 for point in contacts):
                _fail("interior pattern intersections and T junctions are unsupported", patterns=True,
                      pattern_ids=tuple(sorted(set(first.pattern_ids) | set(second.pattern_ids))))
    return chords, np.asarray(anchors).reshape((-1, 2))


@dataclass
class _Corner:
    kind: str
    source_chains: tuple[tuple[int, ...], ...]
    outer: np.ndarray
    left: int
    right: int
    incoming_index: int
    outgoing_index: int


class _Seams:
    def __init__(self, data, anchors):
        self.data, self.anchors, self.items = data, anchors, []

    def add(self, points):
        points = _with_anchors(points, self.anchors, self.data["tol"])
        if len(points) < 2:
            _fail("a section interface has zero length")
        for seam in self.items:
            if _same(points[0], seam.points[0], self.data["tol"]) and _same(points[-1], seam.points[-1], self.data["tol"]):
                return seam.id
        seam_id = len(self.items)
        self.items.append(Seam(seam_id, _readonly(points), (),
                               _pattern_ids(points, self.data["pattern_records"], self.data["tol"])))
        return seam_id


def _nearest_anchor(chain, corner_at_end, target, tol):
    indices = list(range(len(chain) - 1)) if corner_at_end else list(range(1, len(chain)))
    distances = np.linalg.norm(chain[indices] - target, axis=1)
    minimum = float(np.min(distances))
    tied = [index for index, distance in zip(indices, distances) if distance <= minimum + tol]
    return max(tied) if corner_at_end else min(tied)


def _make_corners(data, chords, seams):
    chains, ids, front, normals = data["chains"], data["ids"], data["front"], data["normals"]
    tol, count = data["tol"], len(chains)
    corners = []
    for index in range(count):
        previous = (index - 1) % count
        source_corner = chains[index][0]
        target = front[index]
        turn = float(cross(data["directions"][previous], data["directions"][index]))
        if turn < 0:
            incoming = source_corner + normals[previous] * float((target - source_corner) @ normals[previous])
            outgoing = source_corner + normals[index] * float((target - source_corner) @ normals[index])
            corners.append(_Corner("OUTER", ((int(ids[index][0]),),),
                                   _clean((incoming, target, outgoing), tol),
                                   seams.add((source_corner, incoming)), seams.add((source_corner, outgoing)),
                                   len(ids[previous]) - 1, 0))
            continue
        left_index = _nearest_anchor(chains[previous], True, target, tol)
        right_index = _nearest_anchor(chains[index], False, target, tol)
        left_patterns = [chord for chord in chords if chord.edge == previous
                         and left_index <= chord.index < len(ids[previous]) - 1]
        right_patterns = [chord for chord in chords if chord.edge == index
                          and 0 < chord.index <= right_index]
        left_pattern = max(left_patterns, key=lambda chord: chord.index, default=None)
        right_pattern = min(right_patterns, key=lambda chord: chord.index, default=None)
        left_outer, right_outer = target.copy(), target.copy()
        if left_pattern is not None:
            left_index, left_outer = left_pattern.index, left_pattern.points[1].copy()
        if right_pattern is not None:
            right_index, right_outer = right_pattern.index, right_pattern.points[1].copy()
        # Front arms must follow source traversal and stay on their actual fronts.
        if (float((target - left_outer) @ data["directions"][previous]) < -tol
                or float((right_outer - target) @ data["directions"][index]) < -tol):
            _fail("a forced corner interface reverses the outer boundary", patterns=True, corner=index)
        left_arm, right_arm = not _same(left_outer, target, tol), not _same(right_outer, target, tol)
        kind = "INNER_BOTH" if left_arm and right_arm else "INNER_LEFT" if left_arm else "INNER_RIGHT" if right_arm else "INNER"
        left = seams.add((chains[previous][left_index], left_outer))
        right = seams.add((chains[index][right_index], right_outer))
        corners.append(_Corner(kind, (tuple(map(int, ids[previous][left_index:])),
                                       tuple(map(int, ids[index][:right_index + 1]))),
                               _clean((left_outer, target, right_outer), tol), left, right, left_index, right_index))
    return corners


def _share_repaired_endpoint_seams(data, corners, seams):
    """Join endpoint reservations on a staged source without growing it again.

    A repair can leave an INNER reservation consuming a complete neighboring
    edge. Its OUTER neighbor's normal projection may then lie beyond that
    edge's actual front. Clip that projection to the front and let both corners
    own the same interface, retaining every frozen source segment.
    """
    if "_original_data" not in data:
        return
    tol = data["tol"]
    for edge, first in enumerate(corners):
        following = corners[(edge + 1) % len(corners)]
        if (first.outgoing_index != following.incoming_index
                or first.right == following.left
                or (first.kind == "OUTER") == (following.kind == "OUTER")):
            continue
        before, after = seams.items[first.right], seams.items[following.left]
        outer = before if first.kind == "OUTER" else after
        a, b = data["front"][[edge, (edge + 1) % len(corners)]]
        vector = b - a
        fraction = float((outer.points[-1] - a) @ vector / (vector @ vector))
        target = a if fraction <= 0. else b if fraction >= 1. else outer.points[-1]
        for seam in (before, after):
            if seam.pattern_ids and not _same(seam.points[-1], target, tol):
                _fail("a repaired endpoint reservation would move a pattern interface",
                      patterns=True, source_edge=edge, pattern_ids=seam.pattern_ids)
        source = data["chains"][edge][first.outgoing_index]
        shared = seams.add((source, target))
        first.right = following.left = shared
        for index, corner, left, right in (
                (edge, first, first.outer[0], target),
                ((edge + 1) % len(corners), following, target, following.outer[-1])):
            front = data["front"][index]
            corner.outer = _clean((left, front, right), tol)
            if corner.kind != "OUTER":
                left_arm, right_arm = not _same(left, front, tol), not _same(right, front, tol)
                corner.kind = ("INNER_BOTH" if left_arm and right_arm else
                               "INNER_LEFT" if left_arm else "INNER_RIGHT" if right_arm else "INNER")
    used = {seam for corner in corners for seam in (corner.left, corner.right)}
    remap = {old: new for new, old in enumerate(sorted(used))}
    seams.items = [replace(seams.items[old], id=new) for old, new in remap.items()]
    for corner in corners:
        corner.left, corner.right = remap[corner.left], remap[corner.right]


def _source_points(section, data):
    node_ids = []
    for chain in section.source_chains:
        node_ids.extend(chain[1:] if node_ids and node_ids[-1] == chain[0] else chain)
    return data["xy"][node_ids]


def section_polygon(section, seams, xy, tol=0.):
    """Return the section's directed polygon, retaining frozen source nodes."""
    node_ids = []
    for chain in section.source_chains:
        node_ids.extend(chain[1:] if node_ids and node_ids[-1] == chain[0] else chain)
    source = np.asarray(xy)[node_ids]
    points = _clean([*source, *seams[section.right_seam].points[1:],
                     *section.outer_boundary[::-1][1:], *seams[section.left_seam].points[::-1][1:]], tol)
    if len(points) > 1 and _same(points[0], points[-1], tol):
        points = points[:-1]
    return points


def _validate_sections(sections, seams, data):
    tol = data["tol"]
    expected_source = Counter(tuple(map(int, pair)) for chain in data["ids"] for pair in zip(chain[:-1], chain[1:]))
    represented = Counter(pair for section in sections for chain in section.source_chains for pair in zip(chain[:-1], chain[1:]))
    if represented != expected_source:
        _fail("sections must cover each frozen source segment exactly once")
    for index, section in enumerate(sections):
        if section.right_seam != sections[(index + 1) % len(sections)].left_seam:
            _fail("section interfaces do not form a closed cyclic ring", section=index)
    # All interfaces lie strictly in the open layer except at their endpoints.
    boundary_segments = list(_loop_segments(data["source"])) + list(_loop_segments(data["front"]))
    for seam in seams:
        a, b = seam.points[0], seam.points[-1]
        if _band_location((a + b) / 2., data) != 1:
            _fail("a section interface leaves the open layer", seam=seam.id)
        for c, d in boundary_segments:
            for point in segment_contacts(a, b, c, d, tol):
                if not _same(point, a, tol) and not _same(point, b, tol):
                    _fail("a section interface crosses a source or front boundary", seam=seam.id)
    for index, first in enumerate(seams):
        for second in seams[index + 1:]:
            contacts = segment_contacts(first.points[0], first.points[-1], second.points[0], second.points[-1], tol)
            for point in contacts:
                if (not any(_same(point, p, tol) for p in (first.points[0], first.points[-1]))
                        or not any(_same(point, p, tol) for p in (second.points[0], second.points[-1]))):
                    _fail("section interfaces cross or partially overlap", seams=(first.id, second.id))
    total = 0.
    for section in sections:
        polygon = section_polygon(section, seams, data["xy"], tol)
        if len(polygon) < 3:
            _fail("a section has fewer than three distinct boundary points", section=section.id)
        lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        signed = area(polygon)
        if signed <= tol * float(lengths.sum()):
            _fail("a section has zero or reversed area", section=section.id)
        lower, upper = np.minimum(polygon, np.roll(polygon, -1, axis=0)), np.maximum(polygon, np.roll(polygon, -1, axis=0))
        index = BoxIndex(lower, upper, tol)
        for first, (a, b) in enumerate(_loop_segments(polygon)):
            for second in index.query(np.minimum(a, b), np.maximum(a, b)):
                if second <= first or second == first + 1 or (first == 0 and second == len(polygon) - 1):
                    continue
                if segment_contacts(a, b, polygon[second], polygon[(second + 1) % len(polygon)], tol):
                    _fail("a section boundary intersects itself", section=section.id)
        total += signed
    expected = area(data["source"]) - area(data["front"])
    perimeter = sum(float(np.linalg.norm(b - a)) for a, b in boundary_segments)
    if abs(total - expected) > max(64 * tol * perimeter, abs(expected) * 1e-10):
        _fail("sections do not cover the requested expansion layer", expected_area=expected, section_area=total)


def partition_sections(data) -> QuadSectionPlan:
    """Partition normalized prepared geometry without creating or moving nodes."""
    chords, anchors = _admit_patterns(data)
    if "outline_anchors" in data:
        anchors = np.vstack((data["outline_anchors"], anchors))
    registry = _Seams(data, anchors)
    corners = _make_corners(data, chords, registry)
    _share_repaired_endpoint_seams(data, corners, registry)
    sections = []

    def append(kind, source_chains, outer, left, right, *, source_edge=None, corner=None):
        outer = _with_anchors(outer, anchors, data["tol"])
        section = Section(len(sections), kind, source_chains, _readonly(outer), left, right,
                          source_edge, corner)
        pattern_ids = set(registry.items[left].pattern_ids) | set(registry.items[right].pattern_ids)
        pattern_ids.update(_pattern_ids(outer, data["pattern_records"], data["tol"]))
        pattern_ids.update(_pattern_ids(_source_points(section, data), data["pattern_records"], data["tol"]))
        sections.append(replace(section, pattern_ids=tuple(sorted(pattern_ids))))

    for edge, corner in enumerate(corners):
        append(corner.kind, corner.source_chains, corner.outer, corner.left, corner.right, corner=edge)
        following = corners[(edge + 1) % len(corners)]
        begin, end = corner.outgoing_index, following.incoming_index
        left, right = corner.right, following.left
        if begin > end:
            _fail("neighboring corner reservations overlap", source_edge=edge, begin=begin, end=end)
        if begin == end:
            if left != right:
                _fail("touching source reservations leave a wedge between different interfaces", source_edge=edge)
            continue
        source = data["ids"][edge]
        local = sorted((chord for chord in chords if chord.edge == edge and begin < chord.index < end),
                       key=lambda chord: chord.index)
        boundaries = [(begin, left)] + [(chord.index, registry.add(chord.points)) for chord in local] + [(end, right)]
        for (start, left_seam), (stop, right_seam) in zip(boundaries, boundaries[1:]):
            if start == stop:
                if left_seam != right_seam:
                    _fail("two patterns require different interfaces at one source node", patterns=True, source_edge=edge)
                continue
            outer = (registry.items[left_seam].points[-1], registry.items[right_seam].points[-1])
            if float((outer[1] - outer[0]) @ data["directions"][edge]) < -data["tol"]:
                _fail("a LINE section reverses its outer boundary", source_edge=edge)
            append("LINE", (tuple(map(int, source[start:stop + 1])),), outer, left_seam, right_seam, source_edge=edge)
    owners = [[] for _ in registry.items]
    for section in sections:
        owners[section.left_seam].append(section.id)
        owners[section.right_seam].append(section.id)
    seams = tuple(replace(seam, owners=tuple(owners[seam.id])) for seam in registry.items)
    if any(len(seam.owners) != 2 or len(set(seam.owners)) != 2 for seam in seams):
        _fail("each section interface must have exactly two different owners")
    for chord in chords:
        if not any(_same(chord.points[0], seam.points[0], data["tol"])
                   and _same(chord.points[-1], seam.points[-1], data["tol"]) for seam in seams):
            _fail("a finite pattern crosses a reserved corner section", patterns=True,
                  pattern_ids=chord.pattern_ids, source_edge=chord.edge)
    _validate_sections(sections, seams, data)
    return QuadSectionPlan(tuple(sections), seams, _readonly(data["front"]),
                           tuple(map(float, data["expansion"])),
                           {"path": "corner_first", "sections": dict(Counter(section.kind for section in sections)),
                            "section_count": len(sections), "seam_count": len(seams),
                            "pattern_chord_count": len(chords)})
