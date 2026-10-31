"""Direct Quad4 filling of a slot bounded by frozen chains and a free mouth.

Patterns split the empty patch into faces before any cells are built. Free
mouth intervals and chords are sampled once, with a shared parity assignment,
so neighboring faces always own exactly the same interface segments.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from ..unit.build_unit_1_1_x_y import _validate_patch
from ._geometry import area, cross, on_segment, point_location, segment_contacts
from ._limits import ElementLimitExceeded, check_element_limit
from ._patch import _four_sided_units
from ._patterns import AxisPattern, pattern_contacts
from ._types import QuadLayerError
from ._unit_plan import CoordinatePool


@dataclass
class RepairPatch:
    nodes: np.ndarray
    quads: np.ndarray
    source_ids: np.ndarray
    mouth_ids: np.ndarray
    diagnostics: dict
    polygon: np.ndarray


class _FaceBuildError(ValueError):
    def __init__(self, face_index, causes):
        super().__init__("a repair face cannot be meshed by the bounded four-chain groupings")
        self.face_index = face_index
        self.causes = tuple(dict.fromkeys(causes))[:3]


def _fail(message, *, patterns=False, **context):
    raise QuadLayerError(message, code=("unsupported_pattern_layout" if patterns
                                      else "unmeshable_slot_repair"),
                         stage="slot_repair", **context)


def _source_path(chains, tol):
    if len(chains) != 3:
        _fail("a slot repair needs three frozen source chains")
    values = []
    for chain in chains:
        chain = np.asarray(chain, dtype=float)
        if (chain.ndim != 2 or chain.shape[1] != 3 or not len(chain)
                or not np.all(np.isfinite(chain)) or np.any(chain[:, 2] != 0)):
            _fail("repair source chains must contain finite XYZ points with Z=0")
        if values and not np.array_equal(values[-1][-1], chain[0]):
            _fail("repair source chains must join at identical frozen coordinates")
        if np.any(np.linalg.norm(np.diff(chain[:, :2], axis=0), axis=1) <= tol):
            _fail("repair source contains a zero-length segment")
        values.append(chain.copy())
    source = np.vstack((values[0], *(chain[1:] for chain in values[1:])))
    if len(source) < 3 or np.linalg.norm(source[0] - source[-1]) <= tol:
        _fail("the slot has no usable mouth")
    polygon = source[:, :2]
    if abs(area(polygon)) <= tol * np.linalg.norm(np.ptp(polygon, axis=0)):
        _fail("the slot repair polygon has no usable area")
    for i, (a, b) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
        for j in range(i + 2, len(polygon)):
            if i == 0 and j == len(polygon) - 1:
                continue
            if segment_contacts(a, b, polygon[j], polygon[(j + 1) % len(polygon)], tol):
                _fail("the slot repair polygon intersects itself")
    return source


def _union_patterns(patterns, tol):
    """Coalesce collinear finite pieces before checking complete chords."""
    ordered = sorted(patterns, key=lambda p: (p.axis, p.support, p.lo, p.hi, p.ids))
    result = []
    for pattern in ordered:
        if (result and pattern.axis == result[-1].axis
                and abs(pattern.support - result[-1].support) <= tol
                and pattern.lo <= result[-1].hi + tol):
            previous = result.pop()
            result.append(AxisPattern(previous.axis, previous.support,
                                      min(previous.lo, pattern.lo), max(previous.hi, pattern.hi),
                                      tuple(sorted(set(previous.ids + pattern.ids)))))
        else:
            result.append(pattern)
    return result


def _pattern_geometry(source, patterns, tol):
    """Find complete interior chords and all mandatory mouth contacts."""
    polygon = source[:, :2]
    mouth = (polygon[-1], polygon[0])
    chords, anchors, used_ids = [], [source[-1], source[0]], set()
    for pattern in _union_patterns(patterns, tol):
        events = [(pattern.lo, pattern.point(pattern.lo)),
                  (pattern.hi, pattern.point(pattern.hi))]
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            events.extend((float(p[pattern.axis]), p) for p in pattern_contacts(pattern, a, b, tol))
        ordered = []
        for value, point in sorted(events, key=lambda item: item[0]):
            matches = np.flatnonzero(np.linalg.norm(polygon - point, axis=1) <= tol)
            if len(matches):
                point = polygon[int(matches[0])].copy()
            if not ordered or value - ordered[-1][0] > tol:
                ordered.append((value, point))
            elif len(matches):
                ordered[-1] = (value, point)
            if on_segment(point, *mouth, tol):
                anchors.append(np.r_[point, 0.0])
        pieces = []
        for (_, a), (_, b) in zip(ordered, ordered[1:]):
            if np.linalg.norm(b - a) <= tol or point_location((a + b) * .5, polygon, tol) != 1:
                continue
            if point_location(a, polygon, tol) != 0 or point_location(b, polygon, tol) != 0:
                _fail("a repair pattern terminates inside the slot", patterns=True,
                      pattern_ids=pattern.ids)
            for endpoint in (a, b):
                old = np.any(np.linalg.norm(polygon - endpoint, axis=1) <= tol)
                if not old and not on_segment(endpoint, *mouth, tol):
                    _fail("a repair pattern would split a frozen source segment", patterns=True,
                          pattern_ids=pattern.ids)
            pieces.append((np.r_[a, 0.0], np.r_[b, 0.0], pattern.ids))
        if len(pieces) > 1:
            _fail("a repair pattern must form a single complete chord", patterns=True,
                  pattern_ids=pattern.ids)
        if pieces:
            used_ids.update(pattern.ids)
            chords.extend(pieces)
    for i, (a, b, first_ids) in enumerate(chords):
        for c, d, second_ids in chords[i + 1:]:
            contacts = segment_contacts(a[:2], b[:2], c[:2], d[:2], tol)
            for contact in contacts:
                first_end = min(np.linalg.norm(contact - a[:2]), np.linalg.norm(contact - b[:2])) <= tol
                second_end = min(np.linalg.norm(contact - c[:2]), np.linalg.norm(contact - d[:2])) <= tol
                if not first_end or not second_end or len(contacts) > 1:
                    _fail("repair patterns cross, overlap, or form an interior junction", patterns=True,
                          pattern_ids=tuple(sorted(set(first_ids + second_ids))))
    return chords, anchors, tuple(sorted(used_ids))


def _face_graph(source, chords, anchors, tol):
    pool = CoordinatePool(tol)
    source_ids = pool.chain(source)
    if len(set(source_ids.tolist())) != len(source_ids):
        _fail("repair source contains repeated coordinates")
    delta = source[0, :2] - source[-1, :2]
    anchors = sorted(anchors, key=lambda point: float((point[:2] - source[-1, :2]) @ delta))
    mouth = []
    for point in anchors:
        node = pool.intern(point)
        if not mouth or mouth[-1] != node:
            mouth.append(node)
    if mouth[0] != source_ids[-1] or mouth[-1] != source_ids[0]:
        _fail("mandatory mouth anchors do not close the repair")
    faces = [source_ids.tolist() + mouth[1:-1]]
    free = [tuple(sorted((a, b))) for a, b in zip(mouth, mouth[1:])]
    pattern_edges = []
    for a, b, pattern_ids in chords:
        first, last = pool.intern(a), pool.intern(b)
        edge = tuple(sorted((first, last)))
        if edge in free:
            continue
        for index, face in enumerate(faces):
            if first not in face or last not in face:
                continue
            i, j = sorted((face.index(first), face.index(last)))
            if j == i + 1 or (i == 0 and j == len(face) - 1):
                break
            faces[index:index + 1] = [face[i:j + 1], face[j:] + face[:i + 1]]
            free.append(edge)
            pattern_edges.append((edge, pattern_ids))
            break
        else:
            _fail("repair chords do not split a single face", patterns=True,
                  pattern_ids=pattern_ids)
    return pool.array(), source_ids, mouth, faces, free, pattern_edges


def _count_candidates(points, faces, free, h):
    """At most eight shared count assignments from two parity seeds.

    Gaussian elimination over GF(2) preserves the closest seed on nonpivot
    edges. Mouth variables precede chords, keeping chord sampling close to h.
    Every base assignment also permits a two-segment increase on each free
    interface, including exact-size targets whose lower/upper counts coincide.
    """
    lookup = {edge: index for index, edge in enumerate(free)}
    rows = []
    for face in faces:
        mask, fixed = 0, 0
        for a, b in zip(face, face[1:] + face[:1]):
            index = lookup.get(tuple(sorted((a, b))))
            if index is None:
                fixed ^= 1
            else:
                mask ^= 1 << index
        rows.append((mask, fixed))
    basis = {}
    for mask, rhs in rows:
        while mask:
            pivot = (mask & -mask).bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (mask, rhs)
                break
            previous, value = basis[pivot]
            mask, rhs = mask ^ previous, rhs ^ value
        if not mask and rhs:
            _fail("repair faces have incompatible fixed boundary parity")
    lengths = [float(np.linalg.norm(points[a, :2] - points[b, :2])) for a, b in free]
    targets = [length / h for length in lengths]
    candidates = {}
    for rounder in (floor, ceil):
        parity = [max(1, rounder(target)) % 2 for target in targets]
        for pivot in sorted(basis, reverse=True):
            mask, rhs = basis[pivot]
            rest = mask & ~(1 << pivot)
            while rest:
                index = (rest & -rest).bit_length() - 1
                rhs ^= parity[index]
                rest &= rest - 1
            parity[pivot] = rhs
        choices = []
        for target, bit in zip(targets, parity):
            minimum = 1 if bit else 2
            lower = floor(target)
            lower -= (lower - bit) % 2
            upper = ceil(max(target, minimum))
            upper += (bit - upper) % 2
            counts = {upper}
            if lower >= minimum:
                counts.add(lower)
            choices.append(sorted(counts, key=lambda n: (n * (target / n - 1) ** 2, n)))
        for alternative in (0, 1):
            base = tuple(choice[min(alternative, len(choice) - 1)] for choice in choices)
            for increment in (0, 2):
                counts = tuple(count + increment for count in base)
                error = sum(n * (target / n - 1) ** 2 for target, n in zip(targets, counts))
                candidates[counts] = error
    return sorted(candidates, key=lambda counts: (candidates[counts], counts))


def _sample_edges(points, free, counts):
    sampled = {}
    for (a, b), count in zip(free, counts):
        first, last = points[a], points[b]
        chain = np.asarray([first, *(np.r_[first[:2] + (last[:2] - first[:2]) * (i / count), 0.0]
                                    for i in range(1, count)), last])
        sampled[a, b] = chain
    return sampled


def _edge_chain(a, b, points, sampled):
    key = tuple(sorted((a, b)))
    if key not in sampled:
        return points[[a, b]]
    return sampled[key] if key == (a, b) else sampled[key][::-1]


def _four_chains(boundary, tol):
    """A bounded set of groupings; every grouping retains every perimeter edge."""
    count = len(boundary)
    if count < 4 or count % 2:
        return
    vectors = np.roll(boundary[:, :2], -1, axis=0) - boundary[:, :2]
    previous = np.roll(vectors, 1, axis=0)
    turns = np.abs(cross(previous, vectors))
    corners = np.flatnonzero(turns > tol * np.maximum(np.linalg.norm(previous, axis=1),
                                                   np.linalg.norm(vectors, axis=1))).tolist()
    splits = []
    if len(corners) == 4:
        splits.append(tuple(corners))
    elif len(corners) == 3:
        gaps = [(corners[(i + 1) % 3] - corners[i]) % count for i in range(3)]
        index = int(np.argmax(gaps))
        if gaps[index] > 1:
            splits.append(tuple(sorted((*corners, (corners[index] + gaps[index] // 2) % count))))
    splits.append(tuple((i * count) // 4 for i in range(4)))
    for start in dict.fromkeys([0, *corners[:4]]):
        splits.append(tuple(sorted((start + i) % count for i in range(4))))
    seen = set()
    for split in splits:
        if len(set(split)) != 4 or split in seen:
            continue
        seen.add(split)
        first = split[0]
        rolled = np.vstack((boundary[first:], boundary[:first], boundary[first:first + 1]))
        offsets = [index - first for index in split] + [count]
        yield tuple(rolled[a:b + 1] for a, b in zip(offsets, offsets[1:]))


def _build_faces(points, source_ids, mouth, faces, free, counts, tol, limit, cache,
                 boundary_size, max_size):
    lookup = dict(zip(free, counts))
    lower_bound = sum(sum(lookup.get(tuple(sorted((a, b))), 1)
                          for a, b in zip(face, face[1:] + face[:1])) // 2 - 1 for face in faces)
    check_element_limit(lower_bound, limit, stage="slot_repair_count_preflight", count_kind="lower_bound")
    sampled = _sample_edges(points, free, counts)
    units, face_records = [], []
    total = 0
    for face_index, face in enumerate(faces):
        boundary = np.vstack([_edge_chain(a, b, points, sampled)[:-1]
                              for a, b in zip(face, face[1:] + face[:1])])
        def legacy(face_boundary, available):
            errors, rejected = [], []
            for sides in _four_chains(face_boundary, tol):
                try:
                    return _four_sided_units(sides, tol, available, _cache=cache)
                except ElementLimitExceeded as error:
                    errors.append(error)
                except ValueError as error:
                    rejected.append(str(error))
            if errors:
                raise errors[0]
            raise _FaceBuildError(face_index, rejected or (
                "face boundary needs an even number of at least four segments",))

        available = None if limit is None else limit - total
        sizing = {"ring_count": 0, "ring_target_sizes": [],
                  "stop_reason": "no_coarsening", "core_method": "direct"}
        try:
            from ._repair_adaptive import build_adaptive_face
            try:
                built, sizing = build_adaptive_face(
                    boundary, boundary_size, max_size, tol, available, cache, legacy)
            except ElementLimitExceeded:
                raise
            except ValueError:
                # Preserve the existing bounded direct strategy for faces
                # outside the inset/strip construction's geometric scope.
                built = legacy(boundary, available)
                sizing.update(stop_reason="adaptive_infeasible")
        except ElementLimitExceeded as error:
            error.context['required_elements'] += total
            error.context['limit'] = limit
            raise
        count = sum(len(unit.quads) for unit in built)
        check_element_limit(total + count, limit, stage="slot_repair_builder")
        face_records.append({"face_id": face_index, "element_start": total,
                             "element_end": total + count, "polygon": boundary.tolist(), **sizing})
        units.extend(built)
        total += count
    # Seed all frozen coordinates before importing unit boundaries; these exact
    # values and their input order take precedence over normalized reconstructions.
    pool = CoordinatePool(tol)
    original = pool.chain(points[source_ids])
    mouth_points = np.vstack([_edge_chain(a, b, points, sampled)[:-1]
                              for a, b in zip(mouth, mouth[1:])]
                             + [points[[mouth[-1]]]])
    mouth_ids = pool.chain(mouth_points)
    quads, records = [], []
    offset = 0
    for unit in units:
        mapping = pool.add_unit(unit)
        quads.append(mapping[unit.quads])
        records.append({"unit_id": len(records), "element_start": offset,
                        "element_end": offset + len(unit.quads), "method": unit.method,
                        "quality": unit.quality,
                        "polygon": unit.nodes[np.concatenate([side[:-1] for side in unit.sides])].tolist()})
        offset += len(unit.quads)
    nodes, quads = pool.array(), np.vstack(quads)
    _validate_patch(nodes, quads, (original, mouth_ids))
    if not np.array_equal(nodes[original], points[source_ids]):
        raise ValueError("repair construction changed a frozen coordinate")
    return nodes, quads, original, mouth_ids, records, face_records


def build_repair_patch(source_chains, patterns, h, tol, *, max_added_elements=None,
                       repair_max_element_size=None):
    """Fill three contiguous source chains and a straight closing mouth.

    Coordinates are normalized XYZ; source endpoints and every old segment are
    immutable. The returned mouth runs from the source end back to its start.
    Either input winding is accepted. Only convex CCW Quad4 cells are returned.
    """
    if not np.isfinite(h) or h <= 0 or not np.isfinite(tol) or tol <= 0:
        _fail("repair size and tolerance must be positive and finite")
    source = _source_path(source_chains, tol)
    lengths = np.linalg.norm(np.diff(source[:, :2], axis=0), axis=1)
    boundary_size = min(float(h), float(np.median(lengths[lengths > tol])))
    if repair_max_element_size is None:
        max_size = min(float(h), 4 * boundary_size)
    else:
        value = repair_max_element_size
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0
                or value < boundary_size - max(tol, 64 * np.finfo(float).eps * boundary_size)):
            raise QuadLayerError("repair_max_element_size must be a finite positive scalar "
                                 "at least as large as the repair boundary size",
                                 code="invalid_input", parameter="repair_max_element_size",
                                 stage="input_validation")
        max_size = min(float(h), max(boundary_size, float(value)))
    chords, anchors, pattern_ids = _pattern_geometry(source, patterns, tol)
    points, source_ids, mouth, faces, free, pattern_edges = _face_graph(source, chords, anchors, tol)
    errors, cache, failures = [], {}, []
    for counts in _count_candidates(points, faces, free, boundary_size):
        try:
            nodes, quads, source_ids_out, mouth_ids, units, face_records = _build_faces(
                points, source_ids, mouth, faces, free, counts, tol, max_added_elements, cache,
                boundary_size, max_size)
        except ElementLimitExceeded as error:
            errors.append(error)
            continue
        except ValueError as error:
            failure = {"reason": str(error), "free_edge_counts": list(counts[:16]),
                       "free_edge_count": len(counts)}
            if isinstance(error, _FaceBuildError):
                failure.update(face_index=error.face_index, causes=list(error.causes))
            if failure not in failures:
                failures.append(failure)
            continue
        return RepairPatch(nodes, quads, source_ids_out, mouth_ids,
                           {"units": units, "faces": face_records, "pattern_ids": list(pattern_ids),
                            "pattern_chord_count": len(pattern_edges),
                            "free_edge_counts": list(counts), "face_count": len(faces),
                            "boundary_element_size": boundary_size, "max_element_size": max_size},
                           nodes[np.r_[source_ids_out[:-1], mouth_ids[:-1]]].copy())
    if errors:
        raise errors[0]
    _fail("no bounded shared-count candidate can build every repair face with direct Quad4 patches",
          reason="direct_quad_candidates_exhausted", candidate_failures=failures[:8])


__all__ = ["RepairPatch", "build_repair_patch"]
