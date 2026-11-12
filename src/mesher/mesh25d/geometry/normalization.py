"""Normalize owned source geometry once, with bounded motion and provenance.

Near coordinates are not an equivalence relation. The shared canonicalizer
rejects chained/ambiguous clusters; this boundary additionally rejects source
features that would lose their material volume or keep-out-zone.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import heapq

import numpy as np

from ...tolerance import (
    GeometryToleranceError, canonicalize_points, canonicalize_values, resolve_tolerance,
    within_distance,
)
from .koz import koz_value
from .primitives import _geometry_to_face, _geometry_to_z


def _features(container, path="root"):
    for field in ("bodies", "vias", "circuits", "bumps"):
        for index, feature in enumerate(container.get(field) or []):
            yield feature, f"{path}.{field}[{index}]", field
    for index, child in enumerate(container.get("children") or []):
        yield from _features(child, f"{path}.children[{index}]")


@dataclass(frozen=True)
class _PolygonTopology:
    nesting: frozenset[tuple[int, int]]
    # Loop and source vertex identities survive bounded coordinate motion.
    contacts: frozenset[tuple[int, int, int, int]]


def _touching_polygon_regions(polygons, nesting, contacts):
    """Assemble only shells affected by contact, including all sibling holes."""
    from shapely.geometry import Polygon

    parents = [None] * len(polygons)
    depths = [0] * len(polygons)
    areas = [polygon.area for polygon in polygons]
    for outer, inner in nesting:
        depths[inner] += 1
        parent = parents[inner]
        if parent is None or areas[outer] < areas[parent]:
            parents[inner] = outer
    children = [[] for _ in polygons]
    for inner, outer in enumerate(parents):
        if outer is not None:
            children[outer].append(inner)
    affected = set()
    for left, right, _, _ in contacts:
        for index in (left, right):
            affected.add(parents[index] if depths[index] % 2 else index)
    for shell in sorted(affected):
        holes = children[shell]
        yield [shell, *holes], Polygon(
            polygons[shell].exterior,
            [polygons[hole].exterior for hole in holes],
        )


def _polygon_loop_topology(loops, policy, source, *, normalized=False,
                           stage="geometry_normalization", allow_vertex_contacts=True):
    """Check separation/nesting and retain valid, exact shared-vertex contacts."""
    from shapely.geometry import Polygon, box
    from shapely.strtree import STRtree

    polygons = [Polygon(np.asarray(loop, dtype=float)[:, :2]) for loop in loops]
    for index, polygon in enumerate(polygons):
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise GeometryToleranceError(
                "normalization would invalidate a polygon loop" if normalized
                else "source polygon loop is invalid",
                code="feature_below_resolution" if normalized else "invalid_geometry",
                stage=stage, source=source, loop_index=index, tolerance=policy.absolute,
            )
    tree = STRtree(polygons)
    nesting = set()
    contacts = set()
    vertex_maps = {}

    def vertices(index):
        if index not in vertex_maps:
            mapping = {}
            for vertex, point in enumerate(loops[index]):
                # An optional repeated closing vertex has the first identity.
                mapping.setdefault(tuple(point[:2]), vertex)
            vertex_maps[index] = mapping
        return vertex_maps[index]

    def fail_contact(indices, gap=0.0):
        raise GeometryToleranceError(
            "polygon loops must remain separate or meet only at valid shared vertices"
            if allow_vertex_contacts else "polygon loops must remain separate at geometryTolerance",
            code="invalid_geometry" if not normalized and gap == 0
            else "feature_below_resolution",
            stage=stage, source=source, reason="polygon_loop_contact" if gap == 0
            else "polygon_loop_gap", loop_indices=indices,
            measured_distance=gap, tolerance=policy.absolute,
        )

    for index, polygon in enumerate(polygons):
        xmin, ymin, xmax, ymax = polygon.bounds
        search = box(xmin - policy.absolute, ymin - policy.absolute,
                     xmax + policy.absolute, ymax + policy.absolute)
        for other in tree.query(search):
            other = int(other)
            if other >= index:
                continue
            right = polygons[other]
            gap = float(polygon.boundary.distance(right.boundary))
            if 0 < gap <= policy.absolute:
                # A positive sub-T gap remains a real feature, not a contact.
                fail_contact([other, index], gap)
            if gap == 0 and not allow_vertex_contacts:
                fail_contact([other, index])
            if polygon.contains(right):
                nesting.add((index, other))
            elif right.contains(polygon):
                nesting.add((other, index))
            elif gap == 0 and polygon.overlaps(right):
                # Crossings can occur at vertices of both rings as well.
                fail_contact([other, index])
            if gap == 0:
                intersection = polygon.boundary.intersection(right.boundary)
                if intersection.geom_type == "Point":
                    points = [intersection]
                elif intersection.geom_type == "MultiPoint":
                    points = intersection.geoms
                else:
                    fail_contact([other, index])
                left_vertices, right_vertices = vertices(other), vertices(index)
                for point in points:
                    key = tuple(point.coords[0])
                    if key not in left_vertices or key not in right_vertices:
                        fail_contact([other, index])
                    contacts.add((other, index, left_vertices[key], right_vertices[key]))
    if contacts:
        # Pairwise valid tangencies can jointly disconnect a shell's interior.
        for indices, region in _touching_polygon_regions(polygons, nesting, contacts):
            if region.is_empty or not region.is_valid or region.area <= 0:
                fail_contact(indices)
    return _PolygonTopology(frozenset(nesting), frozenset(contacts))


def _validate_polygon_topology(original_loops, normalized_loops, policy, source,
                               *, stage="geometry_normalization", original_state=None,
                               allow_vertex_contacts=True):
    before = (_polygon_loop_topology(original_loops, policy, source, stage=stage,
                                    allow_vertex_contacts=allow_vertex_contacts)
              if original_state is None else original_state)
    after = _polygon_loop_topology(normalized_loops, policy, source,
                                   normalized=True, stage=stage,
                                   allow_vertex_contacts=allow_vertex_contacts)
    if after.nesting != before.nesting:
        raise GeometryToleranceError(
            "normalization would change polygon hole containment",
            code="feature_below_resolution", stage=stage, source=source,
            reason="polygon_containment_change", tolerance=policy.absolute,
            original_nesting=sorted(before.nesting), normalized_nesting=sorted(after.nesting),
        )
    if after.contacts != before.contacts:
        raise GeometryToleranceError(
            "normalization would change polygon vertex contacts",
            code="feature_below_resolution", stage=stage, source=source,
            reason="polygon_contact_change", tolerance=policy.absolute,
            original_contacts=sorted(before.contacts), normalized_contacts=sorted(after.contacts),
        )


def _check_surface_motion(before, after, policy, source):
    accepted = within_distance(before, after, policy)
    if np.all(accepted):
        return
    index = int(np.flatnonzero(~accepted)[0])
    raise GeometryToleranceError(
        "normalization would exceed the surface displacement budget",
        code="ambiguous_geometry", stage="geometry_normalization", source=source,
        point_index=index, original=before[index].tolist(), proposed=after[index].tolist(),
        displacement=float(np.linalg.norm(after[index] - before[index])),
        tolerance=policy.absolute,
    )


def _footprint_key(geometry):
    """Exact identity after normalization, independent of loop start/winding."""
    face = _geometry_to_face(geometry)
    kind, dim = face["type"], face["dim"]
    if kind == "POLYGON":
        loops = []
        for loop in dim:
            vertices = [tuple(point) for point in loop]
            if vertices[0] == vertices[-1]:
                vertices.pop()
            smallest = min(vertices)
            candidates = []
            for direction in (vertices, vertices[::-1]):
                for index, point in enumerate(direction):
                    if point == smallest:
                        candidates.append(tuple(direction[index:] + direction[:index]))
            loops.append(min(candidates))
        return kind, tuple(sorted(loops))
    if kind == "BOX":
        x1, y1, x2, y2 = dim
        dim = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return kind, tuple(dim)


def _validate_material_boundaries(features, originals, policy):
    groups = {}
    for index, (feature, source, _) in enumerate(features):
        geometry = feature["geometry"]
        groups.setdefault(_footprint_key(geometry), []).append((
            _geometry_to_z(geometry), _geometry_to_z(geometry, False), index,
            source, feature.get("material"), _footprint_key(originals[index]),
        ))
    for group in groups.values():
        if (len({record[4] for record in group}) < 2
                or len({record[5] for record in group}) < 2):
            continue
        active, ends = {}, []
        for record in sorted(group):
            lower, upper, index, source, material, original_key = record
            while ends and ends[0][0] <= lower:
                _, expired = heapq.heappop(ends)
                active.pop(expired)
            for other in active.values():
                if material == other[4] or original_key == other[5]:
                    continue
                raise GeometryToleranceError(
                    "normalization would merge distinct material boundaries",
                    code="feature_below_resolution", stage="geometry_normalization",
                    reason="material_boundary_conflict", source=[source, other[3]],
                    tolerance=policy.absolute,
                )
            active[index] = record
            heapq.heappush(ends, (upper, index))

    _validate_material_boundary_contacts(features, originals, policy)


def _boundary_entities(geometry):
    face = _geometry_to_face(geometry)
    kind, dim = face["type"], face["dim"]
    if kind == "CIRCLE":
        return [("circle", tuple(dim))]
    if kind == "BOX":
        x1, y1, x2, y2 = dim
        loops = [[(min(x1, x2), min(y1, y2)), (max(x1, x2), min(y1, y2)),
                  (max(x1, x2), max(y1, y2)), (min(x1, x2), max(y1, y2))]]
    else:
        loops = dim
    entities = []
    for loop in loops:
        vertices = [tuple(point) for point in loop]
        if vertices[0] == vertices[-1]:
            vertices.pop()
        entities.extend(("segment", (first, last))
                        for first, last in zip(vertices, vertices[1:] + vertices[:1]))
    return entities


def _boundary_contact(left, right):
    """Exact contact dimension: 0 absent, 1 points, 2 coincident support."""
    from shapely.geometry import LineString

    left_kind, a = left
    right_kind, b = right
    if left_kind == right_kind == "segment":
        intersection = LineString(a).intersection(LineString(b))
        return 0 if intersection.is_empty else (2 if intersection.length > 0 else 1)
    if left_kind == "segment":
        return _boundary_contact(right, left)
    x, y, radius = map(lambda value: Fraction(float(value)), a)
    if right_kind == "circle":
        xx, yy, other_radius = map(lambda value: Fraction(float(value)), b)
        squared = (x - xx) ** 2 + (y - yy) ** 2
        if squared == 0 and radius == other_radius:
            return 2
        return int((radius - other_radius) ** 2 <= squared <= (radius + other_radius) ** 2)
    first, last = [[Fraction(float(value)) for value in point] for point in b]
    delta = [last[0] - first[0], last[1] - first[1]]
    residual = [x - first[0], y - first[1]]
    length = sum(value * value for value in delta)
    start_distance = sum(value * value for value in residual)
    end_distance = (x - last[0]) ** 2 + (y - last[1]) ** 2
    position = sum(a * b for a, b in zip(residual, delta))
    if not length or position <= 0:
        minimum = start_distance
    elif position >= length:
        minimum = end_distance
    else:
        minimum = start_distance - position * position / length
    return int(minimum <= radius * radius <= max(start_distance, end_distance))


def _boundary_distance(left, right):
    """Distance between analytic boundary entities before equivalencing."""
    from shapely.geometry import LineString, Point

    left_kind, a = left
    right_kind, b = right
    if left_kind == right_kind == "segment":
        return float(LineString(a).distance(LineString(b)))
    if left_kind == "segment":
        return _boundary_distance(right, left)
    x, y, radius = a
    if right_kind == "circle":
        xx, yy, other_radius = b
        centers = float(np.hypot(x - xx, y - yy))
        return max(0.0, centers - radius - other_radius, abs(radius - other_radius) - centers)
    nearest = float(Point(x, y).distance(LineString(b)))
    farthest = max(float(np.hypot(x - px, y - py)) for px, py in b)
    return max(0.0, nearest - radius, radius - farthest)


def _validate_material_boundary_contacts(features, originals, policy):
    """Allow near interfaces to meet without losing existing shared contacts.

    A spatial index covers both original and normalized entity bounds. Only
    changed boundaries and overlapping material Z intervals are compared.
    Circles use exact analytic contacts, without polygonal approximation.
    """
    from shapely.geometry import box
    from shapely.strtree import STRtree

    def bounds(entity):
        kind, data = entity
        if kind == "circle":
            x, y, radius = data
            return x - radius, y - radius, x + radius, y + radius
        points = np.asarray(data)
        return *np.min(points, axis=0), *np.max(points, axis=0)

    entries, envelopes = [], []
    for index, ((feature, source, _), original) in enumerate(zip(features, originals)):
        geometry = feature["geometry"]
        for entity_index, (before, after) in enumerate(zip(_boundary_entities(original),
                                                          _boundary_entities(geometry))):
            previous, current = bounds(before), bounds(after)
            envelope = box(min(previous[0], current[0]), min(previous[1], current[1]),
                           max(previous[2], current[2]), max(previous[3], current[3]))
            entries.append((index, source, feature.get("material"), _geometry_to_z(geometry),
                            _geometry_to_z(geometry, False), before, after, entity_index))
            envelopes.append(envelope)
    tree = STRtree(envelopes)
    for index, left in enumerate(entries):
        if left[5] == left[6]:
            continue
        for other in tree.query(envelopes[index]):
            other = int(other)
            right = entries[other]
            if (left[0] == right[0] or left[2] == right[2]
                    or max(left[3], right[3]) >= min(left[4], right[4])
                    or (other < index and right[5] != right[6])):
                continue
            before = _boundary_contact(left[5], right[5])
            after = _boundary_contact(left[6], right[6])
            # Coordinate equivalence deliberately closes sub-tolerance gaps
            # across materials. A new point/edge contact is valid when its
            # original boundaries were within T; the whole-footprint check
            # above still prevents two distinct materials becoming one volume.
            aligned_contact = (after > before and within_distance(
                _boundary_distance(left[5], right[5]), 0.0, policy))
            if before != after and not aligned_contact:
                raise GeometryToleranceError(
                    "normalization would separate a shared material boundary or join boundaries beyond geometryTolerance",
                    code="feature_below_resolution", stage="geometry_normalization",
                    reason="material_boundary_conflict", source=[left[1], right[1]],
                    boundary_indices=[left[7], right[7]], original_contact=before,
                    normalized_contact=after, tolerance=policy.absolute,
                )


def normalize_geometry(container, tolerance=None):
    """Normalize XY only on an owned tree; retain source Z and thickness."""
    policy = resolve_tolerance(tolerance)
    features = list(_features(container))
    points, records = [], []
    originals, polygon_states = [], {}

    def fail(message, source, **context):
        raise GeometryToleranceError(message, code="feature_below_resolution",
                                     stage="geometry_normalization", source=source,
                                     tolerance=policy.absolute, **context)

    for feature, source, field in features:
        geometry = feature["geometry"]
        originals.append(deepcopy(geometry))
        kind = geometry["type"]
        # Extracting the face also validates finite XY and supported primitives.
        _geometry_to_face(geometry)
        thickness = float(geometry["thk"])
        if not np.isfinite(thickness) or thickness <= 0:
            fail("material thickness must be finite and positive", source,
                 thickness=thickness)
        base, end = _geometry_to_z(geometry), _geometry_to_z(geometry, False)
        if not np.isfinite(base) or not np.isfinite(end) or end <= base:
            raise GeometryToleranceError(
                "material Z endpoints must be finite and numerically distinct",
                code="numeric_resolution_exceeded", stage="geometry_normalization", source=source,
            )
        koz = koz_value(feature) if field != "bodies" else 0.0
        if 0 < koz <= policy.absolute:
            fail("positive KOZ width must exceed geometryTolerance", source, koz=koz)
        start = len(points)
        if kind == "BoxGeometry":
            lower = np.asarray(geometry["bottom_left"], dtype=float)
            upper = np.asarray(geometry["top_right"], dtype=float)
            if lower.shape != (3,) or upper.shape != (3,):
                fail("box coordinates must be XYZ triples", source)
            widths = np.abs(upper[:2] - lower[:2])
            if np.any(widths <= policy.absolute):
                fail("box width and height must exceed geometryTolerance", source,
                     widths=widths.tolist())
            bottom = [*np.minimum(lower[:2], upper[:2]), float(lower[2])]
            top = [*np.maximum(lower[:2], upper[:2]), float(lower[2]) + thickness]
            points.extend((bottom, top))
            records.append((kind, start, 2, None))
        elif kind == "CylinderGeometry":
            center = np.asarray(geometry["center"], dtype=float)
            radius = float(geometry["bottom_radius"])
            if center.shape != (3,) or radius <= policy.absolute:
                fail("circle radius must exceed geometryTolerance", source, radius=radius)
            points.extend((center.tolist(), [*center[:2], center[2] + thickness]))
            policy.check_resolution([[center[0] - radius, center[1] - radius],
                                     [center[0] + radius, center[1] + radius]],
                                    stage="geometry_normalization", source=source)
            records.append((kind, start, 2, None))
        else:
            loops = geometry["polys"]
            sizes = [len(loop) for loop in loops]
            vertices = np.asarray([point for loop in loops for point in loop], dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                fail("polygon coordinates must be XYZ triples", source)
            policy.check_resolution(vertices[:, :2], stage="geometry_normalization", source=source)
            polygon_states[source] = _polygon_loop_topology(loops, policy, source)
            if np.any(vertices[:, 2] != vertices[0, 2]):
                raise GeometryToleranceError(
                    "2.5D polygon vertices must lie on one Z plane; Z is not normalized",
                    code="invalid_geometry", stage="geometry_normalization", source=source,
                )
            bottom = vertices.copy()
            bottom[:, 2] = np.min(vertices[:, 2])
            points.extend(bottom.tolist())
            points.append([*bottom[0, :2], bottom[0, 2] + thickness])
            records.append((kind, start, len(vertices), sizes))

    if not points:
        return container
    normalized = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("geometry coordinates must be finite")
    policy.check_resolution(normalized[:, :2], stage="geometry_normalization", source="source vertices")
    normalized[:, :2] = canonicalize_points(normalized[:, :2], policy, stage="geometry_normalization",
                                            source=[source for _, source, _ in features])
    for (feature, source, _), original, (kind, start, count, sizes) in zip(features, originals, records):
        geometry = feature["geometry"]
        if kind == "BoxGeometry":
            bottom, top = normalized[start:start + 2]
            if np.any(top[:2] - bottom[:2] <= policy.absolute):
                fail("normalization would collapse a material box", source)
            geometry["bottom_left"] = bottom.tolist()
            geometry["top_right"] = [*top[:2], original["top_right"][2]]
            original_lower = np.minimum(original["bottom_left"][:2], original["top_right"][:2])
            original_upper = np.maximum(original["bottom_left"][:2], original["top_right"][:2])
            before = np.asarray([[x, y, z] for x in (original_lower[0], original_upper[0])
                                 for y in (original_lower[1], original_upper[1])
                                 for z in (_geometry_to_z(original), _geometry_to_z(original, False))])
            after = np.asarray([[x, y, z] for x in (bottom[0], top[0])
                                for y in (bottom[1], top[1]) for z in (bottom[2], top[2])])
            _check_surface_motion(before, after, policy, source)
        elif kind == "CylinderGeometry":
            bottom, top = normalized[start:start + 2]
            geometry["center"] = bottom.tolist()
        else:
            vertices = normalized[start:start + count]
            top = normalized[start + count]
            loops, cursor = [], 0
            for size in sizes:
                loop = vertices[cursor:cursor + size]
                cursor += size
                # Only an explicitly repeated closure is redundant. Distinct
                # source edges may not silently vanish during normalization.
                original_loop = np.asarray(geometry["polys"][len(loops)], dtype=float)
                for i in range(len(loop)):
                    j = (i + 1) % len(loop)
                    if (np.array_equal(loop[i, :2], loop[j, :2])
                            and not np.array_equal(original_loop[i, :2], original_loop[j, :2])):
                        fail("normalization would collapse a polygon edge", source,
                             loop_index=len(loops), edge_index=i)
                loops.append(loop.tolist())
            _validate_polygon_topology(original["polys"], loops, policy, source,
                                       original_state=polygon_states[source])
            geometry["polys"] = loops
            original_vertices = np.asarray([point for loop in original["polys"] for point in loop])
            before = np.concatenate((original_vertices,
                                     original_vertices + [0, 0, original["thk"]]))
            after = np.concatenate((vertices, vertices + [0, 0, geometry["thk"]]))
            _check_surface_motion(before, after, policy, source)

    circle_groups = {}
    for index, (feature, _, _) in enumerate(features):
        geometry = feature["geometry"]
        if geometry["type"] == "CylinderGeometry":
            circle_groups.setdefault(tuple(geometry["center"][:2]), []).append(index)
    for indices in circle_groups.values():
        radii = [features[index][0]["geometry"]["bottom_radius"] for index in indices]
        normalized_radii = canonicalize_values(radii, policy, stage="geometry_normalization",
                                              source=[features[i][1] for i in indices])
        for index, radius in zip(indices, normalized_radii):
            geometry = features[index][0]["geometry"]
            original = originals[index]
            radial_motion = (float(np.linalg.norm(np.asarray(geometry["center"][:2])
                                                  - original["center"][:2]))
                             + abs(float(radius) - original["bottom_radius"]))
            z_motion = max(abs(_geometry_to_z(geometry, start) - _geometry_to_z(original, start))
                           for start in (True, False))
            if not within_distance([radial_motion, z_motion], [0.0, 0.0], policy):
                fail("circle normalization would exceed the surface displacement budget",
                     features[index][1], displacement=float(np.hypot(radial_motion, z_motion)))
            geometry["bottom_radius"] = float(radius)

    # Hash exact normalized footprints, then sweep only overlapping Z intervals.
    _validate_material_boundaries(features, originals, policy)
    return container
