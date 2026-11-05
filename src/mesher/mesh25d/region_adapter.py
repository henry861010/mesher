"""Experimental local XY refinement using the public Quad4 adapter APIs.

Geometry preparation belongs here, at the optional process-flow boundary;
the reusable mesh2d algorithms remain NumPy-only. All meshes are private until
the complete planar construction succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..mesh2d.adapter import (
    QuadLayerError, generate_full_adaptive_mesh, generate_layer_adaptive_mesh,
)
from ..mesh2d.editor import EditorMesh2D
from ..mesh2d.generators import generate_rectilinear_mesh_optimal
from ..mesh2d.model import Mesh2D
from .controls import ResolvedControls
from .domain import face_bounds
from .progress import ProgressCallback, _emit_item_completed

Point = tuple[float, float]
Loop = tuple[Point, ...]
Segment = tuple[Point, Point]
GROWTH_FACTOR = 5.0


@dataclass(frozen=True)
class _RegionDomain:
    outer: Loop
    holes: tuple[Loop, ...]


@dataclass(frozen=True)
class AdapterRegion:
    source: str
    seed_size: float
    transition_sizes: tuple[float, ...]
    domains: tuple[_RegionDomain, ...]
    geometry: Any


@dataclass(frozen=True)
class AdapterPlan:
    regions: tuple[AdapterRegion, ...]
    outer_outline: Loop
    pattern_lines: tuple[Segment, ...]
    element_size: float
    coordinate_tolerance: float

    def region_summary(self) -> list[dict[str, Any]]:
        return [
            {"source": region.source, "seedSize": region.seed_size,
             "transitionSizes": list(region.transition_sizes)}
            for region in self.regions
        ]

    def intersects_annulus(self, center: Point, inner_radius: float, outer_radius: float) -> bool:
        """Whether rebuilding a circular annulus could erase a fine seed."""
        from shapely.geometry import Point as ShapelyPoint, Polygon

        origin = ShapelyPoint(center)
        for region in self.regions:
            for part in region.domains:
                if Polygon(part.outer, part.holes).distance(origin) >= outer_radius:
                    continue
                # A connected polygon's radial range is continuous, and its
                # maximum occurs at an exterior vertex. Check each component
                # separately so disjoint inner/outer parts do not bridge a gap.
                if np.any(np.linalg.norm(np.asarray(part.outer) - center, axis=1) > inner_radius):
                    return True
        return False


def _fail(message: str, *, sources=(), stage="region_planning", **context):
    raise _annotate(
        QuadLayerError(message, code="unsupported_region", stage=stage, **context),
        sources,
    )


def _annotate(error: QuadLayerError, sources, *, stage=None, **context) -> QuadLayerError:
    """Keep typed adapter errors, including their original code and context."""
    if stage is not None:
        error.context.setdefault("stage", stage)
    for key, value in context.items():
        error.context.setdefault(key, value)
    message = getattr(error, "_region_message", str(error))
    error._region_message = message
    error.context.setdefault("region_sources", list(dict.fromkeys(sources)))
    label = ", ".join(error.context["region_sources"]) or "geometry"
    error.args = (
        f"3D_REGION adapter ({label}): {message}; "
        f"code={error.code}, context={error.context}",
    )
    return error


def _run(operation: Callable, *, sources, stage, **context):
    try:
        return operation()
    except QuadLayerError as error:
        raise _annotate(error, sources, stage=stage, **context)
    except (ValueError, TypeError, OverflowError) as error:
        raise _annotate(
            QuadLayerError(str(error), code="region_meshing_failed"),
            sources, stage=stage, **context,
        ) from error


def _rectangle(bounds) -> Loop:
    left, bottom, right, top = bounds
    return ((left, bottom), (right, bottom), (right, top), (left, top))


def _orthogonal_loop(points, *, sources) -> Loop:
    try:
        values = np.asarray(points, dtype=float)
    except (ValueError, TypeError, OverflowError):
        _fail("polygon coordinates must be finite XY pairs", sources=sources)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 3
            or not np.all(np.isfinite(values))):
        _fail("polygon coordinates must be finite XY pairs", sources=sources)
    # Optional closure and redundant consecutive vertices do not change a loop.
    loop = []
    for point in values:
        pair = tuple(map(float, point))
        if not loop or pair != loop[-1]:
            loop.append(pair)
    if len(loop) > 1 and loop[0] == loop[-1]:
        loop.pop()
    if len(loop) < 4:
        _fail("an orthogonal polygon requires at least four corners", sources=sources)
    for first, last in zip(loop, [*loop[1:], loop[0]]):
        if (first[0] == last[0]) == (first[1] == last[1]):
            _fail("polygon edges must be exactly axis aligned", sources=sources)
    return tuple(loop)


def _parents(polygons, tolerance, *, sources, stage):
    for index, polygon in enumerate(polygons):
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            _fail("boundaries must be simple polygons with positive area",
                  sources=sources, stage=stage, boundary_index=index)
        for other_index in range(index):
            if polygon.boundary.distance(polygons[other_index].boundary) <= tolerance:
                _fail("boundaries intersect, touch, or lie within coordinate_tolerance",
                      sources=sources, stage=stage,
                      boundary_indices=(other_index, index))
    return tuple(
        min((other for other, candidate in enumerate(polygons)
             if other != index and candidate.contains(polygon)),
            key=lambda other: polygons[other].area, default=None)
        for index, polygon in enumerate(polygons)
    )


def _shape(polygons):
    from shapely.geometry import Polygon

    shape = Polygon()
    for polygon in polygons:
        shape = shape.symmetric_difference(polygon)
    return shape


def _canonical_ring(coords) -> Loop:
    points = list(map(tuple, list(coords)[:-1]))
    start = min(range(len(points)), key=points.__getitem__)
    return tuple(points[start:] + points[:start])


def _domains(shape) -> tuple[_RegionDomain, ...]:
    from shapely.geometry.polygon import orient

    polygons = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
    polygons.sort(key=lambda polygon: (*polygon.bounds, -polygon.area))
    domains = []
    for polygon in polygons:
        polygon = orient(polygon, sign=1.0)
        domains.append(_RegionDomain(
            _canonical_ring(polygon.exterior.coords),
            tuple(sorted(_canonical_ring(ring.coords) for ring in polygon.interiors)),
        ))
    return tuple(domains)


def _transition_sizes(seed_size: float, global_size: float) -> tuple[float, ...]:
    sizes = []
    current = seed_size
    # Divide the finite cap before multiplying, avoiding overflow at large sizes.
    while current < global_size / GROWTH_FACTOR:
        current *= GROWTH_FACTOR
        sizes.append(current)
    return (*sizes, global_size)


def _check_separation(shapes, sources, domain, tolerance, *, stage):
    for index, shape in enumerate(shapes):
        if not domain.contains(shape) or shape.distance(domain.boundary) <= tolerance:
            _fail("region or transition must lie strictly inside the meshing domain; "
                  "boundary contact and insufficient transition clearance are unsupported",
                  sources=[sources[index]], stage=stage)
        for other in range(index):
            if shape.distance(shapes[other]) <= tolerance:
                _fail("regions or transitions overlap, touch, or have insufficient clearance",
                      sources=[sources[other], sources[index]], stage=stage)


def plan_region_adapter(
    controls: ResolvedControls,
    *,
    faces: list[dict[str, Any]],
    control_bounds: tuple[float, float, float, float],
    outer_bounds: tuple[float, float, float, float],
    pattern_lines: list[Segment],
    element_size: float,
    coordinate_tolerance: float,
) -> AdapterPlan | None:
    """Resolve supported regions without generating meshes or changing inputs."""
    from shapely.geometry import Polygon, box

    domain = box(*control_bounds)
    regions = []
    for index, control in enumerate(controls.xy):
        source = (controls.xy_sources[index] if index < len(controls.xy_sources)
                  else f"controls[{index}]")
        face = control["region"]
        bounds = face_bounds(face)
        if box(*bounds).disjoint(domain):
            continue
        raw_loops = [_rectangle(bounds)] if face["type"] == "BOX" else face["dim"]
        loops = [_orthogonal_loop(loop, sources=[source]) for loop in raw_loops]
        polygons = [Polygon(loop) for loop in loops]
        _parents(polygons, coordinate_tolerance, sources=[source], stage="region_planning")
        shape = _shape(polygons)
        if shape.disjoint(domain):
            continue
        h = min(control["element_size"], element_size)
        regions.append(AdapterRegion(
            source, h, _transition_sizes(h, element_size), _domains(shape), shape,
        ))
    if not regions:
        return None

    sources = [region.source for region in regions]
    _check_separation([region.geometry for region in regions], sources, domain,
                      coordinate_tolerance, stage="region_planning")
    # The circular guide collector skips oblique edges; never silently drop a
    # material/KOZ constraint in this experimental branch.
    for face_index, face in enumerate(faces):
        if face["type"] == "POLYGON":
            for loop in face["dim"]:
                _orthogonal_loop(loop, sources=[*sources, f"geometry face[{face_index}]"])

    return AdapterPlan(tuple(regions), _rectangle(outer_bounds), tuple(pattern_lines),
                       element_size, coordinate_tolerance)


def _flat(outline) -> list[int]:
    return [int(node) for edge in outline for node in edge[:-1]]


def _edge_cycle(outline):
    ids = _flat(outline)
    return frozenset(tuple(sorted((a, b))) for a, b in zip(ids, [*ids[1:], ids[0]]))


def _mesh_state(mesh, outlines, *, sources, tolerance, parents=None):
    from shapely.geometry import Polygon

    actual = EditorMesh2D.outlines_by_edge(mesh)
    if ({_edge_cycle(outline) for outline in actual}
            != {_edge_cycle(outline) for outline in outlines}):
        _fail("a transition joined, removed, or duplicated a boundary",
              sources=sources, stage="intermediate_validation")
    polygons = [Polygon(mesh.nodes[_flat(outline), :2]) for outline in outlines]
    current_parents = _parents(polygons, tolerance, sources=sources,
                               stage="intermediate_validation")
    if parents is not None and parents != current_parents:
        _fail("a transition changed boundary containment", sources=sources,
              stage="intermediate_validation")
    return _shape(polygons), current_parents


def _merge(meshes: list[Mesh2D]) -> tuple[Mesh2D, list[int]]:
    ends = np.cumsum([mesh.node_count for mesh in meshes], dtype=np.int64)
    if ends[-1] - 1 > np.iinfo(np.int32).max:
        raise QuadLayerError("combined node IDs exceed the int32 range",
                             code="numeric_range_exceeded", stage="seed_assembly")
    starts = [0, *ends[:-1]]
    nodes = np.vstack([mesh.nodes for mesh in meshes])
    elements = np.vstack([mesh.elements.astype(np.int64) + start
                          for mesh, start in zip(meshes, starts)])
    return Mesh2D(nodes, elements), ends.tolist()


def build_region_adapter(plan: AdapterPlan, *, progress: ProgressCallback | None = None) -> Mesh2D:
    from shapely.geometry import Polygon

    meshes, outlines, shapes, parents = [], [], [], []
    sources = [region.source for region in plan.regions]
    domain = Polygon(plan.outer_outline)
    for region in plan.regions:
        def seed():
            parts = [generate_rectilinear_mesh_optimal(
                region.seed_size, part.outer, holes=part.holes,
                pattern_lines=plan.pattern_lines,
                coordinate_tolerance=plan.coordinate_tolerance,
            ).mesh for part in region.domains]
            return _merge(parts)[0]

        mesh = _run(seed, sources=[region.source], stage="seed_meshing")
        boundary = EditorMesh2D.outlines_by_edge(mesh)
        shape, parent = _mesh_state(mesh, boundary, sources=[region.source],
                                    tolerance=plan.coordinate_tolerance)
        meshes.append(mesh)
        outlines.append(boundary)
        shapes.append(shape)
        parents.append(parent)

    for region_index, region in enumerate(plan.regions):
        mesh = meshes[region_index]
        for layer_index, size in enumerate(region.transition_sizes[:-1]):
            next_outlines = []
            for boundary_index, outline in enumerate(outlines[region_index]):
                result = _run(
                    lambda: generate_layer_adaptive_mesh(
                        mesh, outline, size, pattern_lines=plan.pattern_lines,
                    ),
                    sources=[region.source], stage="intermediate_layer",
                    region_layer_index=layer_index, boundary_index=boundary_index,
                )
                next_outlines.append(result.outline)
            shapes[region_index], _ = _run(
                lambda: _mesh_state(
                    mesh, next_outlines, sources=[region.source],
                    tolerance=plan.coordinate_tolerance, parents=parents[region_index],
                ), sources=[region.source], stage="intermediate_validation",
                region_layer_index=layer_index,
            )
            _run(
                lambda: _check_separation(shapes, sources, domain, plan.coordinate_tolerance,
                                          stage="intermediate_validation"),
                sources=[region.source], stage="intermediate_validation",
                region_layer_index=layer_index,
            )
            outlines[region_index] = next_outlines
            _emit_item_completed(progress, stage="building_2d_mesh", data={
                "operation": "adapter_layer", "sourceRefs": [region.source],
                "regionLayerIndex": layer_index, "targetSize": size,
            })

    combined, ends = _run(lambda: _merge(meshes), sources=sources, stage="seed_assembly")
    combined_outlines = EditorMesh2D.outlines_by_edge(combined)
    boundary_regions = [int(np.searchsorted(ends, _flat(outline)[0], side="right"))
                        for outline in combined_outlines]
    try:
        result = generate_full_adaptive_mesh(
            combined, [plan.element_size], outer_outline=plan.outer_outline,
            pattern_lines=plan.pattern_lines, coordinate_tolerance=plan.coordinate_tolerance,
        )
    except QuadLayerError as error:
        indices = list(error.context.get("boundary_indices", ()))
        if "boundary_index" in error.context:
            indices.append(error.context["boundary_index"])
        affected = sorted({boundary_regions[index] for index in indices
                           if isinstance(index, int) and 0 <= index < len(boundary_regions)})
        if not affected:
            affected = list(range(len(plan.regions)))
        raise _annotate(error, [sources[index] for index in affected],
                        region_layer_indices={sources[index]: len(plan.regions[index].transition_sizes) - 1
                                              for index in affected})
    _emit_item_completed(progress, stage="building_2d_mesh", data={
        "operation": "adapter_completion", "sourceRefs": sources,
        "targetSize": plan.element_size,
        "node2DCount": result.mesh.node_count, "element2DCount": result.mesh.element_count,
    })
    return result.mesh
