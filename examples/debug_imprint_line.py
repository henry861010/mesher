"""Connect the imprinted cow mesh to the surrounding non-cow grid.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/debug_imprint_line.py

Fillet preparation uses the optional process-flow dependencies (kernel and Shapely).
"""

import json
from pathlib import Path

import numpy as np
from mesher import ElementType2D, Mesh2D
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from mesher.utils import get_mm2_control_area, find_adapter_group
from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.editor import EditorMesh2D

JSON_PATH = Path(__file__).parent / "data" / "package_footprint2.json"
GLOBAL_ELEMENT_SIZE = 1000
FOCUS_ELEMENT_SIZE = 5
ADAPTER_KOZ_SIZE = 1000
COW_CUTOUT_MARGIN = 2000
UF_GAP = 260
FILLET_LAYERS = ((5, 5), (50, 5), (25, 1))  # (spacing, layer count), inside out

COW_TO_STIFFENER = 3000
STIFFENER_WIDTH = 4000
STIFFENER_TO_SBT = 200

def _get_die_boxes():
    with open(JSON_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [value for value in data.values()]


def _get_die_uf_region(die_boxes, gap=UF_GAP):
    """Return the filled footprint and its oriented die + UF boundary loops."""
    from process_flow_kernel.domain.region import Region, TYPE_DIE, TYPE_TARGET
    from shapely import union_all
    from shapely.geometry import box

    region = Region([dict(type="BOX", dim=list(bounds)) for bounds in die_boxes])
    region.set_gap(gap, set_to=TYPE_TARGET, target_mask=TYPE_DIE, is_recursive=True)
    mask = TYPE_DIE | TYPE_TARGET
    cells = [
        box(region.table_x_dim[i], region.table_y_dim[j],
            region.table_x_dim[i + 1], region.table_y_dim[j + 1])
        for i in range(region.cell_num_x)
        for j in range(region.cell_num_y)
        if region.cell_type[i][j] & mask
    ]
    return union_all(cells), region.get_outline(mask)


def _get_fillet_pattern_lines(die_boxes, *, gap=UF_GAP, layers=FILLET_LAYERS):
    """Extend offset sides across corners, retaining only the exterior fillet."""
    from shapely.geometry import LineString

    spacings = []
    for size, count in layers:
        if not np.isfinite(size) or size <= 0 or type(count) is not int or count <= 0:
            raise ValueError("Fillet layers require positive spacing and integer count")
        spacings.extend([size] * count)
    if not spacings:
        raise ValueError("At least one fillet layer is required")
    offsets = np.r_[0.0, np.cumsum(spacings)]
    width = float(offsets[-1])
    if not np.isfinite(width):
        raise ValueError("Fillet width must be finite")

    filled, outlines = _get_die_uf_region(die_boxes, gap)
    # Closed band includes the original outline, but excludes the footprint interior.
    band = filled.buffer(width, join_style="mitre").difference(filled)
    segments = set()

    def collect(geometry):
        if geometry.geom_type == "LineString":
            points = list(geometry.coords)
            for start, end in zip(points, points[1:]):
                if start != end:
                    segments.add(tuple(sorted((start, end))))
        elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
            for part in geometry.geoms:
                collect(part)

    for outline in outlines:
        points = np.asarray(outline, dtype=float)
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            tangent = (end - start) / np.linalg.norm(end - start)
            # Region keeps material on the right, so the left normal is outward.
            normal = np.array([-tangent[1], tangent[0]])
            for offset in offsets:
                line = LineString([start + normal * offset - tangent * width,
                                   end + normal * offset + tangent * width])
                collect(line.intersection(band))
    return sorted(segments)


def _get_cow_outline():
    die_boxes = _get_die_boxes()
    
    pkg_x1, pkg_y1, pkg_x2, pkg_y2 = 100000, 100000, -100000, -100000
    for die_box in die_boxes:
        pkg_x1 = min(pkg_x1, die_box[0])
        pkg_y1 = min(pkg_y1, die_box[1])
        pkg_x2 = max(pkg_x2, die_box[2])
        pkg_y2 = max(pkg_y2, die_box[3])
    pkg_x1 = pkg_x1 - 2000
    pkg_y1 = pkg_y1 - 2000
    pkg_x2 = pkg_x2 + 2000
    pkg_y2 = pkg_y2 + 2000
    return [[pkg_x1, pkg_y1], [pkg_x1, pkg_y2], [pkg_x2, pkg_y2], [pkg_x2, pkg_y1]]
    
def get_cow_mesh() -> Mesh2D:
    cow_outline = _get_cow_outline()

    pattern_lines = []

    # die pattern
    die_boxes = _get_die_boxes()
    for x1, y1, x2, y2 in die_boxes:
        pattern_lines.extend([
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ])

    # fillet pattern
    pattern_lines.extend(_get_fillet_pattern_lines(die_boxes))
    
    # normal box spec
    normal_box_specs, pattern_lines = find_adapter_group(
        group_size = 50,
        lines = pattern_lines, 
        min_group_count = 3
    )

    # focus box spec
    areas = get_mm2_control_area(die_boxes, corner_extent=25, max_gap=UF_GAP)
    focus_box_specs = []
    for box in areas:
        focus_box_specs.append(dict(
            box=box,
            x_points=[[box[0], y] for y in np.r_[np.arange(box[1], box[3], FOCUS_ELEMENT_SIZE), box[3]]],
            y_points=[[x, box[1]] for x in np.r_[np.arange(box[0], box[2], FOCUS_ELEMENT_SIZE), box[2]]]
        ))
        
    # mesh
    line_pairs = get_pair_line(
        boxes = normal_box_specs + focus_box_specs, 
        expand_size = ADAPTER_KOZ_SIZE, 
        outer_outline = cow_outline, 
        distance = 4000
    )
    
    mesh = imprint_lines(
        element_size = GLOBAL_ELEMENT_SIZE, 
        line_pairs = line_pairs, 
        pattern_lines = pattern_lines,
        outer_outline = cow_outline,
        min_transition_size = 20,
    )
    
    return mesh

def get_no_cow_mesh() -> Mesh2D:
    cow_outline = _get_cow_outline()
    
    x_coordinates = []
    y_coordinates = []
    
    cow_x1 = cow_outline[0][0]
    cow_y1 = cow_outline[0][1]
    cow_x2 = cow_outline[2][0]
    cow_y2 = cow_outline[2][1]
    
    # cow to lid
    x_coordinates.append(cow_x1 - COW_TO_STIFFENER)
    y_coordinates.append(cow_y1 - COW_TO_STIFFENER)
    x_coordinates.append(cow_x2 + COW_TO_STIFFENER)
    y_coordinates.append(cow_y2 + COW_TO_STIFFENER)
    
    # stiffener
    x_coordinates.append(cow_x1 - COW_TO_STIFFENER - STIFFENER_WIDTH)
    y_coordinates.append(cow_y1 - COW_TO_STIFFENER - STIFFENER_WIDTH)
    x_coordinates.append(cow_x2 + COW_TO_STIFFENER + STIFFENER_WIDTH)
    y_coordinates.append(cow_y2 + COW_TO_STIFFENER + STIFFENER_WIDTH)
    
    # sbt
    x_coordinates.append(cow_x1 - COW_TO_STIFFENER - STIFFENER_WIDTH - STIFFENER_TO_SBT)
    y_coordinates.append(cow_y1 - COW_TO_STIFFENER - STIFFENER_WIDTH - STIFFENER_TO_SBT)
    x_coordinates.append(cow_x2 + COW_TO_STIFFENER + STIFFENER_WIDTH + STIFFENER_TO_SBT)
    y_coordinates.append(cow_y2 + COW_TO_STIFFENER + STIFFENER_WIDTH + STIFFENER_TO_SBT)
    
    mesh = generate_rectilinear_mesh(
        target_edge_size = GLOBAL_ELEMENT_SIZE,
        x_coordinates = x_coordinates,
        y_coordinates = y_coordinates
    )
    
    return mesh


def merge_by_adapter(cow_mesh: Mesh2D, non_cow_mesh: Mesh2D) -> Mesh2D:
    """Grow two Quad4 layers from the cow boundary to the non-cow cutout."""
    cow = cow_mesh
    grid = non_cow_mesh
    cow_boundary, = EditorMesh2D.outlines(cow)
    xy = cow.nodes[cow_boundary, :2]
    lower = xy.min(axis=0) - COW_CUTOUT_MARGIN
    upper = xy.max(axis=0) + COW_CUTOUT_MARGIN
    cutout = EditorMesh2D.search_element(grid, *lower, *upper, tolerance=0.01)
    EditorMesh2D.delete_element(grid, cutout)

    # Drop nodes used only by deleted cells before assigning shared IDs.
    used, inverse = np.unique(grid.elements, return_inverse=True)
    ring = Mesh2D(grid.nodes[used], inverse.reshape(grid.elements.shape))
    hole, _ = sorted(EditorMesh2D.outlines(ring), key=lambda loop: np.ptp(ring.nodes[loop, 0]))

    # Stage only the cow elements that own an exterior edge. The adapter
    # requires Quad4s, while the cow interior also contains Tri3s.
    successor = np.full(cow.node_count, -1, dtype=np.int64)
    successor[cow_boundary] = np.roll(cow_boundary, -1)
    edge_end = np.roll(cow.elements, -1, axis=1)
    boundary_edges = (successor[cow.elements] == edge_end) | (successor[edge_end] == cow.elements)
    boundary_cells = np.any(boundary_edges, axis=1)
    if np.any(cow.element_types[boundary_cells] != ElementType2D.QUAD4):
        raise ValueError("The adaptive layer requires Quad4s along the cow boundary.")

    ring_elements = ring.elements + cow.node_count
    working = Mesh2D(np.vstack((cow.nodes, ring.nodes)),
                     np.vstack((ring_elements, cow.elements[boundary_cells])))
    original_count = working.element_count
    first_layer = generate_layer_adaptive_mesh(
        mesh=working,
        outline_i=cow_boundary,
        element_size=0.8 * GLOBAL_ELEMENT_SIZE,
        expand_size=0.4 * GLOBAL_ELEMENT_SIZE,
        small_element_size=50,
        small_element_group_count=0,
        max_transition_count=10,
        dense_mapping_angle=5,
        min_size_ratio=0.5,
        max_size_ratio=1.5
    )
    generate_layer_adaptive_mesh(
        mesh=working,
        outline_i=first_layer.outline,
        element_size=GLOBAL_ELEMENT_SIZE,
        outline_o=hole + cow.node_count,
        fixed_outer_boundary=True,
    )
    added_elements = working.elements[original_count:]
    working.elements = np.vstack((cow.elements, ring_elements, added_elements))
    return working


if __name__ == "__main__":
    from mesher.mesh2d.visualization import view_mesh
    j_criteria = 0.4
    
    editor = EditorMesh2D()

    cow_mesh = get_cow_mesh()
    # view_mesh(cow_mesh)
    non_cow_mesh = get_no_cow_mesh()
    mesh = merge_by_adapter(cow_mesh, non_cow_mesh)
    
    jacobian_ratios = editor.jacobian_ratios(mesh, sampling="gauss")
    element_indices = np.where(jacobian_ratios < j_criteria)[0]
    print(f"MIN Jacobian ratio: {np.min(jacobian_ratios)}. {len(element_indices)}element are smaller than {j_criteria}")
    
    view_mesh(mesh, element_indices=element_indices)
    
