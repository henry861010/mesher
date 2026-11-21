"""Connect the imprinted cow mesh to the surrounding non-cow grid.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/debug_imprint_line.py
"""

import json
from pathlib import Path

import numpy as np
from mesher import ElementType2D, Mesh2D
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from mesher.utils import find_adapter_group, get_mm2_control_area
from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.editor import EditorMesh2D

from mesher.mesh2d.visualization import view_mesh

JSON_PATH = Path(__file__).parent / "data" / "mm2_symmetric_package.json"
GLOBAL_ELEMENT_SIZE = 1000
FOCUS_ELEMENT_SIZE = 5
ADAPTER_KOZ_SIZE = 1000
COW_CUTOUT_MARGIN = 2000

COW_TO_STIFFENER = 3000
STIFFENER_WIDTH = 4000
STIFFENER_TO_SBT = 200

def _get_die_boxes():
    with open(JSON_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [value for value in data.values()]

def _get_cow_outline():
    die_boxes = _get_die_boxes()
    
    pkg_x1, pkg_y1, pkg_x2, pkg_y2 = 100000, 100000, -100000, -100000
    for die_box in die_boxes:
        pkg_x1 = min(pkg_x1, die_box[0])
        pkg_y1 = min(pkg_y1, die_box[1])
        pkg_x2 = max(pkg_x2, die_box[2])
        pkg_y2 = max(pkg_y2, die_box[3])
    pkg_x1 = pkg_x1 - 500
    pkg_y1 = pkg_y1 - 500
    pkg_x2 = pkg_x2 + 500
    pkg_y2 = pkg_y2 + 500
    return [[pkg_x1, pkg_y1], [pkg_x1, pkg_y2], [pkg_x2, pkg_y2], [pkg_x2, pkg_y1]]
    
def _find_pair_td_corner(cow_outline):
    die_boxes = _get_die_boxes()

    areas = get_mm2_control_area(die_boxes, corner_extent=25, max_gap=260)

    box_objs = []
    for box in areas:
        box_objs.append(dict(
            box=box,
            x_points=[[box[0], y] for y in np.r_[np.arange(box[1], box[3], FOCUS_ELEMENT_SIZE), box[3]]],
            y_points=[[x, box[1]] for x in np.r_[np.arange(box[0], box[2], FOCUS_ELEMENT_SIZE), box[2]]]
        ))
        
    line_pairs = get_pair_line(
        boxes = box_objs, 
        expand_size = ADAPTER_KOZ_SIZE, 
        outer_outline = cow_outline, 
        distance = 4000
    )
    
    return line_pairs

def _find_pair_fillet(cow_outline, fillet_objs):
    box_objs = []
    remaining_patterns = []
    for fillet_obj in fillet_objs:
        groups, remaining = find_adapter_group(group_size=6, lines=fillet_obj.get_mesh_control())
        box_objs.extend(groups)
        remaining_patterns.extend(remaining)
        
    line_pairs = get_pair_line(
        boxes = box_objs, 
        expand_size = ADAPTER_KOZ_SIZE, 
        outer_outline = cow_outline, 
        distance = 4000
    )
        
    return line_pairs, remaining_patterns

def cow_mesh(fillet_objs=()):
    cow_outline = _get_cow_outline()
    
    # die pattern
    die_boxes = _get_die_boxes()
    pattern_lines = []
    for x1, y1, x2, y2 in die_boxes:
        pattern_lines.extend([
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ])
    
    td_corner_pair = _find_pair_td_corner(cow_outline=cow_outline)
    fillet_pair, fillet_patterns = _find_pair_fillet(cow_outline, fillet_objs)
    pattern_lines.extend(fillet_patterns)
    
    line_pairs = td_corner_pair + fillet_pair
    
    mesh = imprint_lines(
        element_size = GLOBAL_ELEMENT_SIZE, 
        line_pairs = line_pairs, 
        pattern_lines = pattern_lines,
        outer_outline = cow_outline,
        min_transition_size = 20, 
        pattern_snap_distance=0
    )
    
    return mesh

def non_cow_mesh():
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


def build_connected_mesh() -> Mesh2D:
    """Grow two Quad4 layers from the cow boundary to the non-cow cutout."""
    cow = cow_mesh()
    grid = non_cow_mesh()
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
        expand_size=40 * FOCUS_ELEMENT_SIZE,
        small_element_size=30,
        max_transition_count=20,
        dense_mapping_angle=30,
        min_size_ratio=0.9,
        max_size_ratio=1.25
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
    view_mesh(build_connected_mesh())
