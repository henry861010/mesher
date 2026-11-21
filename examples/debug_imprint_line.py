"""Inspect boxes on their boundary grid using Matplotlib.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/pattern2d.py

Edit BOXES below, then use the window toolbar to pan, zoom, or save a picture.
"""

import json
import numpy as np
from mesher.utils import group_lines, get_mm2_control_area
from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.editor import EditorMesh2D

from mesher.mesh2d.visualization import view_mesh

JSON_PATH = "/Users/henry/Desktop/code/mesher/examples/data/mm2_symmetric_package.json"
GLOBAL_ELEMENT_SIZE = 1000
FOCUS_ELEMENT_SIZE = 5
ADAPTER_KOZ_SIZE = 1000

COW_TO_STIFFENER = 3000
STIFFENER_WIDTH = 4000
STIFFENER_TO_SBT = 200

def _get_cow_outline():
    with open(JSON_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    die_boxes = [value for value in data.values()]
    
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
    # find the die coordinates: read the json directly
    with open(JSON_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    die_boxes = [value for value in data.values()]

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

def _find_pair_fillet(cow_outline):
    fillet_objs = []
    
    box_objs = []
    for fillet_obj in fillet_objs:
        lines = fillet_objs.get_mesh_control()
        
        # classify the line to box group
        box_objs += group_lines(lines)
        
    line_pairs = get_pair_line(
        boxes = box_objs, 
        expand_size = ADAPTER_KOZ_SIZE, 
        outer_outline = cow_outline, 
        distance = 4000
    )
        
    return line_pairs

def cow_mesh():
    cow_outline = _get_cow_outline()
    
    td_corner_pair = _find_pair_td_corner(cow_outline=cow_outline)
    fillet_pair = _find_pair_fillet(cow_outline=cow_outline)
    
    line_pairs = td_corner_pair + fillet_pair
    
    mesh = imprint_lines(
        element_size = GLOBAL_ELEMENT_SIZE, 
        line_pairs = line_pairs, 
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


if __name__ == "__main__":
    cow_mesh = cow_mesh()
    non_cow_mesh = non_cow_mesh()
    
    editor = EditorMesh2D()
    editor.merge(cow_mesh, non_cow_mesh)
    
    cow_outline = _get_cow_outline()
    cow_x1 = cow_outline[0][0]
    cow_y1 = cow_outline[0][1]
    cow_x2 = cow_outline[2][0]
    cow_y2 = cow_outline[2][1]
    indices = editor.search_element(non_cow_mesh, cow_x1-2000, cow_y1-2000, cow_x2+2000, cow_y2+2000, tolerance=0.01)
    non_cow_mesh = editor.delete_element(non_cow_mesh, indices)

    # merge
    mesh = editor.merge(cow_mesh, non_cow_mesh)
    
    # adaptive mesh
    rst = generate_layer_adaptive_mesh(
        mesh = mesh, 
        outline_i = outline_i, 
        element_size = element_size, 
        pattern_lines = pattern_lines,
        outline_o = outline_o,
        fixed_outer_boundary = fixed_outer_boundary,
        expand_size = expand_size,
        expand_ratio = expand_ratio,
        tolerance = tolerance,
        max_transition_count = max_transition_count,
        min_size_ratio = min_size_ratio,
        small_element_size = small_element_size,
        dense_mapping_angle = dense_mapping_angle
    )
    mesh = rst.mesh
    
    view_mesh(mesh)
