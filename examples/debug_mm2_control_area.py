"""Inspect boxes on their boundary grid using Matplotlib.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/pattern2d.py

Edit BOXES below, then use the window toolbar to pan, zoom, or save a picture.
"""

import json
import matplotlib.pyplot as plt
from mesher.mesh2d.visualization import plot_pattern2d
from mesher.utils import get_mm2_control_area
from mesher.mesh2d.generators import generate_rectilinear_mesh_optimal

JSON_PATH = "/Users/henry/Desktop/code/mesher/examples/data/mm2_symmetric_package.json"

def main():
    with open(JSON_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    boxes = [value for value in data.values()]
    plot_pattern2d(boxes)
    plt.show()
    
    areas = get_mm2_control_area(boxes, corner_extent=25, max_gap=260)
    plot_pattern2d(areas)
    plt.show()


if __name__ == "__main__":
    main()
