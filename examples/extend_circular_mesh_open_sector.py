"""Extend a non-360-degree circular mesh with automatic topology detection.

Run from the repository root with:

    PYTHONPATH=src python examples/extend_circular_mesh_open_sector.py

Add ``--view`` to display the result when PyVista is installed.
"""

import argparse

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.circular import extend_circular_mesh
from mesher.mesh2d.quality import MeshQualityChecker


def build_inner_sector(
    center,
    radius,
    start_angle,
    end_angle,
    sector_count=24,
):
    """Create a Tri3 fan with one exposed endpoint-bounded circular arc."""
    angles = np.linspace(start_angle, end_angle, sector_count + 1)
    ring_xy = center + radius * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    nodes_xy = np.vstack((center, ring_xy))
    nodes = np.column_stack(
        (nodes_xy, np.zeros(nodes_xy.shape[0], dtype=np.float64))
    )

    ring = np.arange(1, sector_count + 2, dtype=np.int32)
    elements = np.column_stack(
        (
            np.zeros(sector_count, dtype=np.int32),
            ring[:-1],
            ring[1:],
            ring[1:],
        )
    )
    return Mesh2D(nodes=nodes, elements=elements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view",
        action="store_true",
        help="display the generated mesh with PyVista",
    )
    args = parser.parse_args()

    center = np.array([2.0, -1.0])
    inner_radius = 5.0
    outer_radius = 10.0
    element_size = 1.25
    start_angle = np.deg2rad(-30.0)
    end_angle = np.deg2rad(210.0)

    mesh = build_inner_sector(
        center,
        inner_radius,
        start_angle,
        end_angle,
    )
    inner_node_count = mesh.nodes.shape[0]
    inner_element_count = mesh.elements.shape[0]

    extend_circular_mesh(
        mesh,
        element_size=element_size,
        center_x=center[0],
        center_y=center[1],
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        topology="auto",
    )

    quality = MeshQualityChecker(mesh).check_scaled_jacobian(minimum=0.0)
    print("detected topology: open")
    print("angular coverage: 240 degrees")
    print(f"nodes: {inner_node_count} -> {mesh.nodes.shape[0]}")
    print(f"elements: {inner_element_count} -> {mesh.elements.shape[0]}")
    print(f"minimum scaled Jacobian: {quality.summary.minimum:.6f}")

    if args.view:
        from mesher.mesh2d.visualization import view_mesh

        view_mesh(
            mesh,
            reference_circles=[
                [center[0], center[1], inner_radius],
                [center[0], center[1], outer_radius],
            ],
        )


if __name__ == "__main__":
    main()
