"""Imprint a 0-to-180-degree circle into an upper-half-plane mesh.

Run from the repository root with:

    PYTHONPATH=src python examples/imprint_circle_open_sector.py

Add ``--view`` to display the result when the visualization dependency is
installed.
"""

import argparse

import numpy as np

from mesher.circular import imprint_circle
from mesher.generators import generate_rectilinear_mesh
from mesher.quality import MeshQualityChecker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view",
        action="store_true",
        help="display the imprinted half-plane mesh with PyVista",
    )
    args = parser.parse_args()

    center = (0.0, 0.0)
    radius = 5.0
    band_width = 1.0

    # Because y starts at the circle center, the mesh covers only angles from
    # 0 to 180 degrees. The two cut sides lie on rays from center.
    mesh = generate_rectilinear_mesh(
        target_edge_size=0.5,
        x_coordinates=[-8.0, 0.0, 8.0],
        y_coordinates=[0.0, 8.0],
    )
    original_node_count = mesh.nodes.shape[0]
    original_element_count = mesh.elements.shape[0]

    imprint_circle(
        mesh,
        center=center,
        radius=radius,
        band_width=band_width,
        target_edge_size=1.0,
        # This becomes an exact connector through both rebuilt strips.
        guide_segments=[[[0.0, 0.0], [0.0, 8.0]]],
        # topology="auto" is the default and detects this open sector.
    )

    xy = np.asarray(mesh.nodes)[:, :2]
    distances = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
    pattern_nodes = np.flatnonzero(
        np.isclose(distances, radius, rtol=0.0, atol=1.0e-10)
    )
    pattern_angles = np.rad2deg(
        np.arctan2(
            xy[pattern_nodes, 1] - center[1],
            xy[pattern_nodes, 0] - center[0],
        )
    )

    quality = MeshQualityChecker(mesh).check_jacobian()
    print(f"nodes: {original_node_count} -> {mesh.nodes.shape[0]}")
    print(f"elements: {original_element_count} -> {mesh.elements.shape[0]}")
    print(f"pattern arc nodes: {pattern_nodes.size}")
    print(
        "pattern angle range: "
        f"{float(np.min(pattern_angles)):.1f} to "
        f"{float(np.max(pattern_angles)):.1f} degrees"
    )
    print(f"invalid Jacobians: {quality.invalid_indices.size}")

    if args.view:
        from mesher.visualization import view_mesh

        view_mesh(
            mesh,
            reference_circles=[
                [*center, radius - band_width / 2.0],
                [*center, radius],
                [*center, radius + band_width / 2.0],
            ],
            reference_lines=[[[0.0, 0.0], [0.0, 8.0]]],
        )


if __name__ == "__main__":
    main()
