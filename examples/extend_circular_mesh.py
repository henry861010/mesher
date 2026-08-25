"""Build concentric mesh layers from an existing circular boundary.

Run from the repository root with:

    PYTHONPATH=src python examples/extend_circular_mesh.py

Add ``--view`` to display the result when the visualization dependency is
installed. Add ``--half`` to exercise automatic open-arc detection.
"""

import argparse

import numpy as np

from mesher import Mesh2D
from mesher.circular import extend_circular_mesh
from mesher.quality import MeshQualityChecker


def build_inner_disk(
    center_x,
    center_y,
    radius,
    sector_count=32,
    *,
    closed=True,
):
    """Create a full or upper-half Tri3 fan on one circular boundary."""
    ring_node_count = sector_count if closed else sector_count + 1
    angles = np.linspace(
        0.0,
        2.0 * np.pi if closed else np.pi,
        ring_node_count,
        endpoint=not closed,
    )
    ring = np.column_stack(
        (
            center_x + radius * np.cos(angles),
            center_y + radius * np.sin(angles),
        )
    )
    nodes_xy = np.vstack(([center_x, center_y], ring))
    # PyVista requires three coordinates per point. The mesh remains planar
    # because every generated Z coordinate is zero.
    nodes = np.column_stack(
        (nodes_xy, np.zeros(nodes_xy.shape[0], dtype=np.float64))
    )

    ring_indices = np.arange(1, ring_node_count + 1, dtype=np.int32)
    starts = ring_indices if closed else ring_indices[:-1]
    following = np.roll(ring_indices, -1) if closed else ring_indices[1:]
    # Tri3 elements use the padded [n0, n1, n2, n2] representation.
    elements = np.column_stack(
        (
            np.zeros(sector_count, dtype=np.int32),
            starts,
            following,
            following,
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
    parser.add_argument(
        "--half",
        action="store_true",
        help="extend an upper-half circle instead of a complete circle",
    )
    args = parser.parse_args()

    center_x = 2.0
    center_y = -1.0
    inner_radius = 5.0
    outer_radius = 10.0
    element_size = 1.25

    mesh = build_inner_disk(
        center_x,
        center_y,
        inner_radius,
        closed=not args.half,
    )
    inner_node_count = mesh.nodes.shape[0]
    inner_element_count = mesh.elements.shape[0]

    extend_circular_mesh(
        mesh,
        element_size=element_size,
        center_x=center_x,
        center_y=center_y,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        topology="auto",
    )

    layer_count = int(np.ceil((outer_radius - inner_radius) / element_size))
    quality = MeshQualityChecker(mesh).check_scaled_jacobian(minimum=0.0)
    print(f"topology: {'open' if args.half else 'closed'}")
    print(f"radial layers: {layer_count}")
    print(f"nodes: {inner_node_count} -> {mesh.nodes.shape[0]}")
    print(f"elements: {inner_element_count} -> {mesh.elements.shape[0]}")
    print(f"minimum scaled Jacobian: {quality.summary.minimum:.6f}")

    if args.view:
        from mesher.visualization import view_mesh

        view_mesh(
            mesh,
            reference_circles=[
                [center_x, center_y, inner_radius],
                [center_x, center_y, outer_radius],
            ],
        )


if __name__ == "__main__":
    main()
