"""Visual example for meshing the strip between two circular node rings.

Run from the project root:

    venv/bin/python script/test2.py

The constrained connector triangles are highlighted in red and their circle
intersection nodes in yellow.  Use ``--no-view`` to print the mesh statistics
without opening the PyVista window.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# Running this file directly only puts script/ on sys.path.  Add the project
# root so imports work regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle.circle import _mesh_inner_outer_circle
from mesh import Mesh
from mesh_quality import MeshQualityChecker
from viewer import view_mesh


INNER_RADIUS = 4.0
OUTER_RADIUS = 8.0


def _circle_nodes(radius: float, count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros_like(angles),
        )
    )


def build_example():
    """Return the example mesh and data used to annotate the viewer."""
    # The counts deliberately differ.  Both rings contain the four cardinal
    # points, so x=0 and y=0 can be represented as constrained mesh edges.
    inner_count = 12
    outer_count = 20
    nodes = np.vstack(
        (
            _circle_nodes(INNER_RADIUS, inner_count),
            _circle_nodes(OUTER_RADIUS, outer_count),
        )
    )
    mesh = Mesh(
        nodes=nodes,
        elements=np.empty((0, 4), dtype=np.int32),
    )

    inner_nodes = np.arange(0, inner_count, dtype=np.int64)
    outer_nodes = np.arange(
        inner_count,
        inner_count + outer_count,
        dtype=np.int64,
    )

    # Closed rings may be passed in any order.  Shuffle them here to make that
    # behavior visible in a small, reproducible example.
    random_generator = np.random.default_rng(7)
    shuffled_inner_nodes = random_generator.permutation(inner_nodes)
    shuffled_outer_nodes = random_generator.permutation(outer_nodes)

    pattern_lines = np.array(
        [
            [[0.0, -10.0], [0.0, 10.0]],
            [[-10.0, 0.0], [10.0, 0.0]],
        ],
        dtype=np.float64,
    )

    _mesh_inner_outer_circle(
        mesh,
        shuffled_inner_nodes,
        shuffled_outer_nodes,
        lines=pattern_lines,
        closed=True,
    )

    # Cardinal-angle nodes form the four constrained inner-to-outer edges.
    connector_edges = [
        (int(inner_nodes[0]), int(outer_nodes[0])),
        (int(inner_nodes[3]), int(outer_nodes[5])),
        (int(inner_nodes[6]), int(outer_nodes[10])),
        (int(inner_nodes[9]), int(outer_nodes[15])),
    ]
    connector_nodes = np.unique(np.asarray(connector_edges, dtype=np.int64))
    connector_element_indices = np.asarray(
        [
            element_index
            for element_index, element in enumerate(mesh.elements)
            if any(
                {start, end}.issubset(set(map(int, element[:3])))
                for start, end in connector_edges
            )
        ],
        dtype=np.int64,
    )

    return (
        mesh,
        pattern_lines,
        connector_edges,
        connector_nodes,
        connector_element_indices,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="validate and print the example without opening PyVista",
    )
    arguments = parser.parse_args()

    (
        mesh,
        pattern_lines,
        connector_edges,
        connector_nodes,
        connector_element_indices,
    ) = build_example()

    quality = MeshQualityChecker(mesh).calculate_scaled_jacobian()
    connector_use_counts = {}
    for connector in connector_edges:
        connector_key = set(connector)
        connector_use_counts[connector] = sum(
            connector_key.issubset(set(map(int, element[:3])))
            for element in mesh.elements
        )

    print(f"Nodes: {mesh.nodes.shape[0]}")
    print(f"Padded Tri3 elements: {mesh.elements.shape[0]}")
    print(f"Expected triangle count: {mesh.nodes.shape[0]}")
    print(f"Constrained connector edge uses: {connector_use_counts}")
    print(f"Invalid elements: {quality.invalid_indices.tolist()}")
    print(f"Minimum scaled Jacobian: {np.min(quality.values):.6f}")
    print(f"Mean scaled Jacobian: {np.mean(quality.values):.6f}")

    if arguments.no_view:
        return

    view_mesh(
        mesh,
        element_indices=connector_element_indices,
        node_indices=connector_nodes,
        reference_circles=[
            [0.0, 0.0, INNER_RADIUS],
            [0.0, 0.0, OUTER_RADIUS],
        ],
        reference_lines=pattern_lines,
    )


if __name__ == "__main__":
    main()
