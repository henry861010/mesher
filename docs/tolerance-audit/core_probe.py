"""Print current editor/model behavior for the 0.001 tolerance audit.

This is a diagnostic, not a regression test or a proposed implementation.
Run from the repository root with PYTHONPATH=src venv/bin/python <this file>.
"""

import json

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d import EditorMesh2D
from mesher.mesh2d.utils.equivalence import _equivalent_node_mapping


def record(probe, **result):
    print(json.dumps({"probe": probe, **result}))


for offset in (0.0, 1e-10, 0.0005):
    mesh = Mesh2D(
        [[0, 0], [1 + offset, 0], [1 + offset, 1], [0, 1]],
        [[0, 1, 2, 3]],
    )
    record(
        "rectangle selection",
        offset=offset,
        selected=EditorMesh2D.search_element(mesh, 0, 0, 1, 1).tolist(),
    )

xy = np.array([[0, 0], [0.00075, 0], [0.0015, 0]], dtype=float)
record(
    "transitive equivalence",
    mapping=_equivalent_node_mapping(xy, np.arange(3), 0.001).tolist(),
    group_span=0.0015,
)

mesh = Mesh2D([[0, 0], [1, 0], [2, 0.0001]], [[0, 1, 2, 2]])
EditorMesh2D.equivalence(mesh, tol=0.001)
record("thin triangle after equivalence", elements=mesh.element_count, area=0.00005)

for z in (1e-16, 0.0005):
    try:
        Mesh2D([[0, 0, z], [1, 0, 0], [1, 1, 0]], [[0, 1, 2, 2]])
    except ValueError as error:
        record("intentional strict Mesh2D Z contract", z=z, error=str(error))
