"""One-million-element performance check for ``add_midside_node``.

Run from the repository root with::

    venv/bin/python benchmarks/benchmark_add_midside_node.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from mesher import Mesh3D
from mesher.mesh3d.utils import add_midside_node


MAXIMUM_SECONDS = 10.0
NX = 100
NY = 100
NZ = 100


def _structured_hex_mesh() -> Mesh3D:
    x = np.tile(
        np.arange(NX + 1, dtype=np.float64),
        (NY + 1) * (NZ + 1),
    )
    y = np.tile(
        np.repeat(np.arange(NY + 1, dtype=np.float64), NX + 1),
        NZ + 1,
    )
    z = np.repeat(
        np.arange(NZ + 1, dtype=np.float64),
        (NX + 1) * (NY + 1),
    )
    nodes = np.column_stack((x, y, z))

    plane_size = (NX + 1) * (NY + 1)
    lower = (
        np.arange(NZ, dtype=np.int32)[:, None, None] * plane_size
        + np.arange(NY, dtype=np.int32)[None, :, None] * (NX + 1)
        + np.arange(NX, dtype=np.int32)[None, None, :]
    ).reshape(-1)
    elements = np.empty((lower.size, 20), dtype=np.int32)
    elements[:, :8] = np.column_stack(
        (
            lower,
            lower + 1,
            lower + NX + 2,
            lower + NX + 1,
            lower + plane_size,
            lower + plane_size + 1,
            lower + plane_size + NX + 2,
            lower + plane_size + NX + 1,
        )
    )
    elements[:, 8:] = elements[:, 7, None]
    element_count = elements.shape[0]

    return Mesh3D(
        nodes=nodes,
        elements=elements,
        element_comps=np.ones(element_count, dtype=np.int32),
        comps={"solid": 1},
        element_types=np.ones(element_count, dtype=np.int32),
        types={1: 185},
        element_reals=np.zeros(element_count, dtype=np.int32),
        reals={},
        element_sections=np.zeros(element_count, dtype=np.int32),
        sections={},
        element_node_num=np.full(element_count, 8, dtype=np.int32),
    )


def main() -> None:
    mesh = _structured_hex_mesh()
    element_indices = np.arange(mesh.element_count, dtype=np.int64)

    started = time.perf_counter()
    result = add_midside_node(mesh, element_indices)
    elapsed = time.perf_counter() - started

    expected_elements = NX * NY * NZ
    expected_midsides = (
        NX * (NY + 1) * (NZ + 1)
        + (NX + 1) * NY * (NZ + 1)
        + (NX + 1) * (NY + 1) * NZ
    )
    if result.element_count != expected_elements:
        raise AssertionError("element count changed during conversion")
    if result.node_count != mesh.node_count + expected_midsides:
        raise AssertionError("shared midside node count is incorrect")
    if not np.all(result.element_node_num == 20):
        raise AssertionError("not every Hex8 element was converted")
    if elapsed >= MAXIMUM_SECONDS:
        raise AssertionError(
            f"conversion took {elapsed:.3f}s; limit is {MAXIMUM_SECONDS:.3f}s"
        )

    print(
        json.dumps(
            {
                "elements": expected_elements,
                "inputNodes": mesh.node_count,
                "newMidsideNodes": expected_midsides,
                "resultNodes": result.node_count,
                "elapsedSeconds": round(elapsed, 3),
                "maximumSeconds": MAXIMUM_SECONDS,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
