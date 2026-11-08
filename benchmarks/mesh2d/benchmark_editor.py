"""Time and peak-RSS checks for stateless Mesh2D editing.

Run from the repository root::

    PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_editor.py

Each case/size runs in its own process. Timings exclude fixture creation and
validation; peak RSS includes the interpreter, fixtures, results, and NumPy
workspace. A small warm-up precedes three measured fresh-mesh repetitions.
Output is JSON Lines, including scaling across 25k, 50k, and 100k elements.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mesher import GeometryToleranceError, Mesh2D
from mesher.mesh2d import EditorMesh2D


CASES = (
    "delete", "merge", "search", "equivalence", "equivalence_seam",
    "equivalence_dense", "equivalence_chain", "outline",
    "outline_islands", "outline_fan",
)


def _grid(count):
    nx = 250
    ny = max(1, count // nx)
    x, y = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    nodes = np.column_stack((x.ravel(), y.ravel()))
    lower = (np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)).ravel()
    elements = np.column_stack((lower, lower + 1, lower + nx + 2, lower + nx + 1))
    return Mesh2D(nodes, elements)


def _fixture(case, count):
    if case == "equivalence_dense":
        random = np.random.default_rng(0)
        corners = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        nodes = np.tile(corners, (count, 1)) + random.uniform(0, 1e-4, (count * 3, 2))
        ids = np.arange(count, dtype=np.int32) * 3
        mesh = Mesh2D(nodes, np.column_stack((ids, ids + 1, ids + 2, ids + 2)))
    elif case == "equivalence_chain":
        nodes = np.column_stack((np.arange(count) * 0.00075, np.zeros(count)))
        nodes = np.vstack((nodes, [[0, 2], [2, 2]]))
        elements = np.full((count, 4), count + 1, dtype=np.int32)
        elements[:, 0] = np.arange(count)
        elements[:, 1] = count
        mesh = Mesh2D(nodes, elements)
    elif case == "outline_islands":
        nodes = np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (count, 1))
        nodes[:, 0] += np.repeat(np.arange(count) * 3, 4)
        mesh = Mesh2D(nodes, np.arange(count * 4, dtype=np.int32).reshape(-1, 4))
    elif case == "outline_fan":
        angles = np.linspace(0, np.pi, count + 1)
        nodes = np.vstack(([0, 0], np.column_stack((np.cos(angles), np.sin(angles)))))
        ids = np.arange(1, count + 1, dtype=np.int32)
        mesh = Mesh2D(nodes, np.column_stack((np.zeros(count, dtype=np.int32), ids, ids + 1, ids + 1)))
    else:
        mesh = _grid(count)
        if case == "equivalence_seam":
            mesh = EditorMesh2D.merge(mesh, mesh)

    node_count, element_count = mesh.node_count, mesh.element_count
    if case == "delete":
        indices = np.arange(0, element_count, 3, dtype=np.int64)
        operation = lambda: EditorMesh2D.delete_element(mesh, indices)
    elif case == "merge":
        operation = lambda: EditorMesh2D.merge(mesh, mesh)
    elif case == "search":
        operation = lambda: EditorMesh2D.search_element(mesh, 0, 0, 125, count // 250)
    elif case == "equivalence_chain":
        def operation():
            try:
                EditorMesh2D.equivalence(mesh)
            except GeometryToleranceError as error:
                return error
            raise AssertionError("an unbounded proximity chain must fail")
    elif case.startswith("equivalence"):
        operation = lambda: EditorMesh2D.equivalence(mesh)
    else:
        operation = lambda: EditorMesh2D.outline(mesh)
    return mesh, operation, node_count, element_count


def _validate(case, mesh, result, node_count, element_count):
    if case == "delete":
        assert result is mesh and mesh.node_count == node_count
        assert mesh.element_count == element_count - len(range(0, element_count, 3))
    elif case == "merge":
        assert result.node_count == node_count * 2 and mesh.node_count == node_count
        assert result.element_count == element_count * 2
        assert mesh.element_count == element_count
        assert not np.shares_memory(result.nodes, mesh.nodes)
        np.testing.assert_array_equal(result.elements[element_count:], mesh.elements + node_count)
    elif case == "search":
        assert len(result) == element_count // 2
        assert np.all(np.diff(result) > 0)
    elif case == "equivalence_chain":
        assert result.code == "ambiguous_geometry"
        assert mesh.node_count == node_count and mesh.element_count == element_count
    elif case.startswith("equivalence"):
        assert result is mesh and mesh.element_count == element_count
        if case == "equivalence_seam":
            assert mesh.node_count == node_count // 2
        elif case == "equivalence_dense":
            assert mesh.node_count == 3
        else:
            assert mesh.node_count == node_count
    elif case == "outline_islands":
        assert len(result) == element_count
        np.testing.assert_array_equal(np.stack(result), mesh.elements)
    elif case == "outline_fan":
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], np.arange(element_count + 2))
    else:
        assert len(result) == 1 and len(result[0]) == 2 * (250 + element_count // 250)


def _peak_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def _worker(case, count, repeats, maximum_seconds):
    baseline = _peak_mib()
    mesh, operation, nodes, elements = _fixture(case, min(count, 1000))
    result = operation()
    _validate(case, mesh, result, nodes, elements)
    del mesh, operation, result
    gc.collect()

    elapsed = []
    for _ in range(repeats):
        mesh, operation, nodes, elements = _fixture(case, count)
        started = perf_counter()
        result = operation()
        elapsed.append(perf_counter() - started)
        _validate(case, mesh, result, nodes, elements)
        del mesh, operation, result
        gc.collect()
    seconds = median(elapsed)
    peak = _peak_mib()
    print(json.dumps({
        "case": case, "requestedElements": count,
        "inputNodes": nodes, "inputElements": elements,
        "medianSeconds": round(seconds, 6),
        "peakRssMiB": round(peak, 2),
        "peakAboveStartupMiB": round(max(0, peak - baseline), 2),
        "maximumSeconds": maximum_seconds,
        "repetitions": repeats,
    }), flush=True)
    if seconds >= maximum_seconds:
        raise AssertionError(f"{case}: {seconds:.3f}s exceeds {maximum_seconds:.3f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[25_000, 50_000, 100_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maximum-seconds", type=float, default=10.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 250 or size % 250 for size in args.sizes):
        parser.error("sizes must be positive multiples of 250; repeats must be positive")
    if args.worker:
        _worker(args.cases[0], args.sizes[0], args.repeats, args.maximum_seconds)
        return
    for case in args.cases:
        for count in args.sizes:
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--cases", case, "--sizes", str(count),
                "--repeats", str(args.repeats),
                "--maximum-seconds", str(args.maximum_seconds),
            ], check=True)


if __name__ == "__main__":
    main()
