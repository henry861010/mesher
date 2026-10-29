"""Time the full test2 layer, excluding fixture preparation and rendering.

Run from the repository root::

    venv/bin/python benchmarks/mesh2d/benchmark_adapter_layer.py

One warmup precedes five timed calls. The default acceptance limit is one
second for the median full extend_quad_layer call, including validation and
commit. Each call receives a fresh copy of the same L-shaped input mesh.
"""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import extend_quad_layer
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--maximum-seconds", type=float, default=1.)
    args = parser.parse_args()
    if (args.repeats < 1 or args.warmups < 0
            or not np.isfinite(args.maximum_seconds) or args.maximum_seconds <= 0):
        parser.error("repeats and time limit must be positive; warmups must be nonnegative")

    editor = EditorMesh2D()
    left = generate_rectilinear_mesh(target_edge_size=5, x_coordinates=[0, 100],
                                     y_coordinates=[0, 100])
    right = generate_rectilinear_mesh(target_edge_size=5, x_coordinates=[100, 200],
                                      y_coordinates=[0, 200])
    source = editor.equivalence(editor.merge(left, right))
    outline = editor.outlines_by_edge(source)[0]
    timings, trials, repairs = [], [], []
    for iteration in range(args.warmups + args.repeats):
        mesh = Mesh2D(source.nodes.copy(), source.elements.copy())
        started = perf_counter()
        result = extend_quad_layer(mesh, outline, element_size=50, expand_ratio=1.5)
        elapsed = perf_counter() - started
        if iteration >= args.warmups:
            timings.append(elapsed)
            trials.append(result.diagnostics["line_unit_trial_count"])
            repairs.append(result.diagnostics["line_local_repair_count"])

        np.testing.assert_array_equal(mesh.nodes[:len(source.nodes)], source.nodes)
        np.testing.assert_array_equal(mesh.elements[:len(source.elements)], source.elements)
        assert result.diagnostics["section_count"] == 12
        assert result.diagnostics["sections"] == {"OUTER": 5, "LINE": 6, "INNER": 1}
        assert result.diagnostics["quality"]["min_scaled_jacobian"] > 0
        details = result.diagnostics["section_details"]
        for section, following in zip(details, details[1:] + details[:1]):
            assert section["right_ids"] == following["left_ids"]

    seconds = median(timings)
    print(json.dumps({
        "case": "adapter_layer_test2", "completed": True,
        "sourceSpacing": 5, "targetSize": 50, "expandRatio": 1.5,
        "repetitions": args.repeats, "warmups": args.warmups,
        "seconds": timings, "medianSeconds": seconds,
        "maximumSeconds": args.maximum_seconds, "lineUnitTrials": trials,
        "localRepairs": repairs, "planner": result.diagnostics["line_planner"],
        "sectionCount": result.diagnostics["section_count"],
        "unitCount": result.diagnostics["unit_count"],
        "quadCount": result.diagnostics["added_elements"],
        "quality": result.diagnostics["quality"],
    }), flush=True)
    if seconds >= args.maximum_seconds:
        raise AssertionError(f"Layer construction took {seconds:.3f}s; limit is {args.maximum_seconds:.3f}s")


if __name__ == "__main__":
    main()
