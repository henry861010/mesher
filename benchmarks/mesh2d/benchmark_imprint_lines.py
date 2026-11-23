"""End-to-end line imprint benchmark, including validation and assembly.

PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_imprint_lines.py --repeats 5
The time limit is a same-host acceptance target, not a unit-test assertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from mesher.mesh2d.imprint_line import imprint_lines


def fixture(kind):
    pairs = []
    for i in range(10):
        for j in range(10):
            axis = (i + j) % 2
            x, y = 30 + 60 * i, 30 + 60 * j
            if kind == "overlap":
                # 25 aligned two-focus groups, four directional pairs each.
                # Shifted dense grids are tested separately: their coordinate
                # unions can exceed the one-million-element default budget.
                x, y = 30 + 120 * (i // 2), 30 + 120 * (j // 2)
                x += 4 * (j % 2)
                axis = i % 2
                width = 4
            else:
                width = 2
            values = np.linspace(y if axis == 0 else x, (y if axis == 0 else x) + width, 100)
            pairs.append(dict(direction="xy"[axis], line1_expand_size=2, line2_expand_size=2,
                              line1=[[x, y], [x, y + width]] if axis == 0 else [[x, y], [x + width, y]],
                              line2=[[x + width, y], [x + width, y + width]] if axis == 0 else [[x, y + width], [x + width, y + width]],
                              points=[[x, v] for v in values] if axis == 0 else [[v, y] for v in values]))
    rng = np.random.default_rng(91)
    patterns = []
    for i in range(1000):
        x, y = (rng.integers(0, 310, 2) * 2).tolist()
        patterns.append([[x, y], [x + 20, y]] if i % 2 else [[x, y], [x, y + 20]])
    return pairs, patterns


def worker(kind):
    pairs, patterns = fixture(kind)
    start = perf_counter()
    mesh = imprint_lines(2, pairs, patterns, outer_outline=[[0, 0], [640, 0], [640, 640], [0, 640]],
                         tag="mesh_snap", min_transition_size=.5)
    seconds = perf_counter() - start
    digest = hashlib.sha256(mesh.nodes.tobytes() + mesh.elements.tobytes()).hexdigest()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"case": kind, "seconds": seconds, "nodes": mesh.node_count, "elements": mesh.element_count,
            "triangles": int(np.count_nonzero(mesh.element_types == 3)), "sha256": digest,
            "peak_rss_mib": rss / (1024 ** 2 if sys.platform == "darwin" else 1024)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case", choices=("separated", "overlap"), default="separated")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.case)))
        return
    if args.repeats < 1:
        parser.error("repeats must be positive")
    records = []
    for _ in range(args.repeats):
        done = subprocess.run([sys.executable, __file__, "--worker", "--case", args.case],
                              capture_output=True, text=True, timeout=60)
        if done.returncode:
            raise RuntimeError(f"benchmark worker failed:\n{done.stdout}{done.stderr}")
        record = json.loads(done.stdout)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {"platform": platform.platform(), "python": sys.version,
              "numpy": np.__version__, "pairs": 100, "points_per_pair": 100,
              "patterns": 1000, "baseline_grid_cells": 102400, "samples": records,
              "maximum_seconds": max(r["seconds"] for r in records),
              "all_below_10_seconds": all(r["seconds"] < 10 for r in records),
              "deterministic": len({r["sha256"] for r in records}) == 1}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}))
    if not report["all_below_10_seconds"] or not report["deterministic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
