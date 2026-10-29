"""Measure outer-first unit construction for one complete LINE section.

Run from the repository root::

    venv/bin/python benchmarks/mesh2d/benchmark_adapter_line.py --timeout-seconds 30
    venv/bin/python benchmarks/mesh2d/benchmark_adapter_line.py --segments 40 --target-size 10 --height 10

Fixture creation and result validation are outside the timer. Output is one
JSON object containing wall time, unit trial count, result size and quality.
This exercises the production best-only path without splitting the section.
"""

from __future__ import annotations

import argparse
import json
import resource
import signal
import sys
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.adapter.layer import _line
from mesher.mesh2d.adapter.layer._unit_plan import quality_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=128)
    parser.add_argument("--length", type=float, default=None,
                        help="Section length; defaults to the source segment count.")
    parser.add_argument("--height", type=float, default=8.0)
    parser.add_argument("--target-size", type=float, default=8.0)
    parser.add_argument("--left-nodes", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--right-nodes", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--maximum-seconds", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None,
                        help="Interrupt a slow run and emit partial JSON (requires POSIX interval timers).")
    args = parser.parse_args()
    length = float(args.segments) if args.length is None else args.length
    values = [length, args.height, args.target_size]
    if (args.segments < 1 or args.repeats < 1 or args.warmups < 0
            or any(not np.isfinite(v) or v <= 0 for v in values)):
        parser.error("segment count, repeats, lengths and target size must be positive and finite")
    if args.maximum_seconds is not None and (not np.isfinite(args.maximum_seconds) or args.maximum_seconds <= 0):
        parser.error("maximum seconds must be positive and finite")
    if args.timeout_seconds is not None:
        if (not np.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
                or not hasattr(signal, "setitimer")):
            parser.error("timeout requires a positive duration and POSIX interval timers")
    source = np.column_stack((np.linspace(0, length, args.segments + 1),
                              np.zeros(args.segments + 1), np.zeros(args.segments + 1)))
    outer = np.array([[0., args.height, 0.], [length, args.height, 0.]])
    left = np.linspace(source[0], outer[0], args.left_nodes)
    right = np.linspace(source[-1], outer[-1], args.right_nodes)
    tolerance = 128 * np.finfo(float).eps * max(length, args.height)
    timings, trials = [], []
    original_builder = _line.build_unit_patch
    for iteration in range(args.warmups + args.repeats):
        trial_count = 0

        def counted(*values, **kwargs):
            nonlocal trial_count
            trial_count += 1
            return original_builder(*values, **kwargs)

        def timed_out(*_):
            raise TimeoutError("LINE benchmark exceeded its time limit")

        started = perf_counter()
        previous_handler = None
        if args.timeout_seconds is not None:
            previous_handler = signal.signal(signal.SIGALRM, timed_out)
            signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        try:
            with patch.object(_line, "build_unit_patch", counted):
                candidates = _line.plan_line(source, outer, left, right, args.target_size,
                                             tolerance, best_only=True)
            elapsed = perf_counter() - started
        except TimeoutError:
            print(json.dumps({
                "case": "adapter_line", "completed": False,
                "sourceSegments": args.segments, "length": length,
                "height": args.height, "targetSize": args.target_size,
                "leftNodes": args.left_nodes, "rightNodes": args.right_nodes,
                "seconds": perf_counter() - started, "unitTrials": trial_count,
                "timeoutSeconds": args.timeout_seconds,
            }), flush=True)
            raise SystemExit(1)
        finally:
            if args.timeout_seconds is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
        if iteration >= args.warmups:
            timings.append(elapsed)
            trials.append(trial_count)
        if not candidates:
            raise AssertionError("the requested section has no feasible unit plan")
        selected = min(candidates, key=lambda result: (result.size_error, result.quality))
        np.testing.assert_array_equal(selected.nodes[selected.source_ids], source)
        assert len(selected.units) == min(args.segments, len(selected.outer_ids) - 1)
        assert selected.diagnostics["unit_trial_count"] == trial_count
        assert trial_count <= 8 * sum(
            min(args.segments, count) for count in _line.outer_count_candidates(
                args.segments, args.left_nodes, args.right_nodes, length,
                args.target_size, 1))
        assert np.isfinite(quality_score(selected.nodes, selected.quads))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    seconds = median(timings)
    print(json.dumps({
        "case": "adapter_line", "completed": True, "sourceSegments": args.segments,
        "length": length, "height": args.height, "targetSize": args.target_size,
        "leftNodes": args.left_nodes, "rightNodes": args.right_nodes,
        "repetitions": args.repeats, "warmups": args.warmups, "seconds": timings,
        "medianSeconds": round(seconds, 6), "unitTrials": trials,
        "outerSegments": len(selected.outer_ids) - 1, "unitCount": len(selected.units),
        "nodeCount": len(selected.nodes), "quadCount": len(selected.quads),
        "sizeError": selected.size_error, "worstQualityScore": selected.quality,
        "planner": selected.diagnostics["line_planner"],
        "localRepairs": selected.diagnostics["local_repair_count"],
        "peakRssMiB": round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 2),
    }), flush=True)
    if args.maximum_seconds is not None and seconds > args.maximum_seconds:
        raise AssertionError(f"LINE planning took {seconds:.3f}s; limit is {args.maximum_seconds:.3f}s")


if __name__ == "__main__":
    main()
