"""Measure projection-based unit construction for one complete LINE section.

Run from the repository root::

    venv/bin/python benchmarks/mesh2d/benchmark_adapter_line.py --timeout-seconds 30
    venv/bin/python benchmarks/mesh2d/benchmark_adapter_line.py --segments 128 --dense-start 0 --dense-segments 64 --dense-spacing-ratio .6 --target-size 1 --height 1
    venv/bin/python benchmarks/mesh2d/benchmark_adapter_line.py --segments 24 --length 2100 --dense-segments 20 --target-size 500 --height 500

Fixture creation and result validation are outside the timer. Output is one
JSON object containing wall time, unit trial count, result size and quality.
Use --repo for a baseline checkout. unitTrials counts calls to the generic unit
builder; directQuadCount counts the validated one-cell patches on the projection
path. medianPlanningSeconds measures front and cut selection before building
units; the baseline interleaves cut planning with construction. Repeat with
--segments 32, 128, and 512 to compare long sections at a fixed target size.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import resource
import signal
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--segments", type=int, default=128)
    parser.add_argument("--length", type=float, default=None,
                        help="Section length; defaults to the source segment count.")
    parser.add_argument("--height", type=float, default=8.0)
    parser.add_argument("--target-size", type=float, default=8.0)
    parser.add_argument("--max-transition-count", type=int, default=10)
    parser.add_argument("--min-size-ratio", type=float, default=0.5)
    parser.add_argument("--max-size-ratio", type=float, help="Override this checkout's default LINE upper size ratio.")
    parser.add_argument("--dense-segments", type=int, default=0,
                        help="Number of source intervals with reduced spacing.")
    parser.add_argument("--dense-start", type=int, help="First reduced interval (centered by default).")
    parser.add_argument("--dense-spacing-ratio", type=float, default=0.01,
                        help="Dense interval length divided by ordinary interval length.")
    parser.add_argument("--left-nodes", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--right-nodes", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--maximum-seconds", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None,
                        help="Interrupt a slow run and emit partial JSON (requires POSIX interval timers).")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    from mesher.mesh2d.adapter.layer._construction import line
    from mesher.mesh2d.adapter.layer._construction.sampling import LineSampling
    from mesher.mesh2d.adapter.layer._construction.units import quality_score

    projected = hasattr(line, "plan_resampled")
    if projected:
        from mesher.mesh2d.adapter.layer._construction import strip as builder_module
        max_ratio = 1.25 if args.max_size_ratio is None else args.max_size_ratio
        sizing = {"max_size_ratio": max_ratio}
    else:
        if args.max_size_ratio not in (None, 1.):
            parser.error("the baseline checkout has a fixed maximum size ratio of 1")
        builder_module, max_ratio, sizing = line, 1., {}

    length = float(args.segments) if args.length is None else args.length
    values = [length, args.height, args.target_size]
    if (args.segments < 1 or args.repeats < 1 or args.warmups < 0
            or any(not np.isfinite(v) or v <= 0 for v in values)):
        parser.error("segment count, repeats, lengths and target size must be positive and finite")
    if args.max_transition_count < 1:
        parser.error("maximum transition count must be a positive integer")
    if not np.isfinite(args.min_size_ratio) or not 0 < args.min_size_ratio <= 1:
        parser.error("minimum size ratio must be finite and in (0, 1]")
    if not np.isfinite(max_ratio) or max_ratio < 1:
        parser.error("maximum size ratio must be finite and at least one")
    if not 0 <= args.dense_segments <= args.segments:
        parser.error("dense segment count must be between zero and the source segment count")
    if not np.isfinite(args.dense_spacing_ratio) or not 0 < args.dense_spacing_ratio <= 1:
        parser.error("dense spacing ratio must be finite and in (0, 1]")
    if args.maximum_seconds is not None and (not np.isfinite(args.maximum_seconds) or args.maximum_seconds <= 0):
        parser.error("maximum seconds must be positive and finite")
    if args.timeout_seconds is not None:
        if (not np.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
                or not hasattr(signal, "setitimer")):
            parser.error("timeout requires a positive duration and POSIX interval timers")
    weights = np.ones(args.segments)
    dense_start = args.dense_start
    if dense_start is None:
        dense_start = (args.segments - args.dense_segments) // 2
    if not 0 <= dense_start <= args.segments - args.dense_segments:
        parser.error("dense interval run must fit within the source")
    weights[dense_start:dense_start + args.dense_segments] = args.dense_spacing_ratio
    positions = np.r_[0., np.cumsum(weights)]
    positions *= length / positions[-1]
    positions[-1] = length
    source = np.column_stack((positions,
                              np.zeros(args.segments + 1), np.zeros(args.segments + 1)))
    outer = np.array([[0., args.height, 0.], [length, args.height, 0.]])
    left = np.linspace(source[0], outer[0], args.left_nodes)
    right = np.linspace(source[-1], outer[-1], args.right_nodes)
    tolerance = 128 * np.finfo(float).eps * max(length, args.height)
    inputs = (source, outer, left, right)
    snapshots = tuple(value.copy() for value in inputs)
    sampler = LineSampling(source, outer, args.target_size, tolerance,
                           args.max_transition_count, args.min_size_ratio, **sizing)
    metadata = {
        "sourceSegments": args.segments, "length": length,
        "height": args.height, "targetSize": args.target_size,
        "leftNodes": args.left_nodes, "rightNodes": args.right_nodes,
        "maxTransitionCount": args.max_transition_count, "minSizeRatio": args.min_size_ratio,
        "maxSizeRatio": max_ratio,
        "denseSegments": args.dense_segments, "denseSpacingRatio": args.dense_spacing_ratio,
        "denseStartSegment": dense_start if args.dense_segments else None,
    }
    timings, trials, direct_counts, plan_times = [], [], [], []
    original_builder = builder_module.build_unit_patch
    for iteration in range(args.warmups + args.repeats):
        trial_count = 0
        planning_seconds = 0.

        def counted(*values, **kwargs):
            nonlocal trial_count
            trial_count += 1
            return original_builder(*values, **kwargs)

        def timed_plans(*values, **kwargs):
            nonlocal planning_seconds
            iterator = original_planner(*values, **kwargs)
            while True:
                begin = perf_counter()
                try:
                    candidate = next(iterator)
                except StopIteration:
                    return
                finally:
                    planning_seconds += perf_counter() - begin
                yield candidate

        def timed_out(*_):
            raise TimeoutError("LINE benchmark exceeded its time limit")

        started = perf_counter()
        previous_handler = None
        if args.timeout_seconds is not None:
            previous_handler = signal.signal(signal.SIGALRM, timed_out)
            signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        try:
            with ExitStack() as hooks:
                hooks.enter_context(patch.object(builder_module, "build_unit_patch", counted))
                if projected:
                    original_planner = line.plan_resampled
                    hooks.enter_context(patch.object(line, "plan_resampled", timed_plans))
                candidates = line.plan_line(
                    source, outer, left, right, args.target_size, tolerance, best_only=True,
                    max_transition_count=args.max_transition_count,
                    min_size_ratio=args.min_size_ratio, **sizing)
            elapsed = perf_counter() - started
        except TimeoutError:
            print(json.dumps({
                "case": "adapter_line", "completed": False,
                **metadata,
                "seconds": perf_counter() - started, "unitTrials": trial_count,
                "timeoutSeconds": args.timeout_seconds,
            }), flush=True)
            raise SystemExit(1)
        finally:
            if args.timeout_seconds is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
        if not candidates:
            raise AssertionError("the requested section has no feasible unit plan")
        selected = candidates[0]
        if iteration >= args.warmups:
            timings.append(elapsed)
            plan_times.append(planning_seconds)
            trials.append(trial_count)
            direct_counts.append(selected.diagnostics.get("direct_quad_count", 0))
        for actual, snapshot in zip(inputs, snapshots):
            np.testing.assert_array_equal(actual, snapshot)
        np.testing.assert_array_equal(selected.nodes[selected.source_ids], source)
        sampler.validate_sample(selected.nodes[selected.outer_ids])
        for index, unit in enumerate(selected.units):
            source_count, _, outer_count, _ = np.asarray(unit["boundary_counts"]) - 1
            if not projected:
                assert source_count <= args.max_transition_count * outer_count
                continue
            assignments = unit["source_segments_per_outer"]
            assert sum(assignments) == source_count and len(assignments) == outer_count
            assert max(assignments) <= args.max_transition_count
            if index < len(selected.units) - 1:
                assert unit["boundary_counts"][1] == 2
        assert selected.diagnostics["unit_trial_count"] == trial_count
        assert np.isfinite(quality_score(selected.nodes, selected.quads))
    # Reconstruct physical unit cuts for both checkouts, including the older
    # planner whose diagnostics only expose a mean squared shear.
    edge_counts = np.asarray([unit["boundary_counts"] for unit in selected.units])
    cuts = np.cumsum(edge_counts[:, [0, 2]] - 1, axis=0)[:-1]
    delta = selected.nodes[selected.outer_ids[cuts[:, 1]]] - source[cuts[:, 0]]
    angle = float(np.rad2deg(np.arctan2(np.abs(delta[:, 0]), np.abs(delta[:, 1]))).max(initial=0.))
    if projected:
        np.testing.assert_allclose(selected.diagnostics["max_alignment_angle"], angle, atol=1e-10)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    seconds = median(timings)
    print(json.dumps({
        "case": "adapter_line", "completed": True, **metadata,
        "repetitions": args.repeats, "warmups": args.warmups, "seconds": timings,
        "medianSeconds": round(seconds, 6), "unitTrials": trials,
        "directQuadCount": direct_counts,
        "medianPlanningSeconds": median(plan_times) if projected else None,
        "outerSegments": len(selected.outer_ids) - 1, "unitCount": len(selected.units),
        "nodeCount": len(selected.nodes), "quadCount": len(selected.quads),
        "sizeError": selected.size_error, "worstQualityScore": selected.quality,
        "alignmentError": selected.alignment_error,
        "maxActualTransitionCount": selected.diagnostics["max_actual_transition_count"],
        "maxAlignmentAngle": angle,
        "planner": selected.diagnostics["line_planner"],
        "localRepairs": selected.diagnostics["local_repair_count"],
        "peakRssMiB": round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 2),
    }), flush=True)
    if args.maximum_seconds is not None and seconds > args.maximum_seconds:
        raise AssertionError(f"LINE planning took {seconds:.3f}s; limit is {args.maximum_seconds:.3f}s")


if __name__ == "__main__":
    main()
