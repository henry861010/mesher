"""Compare adapter correctness fixes with a saved working-tree baseline.

Run serially on the same machine (fixture preparation is outside the timer)::

    PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_adapter_regressions.py \
        --repo /tmp/mesher_adapter_baseline > /tmp/adapter_before.jsonl
    PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_adapter_regressions.py \
        --compare-to /tmp/adapter_before.jsonl > /tmp/adapter_after.jsonl

Each case has one warmup and five measured runs. Successful cases must stay within 5% of the baseline median. Element
and trial counts are reported alongside their changes from the baseline. Fixed failures have a
one-second target, checked separately from successful cases. The optional
--fixed-failures flag exercises those failures; --timeout caps each failure
benchmark run only, never execution in the library. Fixture helpers require the source tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--compare-to', type=Path)
    parser.add_argument('--fixed-failures', action='store_true')
    parser.add_argument('--timeout', type=float, default=20.)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    sys.path[:0] = [str(args.repo / 'src'), str(args.repo)]
    import numpy as np
    from mesher import Mesh2D
    from mesher.mesh2d.adapter import (
        QuadLayerError, generate_layer_adaptive_mesh as layer,
        generate_full_adaptive_mesh as full, plan_quad_sections as plan,
    )
    from mesher.mesh2d.adapter.layer import _mesher
    from mesher.mesh2d.adapter.layer._construction import line
    from mesher.mesh2d.editor import EditorMesh2D
    from mesher.mesh2d.generators import generate_rectilinear_mesh
    from tests.unit.mesh2d.test_dense_sections import rectangle
    from tests.unit.mesh2d.test_slot_repair import _c_slot
    from examples.full_adaptive_mesh import build_fixture

    left = generate_rectilinear_mesh(5, [0, 100], [0, 100])
    right = generate_rectilinear_mesh(5, [100, 200], [0, 200])
    source = EditorMesh2D.equivalence(EditorMesh2D.merge(left, right))
    outline = EditorMesh2D.outlines(source)[0]
    dense, dense_ids = rectangle()
    slot, slot_ids = _c_slot()
    focus, outer = build_fixture()
    square = Mesh2D([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
    v = np.linspace(-1, 2, 5)
    target = np.vstack((np.column_stack((np.full(4, -1), v[:-1])),
                        np.column_stack((v[:-1], np.full(4, 2))),
                        np.column_stack((np.full(4, 2), v[:0:-1])),
                        np.column_stack((v[:0:-1], np.full(4, -1)))))
    square.nodes = np.vstack((square.nodes, np.column_stack((target, np.zeros(len(target))))))
    cases = [
        ('layer', source, lambda m: layer(m, outline, 50, expand_ratio=1.5)),
        ('dense', dense, lambda m: layer(m, dense_ids, 1000, small_element_size=10,
                                         max_transition_count=1000)),
        ('repair', slot, lambda m: layer(m, slot_ids, .5)),
        ('full', focus, lambda m: full(m, [.25, .5], outer_outline=outer)),
        ('fixed', square, lambda m: layer(m, [0, 3, 2, 1], 1,
                                         outline_o=np.arange(4, len(square.nodes)),
                                         fixed_outer_boundary=True, repair_slots=False)),
    ]
    baseline = ({record['case']: record for text in args.compare_to.read_text().splitlines()
                 if (record := json.loads(text))} if args.compare_to else {})
    failed = []

    def trial_count(diagnostics):
        if 'line_unit_trial_count' in diagnostics:
            return diagnostics['line_unit_trial_count']
        return sum(trial_count(boundary) for layer in diagnostics.get('layers', ())
                   for boundary in layer)

    def emit(name, times, **details):
        record = dict(case=name, median=median(times), times=times, **details)
        if name in baseline and 'elements' in record:
            old = baseline[name]
            record['relative_time'] = record['median'] / old['median']
            record['element_delta'] = median(record['elements']) - median(old['elements'])
            record['trial_delta'] = median(record['trials']) - median(old['trials'])
            if record['relative_time'] > 1.05:
                failed.append(name)
        print(json.dumps(record), flush=True)

    for name, fixture, operation in cases:
        times, elements, trials, validations, searches = [], [], [], [], []
        original_validator = _mesher._validate_patch
        for iteration in range(6):
            mesh = Mesh2D(fixture.nodes.copy(), fixture.elements.copy())
            count = 0

            def validate(*values):
                nonlocal count
                count += 1
                return original_validator(*values)

            with patch.object(_mesher, '_validate_patch', validate):
                started = perf_counter()
                result = operation(mesh)
                elapsed = perf_counter() - started
            np.testing.assert_array_equal(mesh.nodes[:len(fixture.nodes)], fixture.nodes)
            np.testing.assert_array_equal(mesh.elements[:len(fixture.elements)], fixture.elements)
            if iteration:
                times.append(elapsed)
                elements.append(len(mesh.elements))
                trials.append(trial_count(result.diagnostics))
                validations.append(count)
                searches.append(result.diagnostics.get('corner_search'))
        emit(name, times, elements=elements, trials=trials, validations=validations, searches=searches)

    positions = np.arange(129, dtype=float)
    source = np.column_stack((positions, np.zeros((129, 2))))
    outer = np.array([[0., 8., 0.], [128., 8., 0.]])
    left, right = np.array([source[0], outer[0]]), np.array([source[-1], outer[-1]])
    times, elements, trials = [], [], []
    for iteration in range(6):
        diagnostics = {}
        started = perf_counter()
        patches = line.plan_line(source, outer, left, right, 8., 1e-10,
                                 best_only=True, _diagnostics=diagnostics, max_transition_count=10)
        elapsed = perf_counter() - started
        assert patches
        if iteration:
            times.append(elapsed)
            elements.append(len(patches[0].quads))
            trials.append(diagnostics['unit_trial_count'])
    emit('line', times, elements=elements, trials=trials)

    if args.fixed_failures:
        def expired(*_):
            raise TimeoutError('benchmark deadline')

        previous = signal.signal(signal.SIGALRM, expired)
        try:
            for n in (8, 12):
                v = np.linspace(-1, 2, n + 2)
                target = np.vstack(([-1, -1], np.column_stack((np.full(n + 1, -1), v[1:])),
                                    np.column_stack((v[1:], np.full(n + 1, 2))),
                                    np.column_stack((np.full(n + 1, 2), v[-2::-1]))))
                times, visits, outcomes = [], [], []
                for iteration in range(6):
                    mesh = Mesh2D([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
                    mesh.nodes = np.vstack((mesh.nodes, np.column_stack((target, np.zeros(len(target))))))
                    started = perf_counter()
                    signal.setitimer(signal.ITIMER_REAL, args.timeout)
                    try:
                        plan(mesh, [0, 3, 2, 1], 1, outline_o=np.arange(4, len(mesh.nodes)),
                             fixed_outer_boundary=True, repair_slots=False)
                    except QuadLayerError as error:
                        outcome = error.code
                        search = error.context.get('corner_search', {})
                    except TimeoutError:
                        outcome, search = 'benchmark_timeout', {}
                    else:
                        raise AssertionError('impossible target accepted')
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    if iteration:
                        times.append(perf_counter() - started)
                        visits.append(search.get('candidate_visits'))
                        outcomes.append(outcome)
                name = f'fixed_failure_{len(target)}'
                emit(name, times, candidate_visits=visits, outcomes=outcomes)
                if args.compare_to and (median(times) >= 1 or any(v != 0 for v in visits)):
                    failed.append(name)
        finally:
            signal.signal(signal.SIGALRM, previous)
    if failed:
        raise SystemExit('Performance regression: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
