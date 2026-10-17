"""Benchmark full circle imprint cost against the original global pipeline.

Run with venv/bin/python examples/failures/imprint_v2_holes/benchmark.py --verify.
Workers isolate traced peak allocations from timing; mesh construction is
excluded for both algorithms, but every operation-internal copy is included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from mesher import Mesh2D
from mesher.mesh2d.circular.imprint_v2.local import _run_local_imprint
from mesher.mesh2d.circular.imprint_v2.utils import (
    _to_circle, _remove_redundant_element, _imprint_circle,
)

ARGUMENTS = dict(center=[158., 158.], radius=60., projection_tolerance=.2,
                 guide_tolerance=1., merge_tolerance=.002, minimum_area=.0004)


def grid(size):
    x, y = np.meshgrid(np.arange(size+1, dtype=float), np.arange(size+1, dtype=float))
    lower = np.arange(size*(size+1), dtype=np.int32).reshape(size, size+1)[:, :-1].ravel()
    return Mesh2D(np.column_stack((x.ravel(), y.ravel())), np.column_stack((lower, lower+1, lower+size+2, lower+size+1)))


def legacy(mesh, stats):
    started = time.perf_counter()
    working = Mesh2D(mesh.nodes, mesh.elements)
    for name, function, arguments in (
        ('projection_seconds', _to_circle, ([158., 158.], 60., .2, 1., None)),
        ('cleanup_seconds', _remove_redundant_element, (.002, .0004)),
        ('split_seconds', _imprint_circle, ([158., 158.], 60., .002)),
    ):
        phase_started = time.perf_counter()
        function(working, *arguments)
        stats[name] = time.perf_counter() - phase_started
    mesh.replace_data(nodes=working.nodes, elements=working.elements)
    stats['total_seconds'] = time.perf_counter() - started
    stats['patch_elements'] = stats['global_elements'] = stats.pop('_input_elements')


def geometry_digest(mesh):
    """Compare actual cell geometry independently of compact node/element IDs."""
    coordinates = np.round(mesh.nodes, 11)
    order = np.lexsort(coordinates.T[::-1])
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    elements = np.sort(ranks[mesh.elements], axis=1)
    elements = elements[np.lexsort(elements.T[::-1])]
    digest = hashlib.sha256(coordinates[order].tobytes() + elements.tobytes()).hexdigest()
    return digest


def worker(algorithm, size, repeats, memory):
    base = grid(size)
    function = (lambda mesh, stats: _run_local_imprint(mesh, **ARGUMENTS, stats=stats)) if algorithm == 'local' else legacy
    samples, phase_samples = [], []
    result = {}
    for iteration in range(1 if memory else repeats+1):
        mesh = Mesh2D(base.nodes, base.elements)
        stats = {'_input_elements': base.element_count}
        if memory:
            tracemalloc.start()
        started = time.perf_counter()
        function(mesh, stats)
        elapsed = time.perf_counter() - started
        stats.pop('_input_elements', None)
        if memory:
            result['peak_traced_bytes'] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        elif iteration:
            samples.append(elapsed)
            phase_samples.append(stats)
    if not memory:
        result.update(median_seconds=statistics.median(samples), samples_seconds=samples,
                      stages={key: statistics.median(sample[key] for sample in phase_samples)
                              for key in phase_samples[0] if key.endswith('_seconds')},
                      work={key: phase_samples[0][key] for key in ('global_elements', 'patch_elements', 'patch_nodes') if key in phase_samples[0]},
                      geometry_digest=geometry_digest(mesh))
    return result


def launch(algorithm, size, repeats, memory=False):
    args = [sys.executable, str(Path(__file__).resolve()), '--worker', algorithm,
            '--size', str(size), '--repeats', str(repeats)]
    if memory:
        args.append('--memory')
    return json.loads(subprocess.check_output(args, text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('legacy', 'local'))
    parser.add_argument('--size', type=int, default=316)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.size < 219 or args.repeats < 1:
        parser.error('size must be >=219 and repeats must be positive')
    if args.worker:
        print(json.dumps(worker(args.worker, args.size, args.repeats, args.memory)))
        return 0
    report = {'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
              'method': 'one warm-up, median of five runs by default; separate worker for peak traced algorithm allocations; full mesh construction excluded',
              'parameters': ARGUMENTS}
    for algorithm in ('legacy', 'local'):
        report[algorithm] = launch(algorithm, args.size, args.repeats)
        report[algorithm].update(launch(algorithm, args.size, args.repeats, memory=True))
    report['speedup'] = report['legacy']['median_seconds'] / report['local']['median_seconds']
    report['geometry_matches'] = report['legacy']['geometry_digest'] == report['local']['geometry_digest']
    report['scaling'] = [{'size': size, **launch('local', size, args.repeats)} for size in sorted({224, args.size, 448})]
    errors = []
    if report['speedup'] < 3:
        errors.append('full-operation speedup is below 3x')
    if not report['geometry_matches']:
        errors.append('local and global reference geometry differ')
    if len({(row['work']['patch_elements'], row['work']['patch_nodes']) for row in report['scaling']}) != 1:
        errors.append('remote grid growth increased local geometry work')
    if report['local']['work']['patch_elements'] >= .05 * report['local']['work']['global_elements']:
        errors.append('reference patch is not below 5% of the global grid')
    report['verification_errors'] = errors
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)
    return 1 if args.verify and errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
