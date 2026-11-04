"""Benchmark sparse holed meshes and retained collinear anchors.

Run from the repository root::

    PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_holed_rectilinear.py \
        --output /tmp/holed-current.json

Each repetition runs in a fresh process; the default is three repetitions.
Timings include the public generator call, excluding fixture construction and
post-call assertions. Peak RSS includes Python, NumPy, fixtures, and workspace.
Console output is JSON Lines; --output saves metadata and all samples as JSON.

For a same-host comparison, --source-dir accepts saved holed_rectilinear.py,
_holed_geometry.py, and any _holed_*.py helpers that version uses. They replace
the corresponding modules in a temporary package copy, without checkout edits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter


_ROOT = Path(__file__).resolve().parents[2]
_MODULE_FILES = ("holed_rectilinear.py", "_holed_geometry.py")
_CASES = ("hole_row", "collinear_anchors")


def _source_modules(directory):
    return sorted(set(_MODULE_FILES) | {path.name for path in directory.glob("_holed_*.py")})


def _peak_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def _fixture(case, count, np):
    if case == "hole_row":
        outer = [[0, 0], [3 * count + 1, 0], [3 * count + 1, 3], [0, 3]]
        holes = [
            [[3 * i + 1, 1], [3 * i + 2, 1], [3 * i + 2, 2], [3 * i + 1, 2]]
            for i in range(count)
        ]
        expected_elements = 5 * count + 3
    else:
        outer = np.vstack((
            np.column_stack((np.arange(count + 1), np.zeros(count + 1))),
            [[count, 1], [0, 1]],
        ))
        holes = []
        expected_elements = count
    target = max(100_000.0, float(3 * count + 1))
    return outer, holes, target, expected_elements


def _worker(case, count):
    import numpy as np

    module = importlib.import_module("mesher.mesh2d.generators.holed_rectilinear")
    # Capture the domain without adding per-edge instrumentation to the timed
    # operation. Older source snapshots have no stats and report null instead.
    domain_type = module.OrthogonalDomain
    domains = []

    def make_domain(*args, **kwargs):
        domain = domain_type(*args, **kwargs)
        domains.append(domain)
        return domain

    module.OrthogonalDomain = make_domain
    startup_peak = _peak_mib()
    outer, holes, target, expected_elements = _fixture(case, count, np)
    started = perf_counter()
    result = module.generate_rectilinear_mesh_optimal(target, outer, holes=holes)
    seconds = perf_counter() - started
    peak = _peak_mib()

    assert result.mesh.element_count == expected_elements
    assert len(result.hole_outlines) == len(holes)
    assert all(len(loop) % 2 == 0 for loop in [result.outer_outline, *result.hole_outlines])
    assert result.diagnostics["boundary_node_counts_after"] == [
        len(result.outer_outline), *map(len, result.hole_outlines)
    ]
    digest = hashlib.sha256()
    for values in (result.mesh.nodes, result.mesh.elements,
                   result.outer_outline, *result.hole_outlines):
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes())
    stats = getattr(domains[-1], "stats", None)
    print(json.dumps({
        "seconds": seconds,
        "peakRssMiB": peak,
        "peakAboveStartupMiB": max(0.0, peak - startup_peak),
        "nodeCount": result.mesh.node_count,
        "elementCount": result.mesh.element_count,
        "inputVertexCount": len(outer) + sum(map(len, holes)),
        "geometryStats": stats,
        "meshSha256": digest.hexdigest(),
        "numpyVersion": np.__version__,
    }), flush=True)


def _run(args, source_root, source_files):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(source_root), environment.get("PYTHONPATH", ""),
    )))
    report = {
        "schemaVersion": 1,
        "label": args.label,
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repetitions": args.repeats,
        "sourceSha256": {name: hashlib.sha256((source_files / name).read_bytes()).hexdigest()
                         for name in _source_modules(source_files)},
        "timingScope": "public generator call; fixture construction and assertions excluded",
        "rssScope": "fresh worker process including Python, NumPy, fixtures and workspace",
        "records": [],
    }
    cases = []
    if "hole_row" in args.cases:
        cases.extend(("hole_row", count) for count in args.hole_counts)
    if "collinear_anchors" in args.cases:
        cases.extend(("collinear_anchors", count) for count in args.anchor_segments)
    for case, count in cases:
        samples = []
        for repetition in range(args.repeats):
            completed = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--cases", case, "--worker-count", str(count),
            ], env=environment, capture_output=True, text=True, timeout=args.timeout)
            if completed.returncode:
                raise RuntimeError(
                    f"{case} count={count} failed:\n{completed.stdout}{completed.stderr}"
                )
            sample = json.loads(completed.stdout)
            samples.append(sample)
            print(json.dumps({
                "type": "sample", "label": args.label, "case": case,
                "count": count, "repetition": repetition + 1, **sample,
            }), flush=True)
        hashes = {sample["meshSha256"] for sample in samples}
        if len(hashes) != 1:
            raise AssertionError(f"{case} count={count} produced nondeterministic meshes")
        record = {
            "case": case,
            "count": count,
            "countMeaning": "holes" if case == "hole_row" else "bottom boundary segments",
            "medianSeconds": median(sample["seconds"] for sample in samples),
            "medianPeakRssMiB": median(sample["peakRssMiB"] for sample in samples),
            "maximumPeakRssMiB": max(sample["peakRssMiB"] for sample in samples),
            "nodeCount": samples[0]["nodeCount"],
            "elementCount": samples[0]["elementCount"],
            "inputVertexCount": samples[0]["inputVertexCount"],
            "geometryStats": samples[0]["geometryStats"],
            "meshSha256": samples[0]["meshSha256"],
            "samples": samples,
        }
        report["records"].append(record)
        print(json.dumps({"type": "summary", "label": args.label,
                          **{key: value for key, value in record.items() if key != "samples"}}),
              flush=True)
        if args.output:
            # Save completed cases as the run progresses, including long baselines.
            args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=_CASES, default=list(_CASES))
    parser.add_argument("--hole-counts", nargs="+", type=int, default=[100, 200, 400, 800])
    parser.add_argument("--anchor-segments", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="maximum seconds for each independent worker")
    parser.add_argument("--label", default="current")
    parser.add_argument("--source-dir", type=Path,
                        help="directory containing saved versions of the two generator modules")
    parser.add_argument("--output", type=Path, help="save metadata and samples as one JSON file")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.repeats < 1 or any(count < 1 for count in args.hole_counts + args.anchor_segments)
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("counts, repeats, and timeout must be positive")
    if args.worker:
        if args.worker_count is None or args.worker_count < 1:
            parser.error("worker-count must be positive")
        _worker(args.cases[0], args.worker_count)
        return
    if args.source_dir:
        source_files = args.source_dir.resolve()
        if any(not (source_files / name).is_file() for name in _MODULE_FILES):
            parser.error("source-dir must contain holed_rectilinear.py and _holed_geometry.py")
        with tempfile.TemporaryDirectory(prefix="mesher-holed-benchmark-") as directory:
            source_root = Path(directory)
            package = source_root / "mesher"
            shutil.copytree(_ROOT / "src" / "mesher", package,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in _source_modules(source_files):
                shutil.copy2(source_files / name, package / "mesh2d" / "generators" / name)
            _run(args, source_root, source_files)
    else:
        source_root = _ROOT / "src"
        _run(args, source_root, source_root / "mesher" / "mesh2d" / "generators")


if __name__ == "__main__":
    main()
