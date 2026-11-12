"""Synthetic UF normalization and build_mesh -> CDB timings, excluding startup.

Run from the repository root with PYTHONPATH=src. Geometry generation, deep
copies and correctness checks are outside timers. Baseline/candidate order is
alternated to reduce drift; no timings from failed baseline exports are used.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import numpy as np
import shapely
from process_flow_kernel.domain.process_geometry_state import _underfill_gap_polygons

from mesher.mesh25d import build_mesh
from mesher.mesh25d.geometry import context, normalization
from mesher.process_flow.exporters.cdb import write_cdb_text
from mesher.tolerance import GeometryToleranceError, resolve_tolerance


def fixture(kind, side):
    if kind == "regular":
        boxes = [(x * 12, y * 12, x * 12 + 10, y * 12 + 10)
                 for x in range(side) for y in range(side)]
        gap = 3
    else:
        motif = [(0, 0, 6, 6), (0, 10, 6, 16), (10, 10, 16, 16),
                 (10, 20, 16, 26), (20, 0, 26, 6), (20, 20, 26, 26)]
        boxes = [(x0 + x * 48, y0 + y * 48, x1 + x * 48, y1 + y * 48)
                 for x in range(side) for y in range(side) for x0, y0, x1, y1 in motif]
        gap = 14
    loops = _underfill_gap_polygons([dict(type="BOX", dim=list(bounds)) for bounds in boxes], gap)
    geometry = dict(type="PolygonGeometry", polys=[[[x, y, 4] for x, y in loop] for loop in loops], thk=6)
    extent = max(max(bounds) for bounds in boxes) + 2
    root = {"bodies": [
        {"material": "base", "geometry": dict(type="BoxGeometry", bottom_left=[-2, -2, 0],
                                               top_right=[extent, extent, 0], thk=4)},
        {"material": "UF", "geometry": geometry},
    ], "children": [], "bumps": [], "vias": [], "circuits": []}
    settings = dict(schemaVersion="1.1.0", unitSystem="um", mesher="process_flow_2_5d",
                    globalElementSize=4, symmetry="full", geometryTolerance=.001, controls=[])
    # This benchmark isolates UF complexity; source die bodies are unnecessary.
    return {"root": root}, settings, dict(
        case=kind, side=side, loops=len(loops), vertices=sum(map(len, loops)),
        contact_points=side * side if kind == "tangent" else 0,
    )


def load_baseline(ref):
    source = subprocess.check_output([
        "git", "show", f"{ref}:src/mesher/mesh25d/geometry/normalization.py",
    ], text=True)
    module = ModuleType("mesher.mesh25d.geometry._benchmark_baseline")
    module.__package__ = "mesher.mesh25d.geometry"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:normalization.py", "exec"), module.__dict__)
    return module


def summary(samples):
    return dict(median_ms=median(samples), p95_ms=float(np.percentile(samples, 95)), samples_ms=samples)


def measure(model, settings, variants, repeats, output, *, include_export):
    stages = ("normalization", "export") if include_export else ("normalization",)
    samples = {name: {stage: [] for stage in stages} for name in variants}
    for run in range(repeats + 2):
        order = list(variants) if run % 2 else list(reversed(variants))
        for name in order:
            normalize = variants[name].normalize_geometry
            root = deepcopy(model["root"])
            start = perf_counter()
            normalize(root)
            elapsed = (perf_counter() - start) * 1000
            if run >= 2:
                samples[name]["normalization"].append(elapsed)
            if include_export:
                with patch.object(context, "normalize_geometry", normalize):
                    start = perf_counter()
                    mesh = build_mesh(model, settings)
                    write_cdb_text(output, mesh=mesh)
                    elapsed = (perf_counter() - start) * 1000
                if run >= 2:
                    samples[name]["export"].append(elapsed)
    return {name: {stage: summary(values) for stage, values in stages.items()}
            for name, stages in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--max-export-vertices", type=int, default=600,
                        help="larger fixtures measure normalization only (default: 600)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 10:
        parser.error("at least 10 measured repeats are required")
    baseline = load_baseline(args.baseline_ref)
    report = dict(
        baseline_commit=subprocess.check_output(["git", "rev-parse", args.baseline_ref], text=True).strip(),
        candidate_sources_sha256={path.name: sha256(path.read_bytes()).hexdigest() for path in (
            Path(normalization.__file__), Path(normalization.__file__).parents[1] / "controls.py")},
        python=platform.python_version(), platform=platform.platform(), shapely=shapely.__version__,
        repeats=args.repeat, warmups=2, max_export_vertices=args.max_export_vertices,
        export_includes="build_mesh + CDB write; excludes process startup",
        geometry="synthetic UF-only polygons on a base; no database or saved flows", cases=[],
    )
    with tempfile.TemporaryDirectory() as directory:
        for kind, side in (("regular", 6), ("regular", 12), ("regular", 24), ("regular", 48),
                           ("tangent", 1), ("tangent", 5), ("tangent", 10)):
            model, settings, record = fixture(kind, side)
            original = deepcopy(model)
            topology = normalization._polygon_loop_topology(
                model["root"]["bodies"][1]["geometry"]["polys"], resolve_tolerance(.001), "benchmark.UF")
            assert len(topology.contacts) == record["contact_points"]
            variants = {"candidate": normalization}
            if kind == "regular":
                expected = baseline.normalize_geometry(deepcopy(model["root"]))
                assert normalization.normalize_geometry(deepcopy(model["root"])) == expected
                variants["baseline"] = baseline
            else:
                try:
                    baseline.normalize_geometry(deepcopy(model["root"]))
                except GeometryToleranceError as error:
                    assert error.context["reason"] == "polygon_loop_contact"
                    record["baseline_status"] = "rejects polygon_loop_contact; not timed"
                else:
                    raise AssertionError("expected the baseline to reject point tangency")
            include_export = record["vertices"] <= args.max_export_vertices
            record["export_status"] = "timed" if include_export else "not timed: exceeds export vertex limit"
            record["timings"] = measure(model, settings, variants, args.repeat, Path(directory) / "mesh.cdb",
                                        include_export=include_export)
            assert model == original
            if kind == "regular":
                old = record["timings"]["baseline"]["normalization"]["median_ms"]
                new = record["timings"]["candidate"]["normalization"]["median_ms"]
                record["normalization_increase_ms"] = new - old
                record["normalization_limit_ms"] = max(.05 * old, 2)
                record["normalization_budget_pass"] = new - old <= record["normalization_limit_ms"]
            report["cases"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
