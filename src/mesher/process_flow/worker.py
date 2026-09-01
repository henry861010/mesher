from __future__ import annotations

import json
import os
import resource
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

from ..mesh3d import Mesh3D
from .exporters import write_cdb_text
from .pipeline import build_mesh_from_structure

PROGRESS_PREFIX = "PROCESS_FLOW_PROGRESS "


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in {3, 4}:
        print(
            "Usage: python -m mesher.process_flow.worker "
            "<geometry-structure-json> <element-size> <output-cdb> "
            "[symmetry]",
            file=sys.stderr,
        )
        return 2

    input_path, element_size, output_path = args[:3]
    symmetry = args[3] if len(args) == 4 else "full"
    try:
        geometry_structure = json.loads(Path(input_path).read_text(encoding="utf-8"))
        # Keep stdout machine-readable if a downstream callback emits diagnostics.
        with redirect_stdout(sys.stderr):
            mesh = build_mesh_from_structure(
                geometry_structure,
                element_size=float(element_size),
                symmetry=symmetry,
                progress=_emit_progress,
            )
        timer = _start_stage(
            "writing_output",
            "Writing CDB output.",
            current=0,
            total=(
                mesh.node_count
                + mesh.element_count
                + len(mesh.element_comps)
                + mesh.component_count
            ),
            unit="records",
        )
        metadata = _write_cdb_atomically(
            Path(output_path),
            mesh=mesh,
            progress=_emit_progress,
        )
        output_bytes = Path(output_path).stat().st_size
        _complete_stage(
            "writing_output",
            timer,
            data={**metadata, "outputBytes": output_bytes},
        )
        _emit_progress(
            {
                "event": "output.summary",
                "stage": "writing_output",
                "data": {**metadata, "outputBytes": output_bytes},
            }
        )
        print(json.dumps(metadata, separators=(",", ":")), flush=True)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    return 0


def _write_cdb_atomically(
    output_path: Path,
    *,
    mesh: Mesh3D,
    progress: Callable[[dict[str, Any]], None] | None,
) -> dict[str, object]:
    """Write beside the destination and publish only a complete CDB file."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        metadata = write_cdb_text(
            temporary_path,
            mesh=mesh,
            progress=progress,
        )
        os.replace(temporary_path, output_path)
        temporary_path = None
        return {**metadata, "outputPath": str(output_path)}
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _emit_progress(payload: dict[str, object]) -> None:
    print(
        PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def _start_stage(
    stage: str,
    message: str,
    *,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
) -> tuple[float, float]:
    _emit_progress(
        {
            "event": "stage.started",
            "stage": stage,
            "current": current,
            "total": total,
            "unit": unit,
            "message": message,
            "data": {},
        }
    )
    return time.perf_counter(), time.process_time()


def _complete_stage(
    stage: str,
    timer: tuple[float, float],
    *,
    data: dict[str, object],
) -> None:
    wall_started, cpu_started = timer
    metrics = dict(data)
    metrics.update(
        {
            "wallDurationMs": int(round((time.perf_counter() - wall_started) * 1000)),
            "cpuDurationMs": int(round((time.process_time() - cpu_started) * 1000)),
            "peakRssBytes": _peak_rss_bytes(),
        }
    )
    _emit_progress(
        {
            "event": "stage.completed",
            "stage": stage,
            "data": metrics,
        }
    )


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


if __name__ == "__main__":
    raise SystemExit(main())
