"""Stable progress event envelopes shared by mesh25d stages."""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

JsonObject = dict[str, Any]
ProgressCallback = Callable[[JsonObject], None]


@dataclass(frozen=True)
class _StageTimer:
    wall_started: float
    cpu_started: float



def _start_stage(
    progress: ProgressCallback | None,
    stage: str,
    message: str,
    *,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
) -> _StageTimer:
    _emit(
        progress,
        {
            "event": "stage.started",
            "stage": stage,
            "current": current,
            "total": total,
            "unit": unit,
            "message": message,
            "data": {},
        },
    )
    return _StageTimer(time.perf_counter(), time.process_time())


def _complete_stage(
    progress: ProgressCallback | None,
    stage: str,
    timer: _StageTimer,
    *,
    data: JsonObject | None = None,
) -> None:
    metrics = dict(data or {})
    metrics.update(
        {
            "wallDurationMs": int(round((time.perf_counter() - timer.wall_started) * 1000)),
            "cpuDurationMs": int(round((time.process_time() - timer.cpu_started) * 1000)),
            "peakRssBytes": _peak_rss_bytes(),
        }
    )
    _emit(
        progress,
        {
            "event": "stage.completed",
            "stage": stage,
            "data": metrics,
        },
    )


def _emit_progress(
    progress: ProgressCallback | None,
    *,
    current: int | None,
    total: int | None,
    unit: str | None,
    message: str,
    data: JsonObject | None = None,
) -> None:
    _emit(
        progress,
        {
            "event": "progress",
            "current": current,
            "total": total,
            "unit": unit,
            "message": message,
            "data": data or {},
        },
    )


def _emit_item_completed(
    progress: ProgressCallback | None,
    *,
    stage: str,
    data: JsonObject,
) -> None:
    _emit(
        progress,
        {
            "event": "item.completed",
            "stage": stage,
            "data": data,
        },
    )


def _emit(progress: ProgressCallback | None, payload: JsonObject) -> None:
    if progress is not None:
        progress(payload)


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024
