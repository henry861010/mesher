"""Corner-specific unit assembly on fixed source and shared seam chains."""
from __future__ import annotations

import heapq

import numpy as np

from ._limits import (ElementLimitExceeded, bounded_outer_counts,
                      outer_segment_counts)
from ._unit_plan import SectionPatch, assemble_section
from ._patch import _four_sided_units


def _join(*chains: np.ndarray) -> np.ndarray:
    return np.vstack([chains[0], *(chain[1:] for chain in chains[1:])])


def _outer_candidates(outer, source_segments, left_count, right_count, h, tol,
                      max_added_elements=None, limit_errors=None):
    """Two nearest parity-compatible perimeter counts, retaining every anchor."""
    lengths = np.linalg.norm(np.diff(outer[:, :2], axis=0), axis=1)
    if np.any(lengths <= tol):
        raise ValueError("corner outer boundary contains a zero-length interval")
    minimum = len(lengths)
    counts = outer_segment_counts(source_segments, left_count, right_count,
                                  float(lengths.sum()), h, minimum)
    counts, rejected = bounded_outer_counts(
        counts, source_segments, left_count, right_count, max_added_elements)
    if rejected is not None:
        if limit_errors is None:
            raise rejected
        limit_errors.append(rejected)
    if not len(lengths):
        if counts:
            yield outer.copy(), 0.0
        return
    candidates = []
    for total in counts:
        parts = np.ones(len(lengths), dtype=int)

        def cost(index, count):
            return count * (lengths[index] / (count * h) - 1.0) ** 2

        heap = [(cost(i, 2) - cost(i, 1), i) for i in range(len(lengths))]
        heapq.heapify(heap)
        for _ in range(total - minimum):
            _, i = heapq.heappop(heap)
            parts[i] += 1
            n = int(parts[i])
            heapq.heappush(heap, (cost(i, n + 1) - cost(i, n), i))
        points = [outer[0]]
        for a, b, count in zip(outer, outer[1:], parts):
            points.extend(
                np.r_[a[:2] + (b[:2] - a[:2]) * (i / count), 0.0]
                for i in range(1, count)
            )
            points.append(b)
        error = sum(cost(i, int(count)) for i, count in enumerate(parts)) / total
        candidates.append((error, total, np.asarray(points)))
    for error, _, points in sorted(candidates, key=lambda item: item[:2]):
        yield points, float(error)


def _outer_sides(source, outer, left, right, split):
    if len(source) != 1:
        raise ValueError("an OUTER corner must have one source corner node")
    return (right, outer[split:][::-1], outer[:split + 1][::-1], left[::-1])


def _inner_sides(source_chains, outer, left, right, split, kind):
    incoming, outgoing = source_chains
    left_arm, right_arm = outer[:split + 1], outer[split:]
    expected = {"INNER": (False, False), "INNER_LEFT": (True, False),
                "INNER_RIGHT": (False, True), "INNER_BOTH": (True, True)}[kind]
    if (len(left_arm) > 1, len(right_arm) > 1) != expected:
        raise ValueError("corner outer arms disagree with its section kind")
    # Each patterned arm is retained on the corresponding unit boundary.
    # Together with its seam it forms a possibly bent logical side.
    return (incoming, outgoing, _join(right, right_arm[::-1]),
            _join(left_arm[::-1], left[::-1]))


def plan_corner(kind: str, source_chains: tuple[np.ndarray, ...], outer: np.ndarray,
                left: np.ndarray, right: np.ndarray, corner_front: np.ndarray,
                h: float, tol: float, *,
                max_added_elements: int | None = None) -> list[SectionPatch]:
    """Produce feasible concrete patches for one fixed seam-state pair."""
    if kind not in {"OUTER", "INNER", "INNER_LEFT", "INNER_RIGHT", "INNER_BOTH"}:
        raise ValueError("unknown corner section kind")
    source = _join(*source_chains)
    count = len(source) - 1
    results = []
    limit_errors = []
    for sampled, error in _outer_candidates(outer, count, len(left), len(right), h, tol,
                                            max_added_elements, limit_errors):
        matches = np.flatnonzero(np.linalg.norm(sampled[:, :2] - corner_front[:2], axis=1) <= tol)
        if len(matches) != 1:
            raise ValueError("corner front point must remain a mandatory anchor")
        split = int(matches[0])
        try:
            sides = (_outer_sides(source, sampled, left, right, split) if kind == "OUTER"
                     else _inner_sides(source_chains, sampled, left, right, split, kind))
            units = _four_sided_units(sides, tol, max_added_elements)
            results.append(assemble_section(units, source, sampled, left, right, error, tol))
        except ElementLimitExceeded as error:
            limit_errors.append(error)
        except ValueError:
            continue
    if not results and limit_errors:
        raise limit_errors[0]
    return results
