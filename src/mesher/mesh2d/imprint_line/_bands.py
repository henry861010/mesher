"""Shared band geometry and merging for directional line controls."""

import numpy as np

from ._input import contains_box
from ._types import Band, rectangle


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def limits(domain, band):
    levels = sorted({band.low, *(float(v) for v in domain.loops[0][:, 1 - band.axis]
                                if band.low < v < band.high)})
    sections = []
    for value in levels:
        interval = next((i for i in domain.section_after(band.axis, value)
                         if i.start <= band.start and band.end <= i.end), None)
        if interval is None:
            return (band.start, band.end)
        sections.append(interval)
    return max(i.start for i in sections), min(i.end for i in sections)


def envelope(band, margin):
    return rectangle(band.axis, band.fronts[0] - margin, band.fronts[1] + margin,
                     band.low, band.high)


def merge_bands(bands, domain, margin=0):
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(bands):
            for j in range(i + 1, len(bands)):
                second = bands[j]
                if first.axis != second.axis or not overlap(envelope(first, margin), envelope(second, margin)):
                    continue
                merged = Band(first.axis, min(first.start, second.start), max(first.end, second.end),
                              min(first.low, second.low), max(first.high, second.high),
                              np.unique(np.r_[first.coordinates, second.coordinates]),
                              [min(first.fronts[0], second.fronts[0]), max(first.fronts[1], second.fronts[1])],
                              tuple(sorted(first.sources + second.sources)))
                if not contains_box(domain, merged.bounds):
                    continue  # Concave exteriors can separate otherwise overlapping envelopes.
                bands[i] = merged
                bands.pop(j)
                changed = True
                break
            if changed:
                break
    bands.sort(key=lambda b: (b.axis, b.start, b.end, b.low, b.high, tuple(b.coordinates)))
