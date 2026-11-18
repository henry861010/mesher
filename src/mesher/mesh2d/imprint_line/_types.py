"""Input contracts and private planning records for line imprinting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike


class LinePair(TypedDict):
    line1_expand_size: float
    line2_expand_size: float
    direction: Literal["x", "y"]
    line1: ArrayLike
    line2: ArrayLike
    points: ArrayLike


class ImprintLineError(ValueError):
    """An input, planning, construction, or resource-limit failure."""

    def __init__(self, message, *, code, stage, **context):
        self.code = code
        self.context = {"stage": stage, **context}
        super().__init__(f"{message}; code={code}, context={self.context}")


def rectangle(axis, start, end, low, high):
    return ((start, low, end, high) if axis == 0 else (low, start, high, end))


def loop(box):
    a, b, c, d = box
    return np.array(((a, b), (c, b), (c, d), (a, d)), dtype=float)


@dataclass
class Band:
    axis: int
    start: float
    end: float
    low: float
    high: float
    coordinates: np.ndarray
    fronts: list[float]
    sources: tuple[int, ...]
    # Fronts only move outwards. None means that end runs to the domain edge.
    active: list[bool] = field(default_factory=lambda: [True, True])
    splits: list[set[float]] = field(default_factory=lambda: [set(), set()])

    @property
    def bounds(self):
        return rectangle(self.axis, self.start, self.end, self.low, self.high)


@dataclass
class Transition:
    band: int
    axis: int
    side: int
    bounds: tuple[float, float, float, float]
    splits: np.ndarray
    chains: list[np.ndarray] = field(default_factory=list)

    @property
    def inner(self):
        return self.bounds[self.axis + (2 if self.side == 0 else 0)]


@dataclass
class ImprintPlan:
    bands: list[Band]
    transitions: list[Transition]
    rectangles: np.ndarray
    diagnostics: dict
