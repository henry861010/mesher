"""Reservation boundary ordering across Python/NumPy versions."""
import operator
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.generators._holed_geometry import OrthogonalDomain
from mesher.mesh2d.imprint_line._background import ReservedDomain
from mesher.mesh2d.imprint_line._types import Transition


def legacy_sorted(values, *, reverse=False):
    # Python 3.11's sort parses reverse through the integer-index protocol.
    return sorted(values, reverse=operator.index(reverse))


class ReservedDomainTests(unittest.TestCase):
    def test_split_points_follow_both_boundary_directions(self):
        outer = OrthogonalDomain(((0, 0), (10, 0), (10, 10), (0, 10)))
        expected = (
            ((2, 2), (8, 2), (8, 3), (8, 6), (8, 8), (2, 8), (2, 6), (2, 3)),
            ((2, 2), (3, 2), (6, 2), (8, 2), (8, 8), (6, 8), (3, 8), (2, 8)),
        )
        for axis in (0, 1):
            for sorter in (sorted, legacy_sorted):
                with self.subTest(axis=axis, sorter=sorter.__name__):
                    transition = Transition(
                        band=0, axis=axis, side=0, bounds=(2, 2, 8, 8),
                        splits=np.array([2., 3., 6., 8.]),
                    )
                    with warnings.catch_warnings(), patch(
                        "mesher.mesh2d.imprint_line._background.sorted", sorter, create=True
                    ):
                        warnings.simplefilter("error", DeprecationWarning)
                        domain = ReservedDomain(outer, [transition])
                    np.testing.assert_array_equal(domain.loops[1], expected[axis])


if __name__ == "__main__":
    unittest.main()
