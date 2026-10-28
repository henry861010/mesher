"""Small mesh fixtures shared by editor API tests."""

import numpy as np

from mesher import Mesh2D


def two_quads():
    return Mesh2D(
        [[0, 0, 10], [1, 0, 11], [2, 0, 12],
         [0, 1, 13], [1, 1, 14], [2, 1, 15]],
        [[0, 1, 4, 3], [1, 2, 5, 4]],
    )



def empty_mesh():
    return Mesh2D(np.empty((0, 3)), np.empty((0, 4), dtype=np.int32))

