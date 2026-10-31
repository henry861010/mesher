"""Small mesh fixtures shared by editor API tests."""

import numpy as np

from mesher import Mesh2D


def two_quads():
    return Mesh2D(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0],
         [0, 1, 0], [1, 1, 0], [2, 1, 0]],
        [[0, 1, 4, 3], [1, 2, 5, 4]],
    )



def empty_mesh():
    return Mesh2D(np.empty((0, 3)), np.empty((0, 4), dtype=np.int32))

