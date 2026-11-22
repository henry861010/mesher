"""Construct writable-array corruption for tests of diagnostics and atomic errors."""
import numpy as np

from mesher import Mesh2D


def edited_mesh(nodes, elements):
    """Start with valid triangles, then inject the caller's raw array edits."""
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=np.int32)
    seed = np.zeros_like(nodes)
    seed[1, 0] = seed[2, 1] = 1
    mesh = Mesh2D(seed, np.tile([0, 1, 2, 2], (len(elements), 1)))
    mesh.nodes[:, :nodes.shape[1]] = nodes
    mesh.elements[:] = elements
    return mesh


def assert_ccw(mesh):
    xy = mesh.nodes[mesh.elements, :2]
    local = xy - xy[:, :1]
    following = np.roll(local, -1, axis=1)
    area = np.sum(local[..., 0] * following[..., 1] - local[..., 1] * following[..., 0], axis=1)
    np.testing.assert_array_less(0., area)
