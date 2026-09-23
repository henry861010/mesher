"""Utilities for transforming three-dimensional mesh connectivity."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .model import Mesh3D


_EDGE_CORNERS = np.array(
    [
        [0, 1],  # Q: I-J
        [1, 2],  # R: J-K
        [2, 3],  # S: K-L
        [3, 0],  # T: L-I
        [4, 5],  # U: M-N
        [5, 6],  # V: N-O
        [6, 7],  # W: O-P
        [7, 4],  # X: P-M
        [0, 4],  # Y: I-M
        [1, 5],  # Z: J-N
        [2, 6],  # A: K-O
        [3, 7],  # B: L-P
    ],
    dtype=np.intp,
)
_EDGE_SCAN_CHUNK_SIZE = 100_000
_UINT32_SHIFT = np.uint64(32)
_UINT32_MASK = np.uint64(0xFFFFFFFF)


def add_midside_node(mesh: Mesh3D, element_indices) -> Mesh3D:
    """Convert selected ANSYS 8-node solids to 20-slot connectivity.

    Hexahedron, prism, pyramid, and tetrahedron connectivity is accepted in
    the SOLID185 I-P ordering. The generated midside slots use the SOLID186
    Q-B ordering. Midside nodes are shared by every selected element that
    references the same unordered corner-node pair, and by pre-existing
    20-node elements elsewhere in the mesh.

    Args:
        mesh: Source mesh. It is never modified.
        element_indices: One-dimensional integer element-row selection.
            ``None`` selects every element and an empty sequence is a no-op.

    Returns:
        The original mesh when no selected 8-node element needs conversion;
        otherwise, a new ``Mesh3D`` containing the converted connectivity.

    Raises:
        TypeError: If ``mesh`` or the selection has an invalid type.
        ValueError: If the selection shape, element node counts, topology, or
            pre-existing midside connectivity is invalid.
        IndexError: If a selected element index is out of range.
        OverflowError: If generated node ids cannot fit in ``int32``.
    """
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be a Mesh3D instance")

    selected = _normalize_element_indices(
        mesh.element_count,
        element_indices,
    )
    if selected.size == 0:
        return mesh

    selected_node_counts = mesh.element_node_num[selected]
    supported = (selected_node_counts == 8) | (selected_node_counts == 20)
    if not np.all(supported):
        invalid_index = int(selected[np.flatnonzero(~supported)[0]])
        invalid_count = int(mesh.element_node_num[invalid_index])
        raise ValueError(
            "selected elements must have 8 or 20 nodes; "
            f"element {invalid_index} has {invalid_count}"
        )

    converted_indices = selected[selected_node_counts == 8]
    del selected
    del selected_node_counts
    del supported
    if converted_indices.size == 0:
        return mesh

    _validate_connectivity(mesh)

    corners = mesh.elements[converted_indices, :8]
    _validate_solid185_topologies(corners, converted_indices)

    edge_keys = _encode_edge_keys(corners)
    del corners
    unique_keys, inverse = np.unique(edge_keys, return_inverse=True)
    del edge_keys

    lower_nodes = (unique_keys >> _UINT32_SHIFT).astype(
        np.int64,
        copy=False,
    )
    upper_nodes = (unique_keys & _UINT32_MASK).astype(
        np.int64,
        copy=False,
    )
    collapsed = lower_nodes == upper_nodes

    midside_ids = np.full(unique_keys.size, -1, dtype=np.int32)
    midside_ids[collapsed] = lower_nodes[collapsed].astype(
        np.int32,
        copy=False,
    )

    existing_twenty = np.flatnonzero(mesh.element_node_num == 20)
    if existing_twenty.size:
        _reuse_existing_midside_nodes(
            mesh.elements,
            existing_twenty,
            unique_keys,
            midside_ids,
        )
    del existing_twenty

    missing = midside_ids < 0
    new_node_count = int(np.count_nonzero(missing))
    resulting_node_count = mesh.node_count + new_node_count
    if resulting_node_count > np.iinfo(np.int32).max + 1:
        raise OverflowError("converted mesh exceeds the int32 node-id range")

    new_node_ids = np.arange(
        mesh.node_count,
        resulting_node_count,
        dtype=np.int32,
    )
    midside_ids[missing] = new_node_ids
    new_coordinates = (
        mesh.nodes[lower_nodes[missing]] + mesh.nodes[upper_nodes[missing]]
    ) * 0.5

    converted_midsides = midside_ids[inverse].reshape(-1, 12)

    # Release the largest topology buffers before copying the output mesh.
    del unique_keys
    del inverse
    del lower_nodes
    del upper_nodes
    del collapsed
    del midside_ids
    del missing
    del new_node_ids

    new_elements = mesh.elements.copy()
    new_elements[converted_indices, 8:20] = converted_midsides
    del converted_midsides

    new_element_node_num = mesh.element_node_num.copy()
    new_element_node_num[converted_indices] = 20
    new_nodes = np.concatenate((mesh.nodes, new_coordinates), axis=0)
    del new_coordinates

    return Mesh3D(
        nodes=new_nodes,
        elements=new_elements,
        element_comps=mesh.element_comps,
        comps=mesh.comps,
        element_types=mesh.element_types,
        types=mesh.types,
        element_reals=mesh.element_reals,
        reals=mesh.reals,
        element_sections=mesh.element_sections,
        sections=mesh.sections,
        element_node_num=new_element_node_num,
    )


def _normalize_element_indices(
    element_count: int,
    element_indices,
) -> NDArray[np.int64]:
    if element_indices is None:
        return np.arange(element_count, dtype=np.int64)

    normalized = np.asarray(element_indices)
    if normalized.ndim != 1:
        raise ValueError("element_indices must be a one-dimensional sequence")
    if normalized.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(normalized.dtype, np.integer) or np.issubdtype(
        normalized.dtype,
        np.bool_,
    ):
        raise TypeError("element_indices must contain integers")
    if np.any(normalized < 0) or np.any(normalized >= element_count):
        raise IndexError(
            "element_indices contain an element index that is out of range"
        )
    if np.unique(normalized).size != normalized.size:
        raise ValueError("element_indices must not contain duplicates")
    return normalized.astype(np.int64, copy=False)


def _validate_connectivity(mesh: Mesh3D) -> None:
    if mesh.elements.size == 0:
        return
    minimum = int(np.min(mesh.elements))
    maximum = int(np.max(mesh.elements))
    if minimum < 0 or maximum >= mesh.node_count:
        raise ValueError("mesh elements contain an out-of-range node index")


def _validate_solid185_topologies(
    corners: NDArray[np.int32],
    element_indices: NDArray[np.int64],
) -> None:
    sorted_corners = np.sort(corners, axis=1)
    unique_counts = 1 + np.count_nonzero(
        sorted_corners[:, 1:] != sorted_corners[:, :-1],
        axis=1,
    )
    del sorted_corners

    lower_collapsed = corners[:, 2] == corners[:, 3]
    upper_prism_collapsed = corners[:, 6] == corners[:, 7]
    apex_collapsed = (
        (corners[:, 4] == corners[:, 5])
        & (corners[:, 5] == corners[:, 6])
        & (corners[:, 6] == corners[:, 7])
    )

    is_hex = unique_counts == 8
    is_prism = (
        (unique_counts == 6) & lower_collapsed & upper_prism_collapsed
    )
    is_pyramid = (unique_counts == 5) & apex_collapsed
    is_tetra = (
        (unique_counts == 4) & lower_collapsed & apex_collapsed
    )
    valid = is_hex | is_prism | is_pyramid | is_tetra
    if not np.all(valid):
        invalid_index = int(element_indices[np.flatnonzero(~valid)[0]])
        raise ValueError(
            "selected 8-node elements must use ANSYS hex, prism, pyramid, "
            f"or tetra connectivity; element {invalid_index} is invalid"
        )


def _encode_edge_keys(corners: NDArray[np.int32]) -> NDArray[np.uint64]:
    keys = np.empty((corners.shape[0], 12), dtype=np.uint64)
    for slot, (first, second) in enumerate(_EDGE_CORNERS):
        node_a = corners[:, first]
        node_b = corners[:, second]
        lower = np.minimum(node_a, node_b).astype(np.uint64)
        upper = np.maximum(node_a, node_b).astype(np.uint64)
        keys[:, slot] = (lower << _UINT32_SHIFT) | upper
    return keys


def _reuse_existing_midside_nodes(
    elements: NDArray[np.int32],
    element_indices: NDArray[np.int64],
    selected_unique_keys: NDArray[np.uint64],
    midside_ids: NDArray[np.int32],
) -> None:
    """Populate selected edge groups from existing SOLID186 connectivity."""
    unique_key_count = selected_unique_keys.size
    for offset in range(0, element_indices.size, _EDGE_SCAN_CHUNK_SIZE):
        chunk_indices = element_indices[
            offset : offset + _EDGE_SCAN_CHUNK_SIZE
        ]
        chunk_corners = elements[chunk_indices, :8]
        chunk_keys = _encode_edge_keys(chunk_corners).reshape(-1)
        chunk_midsides = elements[chunk_indices, 8:20].reshape(-1)
        del chunk_corners

        lower = chunk_keys >> _UINT32_SHIFT
        upper = chunk_keys & _UINT32_MASK
        noncollapsed = lower != upper
        del lower
        del upper

        positions = np.searchsorted(selected_unique_keys, chunk_keys)
        in_range = positions < unique_key_count
        matching = noncollapsed & in_range
        if np.any(matching):
            matching_locations = np.flatnonzero(matching)
            matching[matching_locations] = (
                selected_unique_keys[positions[matching_locations]]
                == chunk_keys[matching_locations]
            )
        if not np.any(matching):
            continue

        matched_positions = positions[matching]
        matched_midsides = chunk_midsides[matching]
        order = np.argsort(matched_positions, kind="stable")
        matched_positions = matched_positions[order]
        matched_midsides = matched_midsides[order]

        group_starts = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(
                    matched_positions[1:] != matched_positions[:-1]
                )
                + 1,
            )
        )
        group_positions = matched_positions[group_starts]
        minimum_ids = np.minimum.reduceat(matched_midsides, group_starts)
        maximum_ids = np.maximum.reduceat(matched_midsides, group_starts)
        conflicts = minimum_ids != maximum_ids
        if np.any(conflicts):
            conflict_position = int(
                group_positions[np.flatnonzero(conflicts)[0]]
            )
            raise ValueError(
                "existing 20-node elements assign conflicting midside "
                "nodes to edge "
                f"{_decode_edge(selected_unique_keys[conflict_position])}"
            )

        previous_ids = midside_ids[group_positions]
        conflicts = (previous_ids >= 0) & (previous_ids != minimum_ids)
        if np.any(conflicts):
            conflict_position = int(
                group_positions[np.flatnonzero(conflicts)[0]]
            )
            raise ValueError(
                "existing 20-node elements assign conflicting midside "
                "nodes to edge "
                f"{_decode_edge(selected_unique_keys[conflict_position])}"
            )
        unresolved = previous_ids < 0
        midside_ids[group_positions[unresolved]] = minimum_ids[unresolved]


def _decode_edge(key: np.uint64) -> tuple[int, int]:
    return (
        int(key >> _UINT32_SHIFT),
        int(key & _UINT32_MASK),
    )
