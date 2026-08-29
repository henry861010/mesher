"""Typed extrusion of planar mixed-element meshes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..mesh2d import Mesh2D
from .model import ElementType3D, Mesh3D


@dataclass(frozen=True)
class ExtrusionLayer:
    """One Z interval and its per-planar-element component assignment.

    Component id zero means that the corresponding planar element is empty and
    must not be extruded in this interval.
    """

    z_min: float
    z_max: float
    element_component_ids: ArrayLike

    def normalized_component_ids(self, element_count: int) -> NDArray[np.int32]:
        values = np.asarray(self.element_component_ids)
        if values.ndim != 1 or len(values) != element_count:
            raise ValueError(
                "layer element_component_ids must have one entry per Mesh2D element."
            )
        if values.size and not np.issubdtype(values.dtype, np.integer):
            raise ValueError("layer element_component_ids must contain integers.")
        return np.ascontiguousarray(values, dtype=np.int32).copy()


LayerCompletedCallback = Callable[[int, int, int], None]


def extrude_mesh(
    mesh: Mesh2D,
    layers: Iterable[ExtrusionLayer],
    *,
    element_size: float,
    component_ids_by_name: Mapping[str, int],
    _on_layer_completed: LayerCompletedCallback | None = None,
) -> Mesh3D:
    """Extrude typed planar component layers into a fixed-width solid mesh.

    This deliberately preserves the historical subdivision rule used by the
    process-flow mesher so integration does not alter mesh counts or node
    ordering.  ``element_size`` is therefore a nominal Z size rather than a
    newly imposed maximum-height contract.
    """

    if not isinstance(mesh, Mesh2D):
        raise TypeError("mesh must be a Mesh2D instance.")
    try:
        normalized_size = float(element_size)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("element_size must be a real number.") from error
    if not np.isfinite(normalized_size) or normalized_size <= 0.0:
        raise ValueError("element_size must be positive and finite.")

    normalized_layers = _normalize_layers(layers, mesh.element_count)
    builder = _ExtrusionBuilder(mesh)
    for layer_index, layer in enumerate(normalized_layers):
        nodes_before = builder.node_count
        elements_before = builder.element_count
        builder.append_layer(layer, normalized_size)
        if _on_layer_completed is not None:
            _on_layer_completed(
                layer_index,
                builder.node_count - nodes_before,
                builder.element_count - elements_before,
            )
    return builder.build(component_ids_by_name)


def _normalize_layers(
    layers: Iterable[ExtrusionLayer],
    element_count: int,
) -> Iterator[ExtrusionLayer]:
    previous_z_max: float | None = None
    for index, layer in enumerate(layers):
        if not isinstance(layer, ExtrusionLayer):
            raise TypeError(f"layers[{index}] must be an ExtrusionLayer.")
        z_min = _finite_number(layer.z_min, f"layers[{index}].z_min")
        z_max = _finite_number(layer.z_max, f"layers[{index}].z_max")
        if z_max < z_min:
            raise ValueError(f"layers[{index}].z_max must not be below z_min.")
        if previous_z_max is not None and z_min != previous_z_max:
            raise ValueError("extrusion layers must be ordered and contiguous.")
        yield ExtrusionLayer(
            z_min=z_min,
            z_max=z_max,
            element_component_ids=layer.normalized_component_ids(element_count),
        )
        previous_z_max = z_max


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


class _ExtrusionBuilder:
    """Capacity-managed implementation behind :func:`extrude_mesh`."""

    def __init__(self, mesh: Mesh2D) -> None:
        self._source_nodes = mesh.nodes[:, :2]
        self._source_elements = mesh.elements
        self._source_element_types = mesh.element_types
        self._top_node_ids = np.full(mesh.node_count, -1, dtype=np.int32)
        self._nodes = np.empty((0, 3), dtype=np.float64)
        self._elements = np.empty((0, 8), dtype=np.int32)
        self._element_types = np.empty(0, dtype=np.uint8)
        self._element_component_ids = np.empty(0, dtype=np.int32)
        self.node_count = 0
        self.element_count = 0

    def append_layer(self, layer: ExtrusionLayer, element_size: float) -> None:
        distance = round(float(layer.z_max) - float(layer.z_min), 5)
        if distance <= 0:
            return
        subdivision_count = int(max(1, np.floor(distance / element_size)))
        actual_height = distance / subdivision_count

        component_ids = np.asarray(layer.element_component_ids, dtype=np.int32)
        source_element_ids = np.flatnonzero(component_ids != 0)
        if source_element_ids.size == 0:
            self._top_node_ids[:] = -1
            return

        source_element_nodes = self._source_elements[source_element_ids]
        source_node_ids, inverse = np.unique(
            source_element_nodes,
            return_inverse=True,
        )
        local_connectivity = inverse.reshape(source_element_nodes.shape)

        missing_node_ids = source_node_ids[self._top_node_ids[source_node_ids] == -1]
        if missing_node_ids.size:
            self._reserve_nodes(int(missing_node_ids.size))
            destination = self._nodes[
                self.node_count : self.node_count + missing_node_ids.size
            ]
            destination[:, :2] = self._source_nodes[missing_node_ids]
            destination[:, 2] = layer.z_min
            self._top_node_ids[missing_node_ids] = np.arange(
                self.node_count,
                self.node_count + missing_node_ids.size,
                dtype=np.int32,
            )
            self.node_count += int(missing_node_ids.size)

        base_node_ids = self._top_node_ids[source_node_ids]
        nodes_per_plane = int(source_node_ids.size)
        source_element_count = int(source_element_ids.size)

        self._reserve_nodes(nodes_per_plane * subdivision_count)
        node_start = self.node_count
        new_node_ids = (
            node_start
            + np.arange(nodes_per_plane * subdivision_count, dtype=np.int32)
        ).reshape(subdivision_count, nodes_per_plane)
        destination_nodes = self._nodes[
            node_start : node_start + nodes_per_plane * subdivision_count
        ]
        source_xy = self._source_nodes[source_node_ids]
        destination_nodes[:, :2] = np.broadcast_to(
            source_xy,
            (subdivision_count, nodes_per_plane, 2),
        ).reshape(nodes_per_plane * subdivision_count, 2)
        z_values = layer.z_min + actual_height * np.arange(
            1,
            subdivision_count + 1,
            dtype=np.float64,
        )
        destination_nodes[:, 2] = np.repeat(z_values, nodes_per_plane)
        self.node_count += nodes_per_plane * subdivision_count

        self._reserve_elements(source_element_count * subdivision_count)
        layer_nodes = np.empty(
            (subdivision_count + 1, nodes_per_plane),
            dtype=np.int32,
        )
        layer_nodes[0] = base_node_ids
        layer_nodes[1:] = new_node_ids
        bottom = layer_nodes[:-1][:, local_connectivity]
        top = layer_nodes[1:][:, local_connectivity]
        new_elements = np.concatenate([bottom, top], axis=2).reshape(
            subdivision_count * source_element_count,
            8,
        )

        element_start = self.element_count
        element_end = element_start + subdivision_count * source_element_count
        self._elements[element_start:element_end] = new_elements
        source_types = self._source_element_types[source_element_ids]
        solid_types = np.where(
            source_types == 3,
            ElementType3D.WEDGE6,
            ElementType3D.HEX8,
        ).astype(np.uint8, copy=False)
        self._element_types[element_start:element_end] = np.tile(
            solid_types,
            subdivision_count,
        )
        self._element_component_ids[element_start:element_end] = np.tile(
            component_ids[source_element_ids],
            subdivision_count,
        )
        self.element_count = element_end

        self._top_node_ids[:] = -1
        self._top_node_ids[source_node_ids] = layer_nodes[-1]

    def build(self, component_ids_by_name: Mapping[str, int]) -> Mesh3D:
        return Mesh3D(
            nodes=self._nodes[: self.node_count],
            elements=self._elements[: self.element_count],
            element_types=self._element_types[: self.element_count],
            element_component_ids=self._element_component_ids[: self.element_count],
            component_ids_by_name=component_ids_by_name,
        )

    def _reserve_nodes(self, size: int) -> None:
        required = self.node_count + size
        current = len(self._nodes)
        if required <= current:
            return
        capacity = max(required, int(current * 1.5))
        self._nodes = np.vstack(
            (self._nodes, np.empty((capacity - current, 3), dtype=np.float64))
        )

    def _reserve_elements(self, size: int) -> None:
        required = self.element_count + size
        current = len(self._elements)
        if required <= current:
            return
        capacity = max(required, int(current * 1.5))
        extra = capacity - current
        self._elements = np.vstack(
            (self._elements, np.empty((extra, 8), dtype=np.int32))
        )
        self._element_types = np.concatenate(
            (self._element_types, np.empty(extra, dtype=np.uint8))
        )
        self._element_component_ids = np.concatenate(
            (self._element_component_ids, np.empty(extra, dtype=np.int32))
        )


__all__ = ["ExtrusionLayer", "extrude_mesh"]
