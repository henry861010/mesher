"""2.5D extrusion engine for mixed quadrilateral/triangle process meshes.

This module assigns materials to a 2D mesh and extrudes the selected elements
along the z axis into fixed-width 20-slot ANSYS connectivity. Quad4 elements
become SOLID185 hexahedra; padded Tri3 rows become wedge-like degenerate solids
by repeating the third node on the bottom and top faces. Unused slots repeat
the eighth connectivity id.

Design notes:
    * A face is selected only when all four element corners are inside the
      requested shape. Partial overlap is intentionally ignored.
    * Build-time arrays grow by capacity instead of appending one row at a
      time. Use ``element_num`` and ``node_num`` to read the valid slices.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from matplotlib.path import Path

from ..mesh2d.model import _validate_zero_z
from .model import Mesh3D
from ..tolerance import (
    GeometryToleranceError, circle_membership, resolve_tolerance, subdivision_count,
    within_distance, within_segment,
)

ELEMENT_2D_LEN = 4
NODE_2D_LEN = 2

ELEMENT_LEN = 20
SOLID185_NODE_NUM = 8
SOLID185_TYPE_ID = 1
NODE_LEN = 3

START_NORMAL = 3
START_DENSITY = 2
START_CONVERT = 1
END = 0
ProgressCallback = Callable[[dict[str, Any]], None]


def _planar_areas(nodes, elements):
    corners = nodes[elements, :2]
    local = corners - corners[:, :1]
    return .5 * np.abs(np.sum(
        local[..., 0] * np.roll(local[..., 1], -1, axis=1)
        - local[..., 1] * np.roll(local[..., 0], -1, axis=1), axis=1,
    ))


class Dragger:
    """Build a 3D solid mesh from a mixed 2D process layout.

    The workflow is:
        1. Load a 2D Quad4/padded-Tri3 mesh with :meth:`set_2D`.
        2. For each process layer, label 2D elements with :meth:`_organize`.
        3. Extrude the labeled elements along z with :meth:`_drag`.

    Attributes:
        comps (dict[str, int]): Material/component name to numeric component id.
        elements (numpy.ndarray): 3D element connectivity. Valid rows are
            ``elements[:element_num]`` and each row contains 20 node indices.
        element_comps (numpy.ndarray): Component ids for valid 3D elements.
        element_types (numpy.ndarray): ANSYS type ids for valid 3D elements.
        types (dict[int, int]): Type id to ANSYS element number definitions.
        element_reals (numpy.ndarray): Real-constant ids for valid elements.
        reals (dict[int, list]): Real-constant definitions.
        element_sections (numpy.ndarray): Section ids for valid elements.
        sections (dict[int, list]): Section definitions.
        element_node_num (numpy.ndarray): Effective connectivity slot counts.
        nodes (numpy.ndarray): 3D node coordinates. Valid rows are
            ``nodes[:node_num]`` and each row is ``[x, y, z]``.
        element_2D (numpy.ndarray): Source 2D connectivity. Tri3 rows use
            ``[n0, n1, n2, n2]`` padding.
        element_2D_volume (numpy.ndarray): XY area for each 2D element.
        element_2D_comp (numpy.ndarray): Temporary per-layer component id for
            each 2D element. ``0`` means ``EMPTY`` and is not extruded.
        node_2D (numpy.ndarray): Source 2D node coordinates as ``[x, y]``.
        node_2D_to_3D (numpy.ndarray): Mapping from 2D node id to the current
            top-layer 3D node id. ``-1`` means the node is not active.
    """

    def __init__(self, *, tolerance=None):
        """Initialize empty 2D state, 3D output buffers, and component table."""
        self.tolerance = resolve_tolerance(tolerance)
        ### component
        self.comps = {"EMPTY":0}
        
        ### 3D elements
        self.element_num = 0
        self.elements = np.empty((0, ELEMENT_LEN), dtype=np.int32)
        self.element_comps = np.empty((0), dtype=np.int32)
        self.element_types = np.empty((0), dtype=np.int32)
        self.types = {SOLID185_TYPE_ID: 185}
        self.element_reals = np.empty((0), dtype=np.int32)
        self.reals = {}
        self.element_sections = np.empty((0), dtype=np.int32)
        self.sections = {}
        self.element_node_num = np.empty((0), dtype=np.int32)
        
        ### nodes
        self.node_num = 0
        self.nodes = np.empty((0, NODE_LEN), dtype=np.float64)
        
        ### process
        self.element_2D = np.zeros((0, 4), dtype=np.int32)
        self.element_2D_volume = np.empty((0), dtype=np.float64)
        self.element_2D_comp = np.empty((0), dtype=np.int32)
        self.element_2D_priority = np.empty((0), dtype=np.float64)
        self.element_2D_density_occupy = np.empty((0), dtype=np.int32)
        
        self.node_2D = np.empty((0, 2), dtype=np.float64)
        self.node_2D_to_3D = np.zeros((0), dtype=np.int32)
        
    ### initial
    def set_2D(self, nodes, elements):
        """Load clockwise Quad4/padded-Tri3 connectivity and XY source nodes.

        A supplied third coordinate must be exactly zero; extrusion heights
        come from the layer plan. Additional columns retain their legacy
        ignored behavior. Validation completes before replacing source data.
        """
        nodes = np.asarray(nodes, dtype=np.float64)
        elements = np.asarray(elements, dtype=np.int32)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("mesh2D.nodes must have shape (n, 2+)") 
        _validate_zero_z(nodes)
        self.tolerance.check_resolution(nodes[:, :2], stage="extrusion_input")
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("mesh2D.elements must have shape (m, 4)")
        if np.any(elements < 0) or np.any(elements >= len(nodes)):
            raise ValueError("mesh2D.elements contains an out-of-range node index")

        triangle_mask = elements[:, 2] == elements[:, 3]
        triangle_nodes = elements[triangle_mask, :3]
        invalid_triangles = (
            (triangle_nodes[:, 0] == triangle_nodes[:, 1])
            | (triangle_nodes[:, 0] == triangle_nodes[:, 2])
            | (triangle_nodes[:, 1] == triangle_nodes[:, 2])
        )
        quad_nodes = np.sort(elements[~triangle_mask], axis=1)
        invalid_quads = np.any(np.diff(quad_nodes, axis=1) == 0, axis=1)
        if np.any(invalid_triangles) or np.any(invalid_quads):
            raise ValueError(
                "mesh2D.elements must contain Quad4 rows or padded Tri3 rows "
                "formatted as [n0, n1, n2, n2]"
            )
        
        areas = _planar_areas(nodes, elements)
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0):
            raise GeometryToleranceError(
                "extrusion requires source elements with positive finite area",
                code="invalid_source_element", stage="extrusion_input",
            )
        for count, rows in ((3, triangle_mask), (4, ~triangle_mask)):
            corners = nodes[elements[rows, :count], :2]
            if not len(corners):
                continue
            edges = np.roll(corners, -1, axis=1) - corners
            if np.any(within_distance(edges, np.zeros(2), self.tolerance)):
                raise GeometryToleranceError(
                    "source element contains an unresolved edge",
                    code="invalid_source_element", stage="extrusion_input",
                    tolerance=self.tolerance.absolute,
                )
            next_edges = np.roll(edges, -1, axis=1)
            turns = edges[..., 0] * next_edges[..., 1] - edges[..., 1] * next_edges[..., 0]
            scale = np.max(np.abs(edges), axis=(1, 2))
            numerical_area = 64 * np.finfo(float).eps * scale**2
            if np.any(~(np.all(turns > numerical_area[:, None], axis=1)
                        | np.all(turns < -numerical_area[:, None], axis=1))):
                raise GeometryToleranceError(
                    "source element has an invalid planar Jacobian",
                    code="invalid_source_element", stage="extrusion_input",
                )
        self.element_2D = elements.copy()
        self.element_2D_comp = np.zeros(len(elements), dtype=np.int32)
        self.element_2D_volume = areas
        self.element_2D_priority = np.zeros(len(elements), dtype=np.float64)
        self.element_2D_density_occupy = np.zeros(len(elements), dtype=np.float64)
        self.node_2D = nodes[:, :2].copy()
        self.node_2D_to_3D = np.full(len(nodes), -1, dtype=np.int32)
        
    ### buffer growth
    def _pre_allocate_nodes(self, size: int = 1):
        """Ensure the internal 3D node buffer can hold additional rows.

        Args:
            size (int): Number of new node rows that will be appended.

        Notes:
            This method only grows capacity. ``node_num`` still controls how
            many rows are valid.
        """
        required = self.node_num + size
        current_capacity = len(self.nodes)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            
            self.nodes = np.vstack([self.nodes, np.empty((extra, 3), dtype=np.float64)])

    def _pre_allocate_elements(self, size: int = 1):
        """Ensure the internal 3D element buffers can hold additional rows.

        Args:
            size (int): Number of new element rows that will be appended.

        Notes:
            This method grows connectivity and per-element metadata arrays
            together.
            ``element_num`` still controls how many rows are valid.
        """
        required = self.element_num + size
        current_capacity = len(self.elements)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            
            self.elements = np.vstack(
                [self.elements, np.empty((extra, ELEMENT_LEN), dtype=np.int32)]
            )
            self.element_comps = np.concatenate([self.element_comps, np.empty(extra, dtype=np.int32)])
            self.element_types = np.concatenate(
                [self.element_types, np.empty(extra, dtype=np.int32)]
            )
            self.element_reals = np.concatenate(
                [self.element_reals, np.empty(extra, dtype=np.int32)]
            )
            self.element_sections = np.concatenate(
                [self.element_sections, np.empty(extra, dtype=np.int32)]
            )
            self.element_node_num = np.concatenate(
                [self.element_node_num, np.empty(extra, dtype=np.int32)]
            )
        
    ### core
    def _cal_volumes(self):
        """Calculate XY area for every source Quad4 or padded-Tri3 element.

        Returns:
            None: Results are written to ``self.element_2D_volume``.
        """
        self.element_2D_volume = _planar_areas(self.node_2D, self.element_2D)
        
    def _normalize_element_indices(self, indices=None):
        """Normalize optional element indices into a 1D int32 array.

        Args:
            element_indices (int | Sequence[int] | numpy.ndarray | None):
                Optional 2D element indices. ``None`` means all 2D elements.

        Returns:
            numpy.ndarray: 1D array of 2D element indices.
        """
        if indices is None:
            return np.arange(len(self.element_2D), dtype=np.int32)

        indices = np.asarray(indices, dtype=np.int32)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        return indices

    def _element_coordinates(self, indices=None):
        indices = self._normalize_element_indices(indices)
        if len(indices) == 0:
            return np.empty((0, ELEMENT_2D_LEN, NODE_2D_LEN), dtype=self.node_2D.dtype)

        corner_xy = self.node_2D[self.element_2D[indices]]
        return corner_xy
    
    def _signed_area(self, points):
        points = np.asarray(points, dtype=float)
        points = points - points[0]
        total = 0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            total += x1 * y2 - x2 * y1
        return total
    
    def _search_faces(self, face, koz=0, indices=None):
        indices = self._normalize_element_indices(indices)
        
        if face is None:
            return np.ones(len(indices), dtype=bool)
        
        if 0 == len(indices):
            return np.zeros(0, dtype=np.int32)
        
        element_coordinates = self._element_coordinates(indices)
        
        face_type = face["type"]
        face_dim = face["dim"]
        
        if face_type == "BOX":
            lower = np.minimum(face_dim[:2], face_dim[2:]) + koz
            upper = np.maximum(face_dim[:2], face_dim[2:]) - koz
            if np.any(lower > upper):
                return np.zeros(len(indices), dtype=bool)
            closest = np.clip(element_coordinates, lower, upper)
            return np.all(within_distance(element_coordinates, closest, self.tolerance), axis=1)
        elif face_type == "POLYGON":
            flat_coordinates = element_coordinates.reshape(-1, 2)
            flat_mask = np.zeros(len(flat_coordinates), dtype=bool)
            
            boundary = np.zeros(len(flat_coordinates), dtype=bool)
            for poly in face_dim:
                loop = np.asarray(poly, dtype=np.float64)
                if len(loop) < 3:
                    continue
                origin = loop[0]
                flat_mask ^= Path(loop - origin).contains_points(flat_coordinates - origin)
                for first, last in zip(loop, np.roll(loop, -1, axis=0)):
                    boundary |= within_segment(flat_coordinates, first, last, self.tolerance)
            flat_mask |= boundary

            # Combine loop parity per corner before requiring every corner to
            # be inside. Elements partially entering a hole must be excluded.
            return flat_mask.reshape(len(element_coordinates), 4).all(axis=1)
        elif face_type == "CIRCLE":
            center_x, center_y, radius = face_dim
            effective_radius = radius - koz
            if effective_radius < 0:
                return np.zeros(len(indices), dtype=bool)
            return np.all(circle_membership(element_coordinates, [center_x, center_y],
                                            effective_radius, self.tolerance), axis=1)
        else:
            raise ValueError(f"The face type {face_type} is not supported by Dragger")
        
    def _organize_empty(self):
        """Reset temporary 2D component labels before processing one object."""
        self.element_2D_comp[:] = 0
        self.node_2D_to_3D[:] = -1
        
    def _comp(self, comp):
        if comp not in self.comps:
            self.comps[comp] = len(self.comps)
        return self.comps[comp]
        
    def _assign_metal(self, indices, density, total_volume, randomSeed):
        if density == 0:
            return np.empty((0), dtype=np.int32)
            
        volumes = self.element_2D_volume[indices]
        target_indices = np.arange(len(volumes), dtype=np.int32)
        
        # target volume
        target = (density / 100.0) * total_volume

        # random order of candidates (indices only, not rows)
        rng = np.random.default_rng(randomSeed)
        random_indices = target_indices[rng.permutation(len(volumes))]

        # cumulative sum until target
        csum = np.cumsum(volumes[random_indices])
        k = np.searchsorted(csum, target, side="right")  # number to take (may be 0)
        if k > 0:
            chosen_indices  = target_indices[random_indices[:k+1]]
            return indices[chosen_indices]
        else:
            # no assignment if density threshold is 0 or vols too small
            return np.empty((0), dtype=np.int32)
        
    def _organize(
        self,
        assignments,
        layer=1,
        *,
        progress: ProgressCallback | None = None,
        layer_total: int = 0,
    ):
        # sort the assignment (high priority first)
        assignments = sorted(assignments, key=lambda item: (item['type'], -item['areas'][0]['priority']))
        
        for index, assignment in enumerate(assignments):
            assignment_started = time.perf_counter()
            _emit_progress(
                progress,
                current=layer,
                total=layer_total,
                unit="layers",
                message=(
                    f"Assigning feature {index + 1} of {len(assignments)} "
                    f"in layer {layer + 1} of {layer_total}."
                ),
            )
            face = assignment["face"]
            areas = assignment["areas"]
            assign_type = assignment["type"]
        
            ### Select the area once (mask -> indices)
            mask  = self._search_faces(face)
            potential_indices = np.where(mask)[0]
            
            if len(potential_indices) == 0:
                _emit_assignment_completed(
                    progress,
                    assignment,
                    layer=layer,
                    assignment_index=index,
                    duration_ms=int(round((time.perf_counter() - assignment_started) * 1000)),
                    selected_element_count=0,
                )
                continue

            # START_NORMAL
            if assign_type == START_NORMAL:
                area = areas[0]
                material = area["material"]
                priority = area["priority"]
                
                ### filter the lower priority
                mask = (self.element_2D_priority[potential_indices] <= priority)
                target_indices = potential_indices[mask]
                    
                ### get (new) material
                comp_id = self._comp(material)
                
                ### assignment
                self.element_2D_comp[target_indices] = comp_id
                self.element_2D_priority[target_indices] = priority  
                
            # END
            elif assign_type == END:
                priority_o = areas[0]["priority_o"]
                areas = sorted(areas, key=lambda item: item['priority'], reverse=True)

                # relesae density lock
                mask = (self.element_2D_density_occupy[potential_indices] == priority_o)
                self.element_2D_density_occupy[potential_indices[mask]] = 0
                
                # release normal material
                potential_indices = potential_indices[self.element_2D_priority[potential_indices] == priority_o]
                for area in areas:
                    face = area["face"]
                    priority = area["priority"]
                    material = area["material"]
                    
                    ### local area
                    mask  = self._search_faces(face, indices=potential_indices)
                    target_indices = potential_indices[mask]
                    potential_indices = potential_indices[~mask]
                      
                    ### get (new) material
                    comp_id = self._comp(material)
                    
                    ### assignment
                    self.element_2D_comp[target_indices] = comp_id
                    self.element_2D_priority[target_indices] = priority
                    
                    if len(potential_indices) == 0:
                        break
                    
            # START_DENSITY
            elif assign_type == START_DENSITY:
                area = areas[0]
                material = area["material"]
                priority = area["priority"]
                density = area["density"]
                koz = area["koz"]
                
                ### filter the lower priority
                mask = (self.element_2D_priority[potential_indices] <= priority)
                target_indices = potential_indices[mask]
                
                ### filter the density opccupied & set to occupy
                mask = (self.element_2D_density_occupy[target_indices] <= priority)
                target_indices = target_indices[mask]
                self.element_2D_density_occupy[target_indices] = priority
                
                ### koz
                density_face = area.get("face")
                if density_face is None:
                    mask = self._search_faces(face, koz=koz, indices=target_indices)
                else:
                    mask = self._search_faces(
                        density_face,
                        indices=target_indices,
                    )
                target_indices = target_indices[mask]
                
                ### total volume
                total_volume = np.sum(self.element_2D_volume[target_indices])
                target_indices = self._assign_metal(target_indices, density, total_volume, randomSeed=layer*100000+index)
                    
                ### get (new) material
                comp_id = self._comp(material)
                
                ### assignment
                self.element_2D_comp[target_indices] = comp_id
                self.element_2D_priority[target_indices] = priority 

            _emit_assignment_completed(
                progress,
                assignment,
                layer=layer,
                assignment_index=index,
                duration_ms=int(round((time.perf_counter() - assignment_started) * 1000)),
                selected_element_count=len(target_indices),
            )
        
    def _drag(self, element_size: float, begin: float, end: float):
        """Extrude currently labeled 2D elements between two z positions.

        Args:
            element_size (float): Requested z element height. The actual height
                is recalculated so an integer number of layers exactly spans
                ``begin`` to ``end``.
            begin (float): Lower z coordinate of the extrusion interval.
            end (float): Upper z coordinate of the extrusion interval.

        Returns:
            int | None: ``0`` when there is no positive distance or no active
            2D element. Otherwise returns ``None`` after appending nodes and
            elements to the internal buffers.

        Notes:
            ``node_2D_to_3D`` stores the top 3D node for each active 2D node.
            That map lets the next process layer reuse the current top surface
            as its bottom surface, keeping adjacent extrusions connected.
        """
        ### calculate drag_num & element_size
        distance = float(end) - float(begin)
        if distance <= 0:
            return 0
        if within_distance(begin, end, self.tolerance):
            raise GeometryToleranceError(
                "a material layer cannot be resolved at geometryTolerance",
                code="feature_below_resolution", stage="extrusion", begin=begin,
                end=end, measured_distance=distance, tolerance=self.tolerance.absolute,
            )
        self.tolerance.check_resolution([begin, end], stage="extrusion")
        if element_size <= self.tolerance.absolute:
            raise GeometryToleranceError(
                "requested Z element size must exceed geometryTolerance",
                code="feature_below_resolution", stage="extrusion",
                measured_distance=float(element_size), tolerance=self.tolerance.absolute,
                begin=begin, end=end,
            )
        drag_num = subdivision_count(distance, element_size, rounding="floor")
        element_size = distance / drag_num
        if element_size <= self.tolerance.absolute:
            raise GeometryToleranceError(
                "requested Z subdivision would create unresolved layers",
                code="feature_below_resolution", stage="extrusion",
                measured_distance=element_size, tolerance=self.tolerance.absolute,
                begin=begin, end=end, subdivision_count=drag_num,
            )
        # Validate every represented cut before allocating or updating buffers.
        z_cuts = np.linspace(begin, end, drag_num + 1, dtype=self.nodes.dtype)
        gaps = np.diff(z_cuts)
        if (np.any(~np.isfinite(z_cuts)) or np.any(gaps <= 0)
                or np.any(within_distance(z_cuts[:-1, None], z_cuts[1:, None], self.tolerance))):
            raise GeometryToleranceError(
                "Z subdivision cannot represent distinct cuts above geometryTolerance",
                code="numeric_resolution_exceeded", stage="extrusion",
                measured_distance=float(np.min(gaps)), tolerance=self.tolerance.absolute,
                begin=begin, end=end, subdivision_count=drag_num,
            )

        ### target element index
        elem2D_idx = np.flatnonzero(self.element_2D_comp != 0)
        if elem2D_idx.size == 0:
            self.node_2D_to_3D[:] = -1
            return 0

        # unique 2D node ids used by those elements
        elem2D_nodes = self.element_2D[elem2D_idx]
        node2D_idx, inv = np.unique(elem2D_nodes, return_inverse=True)
        elem2D_nodes_local = inv.reshape(elem2D_nodes.shape)   

        ### add the unexisted node
        unexisted_node2D_idx = node2D_idx[self.node_2D_to_3D[node2D_idx] == -1]
        if unexisted_node2D_idx.size:
            self._pre_allocate_nodes(unexisted_node2D_idx.size)
            
            dst = self.nodes[self.node_num : self.node_num + unexisted_node2D_idx.size]
            np.take(self.node_2D, unexisted_node2D_idx, axis=0, out=dst[:, :2]) # xy from 2D nodes -> write directly without temp:
            dst[:, 2] = begin
            
            self.node_2D_to_3D[unexisted_node2D_idx] = np.arange(self.node_num, self.node_num + unexisted_node2D_idx.size, dtype=np.int32)
            self.node_num += unexisted_node2D_idx.size

        ### begin to drag
        base_map = self.node_2D_to_3D[node2D_idx]
        N = node2D_idx.size
        E = elem2D_idx.size

        ### [NODE] allocate all 3D nodes
        self._pre_allocate_nodes(N * drag_num)
        node_start = self.node_num
        new_node_ids = (node_start + np.arange(N * drag_num, dtype=np.int32)).reshape(drag_num, N)

        # fill XY for all layers (broadcast XY), and Z per layer
        xy = self.node_2D[node2D_idx]
        dst_nodes = self.nodes[node_start : node_start + N * drag_num]
        dst_nodes[:, :2] = np.broadcast_to(xy, (drag_num, N, 2)).reshape(N * drag_num, 2)

        z_vals = z_cuts[1:]
        dst_nodes[:, 2] = np.repeat(z_vals, N)

        self.node_num += N * drag_num

        ### [ELEMENT] allocate all 3D elements
        self._pre_allocate_elements(E * drag_num)
        elem_start = self.element_num

        # layer->3D index table (drag_num+1, N): row0 = base, rows1..drag_num = new layers
        layer_nodes = np.empty((drag_num + 1, N), dtype=np.int32)
        layer_nodes[0]  = base_map
        layer_nodes[1:] = new_node_ids

        # For each layer k, bottom = layer_nodes[k-1][cols], top = layer_nodes[k][cols]
        # cols per element are the (E,4) indices into node2D_idx
        bottom = layer_nodes[:-1][:, elem2D_nodes_local]
        top    = layer_nodes[1:][:,  elem2D_nodes_local]
        elems = np.concatenate([bottom, top], axis=2).reshape(
            drag_num * E,
            SOLID185_NODE_NUM,
        )

        ### assign nodes to each element
        element_slice = slice(elem_start, elem_start + drag_num * E)
        target_elements = self.elements[element_slice]
        target_elements[:, :SOLID185_NODE_NUM] = elems.astype(
            np.int32,
            copy=False,
        )
        target_elements[:, SOLID185_NODE_NUM:] = elems[:, -1, None]
        
        ### assign comps to each element
        layer_comps = self.element_2D_comp[elem2D_idx]
        dest = self.element_comps[element_slice].reshape(drag_num, E)
        dest[:] = layer_comps
        self.element_types[element_slice] = SOLID185_TYPE_ID
        self.element_reals[element_slice] = 0
        self.element_sections[element_slice] = 0
        self.element_node_num[element_slice] = SOLID185_NODE_NUM
        
        self.element_num += drag_num * E

        # update map so future ops start from the top layer
        self.node_2D_to_3D[:] = -1
        self.node_2D_to_3D[node2D_idx] = layer_nodes[-1]
        
    def build(
        self,
        layer_infos,
        *,
        verbose=False,
        progress: ProgressCallback | None = None,
    ) -> Mesh3D:
        """Build 3D mesh data from process object definitions.

        Args:
            layer_infos (Sequence[dict]): Ordered z-level dictionaries. Every
                entry except the last must include ``assignments``; every entry
                used as a boundary must include ``z``.

        Returns:
            Mesh3D: An owned snapshot containing only valid 3D mesh rows.

        Notes:
            The method appends to existing 3D buffers. It resets temporary 2D
            labels for each object, but it does not clear previously generated
            3D nodes or elements.
        """

        self._organize_empty()
        layer_total = max(0, len(layer_infos) - 1)
        for index, layer_info in enumerate(layer_infos[:-1]):
            layer_started = time.perf_counter()
            nodes_before = self.node_num
            elements_before = self.element_num
            z_now = layer_infos[index]["z"]
            z_next = layer_infos[index+1]["z"]
            element_size = layer_info["element_size"]
            assignments = layer_info["assignments"]
            
            self._organize(
                assignments,
                index,
                progress=progress,
                layer_total=layer_total,
            )
            
            self._drag(element_size, z_now, z_next)
            if progress is not None:
                progress(
                    {
                        "event": "item.completed",
                        "stage": "building_3d_mesh",
                        "data": {
                            "itemType": "layer",
                            "layerIndex": index + 1,
                            "assignmentCount": len(assignments),
                            "nodesAdded": self.node_num - nodes_before,
                            "elementsAdded": self.element_num - elements_before,
                            "wallDurationMs": int(
                                round((time.perf_counter() - layer_started) * 1000)
                            ),
                        },
                    }
                )
            _emit_progress(
                progress,
                current=index + 1,
                total=layer_total,
                unit="layers",
                message=f"Built layer {index + 1} of {layer_total}.",
            )
            

        if verbose:
            print(f"element_num: {self.element_num}")
            print(f"node_num: {self.node_num}")

        mesh = Mesh3D(
            nodes=self.nodes[: self.node_num],
            elements=self.elements[: self.element_num],
            element_comps=self.element_comps[: self.element_num],
            comps=self.comps,
            element_types=self.element_types[: self.element_num],
            types=self.types,
            element_reals=self.element_reals[: self.element_num],
            reals=self.reals,
            element_sections=self.element_sections[: self.element_num],
            sections=self.sections,
            element_node_num=self.element_node_num[: self.element_num],
        )
        self.tolerance.check_resolution(mesh.nodes, stage="solid_validation")
        if mesh.element_count:
            corners = mesh.nodes[mesh.elements[:, :8]]
            heights = corners[:, 4:, 2] - corners[:, :4, 2]
            if np.any(heights <= 0) or np.any(~np.isfinite(heights)):
                raise GeometryToleranceError(
                    "extrusion produced a solid with non-positive height",
                    code="invalid_solid", stage="solid_validation",
                )
            if np.any(within_distance(corners[:, :4, 2:3], corners[:, 4:, 2:3], self.tolerance)):
                raise GeometryToleranceError(
                    "extrusion produced a solid below geometryTolerance",
                    code="feature_below_resolution", stage="solid_validation",
                    measured_distance=float(np.min(heights)), tolerance=self.tolerance.absolute,
                )
            if not np.array_equal(corners[:, :4, :2], corners[:, 4:, :2]):
                raise GeometryToleranceError(
                    "extrusion produced unmatched lower and upper faces",
                    code="invalid_solid", stage="solid_validation",
                )
        return mesh


def _emit_progress(
    progress: ProgressCallback | None,
    *,
    current: int,
    total: int,
    unit: str,
    message: str,
) -> None:
    if progress is not None:
        progress(
            {
                "event": "progress",
                "current": current,
                "total": total,
                "unit": unit,
                "message": message,
                "data": {},
            }
        )


def _emit_assignment_completed(
    progress: ProgressCallback | None,
    assignment: dict[str, Any],
    *,
    layer: int,
    assignment_index: int,
    duration_ms: int,
    selected_element_count: int,
) -> None:
    if progress is None:
        return
    diagnostic = assignment.get("diagnostic")
    diagnostic_data = diagnostic if isinstance(diagnostic, dict) else {}
    progress(
        {
            "event": "item.completed",
            "stage": "building_3d_mesh",
            "data": {
                "itemType": "assignment",
                "layerIndex": layer + 1,
                "assignmentIndex": assignment_index + 1,
                "sourceRef": diagnostic_data.get("sourceRef"),
                "containerRef": diagnostic_data.get("containerRef"),
                "featureType": diagnostic_data.get("featureType"),
                "geometryType": diagnostic_data.get("geometryType"),
                "operation": diagnostic_data.get("operation"),
                "selectedElementCount": selected_element_count,
                "wallDurationMs": duration_ms,
            },
        }
    )
