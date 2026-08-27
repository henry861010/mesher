"""Coordinate circular strip construction and transactional commit."""

import numpy as np

from ..mesh import Mesh2D
from .pattern_segments import (
    _PatternGuideSet,
    _circle_line_intersections,
    _coerce_pattern_guides,
    _interval_overlap,
)
from .quad_merge import _QuadMergeMixin
from .triangulation import _StripScore, _TriangulationMixin
from .validation import _ValidationMixin


class _CircularStripMesher(
    _TriangulationMixin,
    _ValidationMixin,
    _QuadMergeMixin,
):
    """Build a conforming mixed-element strip between concentric boundaries.

    The object is a staged coordinator. Each private stage validates or derives
    state needed by the next one, while build preserves the original order:
    normalize inputs, fit geometry, prepare pattern guides, read topology,
    order boundaries, validate existing geometry, resolve pattern connectors,
    triangulate, validate the new strip, merge quads, and commit.

    Attributes:
        mesh: Mesh2D receiving elements only after every stage succeeds.
        inner_input: Raw inner-boundary node indices.
        outer_input: Raw outer-boundary node indices.
        guide_segments_input: Raw axis-aligned pattern segments.
        closed_input: Raw closed-ring mode.
        min_quad_scaled_jacobian_input: Raw minimum Quad4 scaled Jacobian.
    """

    def __init__(
        self,
        mesh,
        inner_nodes,
        outer_nodes,
        guide_segments,
        closed,
        min_quad_scaled_jacobian=0.3,
    ):
        """Store raw inputs for deferred, transactional validation.

        Args:
            mesh: Mesh2D that receives generated elements after validation.
            inner_nodes: Node indices on the smaller circular boundary.
            outer_nodes: Node indices on the larger circular boundary.
            guide_segments: Optional axis-aligned pattern segments.
            closed: Whether both boundaries are closed rings rather than arcs.
            min_quad_scaled_jacobian: Minimum scaled Jacobian for Quad4 merging.
        """
        self.mesh = mesh
        self.inner_input = inner_nodes
        self.outer_input = outer_nodes
        self.guide_segments_input = guide_segments
        self.closed_input = closed
        self.min_quad_scaled_jacobian_input = min_quad_scaled_jacobian

    def build(self):
        """Run every meshing stage and commit the validated strip.

        Returns:
            The same Mesh2D instance with generated elements appended.

        Raises:
            TypeError: If an input has an unsupported type.
            ValueError: If inputs, topology, constraints, or geometry are invalid.
            IndexError: If a boundary contains an out-of-range node index.

        Note:
            Mesh2D mutation is deferred until the final commit stage.
        """
        self._read_inputs()
        self._fit_concentric_geometry()
        self._prepare_pattern_guides()
        self._read_existing_topology()
        self._order_boundaries()
        self._validate_existing_geometry()
        self._collect_pattern_connectors()
        triangles = self._triangulate()
        self._validate_completed_strip(triangles)
        new_elements = self._merge_triangle_pairs(triangles)
        self._commit(new_elements)
        return self.mesh

    def _read_inputs(self):
        """Normalize mesh arrays, indices, options, and retain pattern input.

        Raises:
            TypeError: If mesh, closed, or node-index types are invalid.
            ValueError: If array shapes, scalar ranges, or coordinates are invalid.
            IndexError: If a ring contains an out-of-range node index.
        """
        if not isinstance(self.mesh, Mesh2D):
            raise TypeError("mesh must be a Mesh2D instance")
        if not isinstance(self.closed_input, (bool, np.bool_)):
            raise TypeError("closed must be True or False")
        self.closed = bool(self.closed_input)
        try:
            self.minimum_quad_scaled_jacobian = float(
                self.min_quad_scaled_jacobian_input
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "min_quad_scaled_jacobian must be a real number"
            ) from error
        if not np.isfinite(self.minimum_quad_scaled_jacobian):
            raise ValueError("min_quad_scaled_jacobian must be finite")
        if not 0.0 <= self.minimum_quad_scaled_jacobian <= 1.0:
            raise ValueError("min_quad_scaled_jacobian must be between 0 and 1")

        self.nodes = np.asarray(self.mesh.nodes)
        self.elements = np.asarray(self.mesh.elements)
        if self.nodes.ndim != 2 or self.nodes.shape[1] not in (2, 3):
            raise ValueError("nodes must have shape (N, 2) or (N, 3)")
        if (
            not np.issubdtype(self.nodes.dtype, np.number)
            or np.issubdtype(self.nodes.dtype, np.bool_)
            or np.issubdtype(self.nodes.dtype, np.complexfloating)
        ):
            raise ValueError("nodes must have a real numeric dtype")
        if self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise ValueError("elements must have shape (M, 4)")
        if (
            not np.issubdtype(self.elements.dtype, np.integer)
            or np.issubdtype(self.elements.dtype, np.bool_)
        ):
            raise ValueError("elements must have an integer dtype")

        try:
            float_nodes = self.nodes.astype(np.float64, copy=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("nodes must be representable as float64") from error
        if not np.all(np.isfinite(float_nodes)):
            raise ValueError("nodes must contain finite float64 coordinates")
        self.xy = float_nodes[:, :2]

        self.inner_indices = self._normalize_indices(
            self.inner_input, "inner_nodes"
        )
        self.outer_indices = self._normalize_indices(
            self.outer_input, "outer_nodes"
        )
        if np.intersect1d(self.inner_indices, self.outer_indices).size:
            raise ValueError("inner_nodes and outer_nodes must be disjoint")
        if np.any(self.elements < 0) or np.any(
            self.elements >= self.nodes.shape[0]
        ):
            raise ValueError("elements contain an out-of-range node index")

        # Pattern geometry is prepared after the concentric fit establishes
        # the local coordinate scale. A set prepared by imprint_circle is
        # retained directly and therefore is not parsed again.
        if isinstance(self.guide_segments_input, _PatternGuideSet):
            self.pattern_guides = self.guide_segments_input
            self.raw_pattern_lines = None
        else:
            self.raw_pattern_lines = self.guide_segments_input

    def _normalize_indices(self, values, name):
        """Validate one ring or arc index sequence.

        Args:
            values: Candidate node-index sequence.
            name: Input name used in validation errors.

        Returns:
            Unique int64 indices with the original input order preserved.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If the sequence shape, size, or uniqueness is invalid.
            IndexError: If an index is outside the mesh node range.
        """
        indices = np.asarray(values)
        if indices.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        minimum = 3 if self.closed else 2
        if indices.size < minimum:
            kind = "ring" if self.closed else "arc"
            raise ValueError(
                f"{name} must contain at least {minimum} nodes for a {kind}"
            )
        if (
            not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
        ):
            raise TypeError(f"{name} must contain integers")
        if np.any(indices < 0) or np.any(indices >= self.nodes.shape[0]):
            raise IndexError(f"{name} contain an out-of-range node index")
        indices = indices.astype(np.int64, copy=False)
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} must not contain duplicates")
        return indices

    def _fit_concentric_geometry(self):
        """Fit the common center and the two boundary radii.

        Raises:
            ValueError: If the selected nodes do not define two finite, distinct,
                concentric circles.

        Note:
            This stage also derives all linear, radial, and angular tolerances used by
            later geometry operations.
        """
        selected = np.vstack(
            (self.xy[self.inner_indices], self.xy[self.outer_indices])
        )
        origin = selected[0].copy()
        centered = selected - origin
        if not np.all(np.isfinite(centered)):
            raise ValueError("selected circle-coordinate differences exceed float64")
        scale = float(np.max(np.abs(centered)))
        if scale == 0.0:
            raise ValueError("the selected nodes do not determine circle radii")

        normalized = centered / scale
        design = np.zeros((selected.shape[0], 4), dtype=np.float64)
        design[:, :2] = 2.0 * normalized
        design[: self.inner_indices.size, 2] = 1.0
        design[self.inner_indices.size :, 3] = 1.0
        rhs = np.einsum("ij,ij->i", normalized, normalized)
        try:
            fit, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
        except np.linalg.LinAlgError as error:
            raise ValueError("the common circle fit could not be solved") from error
        if rank != 4 or not np.all(np.isfinite(fit)):
            raise ValueError(
                "the selected arcs do not determine two concentric circles"
            )

        normalized_center = fit[:2]
        center_norm_squared = float(np.dot(normalized_center, normalized_center))
        inner_squared = float(fit[2] + center_norm_squared)
        outer_squared = float(fit[3] + center_norm_squared)
        if (
            not np.isfinite(inner_squared)
            or not np.isfinite(outer_squared)
            or inner_squared <= 0.0
            or outer_squared <= 0.0
        ):
            raise ValueError("the fitted circle radii must be positive")

        with np.errstate(over="ignore", invalid="ignore"):
            self.center = origin + scale * normalized_center
            self.inner_radius = scale * np.sqrt(inner_squared)
            self.outer_radius = scale * np.sqrt(outer_squared)
        if (
            not np.all(np.isfinite(self.center))
            or not np.isfinite(self.inner_radius)
            or not np.isfinite(self.outer_radius)
        ):
            raise ValueError("the fitted circle geometry exceeds float64")

        coordinate_scale = max(
            float(np.max(np.abs(selected))),
            float(np.max(np.abs(self.center))),
            self.outer_radius,
        )
        coordinate_ulp = abs(float(np.spacing(coordinate_scale)))
        local_scale = max(scale, self.outer_radius)
        machine_tolerance = max(
            8.0 * coordinate_ulp,
            256.0 * np.finfo(np.float64).eps * local_scale,
        )
        self.geometry_tolerance = max(
            machine_tolerance, 1.0e-10 * self.outer_radius
        )
        if (
            self.inner_radius
            >= self.outer_radius - self.geometry_tolerance
        ):
            raise ValueError("inner_nodes must lie on the smaller fitted circle")

        inner_distances = np.hypot(
            self.xy[self.inner_indices, 0] - self.center[0],
            self.xy[self.inner_indices, 1] - self.center[1],
        )
        outer_distances = np.hypot(
            self.xy[self.outer_indices, 0] - self.center[0],
            self.xy[self.outer_indices, 1] - self.center[1],
        )
        self.fit_tolerance = max(
            self.geometry_tolerance, 1.0e-9 * self.outer_radius
        )
        if np.any(
            np.abs(inner_distances - self.inner_radius) > self.fit_tolerance
        ) or np.any(
            np.abs(outer_distances - self.outer_radius) > self.fit_tolerance
        ):
            raise ValueError(
                "inner_nodes and outer_nodes must lie on concentric circles"
            )
        self.angular_tolerance = max(
            256.0 * np.finfo(np.float64).eps,
            self.geometry_tolerance / self.inner_radius,
        )
        offsets = self.xy - self.center
        self.node_angles = np.mod(
            np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * np.pi
        )

    def _prepare_pattern_guides(self):
        """Prepare raw guide segments using the fitted circle scale."""
        if hasattr(self, "pattern_guides"):
            return
        coordinate_scale = max(
            1.0,
            float(np.max(np.abs(self.center))),
            self.outer_radius,
        )
        self.pattern_guides = _coerce_pattern_guides(
            self.raw_pattern_lines,
            coordinate_scale=coordinate_scale,
            minimum_tolerance=self.geometry_tolerance,
        )

    def _read_existing_topology(self):
        """Decode existing perimeters and collect directed edge uses.

        Raises:
            ValueError: If an element violates the mixed Tri3/Quad4 contract, an edge
                is non-manifold, or adjacent elements use inconsistent directions.
        """
        self.existing_edge_uses = {}
        self.existing_perimeters = []
        for element in self.elements:
            values = [int(value) for value in element]
            if values[2] == values[3]:
                perimeter = values[:3]
                if len(set(perimeter)) != 3:
                    raise ValueError(
                        "a padded Tri3 must contain three distinct perimeter nodes"
                    )
            else:
                perimeter = values
                if len(set(perimeter)) != 4:
                    raise ValueError(
                        "a Quad4 must contain four distinct nodes"
                    )
            self.existing_perimeters.append(tuple(perimeter))
            for start, end in zip(
                perimeter, perimeter[1:] + perimeter[:1]
            ):
                key = self._edge_key((start, end))
                uses = self.existing_edge_uses.setdefault(key, [])
                uses.append((start, end))
                if len(uses) > 2:
                    raise ValueError(
                        "mesh contains a non-manifold edge shared by more than "
                        "two elements"
                    )
        for uses in self.existing_edge_uses.values():
            if len(uses) == 2 and uses[0] != uses[1][::-1]:
                raise ValueError(
                    "elements sharing an edge must traverse it in opposite "
                    "directions"
                )

    @staticmethod
    def _edge_key(edge):
        """Return a direction-independent integer key for one edge.

        Args:
            edge: Pair of node indices.

        Returns:
            The endpoint pair in ascending order.
        """
        first, second = map(int, edge)
        return (min(first, second), max(first, second))

    @staticmethod
    def _cross(first, second):
        """Return the scalar two-dimensional cross product.

        Args:
            first: First XY vector.
            second: Second XY vector.

        Returns:
            The signed scalar cross product.
        """
        return first[0] * second[1] - first[1] * second[0]

    def _polygon_area(self, indices):
        """Return the signed area of an indexed polygon.

        Args:
            indices: Ordered polygon node indices.

        Returns:
            Positive area for counter-clockwise winding and negative area for
            clockwise winding.
        """
        points = self.xy[np.asarray(indices, dtype=np.intp)]
        points = points - points[0]
        following = np.roll(points, -1, axis=0)
        return 0.5 * float(
            np.sum(
                points[:, 0] * following[:, 1]
                - following[:, 0] * points[:, 1]
            )
        )

    def _area_tolerance(self, vertex_count):
        """Scale an area tolerance by polygon size.

        Args:
            vertex_count: Number of polygon vertices.

        Returns:
            A non-negative area tolerance for the current fitted geometry.
        """
        return (
            self.geometry_tolerance
            * max(self.geometry_tolerance, self.outer_radius)
            * max(1, int(vertex_count))
        )

    @staticmethod
    def _boundary_edges(indices, closing):
        """Build directed consecutive edges from a node sequence.

        Args:
            indices: Ordered boundary node indices.
            closing: Whether to include the last-to-first edge.

        Returns:
            Directed integer node-index pairs.
        """
        indices = np.asarray(indices, dtype=np.int64)
        count = indices.size if closing else indices.size - 1
        return [
            (
                int(indices[position]),
                int(indices[(position + 1) % indices.size]),
            )
            for position in range(count)
        ]

    def _segment_relation(self, first_edge, second_edge):
        """Classify the geometric relation between two edges.

        Args:
            first_edge: First node-index pair.
            second_edge: Second node-index pair.

        Returns:
            One of none, touch, cross, or overlap.
        """
        first_start, first_end = self.xy[list(first_edge)]
        second_start, second_end = self.xy[list(second_edge)]
        first_vector = first_end - first_start
        second_vector = second_end - second_start
        length_scale = max(
            self.geometry_tolerance,
            float(np.hypot(*first_vector)),
            float(np.hypot(*second_vector)),
        )
        area_tolerance = self.geometry_tolerance * length_scale

        if (
            max(
                min(first_start[0], first_end[0]),
                min(second_start[0], second_end[0]),
            )
            > min(
                max(first_start[0], first_end[0]),
                max(second_start[0], second_end[0]),
            )
            + self.geometry_tolerance
            or max(
                min(first_start[1], first_end[1]),
                min(second_start[1], second_end[1]),
            )
            > min(
                max(first_start[1], first_end[1]),
                max(second_start[1], second_end[1]),
            )
            + self.geometry_tolerance
        ):
            return "none"

        orientations = np.asarray(
            [
                self._cross(first_vector, second_start - first_start),
                self._cross(first_vector, second_end - first_start),
                self._cross(second_vector, first_start - second_start),
                self._cross(second_vector, first_end - second_start),
            ],
            dtype=np.float64,
        )
        if np.all(np.abs(orientations) <= area_tolerance):
            axis = int(np.argmax(np.abs(first_vector)))
            first_interval = sorted((first_start[axis], first_end[axis]))
            second_interval = sorted((second_start[axis], second_end[axis]))
            overlap = (
                min(first_interval[1], second_interval[1])
                - max(first_interval[0], second_interval[0])
            )
            if overlap > self.geometry_tolerance:
                return "overlap"
            if overlap >= -self.geometry_tolerance:
                return "touch"
            return "none"

        signs = np.where(
            orientations > area_tolerance,
            1,
            np.where(orientations < -area_tolerance, -1, 0),
        )
        if signs[0] * signs[1] <= 0 and signs[2] * signs[3] <= 0:
            return "cross"
        return "none"

    def _edges_are_compatible(self, first_edge, second_edge):
        """Return whether two edges may coexist in one mesh.

        Args:
            first_edge: First node-index pair.
            second_edge: Second node-index pair.

        Returns:
            True for disjoint edges, the same edge, or a permitted shared endpoint.
        """
        if self._edge_key(first_edge) == self._edge_key(second_edge):
            return True
        relation = self._segment_relation(first_edge, second_edge)
        if relation == "none":
            return True
        shared = set(first_edge).intersection(second_edge)
        return bool(shared) and relation != "overlap"

    def _point_on_segment(self, point, start, end):
        """Return whether a point lies on a closed segment.

        Args:
            point: XY coordinate to test.
            start: Segment start coordinate.
            end: Segment end coordinate.

        Returns:
            True when the point lies on the segment within geometry tolerance.
        """
        vector = end - start
        offset = point - start
        length = float(np.hypot(*vector))
        if length <= self.geometry_tolerance:
            return float(np.hypot(*offset)) <= self.geometry_tolerance
        if (
            abs(self._cross(vector, offset))
            > self.geometry_tolerance * length
        ):
            return False
        projection = float(np.dot(offset, vector))
        margin = self.geometry_tolerance * length
        return -margin <= projection <= length * length + margin

    def _point_on_polygon(self, point, polygon_ids):
        """Return whether a point lies on any polygon edge.

        Args:
            point: XY coordinate to test.
            polygon_ids: Ordered polygon node indices.

        Returns:
            True when the point lies on the polygon boundary.
        """
        polygon = self.xy[np.asarray(polygon_ids, dtype=np.intp)]
        return any(
            self._point_on_segment(point, start, end)
            for start, end in zip(
                polygon, np.roll(polygon, -1, axis=0)
            )
        )

    def _point_in_polygon(self, point, polygon_ids):
        """Classify a point using the polygon's odd-even crossing rule.

        Call :meth:`_point_on_polygon` first when boundary points must be
        distinguished from interior points.

        Args:
            point: XY coordinate to test.
            polygon_ids: Ordered polygon node indices.

        Returns:
            The odd-even ray-test result. Some boundary points can also
            produce True.
        """
        polygon = self.xy[np.asarray(polygon_ids, dtype=np.intp)]
        following = np.roll(polygon, -1, axis=0)
        crosses_y = (polygon[:, 1] > point[1]) != (
            following[:, 1] > point[1]
        )
        positions = np.flatnonzero(crosses_y)
        if positions.size == 0:
            return False
        starts = polygon[positions]
        ends = following[positions]
        intersection_x = starts[:, 0] + (
            (point[1] - starts[:, 1])
            * (ends[:, 0] - starts[:, 0])
            / (ends[:, 1] - starts[:, 1])
        )
        return bool(np.count_nonzero(point[0] < intersection_x) % 2)

    def _polygon_is_simple(self, polygon_ids):
        """Return whether a polygon has no invalid edge intersections.

        Args:
            polygon_ids: Ordered polygon node indices.

        Returns:
            True when only adjacent edges touch at shared endpoints.
        """
        edges = self._boundary_edges(polygon_ids, True)
        for first_position, first_edge in enumerate(edges):
            for second_position in range(first_position + 1, len(edges)):
                second_edge = edges[second_position]
                adjacent = (
                    second_position == first_position + 1
                    or (
                        first_position == 0
                        and second_position == len(edges) - 1
                    )
                )
                relation = self._segment_relation(first_edge, second_edge)
                if relation == "none":
                    continue
                if adjacent and relation != "overlap":
                    continue
                return False
        return True

    def _closed_order(self, indices, name):
        """Normalize a ring to deterministic counter-clockwise order.

        Args:
            indices: Unordered ring node indices.
            name: Boundary name used in validation errors.

        Returns:
            Counter-clockwise int64 node indices.

        Raises:
            ValueError: If two nodes have indistinguishable angular positions.
        """
        angles = self.node_angles[indices]
        order = np.lexsort((indices, angles))
        ordered = indices[order]
        ordered_angles = angles[order]
        steps = np.mod(
            np.roll(ordered_angles, -1) - ordered_angles, 2.0 * np.pi
        )
        if np.any(steps <= self.angular_tolerance):
            raise ValueError(f"{name} contain duplicate angular positions")
        return ordered.astype(np.int64, copy=False)

    def _open_orders(self, indices, name):
        """Enumerate valid endpoint-preserving orders for an open arc.

        Args:
            indices: Arc nodes whose first and last entries are fixed endpoints.
            name: Boundary name used in validation errors.

        Returns:
            Deterministic candidate node orders along the endpoint-bounded arc.

        Raises:
            ValueError: If the interior nodes do not lie on either valid directed arc.
        """
        angles = self.node_angles[indices]
        interior_positions = np.arange(1, indices.size - 1, dtype=np.intp)
        candidates = []
        endpoints = ((0, indices.size - 1), (indices.size - 1, 0))
        for start_position, end_position in endpoints:
            start_angle = angles[start_position]
            span = float(
                np.mod(angles[end_position] - start_angle, 2.0 * np.pi)
            )
            if span <= self.angular_tolerance:
                continue
            steps = np.mod(
                angles[interior_positions] - start_angle, 2.0 * np.pi
            )
            if steps.size and (
                np.any(steps <= self.angular_tolerance)
                or np.any(steps >= span - self.angular_tolerance)
            ):
                continue
            order = interior_positions[
                np.lexsort((indices[interior_positions], steps))
            ]
            if order.size:
                sorted_steps = np.mod(
                    angles[order] - start_angle, 2.0 * np.pi
                )
                if np.any(
                    np.diff(sorted_steps) <= self.angular_tolerance
                ):
                    continue
            candidate = np.concatenate(
                (
                    indices[[start_position]],
                    indices[order],
                    indices[[end_position]],
                )
            ).astype(np.int64, copy=False)
            signature = tuple(map(int, candidate))
            if all(tuple(map(int, item)) != signature for item in candidates):
                candidates.append(candidate)
        if not candidates:
            raise ValueError(
                f"the interior nodes of {name} do not lie on one "
                "endpoint-bounded arc"
            )
        return candidates

    def _order_boundaries(self):
        """Build closed-ring or open-arc ordering candidates.

        Open-arc pairs that do not enclose a valid strip polygon are omitted.
        The triangulation stage reports an error if no candidate remains.
        """
        if self.closed:
            self.inner_order_candidates = [
                self._closed_order(self.inner_indices, "inner_nodes")
            ]
            self.outer_order_candidates = [
                self._closed_order(self.outer_indices, "outer_nodes")
            ]
            self.open_geometry_candidates = []
            return

        self.inner_order_candidates = self._open_orders(
            self.inner_indices, "inner_nodes"
        )
        self.outer_order_candidates = self._open_orders(
            self.outer_indices, "outer_nodes"
        )
        self.open_geometry_candidates = []
        for outer_order in self.outer_order_candidates:
            for inner_order in self.inner_order_candidates:
                boundary = np.concatenate(
                    (outer_order, inner_order[::-1])
                ).astype(np.int64, copy=False)
                if self._polygon_area(boundary) <= self._area_tolerance(
                    boundary.size
                ):
                    continue
                if not self._polygon_is_simple(boundary):
                    continue
                self.open_geometry_candidates.append(
                    (outer_order, inner_order, boundary)
                )

    def _triangle_quality(self, first, second, third):
        """Measure orientation-normalized quality of one triangle.

        Args:
            first: First node index.
            second: Second node index.
            third: Third node index.

        Returns:
            Positive quality for a valid counter-clockwise triangle, or negative
            infinity for invalid geometry.
        """
        if len({int(first), int(second), int(third)}) != 3:
            return -np.inf
        points = self.xy[[first, second, third]]
        forward = np.roll(points, -1, axis=0) - points
        lengths = np.hypot(forward[:, 0], forward[:, 1])
        if np.any(lengths <= self.geometry_tolerance):
            return -np.inf
        backward = np.roll(points, 1, axis=0) - points
        corner_cross = (
            forward[:, 0] * backward[:, 1]
            - forward[:, 1] * backward[:, 0]
        )
        length_scale = max(
            self.geometry_tolerance, float(np.max(lengths))
        )
        if np.any(
            corner_cross <= self.geometry_tolerance * length_scale
        ):
            return -np.inf
        quality = (
            (2.0 / np.sqrt(3.0))
            * corner_cross
            / (lengths * np.roll(lengths, 1))
        )
        if not np.all(np.isfinite(quality)):
            return -np.inf
        return float(np.min(quality))

    def _validate_existing_geometry(self):
        """Reject malformed existing Tri3 and Quad4 elements.

        Raises:
            ValueError: If an existing cell is non-simple, inverted, or
                degenerate.
        """
        for perimeter in self.existing_perimeters:
            if not self._polygon_is_simple(perimeter):
                raise ValueError(
                    "existing elements must have simple perimeters"
                )
            if len(perimeter) == 3:
                if not np.isfinite(self._triangle_quality(*perimeter)):
                    raise ValueError(
                        "existing Tri3 elements must be non-degenerate and "
                        "counter-clockwise"
                    )
                continue

            points = self.xy[np.asarray(perimeter, dtype=np.intp)]
            forward = np.roll(points, -1, axis=0) - points
            backward = np.roll(points, 1, axis=0) - points
            forward_lengths = np.hypot(
                forward[:, 0], forward[:, 1]
            )
            backward_lengths = np.hypot(
                backward[:, 0], backward[:, 1]
            )
            if np.any(forward_lengths <= self.geometry_tolerance):
                raise ValueError(
                    "existing Quad4 elements have a zero-length edge"
                )
            corner_cross = (
                forward[:, 0] * backward[:, 1]
                - forward[:, 1] * backward[:, 0]
            )
            normalized_turn = corner_cross / (
                forward_lengths * backward_lengths
            )
            if (
                not np.all(np.isfinite(normalized_turn))
                or np.any(
                    normalized_turn <= self.angular_tolerance
                )
            ):
                raise ValueError(
                    "existing Quad4 elements must be non-degenerate and "
                    "counter-clockwise"
                )

    def _axis_interval_is_active(
        self,
        fixed_axis,
        varying_axis,
        fixed_value,
        interval_lower,
        interval_upper,
        tolerance,
    ):
        """Return whether an axis-aligned interval is active in the strip.

        Closed rings accept every interval longer than tolerance. Open arcs
        additionally require an interior portion inside a candidate polygon.

        Args:
            fixed_axis: Axis held constant by the pattern segment.
            varying_axis: Axis along which the segment interval varies.
            fixed_value: Coordinate on the fixed axis.
            interval_lower: Lower coordinate on the varying axis.
            interval_upper: Upper coordinate on the varying axis.
            tolerance: Linear pattern-segment tolerance.

        Returns:
            True for a nontrivial closed-ring interval, or when an open-arc
            interval enters a candidate strip polygon.
        """
        if self.closed:
            return interval_upper - interval_lower > tolerance
        if interval_upper - interval_lower <= tolerance:
            return False

        for _, _, polygon_ids in self.open_geometry_candidates:
            polygon = self.xy[np.asarray(polygon_ids, dtype=np.intp)]
            cuts = [float(interval_lower), float(interval_upper)]
            for start, end in zip(
                polygon, np.roll(polygon, -1, axis=0)
            ):
                fixed_start = float(start[fixed_axis])
                fixed_end = float(end[fixed_axis])
                if (
                    fixed_value
                    < min(fixed_start, fixed_end) - tolerance
                    or fixed_value
                    > max(fixed_start, fixed_end) + tolerance
                ):
                    continue
                fixed_delta = fixed_end - fixed_start
                if abs(fixed_delta) <= tolerance:
                    if abs(fixed_value - fixed_start) <= tolerance:
                        cuts.extend(
                            (
                                float(start[varying_axis]),
                                float(end[varying_axis]),
                            )
                        )
                    continue
                fraction = (
                    fixed_value - fixed_start
                ) / fixed_delta
                if -tolerance <= fraction <= 1.0 + tolerance:
                    cuts.append(
                        float(
                            start[varying_axis]
                            + fraction
                            * (end[varying_axis] - start[varying_axis])
                        )
                    )

            clipped = sorted(
                max(interval_lower, min(interval_upper, value))
                for value in cuts
                if interval_lower - tolerance
                <= value
                <= interval_upper + tolerance
            )
            distinct = []
            for value in clipped:
                if (
                    not distinct
                    or value - distinct[-1] > tolerance
                ):
                    distinct.append(value)
                else:
                    distinct[-1] = 0.5 * (
                        distinct[-1] + value
                    )
            for lower, upper in zip(distinct, distinct[1:]):
                if upper - lower <= tolerance:
                    continue
                midpoint = np.empty(2, dtype=np.float64)
                midpoint[fixed_axis] = fixed_value
                midpoint[varying_axis] = 0.5 * (lower + upper)
                if self._point_on_polygon(midpoint, polygon_ids):
                    continue
                if self._point_in_polygon(midpoint, polygon_ids):
                    return True
        return False

    def _match_circle_node(
        self, indices, target, description, tolerance
    ):
        """Match an exact pattern intersection to one ring node.

        Args:
            indices: Candidate ring-node indices.
            target: Exact XY intersection coordinate.
            description: Ring name used in validation errors.
            tolerance: Maximum distance from the target for a match.

        Returns:
            The uniquely matching node index, or None when no node matches.

        Raises:
            ValueError: If multiple candidates match.
        """
        distances = np.hypot(
            self.xy[indices, 0] - target[0],
            self.xy[indices, 1] - target[1],
        )
        matches = np.flatnonzero(distances <= tolerance)
        if matches.size == 0:
            return None
        if matches.size > 1:
            raise ValueError(
                f"an active pattern segment has an ambiguous "
                f"{description} node"
            )
        return int(indices[int(matches[0])])

    def _collect_pattern_connectors(self):
        """Resolve active pattern segments into mandatory ring connectors.

        Raises:
            ValueError: If a segment is malformed, crosses the strip without supported
                anchors, creates conflicting connectors, or violates connector order.
        """
        connectors = set()
        for segment in self.pattern_guides:
            fixed_axis = segment.fixed_axis
            varying_axis = segment.varying_axis
            fixed_tolerance = max(
                self.geometry_tolerance,
                segment.fixed_tolerance,
            )
            bound_tolerance = max(
                self.geometry_tolerance,
                segment.bound_tolerance,
            )
            intersection_tolerance = fixed_tolerance
            fixed = segment.fixed_value
            segment_lower = segment.lower
            segment_upper = segment.upper

            outer_points = _circle_line_intersections(
                segment,
                self.center,
                self.outer_radius,
                tolerance=intersection_tolerance,
            )
            # An outer tangent has no nontrivial interval inside the strip.
            if outer_points.shape[0] < 2:
                continue
            outer_lower, outer_upper = outer_points[:, varying_axis]

            inner_points = _circle_line_intersections(
                segment,
                self.center,
                self.inner_radius,
                tolerance=intersection_tolerance,
            )
            if inner_points.shape[0] == 0:
                overlap_lower, overlap_upper, overlaps = _interval_overlap(
                    segment_lower,
                    segment_upper,
                    outer_lower,
                    outer_upper,
                    intersection_tolerance,
                )
                active = overlaps and self._axis_interval_is_active(
                    fixed_axis,
                    varying_axis,
                    fixed,
                    overlap_lower,
                    overlap_upper,
                    intersection_tolerance,
                )
                if active:
                    raise ValueError(
                        "an active pattern segment forms an unsupported "
                        "outer-circle chord"
                    )
                continue

            if inner_points.shape[0] == 1:
                inner_lower = inner_upper = float(
                    inner_points[0, varying_axis]
                )
            else:
                inner_lower, inner_upper = inner_points[:, varying_axis]

            for outer_value, inner_value in (
                (outer_lower, inner_lower),
                (outer_upper, inner_upper),
            ):
                branch_lower = min(outer_value, inner_value)
                branch_upper = max(outer_value, inner_value)
                if branch_upper - branch_lower <= intersection_tolerance:
                    continue
                overlap_lower, overlap_upper, overlaps = _interval_overlap(
                    segment_lower,
                    segment_upper,
                    branch_lower,
                    branch_upper,
                    intersection_tolerance,
                )
                if not overlaps:
                    continue
                active = self._axis_interval_is_active(
                    fixed_axis,
                    varying_axis,
                    fixed,
                    overlap_lower,
                    overlap_upper,
                    intersection_tolerance,
                )
                if not active:
                    continue
                if not (
                    segment_lower
                    <= branch_lower + bound_tolerance
                    and segment_upper
                    >= branch_upper - bound_tolerance
                ):
                    raise ValueError(
                        "a pattern segment ends inside the annular material"
                    )

                outer_target = np.empty(2, dtype=np.float64)
                inner_target = np.empty(2, dtype=np.float64)
                outer_target[fixed_axis] = fixed
                outer_target[varying_axis] = outer_value
                inner_target[fixed_axis] = fixed
                inner_target[varying_axis] = inner_value
                match_tolerance = max(
                    self.fit_tolerance,
                    fixed_tolerance,
                    self.geometry_tolerance,
                )
                outer_node = self._match_circle_node(
                    self.outer_indices,
                    outer_target,
                    "outer circle",
                    match_tolerance,
                )
                inner_node = self._match_circle_node(
                    self.inner_indices,
                    inner_target,
                    "inner circle",
                    match_tolerance,
                )
                if outer_node is None:
                    raise ValueError(
                        "an active pattern segment reaches outer circle "
                        "without a node"
                    )
                if inner_node is None:
                    raise ValueError(
                        "an active pattern segment reaches inner circle "
                        "without a node"
                    )
                connectors.add((outer_node, inner_node))

        self.forced_connectors = sorted(connectors)
        for position, first in enumerate(self.forced_connectors):
            for second in self.forced_connectors[position + 1 :]:
                if not self._edges_are_compatible(first, second):
                    raise ValueError(
                        "pattern connectors cross or overlap"
                    )

    def _commit(self, new_elements):
        """Append validated elements while preserving index capacity.

        Args:
            new_elements: Final mixed Tri3/Quad4 connectivity to append.

        Note:
            This is the only pipeline stage that mutates the caller's mesh.
        """
        element_dtype = self.elements.dtype
        dtype_info = np.iinfo(element_dtype)
        minimum = int(np.min(new_elements))
        maximum = int(np.max(new_elements))
        if minimum < dtype_info.min or maximum > dtype_info.max:
            element_dtype = np.dtype(np.int64)
        self.mesh.elements = np.concatenate(
            (
                self.elements.astype(element_dtype, copy=False),
                new_elements.astype(element_dtype, copy=False),
            ),
            axis=0,
        )


def _mesh_inner_outer_circle(
    mesh: Mesh2D,
    inner_nodes,
    outer_nodes,
    guide_segments=None,
    closed: bool = True,
    min_quad_scaled_jacobian: float = 0.3,
) -> Mesh2D:
    """Mesh the strip between two concentric circular boundaries.

    Existing nodes remain fixed.  Cross-ring connectors are selected by a
    global monotone matching that first minimizes the worst angular mismatch,
    then the total squared mismatch, and finally maximizes the worst triangle
    quality.  Valid adjacent triangle pairs are then greedily merged into
    Quad4 elements when their minimum scaled Jacobian is at least
    ``min_quad_scaled_jacobian``. Pattern guide segments become mandatory
    connector edges and are never removed by merging.

    Args:
        mesh: Mesh2D whose existing nodes remain fixed and whose element array is
            extended only after successful validation.
        inner_nodes: Node indices on the smaller circular ring or arc.
        outer_nodes: Node indices on the larger circular ring or arc.
        guide_segments: Optional finite horizontal or vertical pattern segments with
            shape (L, 2, 2).
        closed: True for complete rings or False for endpoint-bounded arcs.
        min_quad_scaled_jacobian: Minimum scaled Jacobian in [0, 1] for Quad4 merging.

    Returns:
        The same mesh instance with the validated strip elements appended.

    Raises:
        TypeError: If mesh, closed, or an index sequence has an invalid type.
        ValueError: If inputs, topology, fitted geometry, constraints, or the
            generated strip are invalid.
        IndexError: If a boundary contains an out-of-range node index.

    Note:
        The node array is never replaced or modified. Element mutation occurs
        only after the complete strip has passed validation.
    """
    return _CircularStripMesher(
        mesh, inner_nodes, outer_nodes, guide_segments, closed, min_quad_scaled_jacobian
    ).build()
