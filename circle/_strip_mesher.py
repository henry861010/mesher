"""Generate and validate mixed-element strips between circular rings."""

from functools import partial
from typing import NamedTuple

import numpy as np

from mesh import Mesh


class _StripScore(NamedTuple):
    """Lexicographic objective used to compare triangulations.

    Attributes:
        worst_mismatch: Largest angular mismatch among chosen connectors.
        total_squared_mismatch: Sum of squared connector mismatches.
        minimum_quality: Worst normalized quality among generated triangles.
    """
    worst_mismatch: float
    total_squared_mismatch: float
    minimum_quality: float


class _CircularStripMesher:
    """Build a conforming mixed-element strip between concentric boundaries.

    The object is a staged coordinator. Each private stage validates or derives
    state needed by the next one, while build preserves the original order:
    normalize inputs, fit geometry, read topology, order boundaries, validate
    existing geometry, resolve pattern connectors, triangulate, validate the
    new strip, merge quads, and commit.

    Attributes:
        mesh: Mesh receiving elements only after every stage succeeds.
        inner_input: Raw inner-boundary node indices.
        outer_input: Raw outer-boundary node indices.
        lines_input: Raw axis-aligned pattern segments.
        closed_input: Raw closed-ring mode.
        jacobian_input: Raw minimum Quad4 scaled Jacobian.
    """

    def __init__(
        self,
        mesh,
        inner_nodes,
        outer_nodes,
        lines,
        closed,
        jacobian=0.3,
    ):
        """Store raw inputs for deferred, transactional validation.

        Args:
            mesh: Mesh that receives generated elements after validation.
            inner_nodes: Node indices on the smaller circular boundary.
            outer_nodes: Node indices on the larger circular boundary.
            lines: Optional axis-aligned pattern segments.
            closed: Whether both boundaries are closed rings rather than arcs.
            jacobian: Minimum scaled Jacobian for Quad4 merging.
        """
        self.mesh = mesh
        self.inner_input = inner_nodes
        self.outer_input = outer_nodes
        self.lines_input = lines
        self.closed_input = closed
        self.jacobian_input = jacobian

    def build(self):
        """Run every meshing stage and commit the validated strip.

        Returns:
            The same Mesh instance with generated elements appended.

        Raises:
            TypeError: If an input has an unsupported type.
            ValueError: If inputs, topology, constraints, or geometry are invalid.
            IndexError: If a boundary contains an out-of-range node index.

        Note:
            Mesh mutation is deferred until the final commit stage.
        """
        self._read_inputs()
        self._fit_concentric_geometry()
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
        """Normalize mesh arrays, indices, options, and pattern lines.

        Raises:
            TypeError: If mesh, closed, or node-index types are invalid.
            ValueError: If array shapes, scalar ranges, or coordinates are invalid.
            IndexError: If a ring contains an out-of-range node index.
        """
        if not isinstance(self.mesh, Mesh):
            raise TypeError("mesh must be a Mesh instance")
        if not isinstance(self.closed_input, (bool, np.bool_)):
            raise TypeError("closed must be True or False")
        self.closed = bool(self.closed_input)
        try:
            self.minimum_quad_scaled_jacobian = float(
                self.jacobian_input
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("jacobian must be a real number") from error
        if not np.isfinite(self.minimum_quad_scaled_jacobian):
            raise ValueError("jacobian must be finite")
        if not 0.0 <= self.minimum_quad_scaled_jacobian <= 1.0:
            raise ValueError("jacobian must be between 0 and 1")

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

        if self.lines_input is None:
            self.pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
            return
        try:
            pattern_lines = np.asarray(self.lines_input, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("lines must have shape (L, 2, 2)") from error
        if pattern_lines.size == 0:
            if pattern_lines.shape not in ((0,), (0, 2, 2)):
                raise ValueError("lines must have shape (L, 2, 2)")
            pattern_lines = np.empty((0, 2, 2), dtype=np.float64)
        elif pattern_lines.ndim != 3 or pattern_lines.shape[1:] != (2, 2):
            raise ValueError("lines must have shape (L, 2, 2)")
        if not np.all(np.isfinite(pattern_lines)):
            raise ValueError("lines must contain finite coordinates")
        self.pattern_lines = pattern_lines

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
        for segment in self.pattern_lines:
            delta = segment[1] - segment[0]
            axis_scales = np.max(np.abs(segment), axis=0)
            axis_tolerances = np.maximum(
                self.geometry_tolerance,
                8.0 * np.abs(np.spacing(axis_scales)),
            )
            vertical = (
                abs(delta[0]) <= axis_tolerances[0]
                and abs(delta[1]) > axis_tolerances[1]
            )
            horizontal = (
                abs(delta[1]) <= axis_tolerances[1]
                and abs(delta[0]) > axis_tolerances[0]
            )
            if vertical == horizontal:
                raise ValueError(
                    "each pattern segment must be non-zero and horizontal "
                    "or vertical"
                )

            fixed_axis = 0 if vertical else 1
            varying_axis = 1 - fixed_axis
            fixed_tolerance = float(axis_tolerances[fixed_axis])
            bound_tolerance = float(axis_tolerances[varying_axis])
            intersection_tolerance = max(
                self.geometry_tolerance, fixed_tolerance
            )
            fixed = float(segment[0, fixed_axis])
            segment_lower = float(
                np.min(segment[:, varying_axis])
            )
            segment_upper = float(
                np.max(segment[:, varying_axis])
            )
            fixed_distance = abs(fixed - self.center[fixed_axis])
            if (
                fixed_distance
                > self.outer_radius + fixed_tolerance
            ):
                continue
            if (
                abs(fixed_distance - self.outer_radius)
                <= fixed_tolerance
            ):
                continue

            normalized_outer = min(
                fixed_distance / self.outer_radius, 1.0
            )
            outer_root = self.outer_radius * np.sqrt(
                max(0.0, 1.0 - normalized_outer**2)
            )
            if outer_root <= intersection_tolerance:
                continue
            outer_lower = (
                self.center[varying_axis] - outer_root
            )
            outer_upper = (
                self.center[varying_axis] + outer_root
            )

            if (
                fixed_distance
                > self.inner_radius + fixed_tolerance
            ):
                overlap_lower = max(segment_lower, outer_lower)
                overlap_upper = min(segment_upper, outer_upper)
                active = (
                    overlap_upper - overlap_lower
                    > intersection_tolerance
                    and self._axis_interval_is_active(
                        fixed_axis,
                        varying_axis,
                        fixed,
                        overlap_lower,
                        overlap_upper,
                        intersection_tolerance,
                    )
                )
                if active:
                    raise ValueError(
                        "an active pattern segment forms an unsupported "
                        "outer-circle chord"
                    )
                continue

            if (
                abs(fixed_distance - self.inner_radius)
                <= fixed_tolerance
            ):
                inner_lower = inner_upper = self.center[varying_axis]
            else:
                normalized_inner = min(
                    fixed_distance / self.inner_radius, 1.0
                )
                inner_root = self.inner_radius * np.sqrt(
                    max(0.0, 1.0 - normalized_inner**2)
                )
                inner_lower = (
                    self.center[varying_axis] - inner_root
                )
                inner_upper = (
                    self.center[varying_axis] + inner_root
                )

            for first_value, second_value in (
                (outer_lower, inner_lower),
                (inner_upper, outer_upper),
            ):
                branch_lower = min(first_value, second_value)
                branch_upper = max(first_value, second_value)
                if (
                    branch_upper - branch_lower
                    <= intersection_tolerance
                ):
                    continue
                overlap_lower = max(segment_lower, branch_lower)
                overlap_upper = min(segment_upper, branch_upper)
                if (
                    overlap_upper - overlap_lower
                    <= intersection_tolerance
                ):
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

                if first_value in (outer_lower, outer_upper):
                    outer_value = first_value
                    inner_value = second_value
                else:
                    outer_value = second_value
                    inner_value = first_value
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

    def _connector_mismatch(self, outer_node, inner_node):
        """Return angular mismatch for one cross-ring connector.

        Args:
            outer_node: Node index on the outer boundary.
            inner_node: Node index on the inner boundary.

        Returns:
            Absolute wrapped angular separation in radians.
        """
        difference = (
            self.node_angles[int(outer_node)]
            - self.node_angles[int(inner_node)]
        )
        return abs(
            float(np.arctan2(np.sin(difference), np.cos(difference)))
        )

    def _score_better(self, candidate, current):
        """Compare scores in connector-locality priority order.

        Args:
            candidate: Score being considered.
            current: Best score so far, or None when no path exists yet.

        Returns:
            True when the candidate has a lower worst mismatch, then a lower
            total squared mismatch, then a higher minimum triangle quality.
        """
        if current is None:
            return True
        angular_epsilon = max(
            self.angular_tolerance, 32.0 * np.finfo(np.float64).eps
        )
        if (
            candidate.worst_mismatch
            < current.worst_mismatch - angular_epsilon
        ):
            return True
        if (
            candidate.worst_mismatch
            > current.worst_mismatch + angular_epsilon
        ):
            return False
        sum_epsilon = max(
            1.0e-15,
            angular_epsilon
            * max(
                1.0,
                abs(candidate.total_squared_mismatch),
                abs(current.total_squared_mismatch),
            ),
        )
        if (
            candidate.total_squared_mismatch
            < current.total_squared_mismatch - sum_epsilon
        ):
            return True
        if (
            candidate.total_squared_mismatch
            > current.total_squared_mismatch + sum_epsilon
        ):
            return False
        quality_epsilon = 1.0e-15
        return (
            candidate.minimum_quality
            > current.minimum_quality + quality_epsilon
        )

    @staticmethod
    def _combine_scores(scores):
        """Combine independent sector scores into one strip score.

        Args:
            scores: Per-sector triangulation scores.

        Returns:
            A score using the worst mismatch, summed squared mismatch, and worst
            triangle quality across all sectors.
        """
        return _StripScore(
            max(score.worst_mismatch for score in scores),
            sum(score.total_squared_mismatch for score in scores),
            min(score.minimum_quality for score in scores),
        )

    def _triangulate_sector(
        self, outer_arc, inner_arc, connector_is_valid
    ):
        """Find the globally best monotone triangulation of one sector.

        Args:
            outer_arc: Ordered outer nodes including both sector endpoints.
            inner_arc: Ordered inner nodes including both sector endpoints.
            connector_is_valid: Predicate for candidate cross-ring edges.

        Returns:
            A pair containing padded Tri3 connectivity and its global score.

        Raises:
            ValueError: If the endpoints or dynamic-programming grid admit no
                valid monotone triangulation.
            RuntimeError: If a reachable grid state has no predecessor during
                reconstruction.
        """
        outer_arc = np.asarray(outer_arc, dtype=np.int64)
        inner_arc = np.asarray(inner_arc, dtype=np.int64)
        outer_steps = outer_arc.size - 1
        inner_steps = inner_arc.size - 1
        worst = np.full(
            (outer_steps + 1, inner_steps + 1),
            np.inf,
            dtype=np.float64,
        )
        total = np.full_like(worst, np.inf)
        minimum_quality = np.full_like(worst, -np.inf)
        predecessor = np.zeros(worst.shape, dtype=np.int8)

        start_outer = int(outer_arc[0])
        start_inner = int(inner_arc[0])
        if not connector_is_valid(start_outer, start_inner):
            raise ValueError(
                "a sector start connector leaves the annular material"
            )
        start_mismatch = self._connector_mismatch(
            start_outer, start_inner
        )
        worst[0, 0] = start_mismatch
        total[0, 0] = start_mismatch**2
        minimum_quality[0, 0] = np.inf

        for outer_position in range(outer_steps + 1):
            for inner_position in range(inner_steps + 1):
                if outer_position == 0 and inner_position == 0:
                    continue
                outer_node = int(outer_arc[outer_position])
                inner_node = int(inner_arc[inner_position])
                if not connector_is_valid(outer_node, inner_node):
                    continue
                mismatch = self._connector_mismatch(
                    outer_node, inner_node
                )
                chosen_score = None
                chosen_source = 0

                if (
                    outer_position > 0
                    and np.isfinite(
                        worst[outer_position - 1, inner_position]
                    )
                ):
                    quality = self._triangle_quality(
                        int(outer_arc[outer_position - 1]),
                        outer_node,
                        inner_node,
                    )
                    if np.isfinite(quality):
                        candidate = _StripScore(
                            max(
                                worst[
                                    outer_position - 1, inner_position
                                ],
                                mismatch,
                            ),
                            total[
                                outer_position - 1, inner_position
                            ]
                            + mismatch**2,
                            min(
                                minimum_quality[
                                    outer_position - 1, inner_position
                                ],
                                quality,
                            ),
                        )
                        if self._score_better(
                            candidate, chosen_score
                        ):
                            chosen_score = candidate
                            chosen_source = 1

                if (
                    inner_position > 0
                    and np.isfinite(
                        worst[outer_position, inner_position - 1]
                    )
                ):
                    quality = self._triangle_quality(
                        outer_node,
                        inner_node,
                        int(inner_arc[inner_position - 1]),
                    )
                    if np.isfinite(quality):
                        candidate = _StripScore(
                            max(
                                worst[
                                    outer_position, inner_position - 1
                                ],
                                mismatch,
                            ),
                            total[
                                outer_position, inner_position - 1
                            ]
                            + mismatch**2,
                            min(
                                minimum_quality[
                                    outer_position,
                                    inner_position - 1,
                                ],
                                quality,
                            ),
                        )
                        if self._score_better(
                            candidate, chosen_score
                        ):
                            chosen_score = candidate
                            chosen_source = 2

                if chosen_score is not None:
                    worst[outer_position, inner_position] = (
                        chosen_score.worst_mismatch
                    )
                    total[outer_position, inner_position] = (
                        chosen_score.total_squared_mismatch
                    )
                    minimum_quality[
                        outer_position, inner_position
                    ] = chosen_score.minimum_quality
                    predecessor[
                        outer_position, inner_position
                    ] = chosen_source

        score = _StripScore(
            worst[outer_steps, inner_steps],
            total[outer_steps, inner_steps],
            minimum_quality[outer_steps, inner_steps],
        )
        if not np.all(np.isfinite(score)):
            raise ValueError(
                "the constrained sector has no valid triangulation"
            )

        rows = []
        outer_position = outer_steps
        inner_position = inner_steps
        while outer_position or inner_position:
            source = int(
                predecessor[outer_position, inner_position]
            )
            if source == 1:
                triangle = [
                    int(outer_arc[outer_position - 1]),
                    int(outer_arc[outer_position]),
                    int(inner_arc[inner_position]),
                ]
                outer_position -= 1
            elif source == 2:
                triangle = [
                    int(outer_arc[outer_position]),
                    int(inner_arc[inner_position]),
                    int(inner_arc[inner_position - 1]),
                ]
                inner_position -= 1
            else:
                raise RuntimeError(
                    "missing triangulation predecessor"
                )
            rows.append([*triangle, triangle[-1]])
        rows.reverse()
        return np.asarray(rows, dtype=np.int64), score

    def _protected_connector_is_valid(self, edge, protected_edges):
        """Return whether an edge avoids every protected connector.

        Args:
            edge: Candidate connector edge.
            protected_edges: Pattern or side edges that cannot be crossed.

        Returns:
            True when the candidate is compatible with all protected edges.
        """
        return all(
            self._edges_are_compatible(edge, protected)
            for protected in protected_edges
        )

    def _validate_closed_boundaries(self, outer_order, inner_order):
        """Validate winding, convexity, and strict ring containment.

        Args:
            outer_order: Counter-clockwise outer-ring node order.
            inner_order: Counter-clockwise inner-ring node order.

        Raises:
            ValueError: If either ring is misoriented or non-convex, or the outer
                polygon does not strictly contain the inner polygon.
        """
        if (
            self._polygon_area(outer_order) <= 0.0
            or self._polygon_area(inner_order) <= 0.0
        ):
            raise ValueError(
                "circle rings could not be normalized counter-clockwise"
            )
        for order, name in (
            (outer_order, "outer"),
            (inner_order, "inner"),
        ):
            points = self.xy[order]
            previous = points - np.roll(points, 1, axis=0)
            following = np.roll(points, -1, axis=0) - points
            turns = (
                previous[:, 0] * following[:, 1]
                - previous[:, 1] * following[:, 0]
            )
            local_scale = np.maximum(
                self.geometry_tolerance,
                np.maximum(
                    np.hypot(previous[:, 0], previous[:, 1]),
                    np.hypot(following[:, 0], following[:, 1]),
                ),
            )
            if np.any(
                turns <= self.geometry_tolerance * local_scale
            ):
                raise ValueError(
                    f"the {name} circle polygon must be strictly convex"
                )

        outer_points = self.xy[outer_order]
        inner_points = self.xy[inner_order]
        for start, end in zip(
            outer_points, np.roll(outer_points, -1, axis=0)
        ):
            edge = end - start
            signed_offsets = (
                edge[0] * (inner_points[:, 1] - start[1])
                - edge[1] * (inner_points[:, 0] - start[0])
            )
            threshold = self.geometry_tolerance * max(
                self.geometry_tolerance, float(np.hypot(*edge))
            )
            if np.any(signed_offsets <= threshold):
                raise ValueError(
                    "the outer circle polygon must strictly contain the "
                    "inner polygon"
                )

    def _closed_connector_is_valid(
        self,
        outer_node,
        inner_node,
        *,
        inner_order,
        inner_positions,
        protected_edges,
        cache,
    ):
        """Return whether a connector stays outside the inner polygon.

        Args:
            outer_node: Candidate node on the outer ring.
            inner_node: Candidate node on the inner ring.
            inner_order: Counter-clockwise inner-ring node order.
            inner_positions: Mapping from inner node to its ring position.
            protected_edges: Pattern connectors that may not be crossed.
            cache: Mutable connector-result cache for this triangulation.

        Returns:
            True when the connector leaves the inner polygon outward and is
            compatible with every protected edge.
        """
        key = (int(outer_node), int(inner_node))
        if key in cache:
            return cache[key]
        inner_position = inner_positions[int(inner_node)]
        inner_point = self.xy[int(inner_node)]
        vector = self.xy[int(outer_node)] - inner_point
        incoming = inner_point - self.xy[
            inner_order[(inner_position - 1) % inner_order.size]
        ]
        outgoing = (
            self.xy[inner_order[(inner_position + 1) % inner_order.size]]
            - inner_point
        )
        local_scale = max(
            self.geometry_tolerance,
            float(np.hypot(*vector)),
            float(np.hypot(*incoming)),
            float(np.hypot(*outgoing)),
        )
        tolerance = self.geometry_tolerance * local_scale
        valid = (
            self._cross(incoming, vector) < -tolerance
            or self._cross(outgoing, vector) < -tolerance
        )
        if valid:
            valid = self._protected_connector_is_valid(
                key,
                protected_edges,
            )
        cache[key] = valid
        return valid

    def _open_connector_is_valid(
        self,
        outer_node,
        inner_node,
        *,
        side_keys,
        polygon_edges,
        edge_minimums,
        edge_maximums,
        boundary,
        protected_edges,
        cache,
    ):
        """Return whether a connector lies within one open strip polygon.

        Args:
            outer_node: Candidate node on the outer arc.
            inner_node: Candidate node on the inner arc.
            side_keys: Undirected keys of the two strip side edges.
            polygon_edges: Directed perimeter edges of the open strip.
            edge_minimums: Per-edge XY bounding-box minima.
            edge_maximums: Per-edge XY bounding-box maxima.
            boundary: Ordered node polygon enclosing the open strip.
            protected_edges: Pattern and side connectors that may not cross.
            cache: Mutable connector-result cache for this candidate geometry.

        Returns:
            True when the connector is contained by the strip and compatible
            with every boundary and protected edge.
        """
        key = (int(outer_node), int(inner_node))
        if key in cache:
            return cache[key]
        if self._edge_key(key) in side_keys:
            cache[key] = True
            return True

        points = self.xy[list(key)]
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        candidates = np.flatnonzero(
            np.all(
                (
                    minimum
                    <= edge_maximums + self.geometry_tolerance
                )
                & (
                    maximum
                    >= edge_minimums - self.geometry_tolerance
                ),
                axis=1,
            )
        )
        valid = True
        for position in candidates:
            polygon_edge = polygon_edges[int(position)]
            if self._edge_key(key) == self._edge_key(polygon_edge):
                continue
            if not self._edges_are_compatible(key, polygon_edge):
                valid = False
                break
        if valid:
            midpoint = 0.5 * (self.xy[key[0]] + self.xy[key[1]])
            valid = self._point_in_polygon(midpoint, boundary)
        if valid:
            valid = self._protected_connector_is_valid(
                key,
                protected_edges,
            )
        cache[key] = valid
        return valid

    def _triangulate_closed(self):
        """Choose a globally scored triangulation for a closed annulus.

        Returns:
            Padded Tri3 rows covering the complete annulus.

        Raises:
            ValueError: If pattern connectors cannot be cyclically ordered or no valid
                seam and sector triangulation exists.
        """
        outer_order = self.outer_order_candidates[0]
        inner_order = self.inner_order_candidates[0]
        self._validate_closed_boundaries(outer_order, inner_order)
        protected = [tuple(edge) for edge in self.forced_connectors]
        inner_positions = {
            int(node): position
            for position, node in enumerate(inner_order)
        }
        connector_cache = {}
        connector_is_valid = partial(
            self._closed_connector_is_valid,
            inner_order=inner_order,
            inner_positions=inner_positions,
            protected_edges=protected,
            cache=connector_cache,
        )

        outer_positions = {
            int(node): position
            for position, node in enumerate(outer_order)
        }
        anchors = [
            (
                outer_positions[int(outer_node)],
                inner_positions[int(inner_node)],
                int(outer_node),
                int(inner_node),
            )
            for outer_node, inner_node in self.forced_connectors
        ]
        if not anchors:
            seam_candidates = []
            for outer_node in outer_order:
                for inner_node in inner_order:
                    outer_node = int(outer_node)
                    inner_node = int(inner_node)
                    if not connector_is_valid(
                        outer_node, inner_node
                    ):
                        continue
                    seam_candidates.append(
                        (
                            self._connector_mismatch(
                                outer_node, inner_node
                            ),
                            float(
                                np.hypot(
                                    *(
                                        self.xy[outer_node]
                                        - self.xy[inner_node]
                                    )
                                )
                            ),
                            outer_node,
                            inner_node,
                        )
                    )
            if not seam_candidates:
                raise ValueError(
                    "no valid seam exists between the two circles"
                )
            _, _, outer_node, inner_node = min(seam_candidates)
            anchors = [
                (
                    outer_positions[outer_node],
                    inner_positions[inner_node],
                    outer_node,
                    inner_node,
                )
            ]

        outer_count = outer_order.size
        inner_count = inner_order.size
        anchor_orders = []
        for seed in anchors:
            seed_outer, seed_inner = seed[:2]
            ordered = sorted(
                anchors,
                key=lambda anchor: (
                    (anchor[0] - seed_outer) % outer_count,
                    (anchor[1] - seed_inner) % inner_count,
                    anchor[2],
                    anchor[3],
                ),
            )
            if ordered[0] != seed:
                continue
            outer_offsets = np.asarray(
                [
                    (anchor[0] - seed_outer) % outer_count
                    for anchor in ordered
                ],
                dtype=np.int64,
            )
            inner_offsets = np.asarray(
                [
                    (anchor[1] - seed_inner) % inner_count
                    for anchor in ordered
                ],
                dtype=np.int64,
            )
            if np.any(np.diff(inner_offsets) < 0):
                continue
            if any(
                outer_offsets[position]
                == outer_offsets[position - 1]
                and inner_offsets[position]
                == inner_offsets[position - 1]
                for position in range(1, len(ordered))
            ):
                continue
            signature = tuple(
                (anchor[2], anchor[3]) for anchor in ordered
            )
            anchor_orders.append(
                (
                    signature,
                    seed_outer,
                    seed_inner,
                    ordered,
                    outer_offsets,
                    inner_offsets,
                )
            )
        if not anchor_orders:
            raise ValueError(
                "pattern connectors do not preserve cyclic order"
            )

        best_rows = None
        best_score = None
        best_signature = None
        for (
            signature,
            seed_outer,
            seed_inner,
            ordered,
            outer_offsets,
            inner_offsets,
        ) in anchor_orders:
            outer_limits = np.append(outer_offsets, outer_count)
            inner_limits = np.append(inner_offsets, inner_count)
            sector_rows = []
            sector_scores = []
            try:
                for position in range(len(ordered)):
                    outer_start = int(outer_limits[position])
                    outer_end = int(outer_limits[position + 1])
                    inner_start = int(inner_limits[position])
                    inner_end = int(inner_limits[position + 1])
                    outer_arc = outer_order[
                        (
                            seed_outer
                            + np.arange(
                                outer_start, outer_end + 1
                            )
                        )
                        % outer_count
                    ]
                    inner_arc = inner_order[
                        (
                            seed_inner
                            + np.arange(
                                inner_start, inner_end + 1
                            )
                        )
                        % inner_count
                    ]
                    rows, score = self._triangulate_sector(
                        outer_arc, inner_arc, connector_is_valid
                    )
                    sector_rows.append(rows)
                    sector_scores.append(score)
            except ValueError:
                continue

            score = self._combine_scores(sector_scores)
            if self._score_better(score, best_score) or (
                best_score is not None
                and not self._score_better(best_score, score)
                and not self._score_better(score, best_score)
                and (
                    best_signature is None
                    or signature < best_signature
                )
            ):
                best_rows = np.concatenate(sector_rows, axis=0)
                best_score = score
                best_signature = signature

        if best_rows is None:
            raise ValueError(
                "pattern connectors do not bound triangulatable sectors"
            )
        self.chosen_outer_order = outer_order
        self.chosen_inner_order = inner_order
        self.chosen_boundary_polygon = None
        return best_rows

    def _triangulate_open(self):
        """Choose a globally scored triangulation for two open arcs.

        Returns:
            Padded Tri3 rows covering the endpoint-bounded strip.

        Raises:
            ValueError: If no candidate arc ordering admits a valid constrained
                triangulation.
        """
        best_rows = None
        best_score = None
        best_signature = None
        best_orders = None
        last_error = None

        for outer_order, inner_order, boundary in (
            self.open_geometry_candidates
        ):
            polygon_edges = self._boundary_edges(boundary, True)
            side_edges = [
                (int(outer_order[0]), int(inner_order[0])),
                (int(outer_order[-1]), int(inner_order[-1])),
            ]
            side_keys = {
                self._edge_key(edge) for edge in side_edges
            }
            protected = [
                *[tuple(edge) for edge in self.forced_connectors],
                *side_edges,
            ]
            if any(
                not self._edges_are_compatible(first, second)
                for position, first in enumerate(protected)
                for second in protected[position + 1 :]
            ):
                continue

            edge_array = np.asarray(polygon_edges, dtype=np.int64)
            edge_points = self.xy[edge_array]
            edge_minimums = np.min(edge_points, axis=1)
            edge_maximums = np.max(edge_points, axis=1)
            connector_cache = {}
            connector_is_valid = partial(
                self._open_connector_is_valid,
                side_keys=side_keys,
                polygon_edges=polygon_edges,
                edge_minimums=edge_minimums,
                edge_maximums=edge_maximums,
                boundary=boundary,
                protected_edges=protected,
                cache=connector_cache,
            )

            outer_positions = {
                int(node): position
                for position, node in enumerate(outer_order)
            }
            inner_positions = {
                int(node): position
                for position, node in enumerate(inner_order)
            }
            anchors = [(0, 0)]
            anchors.extend(
                (
                    outer_positions[int(outer_node)],
                    inner_positions[int(inner_node)],
                )
                for outer_node, inner_node
                in self.forced_connectors
            )
            anchors.append(
                (outer_order.size - 1, inner_order.size - 1)
            )
            anchors = sorted(set(anchors))
            if (
                anchors[0] != (0, 0)
                or anchors[-1]
                != (
                    outer_order.size - 1,
                    inner_order.size - 1,
                )
                or any(
                    anchors[position][1]
                    < anchors[position - 1][1]
                    for position in range(1, len(anchors))
                )
            ):
                continue

            sector_rows = []
            sector_scores = []
            try:
                for start, end in zip(anchors, anchors[1:]):
                    rows, score = self._triangulate_sector(
                        outer_order[start[0] : end[0] + 1],
                        inner_order[start[1] : end[1] + 1],
                        connector_is_valid,
                    )
                    sector_rows.append(rows)
                    sector_scores.append(score)
            except ValueError as error:
                last_error = error
                continue

            score = self._combine_scores(sector_scores)
            signature = (
                tuple(map(int, outer_order)),
                tuple(map(int, inner_order)),
            )
            if self._score_better(score, best_score) or (
                best_score is not None
                and not self._score_better(best_score, score)
                and not self._score_better(score, best_score)
                and (
                    best_signature is None
                    or signature < best_signature
                )
            ):
                best_rows = np.concatenate(sector_rows, axis=0)
                best_score = score
                best_signature = signature
                best_orders = (
                    outer_order,
                    inner_order,
                    boundary,
                )

        if best_rows is None:
            if last_error is not None:
                raise ValueError(
                    "the open circular arcs have no valid constrained "
                    "triangulation"
                ) from last_error
            raise ValueError(
                "the open circular arcs do not bound a simple strip"
            )
        (
            self.chosen_outer_order,
            self.chosen_inner_order,
            self.chosen_boundary_polygon,
        ) = best_orders
        return best_rows

    def _triangulate(self):
        """Dispatch to the closed-ring or open-arc triangulator.

        Returns:
            Padded Tri3 rows for the complete selected strip.
        """
        if self.closed:
            return self._triangulate_closed()
        return self._triangulate_open()

    def _geometry_bounds(self, connectivity):
        """Compute XY bounding boxes for indexed geometry.

        Args:
            connectivity: Iterable of node-index collections.

        Returns:
            A pair of arrays containing per-item minima and maxima.
        """
        connectivity = [
            tuple(map(int, item)) for item in connectivity
        ]
        if not connectivity:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty
        minimums = np.empty((len(connectivity), 2))
        maximums = np.empty((len(connectivity), 2))
        for position, item in enumerate(connectivity):
            points = self.xy[np.asarray(item, dtype=np.intp)]
            minimums[position] = np.min(points, axis=0)
            maximums[position] = np.max(points, axis=0)
        return minimums, maximums

    def _bbox_pairs(
        self,
        first_minimums,
        first_maximums,
        second_minimums,
        second_maximums,
        same_collection=False,
    ):
        """Yield pairs of bounding boxes that may overlap.

        Args:
            first_minimums: Minima for the first geometry collection.
            first_maximums: Maxima for the first geometry collection.
            second_minimums: Minima for the second geometry collection.
            second_maximums: Maxima for the second geometry collection.
            same_collection: Whether both inputs describe the same collection.

        Yields:
            Arrays of candidate positions from the first and second collections.
        """
        chunk_size = 256
        for offset in range(
            0, first_minimums.shape[0], chunk_size
        ):
            stop = min(
                offset + chunk_size, first_minimums.shape[0]
            )
            overlaps = np.all(
                (
                    first_minimums[offset:stop, None, :]
                    <= second_maximums[None, :, :]
                    + self.geometry_tolerance
                )
                & (
                    first_maximums[offset:stop, None, :]
                    >= second_minimums[None, :, :]
                    - self.geometry_tolerance
                ),
                axis=2,
            )
            first_positions, second_positions = np.nonzero(
                overlaps
            )
            first_positions += offset
            if same_collection:
                keep = second_positions > first_positions
                first_positions = first_positions[keep]
                second_positions = second_positions[keep]
            yield first_positions, second_positions

    def _validate_triangle_topology(self, triangles):
        """Validate triangle rows, edge uses, boundaries, and connectors.

        Args:
            triangles: Complete padded-Tri3 strip connectivity.

        Returns:
            Directed uses of every new edge and total signed triangle area.

        Raises:
            RuntimeError: If the triangulator violates its internal row shape.
            ValueError: If rows, edges, boundaries, or mandatory connectors are
                invalid or incomplete.
        """
        expected_count = (
            self.chosen_outer_order.size
            + self.chosen_inner_order.size
            if self.closed
            else self.chosen_outer_order.size
            + self.chosen_inner_order.size
            - 2
        )
        if triangles.shape != (expected_count, 4):
            raise RuntimeError(
                "the triangulation produced an unexpected element count"
            )
        if np.any(triangles[:, 2] != triangles[:, 3]):
            raise RuntimeError(
                "the triangulation did not produce padded Tri3 elements"
            )
        triangle_signatures = {
            tuple(sorted(map(int, row[:3]))) for row in triangles
        }
        if len(triangle_signatures) != triangles.shape[0]:
            raise ValueError(
                "the triangulation contains duplicate triangles"
            )

        new_edge_uses = {}
        triangle_area = 0.0
        for row in triangles:
            first, second, third = map(int, row[:3])
            quality = self._triangle_quality(
                first, second, third
            )
            if not np.isfinite(quality) or quality <= 0.0:
                raise ValueError(
                    "the triangulation contains an invalid triangle"
                )
            triangle_area += 0.5 * self._cross(
                self.xy[second] - self.xy[first],
                self.xy[third] - self.xy[first],
            )
            for edge in (
                (first, second),
                (second, third),
                (third, first),
            ):
                new_edge_uses.setdefault(
                    self._edge_key(edge), []
                ).append(edge)

        outer_edges = self._boundary_edges(
            self.chosen_outer_order, self.closed
        )
        inner_edges = self._boundary_edges(
            self.chosen_inner_order, self.closed
        )
        expected_boundary = {
            self._edge_key(edge)
            for edge in (*outer_edges, *inner_edges)
        }
        open_side_keys = set()
        if not self.closed:
            open_side_keys = {
                self._edge_key(
                    (
                        int(self.chosen_outer_order[0]),
                        int(self.chosen_inner_order[0]),
                    )
                ),
                self._edge_key(
                    (
                        int(self.chosen_outer_order[-1]),
                        int(self.chosen_inner_order[-1]),
                    )
                ),
            }
            expected_boundary.update(open_side_keys)

        for key, uses in new_edge_uses.items():
            expected_uses = 1 if key in expected_boundary else 2
            if len(uses) != expected_uses:
                raise ValueError(
                    "the triangulation has an unexpected boundary edge"
                )
            if len(uses) == 2 and uses[0] != uses[1][::-1]:
                raise ValueError(
                    "new triangles sharing an edge must traverse it "
                    "oppositely"
                )
        for key in expected_boundary:
            if len(new_edge_uses.get(key, ())) != 1:
                raise ValueError(
                    "a circular boundary edge is missing from the mesh"
                )
        for connector in self.forced_connectors:
            key = self._edge_key(connector)
            expected_uses = 1 if key in open_side_keys else 2
            if len(new_edge_uses.get(key, ())) != expected_uses:
                raise ValueError(
                    "a pattern connector is missing from the triangulation"
                )

        return new_edge_uses, triangle_area

    def _validate_new_edge_geometry(self, new_edge_uses):
        """Reject crossings and overlaps among generated edges.

        Args:
            new_edge_uses: Directed uses of every new topological edge.

        Returns:
            Representative new edges and their bounding-box minima and maxima.

        Raises:
            ValueError: If two non-neighbouring generated edges are
                geometrically incompatible.
        """
        new_edges = [uses[0] for uses in new_edge_uses.values()]
        new_minimums, new_maximums = self._geometry_bounds(
            new_edges
        )
        for first_positions, second_positions in self._bbox_pairs(
            new_minimums,
            new_maximums,
            new_minimums,
            new_maximums,
            same_collection=True,
        ):
            for first_position, second_position in zip(
                first_positions, second_positions
            ):
                if not self._edges_are_compatible(
                    new_edges[int(first_position)],
                    new_edges[int(second_position)],
                ):
                    raise ValueError(
                        "non-neighbouring generated edges cross or overlap"
                    )

        return new_edges, new_minimums, new_maximums

    def _validate_existing_mesh_compatibility(
        self,
        triangles,
        new_edges,
        new_minimums,
        new_maximums,
    ):
        """Reject generated geometry that intersects existing mesh cells.

        Args:
            triangles: Complete padded-Tri3 strip connectivity.
            new_edges: One representative of every generated edge.
            new_minimums: Per-edge bounding-box minima.
            new_maximums: Per-edge bounding-box maxima.

        Raises:
            ValueError: If a generated edge crosses existing geometry or a
                generated and existing element overlap.
        """
        existing_edges = [
            uses[0] for uses in self.existing_edge_uses.values()
        ]
        existing_minimums, existing_maximums = (
            self._geometry_bounds(existing_edges)
        )
        for new_positions, existing_positions in self._bbox_pairs(
            new_minimums,
            new_maximums,
            existing_minimums,
            existing_maximums,
        ):
            for new_position, existing_position in zip(
                new_positions, existing_positions
            ):
                new_edge = new_edges[int(new_position)]
                existing_edge = existing_edges[
                    int(existing_position)
                ]
                if (
                    self._edge_key(new_edge)
                    == self._edge_key(existing_edge)
                ):
                    continue
                if not self._edges_are_compatible(
                    new_edge, existing_edge
                ):
                    raise ValueError(
                        "generated edges intersect existing mesh geometry"
                    )

        new_perimeters = [
            tuple(map(int, row[:3])) for row in triangles
        ]
        new_element_minimums, new_element_maximums = (
            self._geometry_bounds(new_perimeters)
        )
        (
            existing_element_minimums,
            existing_element_maximums,
        ) = self._geometry_bounds(self.existing_perimeters)
        for new_positions, existing_positions in self._bbox_pairs(
            new_element_minimums,
            new_element_maximums,
            existing_element_minimums,
            existing_element_maximums,
        ):
            for new_position, existing_position in zip(
                new_positions, existing_positions
            ):
                new_perimeter = new_perimeters[int(new_position)]
                existing_perimeter = self.existing_perimeters[
                    int(existing_position)
                ]
                existing_inside_new = any(
                    not self._point_on_polygon(
                        self.xy[node], new_perimeter
                    )
                    and self._point_in_polygon(
                        self.xy[node], new_perimeter
                    )
                    for node in existing_perimeter
                )
                new_inside_existing = any(
                    not self._point_on_polygon(
                        self.xy[node], existing_perimeter
                    )
                    and self._point_in_polygon(
                        self.xy[node], existing_perimeter
                    )
                    for node in new_perimeter
                )
                if existing_inside_new or new_inside_existing:
                    raise ValueError(
                        "generated elements overlap existing mesh elements"
                    )

    def _validate_strip_area(self, triangle_area):
        """Verify that triangles cover exactly the selected strip area.

        Args:
            triangle_area: Sum of signed generated triangle areas.

        Raises:
            ValueError: If the expected strip area is invalid or differs from
                the triangulated area beyond geometry tolerance.
        """
        if self.closed:
            expected_area = self._polygon_area(
                self.chosen_outer_order
            ) - self._polygon_area(self.chosen_inner_order)
            area_boundaries = (
                self.chosen_outer_order,
                self.chosen_inner_order,
            )
        else:
            expected_area = self._polygon_area(
                self.chosen_boundary_polygon
            )
            area_boundaries = (self.chosen_boundary_polygon,)
        boundary_length = 0.0
        for boundary in area_boundaries:
            points = self.xy[np.asarray(boundary, dtype=np.intp)]
            edges = np.roll(points, -1, axis=0) - points
            boundary_length += float(
                np.sum(np.hypot(edges[:, 0], edges[:, 1]))
            )
        area_tolerance = max(
            self.geometry_tolerance
            * max(self.geometry_tolerance, boundary_length),
            1.0e-10
            * max(
                self.geometry_tolerance**2,
                abs(expected_area),
            ),
        )
        if (
            expected_area <= area_tolerance
            or abs(triangle_area - expected_area) > area_tolerance
        ):
            raise ValueError(
                "triangles do not cover the complete circular strip"
            )

    def _validate_completed_manifold(self, new_edge_uses):
        """Validate combined existing and generated edge manifoldness.

        Args:
            new_edge_uses: Directed uses of every generated topological edge.

        Raises:
            ValueError: If a combined edge is non-manifold or paired in the
                same direction.
        """
        combined_uses = {
            key: list(uses)
            for key, uses in self.existing_edge_uses.items()
        }
        for key, uses in new_edge_uses.items():
            combined = combined_uses.setdefault(key, [])
            combined.extend(uses)
            if len(combined) > 2:
                raise ValueError(
                    "the completed mesh would contain a non-manifold edge"
                )
            if len(combined) == 2 and combined[0] != combined[1][::-1]:
                raise ValueError(
                    "new elements would traverse an existing shared edge "
                    "incorrectly"
                )

    def _validate_completed_strip(self, triangles):
        """Validate topology, geometry, area, and mesh compatibility.

        Args:
            triangles: Complete padded-Tri3 strip connectivity.

        Raises:
            RuntimeError: If the triangulator violates its internal row
                contract.
            ValueError: If the strip is invalid, incomplete, intersecting,
                overlapping, non-manifold, or incompatible with existing
                elements.
        """
        new_edge_uses, triangle_area = self._validate_triangle_topology(
            triangles
        )
        (
            new_edges,
            new_minimums,
            new_maximums,
        ) = self._validate_new_edge_geometry(new_edge_uses)
        self._validate_existing_mesh_compatibility(
            triangles,
            new_edges,
            new_minimums,
            new_maximums,
        )
        self._validate_strip_area(triangle_area)
        self._validate_completed_manifold(new_edge_uses)

    def _quad_is_valid(self, quad):
        """Return whether four perimeter nodes form an acceptable Quad4.

        Args:
            quad: Ordered four-node perimeter.

        Returns:
            True when the perimeter is distinct, simple, convex,
            counter-clockwise, and meets the configured Jacobian threshold.
        """
        if len(set(map(int, quad))) != 4:
            return False
        if not self._polygon_is_simple(quad):
            return False

        points = self.xy[np.asarray(quad, dtype=np.intp)]
        forward = np.roll(points, -1, axis=0) - points
        backward = np.roll(points, 1, axis=0) - points
        forward_lengths = np.hypot(forward[:, 0], forward[:, 1])
        backward_lengths = np.hypot(backward[:, 0], backward[:, 1])
        if np.any(forward_lengths <= self.geometry_tolerance):
            return False

        corner_cross = (
            forward[:, 0] * backward[:, 1]
            - forward[:, 1] * backward[:, 0]
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized_turn = corner_cross / (
                forward_lengths * backward_lengths
            )
        return bool(
            np.all(np.isfinite(normalized_turn))
            and np.all(normalized_turn > self.angular_tolerance)
            and float(np.min(normalized_turn))
            >= self.minimum_quad_scaled_jacobian
        )

    def _merge_triangle_pairs(self, triangles):
        """Greedily replace adjacent Tri3 pairs with valid Quad4 elements.

        Args:
            triangles: Valid padded-Tri3 connectivity for the complete strip.

        Returns:
            Mixed Tri3/Quad4 connectivity in deterministic source-row order.

        Raises:
            RuntimeError: If adjacent triangles traverse a shared edge in the
                same direction.
        """
        edge_uses = {}
        for position, row in enumerate(triangles):
            first, second, third = map(int, row[:3])
            for start, end, opposite in (
                (first, second, third),
                (second, third, first),
                (third, first, second),
            ):
                edge_uses.setdefault(
                    self._edge_key((start, end)), []
                ).append((position, start, end, opposite))

        protected_edges = {
            self._edge_key(connector)
            for connector in self.forced_connectors
        }
        partners = np.full(triangles.shape[0], -1, dtype=np.int64)
        merged_at = {}

        for edge_key in sorted(edge_uses):
            uses = edge_uses[edge_key]
            if len(uses) != 2 or edge_key in protected_edges:
                continue
            first_use, second_use = uses
            first_position = int(first_use[0])
            second_position = int(second_use[0])
            if (
                partners[first_position] >= 0
                or partners[second_position] >= 0
            ):
                continue
            if (first_use[1], first_use[2]) != (
                second_use[2], second_use[1]
            ):
                raise RuntimeError(
                    "adjacent triangles must traverse their shared edge "
                    "oppositely"
                )

            quad = np.asarray(
                [
                    first_use[3],
                    first_use[1],
                    second_use[3],
                    first_use[2],
                ],
                dtype=np.int64,
            )
            if not self._quad_is_valid(quad):
                continue

            partners[first_position] = second_position
            partners[second_position] = first_position
            merged_at[min(first_position, second_position)] = quad

        merged = []
        for position, triangle in enumerate(triangles):
            quad = merged_at.get(position)
            if quad is not None:
                merged.append(quad)
            elif partners[position] < 0:
                merged.append(triangle)
        return np.asarray(merged, dtype=np.int64)

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
    mesh: Mesh,
    inner_nodes,
    outer_nodes,
    lines=None,
    closed: bool = True,
    jacobian: float = 0.3,
) -> Mesh:
    """Mesh the strip between two concentric circular boundaries.

    Existing nodes remain fixed.  Cross-ring connectors are selected by a
    global monotone matching that first minimizes the worst angular mismatch,
    then the total squared mismatch, and finally maximizes the worst triangle
    quality.  Valid adjacent triangle pairs are then greedily merged into
    Quad4 elements when their minimum scaled Jacobian is at least
    jacobian. Pattern lines become mandatory connector edges and are never
    removed by merging.

    Args:
        mesh: Mesh whose existing nodes remain fixed and whose element array is
            extended only after successful validation.
        inner_nodes: Node indices on the smaller circular ring or arc.
        outer_nodes: Node indices on the larger circular ring or arc.
        lines: Optional finite horizontal or vertical pattern segments with
            shape (L, 2, 2).
        closed: True for complete rings or False for endpoint-bounded arcs.
        jacobian: Minimum scaled Jacobian in [0, 1] for Quad4 merging.

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
        mesh, inner_nodes, outer_nodes, lines, closed, jacobian
    ).build()
