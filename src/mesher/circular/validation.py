"""Topology and geometry validation for generated circular strips."""

import numpy as np


class _ValidationMixin:
    """Validate a generated strip before it is committed."""

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
