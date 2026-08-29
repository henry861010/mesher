"""Constrained triangulation for circular mesh strips."""

from functools import partial
from typing import NamedTuple

import numpy as np


class _StripScore(NamedTuple):
    """Lexicographic objective used to compare triangulations."""

    worst_mismatch: float
    total_squared_mismatch: float
    minimum_quality: float


class _TriangulationMixin:
    """Select valid, high-quality connectors between circular rings."""

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
