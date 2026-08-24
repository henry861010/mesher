"""Quality-gated Tri3-to-Quad4 merging for circular strips."""

import numpy as np


class _QuadMergeMixin:
    """Merge adjacent triangles when they form an acceptable Quad4."""

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
