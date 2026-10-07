import unittest

from mesher.process_flow.circle_planning import (
    _CirclePattern,
    _build_circle_meshing_plan,
    _collect_pattern_segments,
    _collect_circle_source_refs,
    _planar_element_size,
    _validate_circle_clearances,
    _segment_intersects_annulus,
)


def _circle(x, y, radius):
    return {"type": "CIRCLE", "dim": [x, y, radius]}


class CircleMeshingPlanTests(unittest.TestCase):
    def test_koz_circle_clearance_refines_xy_size(self):
        outer, inner = _CirclePattern(0, 0, 1), _CirclePattern(0, 0, 0.9)
        self.assertAlmostEqual(
            _planar_element_size(1, [outer, inner], koz_patterns={inner}), 0.025
        )
        self.assertEqual(_planar_element_size(1, [outer, inner]), 0.3)

    def test_koz_boundary_intersections_and_tangency_still_fail(self):
        inner = _CirclePattern(0, 0, 1)
        for other in (_CirclePattern(1, 0, 1), _CirclePattern(2, 0, 1)):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "intersecting or tangent"):
                _planar_element_size(1, [inner, other], koz_patterns={inner})

    def test_neighbor_gap_can_drive_finer_koz_resolution(self):
        inner = _CirclePattern(0, 0, 1)
        neighbor = _CirclePattern(2.0000001, 0, 1)
        size = _planar_element_size(1, [inner, neighbor], koz_patterns={inner})
        self.assertAlmostEqual(size, 2.5e-8)
        _validate_circle_clearances([inner, neighbor], 2 * size, tolerance=size / 100)

    def test_koz_circles_retain_original_feature_refs(self):
        container = {"children": [{"vias": [{
            "id": "via-1", "koz": 0.2, "geometry": {"type": "CylinderGeometry",
            "center": [0, 0, 0], "bottom_radius": 1, "thk": 1},
        }]}]}
        refs = _collect_circle_source_refs(container)
        self.assertEqual(refs[_CirclePattern(0, 0, 1)], ["via-1"])
        self.assertEqual(refs[_CirclePattern(0, 0, 0.8)], ["via-1"])
        self.assertEqual(set(_collect_circle_source_refs(container, koz_only=True)),
                         {_CirclePattern(0, 0, 0.8)})

    def test_collects_box_and_polygon_segments_but_not_circles(self):
        faces = [
            _circle(0.0, 0.0, 10.0),
            {"type": "BOX", "dim": [-2.0, -1.0, 2.0, 1.0]},
            {
                "type": "POLYGON",
                "dim": [
                    [[3.0, -1.0], [5.0, -1.0], [5.0, 1.0], [3.0, 1.0]],
                ],
            },
        ]

        segments = _collect_pattern_segments(faces)

        self.assertEqual(
            segments,
            [
                ((-2.0, -1.0), (2.0, -1.0)),
                ((2.0, -1.0), (2.0, 1.0)),
                ((2.0, 1.0), (-2.0, 1.0)),
                ((-2.0, 1.0), (-2.0, -1.0)),
                ((3.0, -1.0), (5.0, -1.0)),
                ((5.0, -1.0), (5.0, 1.0)),
                ((5.0, 1.0), (3.0, 1.0)),
                ((3.0, 1.0), (3.0, -1.0)),
            ],
        )

    def test_box_base_keeps_all_circles_as_imprints(self):
        base = {"type": "BOX", "dim": [-20.0, -20.0, 20.0, 20.0]}
        faces = [base, _circle(0.0, 0.0, 10.0), _circle(0.0, 0.0, 13.0)]
        patterns = [_CirclePattern(0.0, 0.0, 10.0), _CirclePattern(0.0, 0.0, 13.0)]

        plan = _build_circle_meshing_plan(
            base,
            faces,
            patterns,
            band_width=1.0,
        )

        self.assertEqual(plan.imprint_patterns, tuple(patterns))
        self.assertEqual(plan.extensions, ())

    def test_protruding_footprint_disables_circular_extension(self):
        base = _circle(0.0, 0.0, 13.0)
        protruding_box = {
            "type": "BOX",
            "dim": [12.0, -0.1, 14.0, 0.1],
        }
        faces = [_circle(0.0, 0.0, 10.0), base, protruding_box]
        patterns = [_CirclePattern(0.0, 0.0, 10.0), _CirclePattern(0.0, 0.0, 13.0)]

        plan = _build_circle_meshing_plan(
            base,
            faces,
            patterns,
            band_width=1.0,
        )

        self.assertEqual(plan.imprint_patterns, tuple(patterns))
        self.assertEqual(plan.extensions, ())

    def test_nonconcentric_band_moves_extension_source_outward(self):
        base = _circle(0.0, 0.0, 13.0)
        faces = [
            _circle(0.0, 0.0, 5.0),
            _circle(0.0, 0.0, 10.0),
            base,
            _circle(4.0, 0.0, 1.0),
        ]
        patterns = [
            _CirclePattern(0.0, 0.0, 5.0),
            _CirclePattern(0.0, 0.0, 10.0),
            _CirclePattern(0.0, 0.0, 13.0),
            _CirclePattern(4.0, 0.0, 1.0),
        ]

        plan = _build_circle_meshing_plan(
            base,
            faces,
            patterns,
            band_width=1.0,
        )

        self.assertEqual(len(plan.extensions), 1)
        self.assertEqual(plan.extensions[0].inner.radius, 10.0)
        self.assertEqual(plan.extensions[0].outer.radius, 13.0)
        self.assertIn(_CirclePattern(4.0, 0.0, 1.0), plan.imprint_patterns)

    def test_segment_annulus_intersection_includes_tangency(self):
        kwargs = {
            "center": (0.0, 0.0),
            "inner_radius": 10.0,
            "outer_radius": 13.0,
        }

        self.assertTrue(
            _segment_intersects_annulus((10.0, 0.0), (10.0, 1.0), **kwargs)
        )
        self.assertTrue(
            _segment_intersects_annulus((-14.0, 0.0), (14.0, 0.0), **kwargs)
        )
        self.assertFalse(
            _segment_intersects_annulus((9.0, 0.0), (9.0, 1.0), **kwargs)
        )
        self.assertFalse(
            _segment_intersects_annulus((14.0, 0.0), (14.0, 1.0), **kwargs)
        )


if __name__ == "__main__":
    unittest.main()
