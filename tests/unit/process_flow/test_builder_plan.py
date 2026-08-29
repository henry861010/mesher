import unittest

from mesher.process_flow.circle_planning import (
    _CirclePattern,
    _build_circle_meshing_plan,
    _collect_pattern_segments,
    _segment_intersects_annulus,
)


def _circle(x, y, radius):
    return {"type": "CIRCLE", "dim": [x, y, radius]}


class CircleMeshingPlanTests(unittest.TestCase):
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
