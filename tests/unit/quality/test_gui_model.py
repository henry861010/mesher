import math
import unittest

import numpy as np

from mesher.quality import ElementType, QualityReason
from mesher.quality.gui_model import (
    QualityThresholds,
    classify_quality,
    evaluate_element,
    format_metric,
    preset_nodes,
)


class GuiModelTests(unittest.TestCase):
    def test_equilateral_triangle_uses_checker_metrics(self):
        quality = evaluate_element(
            [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]]
        )

        self.assertEqual(quality.element_type, ElementType.TRI3)
        self.assertAlmostEqual(quality.scaled_jacobian_minimum, 1.0)
        self.assertAlmostEqual(quality.aspect_ratio, 1.0)
        self.assertEqual(quality.reasons, QualityReason.NONE)

    def test_square_passes_default_thresholds(self):
        quality = evaluate_element(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )

        status = classify_quality(quality, QualityThresholds())

        self.assertEqual(quality.element_type, ElementType.QUAD4)
        self.assertEqual(status.key, "pass")

    def test_valid_stretched_quad_is_below_aspect_threshold(self):
        quality = evaluate_element(
            [[0.0, 0.0], [5.0, 0.0], [5.0, 1.0], [0.0, 1.0]]
        )

        status = classify_quality(quality, QualityThresholds())

        self.assertTrue(quality.is_valid)
        self.assertEqual(status.key, "warning")
        self.assertIn("aspect ratio", status.detail)

    def test_folded_quad_is_invalid(self):
        quality = evaluate_element(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        )

        status = classify_quality(quality, QualityThresholds())

        self.assertEqual(status.key, "invalid")
        self.assertTrue(quality.reasons & QualityReason.FOLDED)
        self.assertIn("folded geometry", status.detail)

    def test_preset_returns_a_copy(self):
        first = preset_nodes(ElementType.TRI3)
        first[0, 0] = 99.0

        second = preset_nodes("TRI3")

        self.assertNotEqual(second[0, 0], 99.0)

    def test_invalid_node_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_element(np.zeros((5, 2)))

    def test_threshold_ranges_are_validated(self):
        with self.assertRaisesRegex(ValueError, "between -1 and 1"):
            QualityThresholds(minimum_scaled_jacobian=1.1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            QualityThresholds(maximum_aspect_ratio=0.9)

    def test_nonfinite_metric_formatting(self):
        self.assertEqual(format_metric(float("nan")), "undefined")
        self.assertEqual(format_metric(float("inf")), "+infinity")


if __name__ == "__main__":
    unittest.main()
