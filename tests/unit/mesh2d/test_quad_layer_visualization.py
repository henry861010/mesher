"""Committed repair cells remain visible alongside expansion sections."""
from collections import Counter
from dataclasses import replace
import importlib.util
import unittest

import numpy as np

from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from mesher.mesh2d.adapter.visualization import plot_quad_layer
from tests.unit.mesh2d.test_quad_sections import _cells
from tests.unit.mesh2d.test_slot_repair import _c_slot


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional")
class QuadLayerVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        cls.plt = plt
        mesh, cls.slot_outline = _c_slot()
        cls.slot_result = generate_layer_adaptive_mesh(mesh, cls.slot_outline, .5)

    def tearDown(self):
        self.plt.close("all")

    def assert_all_cells_drawn(self, axis, mesh):
        from matplotlib.collections import PolyCollection

        def key(points):
            return tuple(sorted(map(tuple, points)))

        rendered = Counter(key(path.vertices[:-1]) for collection in axis.collections
                           if isinstance(collection, PolyCollection)
                           for path in collection.get_paths())
        expected = Counter(key(cell) for cell in mesh.nodes[mesh.elements, :2])
        self.assertEqual(rendered, expected)

    def test_legacy_slot_diagnostics_without_corner_key_render_every_cell(self):
        diagnostics = dict(self.slot_result.diagnostics)
        diagnostics.pop("corner_repairs", None)
        result = replace(self.slot_result, diagnostics=diagnostics)
        figure = plot_quad_layer(result, self.slot_outline)
        figure.canvas.draw()
        self.assert_all_cells_drawn(figure.axes[2], result.mesh)
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        self.assertIn("Slot repair", labels)
        self.assertNotIn("Corner repair", labels)

    def test_corner_cells_and_bent_boundary_render_with_shared_repair_legend(self):
        mesh, outline = _cells({(0, 0), (1, 0), (1, 1)})
        result = generate_layer_adaptive_mesh(mesh, outline, 1.)
        self.assertEqual(len(result.diagnostics["corner_repairs"]), 1)
        figure, axes = self.plt.subplots(2, 3, layout="constrained")
        plot_quad_layer(self.slot_result, self.slot_outline, axes=axes[0])
        plot_quad_layer(result, outline, axes=axes[1])
        figure.canvas.draw()
        self.assert_all_cells_drawn(axes[0, 2], self.slot_result.mesh)
        self.assert_all_cells_drawn(axes[1, 2], result.mesh)
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        self.assertEqual(labels.count("Slot repair"), 1)
        self.assertEqual(labels.count("Corner repair"), 1)
        renderer = figure.canvas.get_renderer()
        legend_top = figure.legends[0].get_window_extent(renderer).y1
        self.assertTrue(all(axis.xaxis.label.get_window_extent(renderer).y0 > legend_top
                            for axis in axes[-1]))
        self.assertIn("corner repair", axes[1, 0].get_title())
        mouth = np.asarray(result.diagnostics["corner_repairs"][0]["mouth_points"])
        self.assertGreaterEqual(len(mouth), 3)
        for axis in axes[1]:
            self.assertTrue(any(len(line.get_xdata()) == len(mouth)
                                and np.array_equal(line.get_xdata(), mouth[:, 0])
                                and np.array_equal(line.get_ydata(), mouth[:, 1])
                                for line in axis.lines))


if __name__ == "__main__":
    unittest.main()
