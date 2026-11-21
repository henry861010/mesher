"""Box-grid coverage, complete grid lines, and optional plotting dependencies."""

import importlib.util
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from mesher.mesh2d.visualization import plot_pattern2d


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional")
class PatternVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        cls.plt = plt

    def tearDown(self):
        self.plt.close("all")

    def test_coverage_and_cut_coordinates(self):
        from matplotlib.collections import QuadMesh

        cases = [
            [[0, 0, 3, 2]],
            [[0, 0, 1, 1], [2, 2, 3, 3]],
            [[0, 0, 3, 2], [2, 1, 5, 4]],
            [[0, 0, 4, 4], [1, 1, 2, 2]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [1, 0, 2, 1]],
            [[-.5, -.25, .5, .75], [.500001, -.25, .8, .75]],
        ]
        for boxes in cases:
            with self.subTest(boxes=boxes):
                original = np.array(boxes, dtype=float)
                fig, ax = plot_pattern2d(original)
                np.testing.assert_array_equal(original, boxes)
                mesh, = [c for c in ax.collections if isinstance(c, QuadMesh)]
                coords = mesh.get_coordinates()
                x = np.unique(original[:, (0, 2)])
                y = np.unique(original[:, (1, 3)])
                np.testing.assert_array_equal(coords[0, :, 0], x)
                np.testing.assert_array_equal(coords[:, 0, 1], y)
                cx, cy = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2)
                expected = np.zeros(cx.shape, dtype=bool)
                for x1, y1, x2, y2 in boxes:
                    expected |= (cx > x1) & (cx < x2) & (cy > y1) & (cy < y2)
                np.testing.assert_array_equal(mesh.get_array(), expected)
                fig.canvas.draw()
                # All-covered cases must still map to blue, rather than white.
                colors = mesh.get_facecolors()
                np.testing.assert_allclose(colors[~expected.ravel(), :3], 1)
                self.assertTrue(np.all(colors[expected.ravel(), :3] < 1))
                self.plt.close(fig)

    def test_grid_lines_span_covered_and_uncovered_cells_above_fill(self):
        from matplotlib.collections import LineCollection, QuadMesh

        fig, ax = plot_pattern2d([[0, 0, 1, 1], [2, 2, 3, 3]])
        lines, = [c for c in ax.collections if isinstance(c, LineCollection)]
        fill, = [c for c in ax.collections if isinstance(c, QuadMesh)]
        expected = [((x, 0), (x, 3)) for x in (0, 1, 2, 3)]
        expected += [((0, y), (3, y)) for y in (0, 1, 2, 3)]
        np.testing.assert_array_equal(lines.get_segments(), expected)
        self.assertGreater(lines.get_zorder(), fill.get_zorder())
        self.assertTrue(np.all(lines.get_linewidths() > 0))
        self.assertEqual(ax.get_aspect(), 1)
        self.assertFalse(ax.yaxis_inverted())
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                         ["Covered", "Uncovered"])
        output = BytesIO()
        fig.savefig(output, format="png")
        self.assertTrue(output.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))
        renderer = fig.canvas.get_renderer()
        self.assertGreater(ax.get_legend().get_window_extent(renderer).x0,
                           ax.get_window_extent(renderer).x1)

    def test_reversed_endpoints_and_existing_axes(self):
        fig, ax = self.plt.subplots()
        boxes = np.array([[3, 2, -1, -2]])
        original = boxes.copy()
        returned_fig, returned_ax = plot_pattern2d(boxes, ax=ax)
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        np.testing.assert_array_equal(boxes, original)
        self.assertEqual(ax.get_xlim(), (-1, 3))
        self.assertEqual(ax.get_ylim(), (-2, 2))

    def test_empty_input_produces_blank_labeled_axes(self):
        for boxes in ([], np.empty((0, 4))):
            fig, ax = plot_pattern2d(boxes)
            self.assertEqual(len(ax.collections), 0)
            self.assertEqual((ax.get_xlabel(), ax.get_ylabel()), ("X", "Y"))
            fig.canvas.draw()

    def test_invalid_input_raises_before_creating_figure(self):
        for boxes in ([0, 0, 1, 1], [[0, 0, 1]], [[0, 0, 1, 1], [2]],
                      np.empty((0, 3)), [[0, 0, "x", 1]], [[0, 0, 1j, 1]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                plot_pattern2d(boxes)
        for invalid in ([0, 0, 0, 1], [0, 0, 1, 0], [0, np.nan, 1, 1],
                        [0, 0, np.inf, 1]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, r"boxes\[1\]"):
                plot_pattern2d([[0, 0, 1, 1], invalid])
        self.assertEqual(self.plt.get_fignums(), [])


class PatternImportTests(unittest.TestCase):
    def test_import_keeps_optional_dependencies_unloaded_and_plot_needs_no_pyvista(self):
        script = """
import importlib.abc
import importlib.util
import sys
class BlockPyVista(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyvista' or fullname.startswith('pyvista.'):
            raise ImportError('PyVista is unavailable for this test')
sys.meta_path.insert(0, BlockPyVista())
from mesher.mesh2d.visualization import plot_pattern2d
assert not {'matplotlib', 'pyvista', 'PySide6', 'shapely', 'process_flow_kernel'}.intersection(sys.modules)
if importlib.util.find_spec('matplotlib'):
    import matplotlib
    matplotlib.use('Agg')
    fig, ax = plot_pattern2d([[0, 0, 1, 1]])
    fig.canvas.draw()
    assert 'pyvista' not in sys.modules
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
        result = subprocess.run([sys.executable, "-c", script], env=env,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("pyvista"), "PyVista is optional")
    def test_existing_pyvista_exports_are_preserved(self):
        from mesher.mesh2d.visualization import build_faces, view_mesh
        from mesher.mesh2d.visualization import pyvista

        self.assertIs(build_faces, pyvista.build_faces)
        self.assertIs(view_mesh, pyvista.view_mesh)


if __name__ == "__main__":
    unittest.main()
