"""Complete source loops are normalized without changing frozen topology."""

import unittest

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d import EditorMesh2D
from mesher.mesh2d.adapter import QuadLayerError, generate_layer_adaptive_mesh, plan_quad_sections
from tests.unit.mesh2d.test_extend_quad_layer import _area, _polyomino


class OutlineLoopTests(unittest.TestCase):
    @staticmethod
    def fixtures():
        rectangle, loop, _ = _polyomino({(x, y) for x in range(4) for y in range(3)})
        yield rectangle, loop, .5
        coarse = {(0, 0), (1, 0), (0, 1)}
        cells = {(3*x+dx, 3*y+dy) for x, y in coarse for dx in range(3) for dy in range(3)}
        concave, loop, _ = _polyomino(cells)
        concave.nodes[:, :2] /= 3
        yield concave, loop, .2
        cells = {(x, y) for x in range(12) for y in range(12)
                 if x < 4 or x >= 8 or y < 4 or y >= 8}
        hole, _, _ = _polyomino(cells)
        hole.nodes[:, :2] /= 4
        loop = next(ids for ids in EditorMesh2D.outlines(hole)
                    if _area(hole.nodes[ids, :2]) > 0)
        yield hole, loop, .2

    def test_starts_windings_and_explicit_closure_produce_identical_results(self):
        for source, loop, size in self.fixtures():
            baseline = None
            for reverse, start, closed in ((False, 0, False), (False, 1, True),
                                           (False, len(loop)-1, False), (True, 0, True),
                                           (True, 1, False), (True, len(loop)-1, True)):
                with self.subTest(nodes=source.node_count, reverse=reverse, start=start, closed=closed):
                    supplied = np.roll(np.asarray(loop)[::-1] if reverse else loop, start)
                    if closed:
                        supplied = np.r_[supplied, supplied[0]]
                    supplied.setflags(write=False)
                    saved = supplied.copy()
                    mesh = Mesh2D(source.nodes, source.elements)
                    plan = plan_quad_sections(mesh, outline_i=supplied, element_size=size)
                    np.testing.assert_array_equal(mesh.nodes, source.nodes)
                    np.testing.assert_array_equal(mesh.elements, source.elements)
                    result = generate_layer_adaptive_mesh(mesh, outline_i=supplied, element_size=size)
                    np.testing.assert_array_equal(supplied, saved)
                    self.assertIsInstance(result.outline, list)
                    self.assertTrue(all(isinstance(node, int) for node in result.outline))
                    if baseline is None:
                        baseline = mesh.nodes.copy(), mesh.elements.copy(), result.outline, plan.front
                    else:
                        np.testing.assert_array_equal(mesh.nodes, baseline[0])
                        np.testing.assert_array_equal(mesh.elements, baseline[1])
                        self.assertEqual(result.outline, baseline[2])
                        np.testing.assert_array_equal(plan.front, baseline[3])

    def test_clockwise_cells_and_counterclockwise_input_normalize_to_same_growth(self):
        source, loop, _ = next(self.fixtures())
        ccw = Mesh2D(source.nodes, source.elements)
        cw = Mesh2D(source.nodes, source.elements[:, ::-1])
        first = generate_layer_adaptive_mesh(ccw, loop, .5)
        second = generate_layer_adaptive_mesh(cw, np.asarray(loop)[::-1], .5)
        np.testing.assert_array_equal(ccw.nodes, cw.nodes)
        np.testing.assert_array_equal(ccw.elements[source.element_count:], cw.elements[source.element_count:])
        self.assertEqual(first.outline, second.outline)

    def test_missing_intermediate_nodes_fail_atomically(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            source, loop, _ = next(self.fixtures())
            mesh = Mesh2D(source.nodes, source.elements)
            nodes, elements = mesh.nodes, mesh.elements
            with self.subTest(api=api.__name__), self.assertRaises(QuadLayerError) as caught:
                api(mesh, np.delete(loop, 1), .5)
            self.assertEqual(caught.exception.code, "fixed_boundary_conflict")
            self.assertEqual(caught.exception.context["parameter"], "outline_i")
            self.assertIs(mesh.nodes, nodes)
            self.assertIs(mesh.elements, elements)
            np.testing.assert_array_equal(nodes, source.nodes)
            np.testing.assert_array_equal(elements, source.elements)

    def test_none_source_is_a_typed_input_failure(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            source, _, _ = next(self.fixtures())
            with self.subTest(api=api.__name__), self.assertRaises(QuadLayerError) as caught:
                api(source, None, .5)
            self.assertEqual(caught.exception.context["parameter"], "outline_i")
            self.assertEqual(caught.exception.code, "invalid_input")

    def test_axis_tolerance_applies_to_complete_automatically_grouped_runs(self):
        for shear, accepted in ((1e-4, True), (.02, False)):
            mesh, loop, _ = next(self.fixtures())
            mesh.nodes[:, 0] += shear * mesh.nodes[:, 1]
            original = mesh.nodes.copy()
            if accepted:
                result = generate_layer_adaptive_mesh(mesh, loop, .5)
                np.testing.assert_array_equal(mesh.nodes[:len(original)], original)
                points = mesh.nodes[result.outline, :2]
                vectors = np.roll(points, -1, axis=0) - points
                self.assertTrue(np.all(np.min(np.abs(vectors), axis=1) < 1e-12))
            else:
                with self.assertRaises(QuadLayerError):
                    generate_layer_adaptive_mesh(mesh, loop, .5)
                np.testing.assert_array_equal(mesh.nodes, original)


if __name__ == "__main__":
    unittest.main()
