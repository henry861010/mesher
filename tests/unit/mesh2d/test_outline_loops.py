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

    def test_starts_and_explicit_closure_produce_identical_results(self):
        for source, loop, size in self.fixtures():
            baseline = None
            for start, closed in ((0, False), (1, True), (len(loop)-1, False),
                                  (0, True), (1, False), (len(loop)-1, True)):
                with self.subTest(nodes=source.node_count, start=start, closed=closed):
                    supplied = np.roll(loop, start)
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

    def test_cell_winding_does_not_change_caller_directed_growth(self):
        source, loop, _ = next(self.fixtures())
        ccw = Mesh2D(source.nodes, source.elements)
        cw = Mesh2D(source.nodes, source.elements[:, ::-1])
        first = generate_layer_adaptive_mesh(ccw, loop, .5)
        second = generate_layer_adaptive_mesh(cw, loop, .5)
        np.testing.assert_array_equal(ccw.nodes, cw.nodes)
        np.testing.assert_array_equal(ccw.elements[source.element_count:], cw.elements[source.element_count:])
        self.assertEqual(first.outline, second.outline)

    def test_reversing_source_reverses_growth_side_without_normalization(self):
        source, loop, _ = next(self.fixtures())
        for reverse in (False, True):
            supplied = np.asarray(loop)[::-1] if reverse else np.asarray(loop)
            plan = plan_quad_sections(source, supplied, .5, repair_slots=False)
            lower = source.nodes[loop, :2].min(axis=0)
            upper = source.nodes[loop, :2].max(axis=0)
            distance = -.5 if reverse else .5
            np.testing.assert_allclose(plan.front.min(axis=0), lower - distance)
            np.testing.assert_allclose(plan.front.max(axis=0), upper + distance)
            self.assertEqual(np.sign(_area(plan.front)), np.sign(_area(source.nodes[supplied, :2])))

    def test_source_ownership_is_a_caller_precondition_for_planning(self):
        source, loop, _ = next(self.fixtures())
        # A missing collinear node violates the caller contract, but does not
        # prevent a geometric preview. No ownership lookup repairs this input.
        supplied = np.delete(loop, 1)
        plan = plan_quad_sections(source, supplied, .5, repair_slots=False)
        self.assertEqual(len(plan.front), 4)

    def test_none_source_is_a_typed_input_failure(self):
        for api in (plan_quad_sections, generate_layer_adaptive_mesh):
            source, _, _ = next(self.fixtures())
            with self.subTest(api=api.__name__), self.assertRaises(QuadLayerError) as caught:
                api(source, None, .5)
            self.assertEqual(caught.exception.context["parameter"], "outline_i")
            self.assertEqual(caught.exception.code, "invalid_input")

    def test_tolerance_applies_to_complete_automatically_grouped_runs(self):
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
