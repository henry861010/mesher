"""Die/UF footprint and clipped corner grids in the debug imprint example."""

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh2d.imprint_line import imprint_lines
from mesher.mesh2d.imprint_line.mesher import _prepare_plan


HAS_GEOMETRY = all(importlib.util.find_spec(name) is not None
                   for name in ("process_flow_kernel", "shapely"))
if HAS_GEOMETRY:
    from shapely import union_all
    from shapely.geometry import LineString, Point, Polygon, box

    path = Path(__file__).resolve().parents[4] / "examples/debug_imprint_line.py"
    spec = importlib.util.spec_from_file_location("debug_imprint_line", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)


@unittest.skipUnless(HAS_GEOMETRY, "fillet example requires process-flow dependencies")
class DebugFilletTests(unittest.TestCase):
    def test_gap_threshold_is_inclusive(self):
        for gap, components in ((260, 1), (260.1, 2)):
            with self.subTest(gap=gap):
                filled, loops = example._get_die_uf_region(
                    [[0, 0, 1000, 1000], [1000 + gap, 0, 2000 + gap, 1000]])
                self.assertEqual(len(loops), components)
                self.assertEqual(filled.covers(Point(1130, 500)), components == 1)

    def test_recursive_fill_closes_four_die_intersection(self):
        boxes = [[0, 0, 10, 10], [14, 0, 24, 10],
                 [0, 14, 10, 24], [14, 14, 24, 24]]
        filled, loops = example._get_die_uf_region(boxes, gap=4)
        self.assertTrue(filled.equals(box(0, 0, 24, 24)))
        self.assertEqual(len(loops), 1)

    def test_rectangle_has_exact_layers_and_crossing_corner_lines(self):
        lines = example._get_fillet_pattern_lines([[0, 0, 1000, 1000]])
        offsets = (0, 5, 10, 15, 20, 25, 75, 125, 175, 225, 275, 300)
        coverage = union_all([LineString(line) for line in lines])
        self.assertEqual({-start[1] for start, end in lines
                          if start[1] == end[1] and start[1] <= 0}, set(offsets))
        for distance in offsets:
            for line in (((-300, -distance), (1300, -distance)),
                         ((-distance, -300), (-distance, 1300)),
                         ((-300, 1000 + distance), (1300, 1000 + distance)),
                         ((1000 + distance, -300), (1000 + distance, 1300))):
                self.assertTrue(coverage.covers(LineString(line)))

    def test_short_steps_clip_interior_but_keep_exterior_grid(self):
        boxes = [[0, 0, 1000, 1000], [1000, 0, 1225, 990]]
        filled, _ = example._get_die_uf_region(boxes, gap=0)
        band = filled.buffer(300, join_style="mitre").difference(filled)
        lines = example._get_fillet_pattern_lines(boxes, gap=0)
        for start, end in lines:
            line = LineString([start, end])
            self.assertTrue(band.covers(line))
            self.assertTrue(line.relate_pattern(filled, "F********"))
            self.assertGreater(line.length, 0)
            self.assertEqual(sum(a == b for a, b in zip(start, end)), 1)
        # The line above the short right-hand step stops at the taller die.
        self.assertIn(((1000, 995), (1525, 995)), lines)
        # Both directions extend through the bottom-right exterior corner.
        self.assertIn(((1230, -300), (1230, 1290)), lines)
        self.assertIn(((-300, -10), (1525, -10)), lines)

    def test_actual_package_outline_and_cow_containment(self):
        boxes = example._get_die_boxes()
        before = copy.deepcopy(boxes)
        filled, loops = example._get_die_uf_region(boxes)
        lines = example._get_fillet_pattern_lines(boxes)
        self.assertEqual(boxes, before)
        self.assertEqual([len(loop) for loop in loops], [20])
        self.assertTrue(Polygon(loops[0]).equals(filled))
        bounds = np.asarray(boxes)
        np.testing.assert_array_equal(filled.bounds, [*bounds[:, :2].min(axis=0),
                                                     *bounds[:, 2:].max(axis=0)])
        cow = Polygon(example._get_cow_outline())
        band = filled.buffer(300, join_style="mitre").difference(filled)
        self.assertEqual(len(lines), len(set(lines)))
        for line in map(LineString, lines):
            self.assertTrue(cow.covers(line))
            self.assertTrue(band.covers(line))
        self.assertEqual(lines, example._get_fillet_pattern_lines(list(reversed(boxes))))

    def test_patterns_survive_meshing_as_exact_edges(self):
        lines = example._get_fillet_pattern_lines([[0, 0, 1000, 1000]])
        mesh = imprint_lines(1000, [], lines,
                             outer_outline=[[-500, -500], [1500, -500],
                                            [1500, 1500], [-500, 1500]],
                             min_transition_size=20)
        edges = np.stack((mesh.elements, np.roll(mesh.elements, -1, axis=1)), axis=-1)
        edges = np.unique(np.sort(edges.reshape(-1, 2), axis=1), axis=0)
        coverage = union_all([LineString(points) for points in mesh.nodes[edges, :2]])
        for line in map(LineString, lines):
            self.assertTrue(line.difference(coverage).is_empty)

    def test_package_corner_uses_one_outward_layer(self):
        with patch.object(example, "imprint_lines", _prepare_plan):
            _, plan, _, _ = example.get_cow_mesh()
        corner, = (end for end in plan.transitions if end.axis == 0 and end.side == 0
                   and plan.bands[end.band].start == 12867.5 and end.bounds[1] == 10935)
        self.assertAlmostEqual(corner.inner, 11623.846153846154)
        self.assertAlmostEqual(corner.bounds[0], 10655.192307692309)
        self.assertEqual(len(corner.cells), 3)
        for end in plan.transitions:
            axis = end.axis
            self.assertEqual({(cell[axis], cell[axis + 2]) for cell in end.cells},
                             {(end.bounds[axis], end.bounds[axis + 2])})

    def test_package_axes_and_two_quad_adapter_layers(self):
        cow = example.get_cow_mesh()
        exterior, = example.EditorMesh2D.outlines(cow)
        self.assertEqual(len(exterior) % 2, 0)
        points = cow.nodes[cow.elements, :2]
        self.assertFalse(np.any((points.min(axis=1) < 0) & (points.max(axis=1) > 0)))
        self.assertEqual(np.count_nonzero(np.all(cow.nodes[:, :2] == 0, axis=1)), 1)
        generate = example.generate_layer_adaptive_mesh
        first_outline = []

        def boundary_edges(loop):
            return {tuple(sorted((a, b))) for a, b in zip(loop, np.roll(loop, -1))}

        def check_layer(**kwargs):
            start = kwargs["mesh"].element_count
            result = generate(**kwargs)
            quads = result.mesh.elements[start:]
            edges = np.stack((quads, np.roll(quads, -1, axis=1)), axis=2)
            keys, counts = np.unique(np.sort(edges.reshape(-1, 2), axis=1), axis=0, return_counts=True)
            self.assertLessEqual(counts.max(), 2)
            self.assertEqual(set(map(tuple, keys[counts == 1])),
                             boundary_edges(kwargs["outline_i"]) | boundary_edges(result.outline))
            self.assertGreater(result.diagnostics["quality"]["min_scaled_jacobian"], 0.)
            if kwargs.get("fixed_outer_boundary"):
                self.assertEqual(kwargs["outline_i"], first_outline)
                self.assertEqual(boundary_edges(result.outline), boundary_edges(kwargs["outline_o"]))
                return result
            first_outline.extend(result.outline)
            ratios = {"dense": [], "ordinary": []}
            splits = []
            dense_layouts = []
            for section in result.diagnostics["section_details"]:
                if section["kind"] != "LINE":
                    continue
                cells = result.mesh.elements[section["element_start"]:section["element_end"]]
                xy = result.mesh.nodes[cells, :2]
                lengths = np.linalg.norm(np.roll(xy, -1, axis=1) - xy, axis=2)
                group = "dense" if "midline_ids" in section else "ordinary"
                aspect = lengths.max(axis=1) / lengths.min(axis=1)
                ratios[group].append(aspect)
                if group == "dense":
                    middle = result.mesh.nodes[section["midline_ids"], :2]
                    gaps = np.linalg.norm(np.diff(middle, axis=0), axis=1)
                    self.assertGreater(gaps.min(), 5. + 1e-8)
                    for i in section["unit_ids"]:
                        unit = result.diagnostics["units"][i]
                        if unit["sublayer"] == 2:
                            a = unit["element_start"] - section["element_start"]
                            b = unit["element_end"] - section["element_start"]
                            self.assertGreaterEqual(lengths[a:b].min(), 10. - 1e-8)
                            self.assertLessEqual(aspect[a:b].max(), 20. + 1e-8)
                    dense_layouts.append((len(section["source_chains"][0]) - 1, len(gaps),
                                          len(section["outer_ids"]) - 1))
                if "parity_split_node" in section:
                    splits.append(section["parity_split_node"])
            self.assertEqual(len(set(splits)), 4)
            self.assertIn((37, 5, 1), dense_layouts)
            self.assertTrue(set(splits).issubset(first_outline))
            self.assertLess(np.concatenate(ratios["dense"]).max(), 80.)
            self.assertLess(np.median(np.concatenate(ratios["ordinary"])), 4.91)
            return result

        with patch.object(example, "generate_layer_adaptive_mesh", side_effect=check_layer):
            merged = example.merge_by_adapter(cow, example.get_no_cow_mesh())
        self.assertGreater(merged.element_count, cow.element_count)
        self.assertTrue(np.all(merged.element_types[cow.element_count:] == 4))
        self.assertEqual(len(example.EditorMesh2D.outlines(merged)), 1)

    def test_custom_layers_and_empty_input(self):
        lines = example._get_fillet_pattern_lines([[0, 0, 100, 100]], layers=((10, 2),))
        self.assertIn(((-20, -10), (120, -10)), lines)
        self.assertEqual(example._get_fillet_pattern_lines([]), [])
        for layers in ((), ((0, 1),), ((5, 0),), ((5, 1.5),), ((float("inf"), 1),)):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                example._get_fillet_pattern_lines([], layers=layers)


if __name__ == "__main__":
    unittest.main()
