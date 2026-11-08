"""Unmodified user case: interfaces, material coverage, volume and export.

This deliberately uses the original 1.0.0 fixture with no tolerance override.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mesher.mesh25d import build_mesh, region_adapter
from mesher.process_flow.exporters.cdb import write_cdb_text


class Case1ToleranceAcceptanceTests(unittest.TestCase):
    metrics = {}

    def test_original_case1_all_interfaces_material_volumes_and_export(self):
        fixture = Path(__file__).resolve().parents[3] / "docs/tolerance-audit/case1"
        geometry = json.loads((fixture / "model.geometry.json").read_text())
        settings = json.loads((fixture / "model.mesh-control.json").read_text())
        original_inputs = copy.deepcopy((geometry, settings))
        original = region_adapter.generate_full_adaptive_mesh
        metrics = {}

        def verify_interfaces(mesh, *args, **kwargs):
            nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
            result = original(mesh, *args, **kwargs)
            np.testing.assert_array_equal(mesh.nodes[:len(nodes)], nodes)
            np.testing.assert_array_equal(mesh.elements[:len(elements)], elements)
            interfaces = result.layer_outlines[-1]
            self.assertEqual(len(interfaces), 8)
            self.assertTrue(all(len(loop) % 2 == 0 for loop in interfaces))
            directed = np.stack((mesh.elements, np.roll(mesh.elements, -1, axis=1)), axis=2).reshape(-1, 2)
            keys, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                             return_inverse=True, return_counts=True)
            balance = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
            self.assertTrue(np.all(counts <= 2))
            np.testing.assert_array_equal(balance[counts == 2], 0)
            exterior = {tuple(sorted((int(a), int(b)))) for a, b in
                        zip(result.outer_outline, np.roll(result.outer_outline, -1))}
            self.assertEqual(set(map(tuple, keys[counts == 1])), exterior)
            lookup = {tuple(edge): i for i, edge in enumerate(keys)}
            for loop in interfaces:
                for a, b in zip(loop, np.roll(loop, -1)):
                    self.assertEqual(counts[lookup[tuple(sorted((int(a), int(b))))]], 2)
            cells = mesh.nodes[mesh.elements, :2]
            local = cells - cells[:, :1]
            following = np.roll(local, -1, axis=1)
            area = .5 * np.sum(local[..., 0] * following[..., 1] - local[..., 1] * following[..., 0])
            self.assertAlmostEqual(area, 1_600_000_000., delta=.001)
            edges = np.roll(cells, -1, axis=1) - cells
            next_edges = np.roll(edges, -1, axis=1)
            self.assertTrue(np.all(edges[..., 0] * next_edges[..., 1] - edges[..., 1] * next_edges[..., 0] > 0))
            minimum = np.hypot(edges[..., 0], edges[..., 1]).min()
            self.assertGreater(minimum, .001)
            metrics.update(frozen_nodes=len(nodes), frozen_elements=len(elements), interfaces=len(interfaces),
                           interface_node_counts=list(map(len, interfaces)), area=float(area),
                           minimum_edge=float(minimum), source_preserved_exactly=True)
            return result

        with patch.object(region_adapter, "generate_full_adaptive_mesh", verify_interfaces):
            mesh = build_mesh(geometry, settings)
        self.assertEqual((geometry, settings), original_inputs)
        corners = mesh.nodes[mesh.elements[:, :8]]
        np.testing.assert_array_equal(corners[:, :4, :2], corners[:, 4:8, :2])
        np.testing.assert_array_equal(corners[:, :4, 2], np.broadcast_to(corners[:, :1, 2], corners[:, :4, 2].shape))
        np.testing.assert_array_equal(corners[:, 4:8, 2], np.broadcast_to(corners[:, 4:5, 2], corners[:, 4:8, 2].shape))
        height = corners[:, 4, 2] - corners[:, 0, 2]
        self.assertTrue(np.all(height > .001))
        base = corners[:, :4, :2] - corners[:, :1, :2]
        following = np.roll(base, -1, axis=1)
        areas = .5 * np.abs(np.sum(base[..., 0] * following[..., 1] - base[..., 1] * following[..., 0], axis=1))
        volumes = areas * height
        self.assertTrue(np.all(volumes > 0))
        expected = {"Si": 40_000 * 40_000 * 775,
                    "si": (10_000 * 10_000 * 2 + 10_000 * 19_500) * 200}
        measured = {}
        for material, volume in expected.items():
            selected = mesh.element_comps == mesh.comps[material]
            measured[material] = float(volumes[selected].sum())
            self.assertAlmostEqual(measured[material], volume, delta=.01)
        self.assertAlmostEqual(float(volumes.sum()), sum(expected.values()), delta=.01)
        # Every child-material corner must lie in one actual child footprint,
        # and every carrier cell in its own Z interval.
        carrier = corners[mesh.element_comps == mesh.comps["Si"]]
        self.assertTrue(np.all((carrier[..., 2] >= -387.5) & (carrier[..., 2] <= 387.5)))
        children = corners[mesh.element_comps == mesh.comps["si"]]
        contained = np.zeros(len(children), dtype=bool)
        for child in geometry["root"]["children"]:
            box = child["bodies"][0]["geometry"]
            low = np.asarray(box["bottom_left"], dtype=float)
            high = np.asarray(box["top_right"], dtype=float)
            high[2] += box["thk"]
            contained |= np.all((children >= low - .001) & (children <= high + .001), axis=(1, 2))
        self.assertTrue(np.all(contained))
        self.assertEqual(len(np.unique(np.sort(mesh.elements[:, :8], axis=1), axis=0)), mesh.element_count)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case1.cdb"
            write_cdb_text(output, mesh=mesh)
            parsed = np.empty_like(mesh.nodes)
            section = None
            count = 0
            with output.open() as stream:
                for line in stream:
                    if line.startswith("*"):
                        section = line.split(",", 1)[0]
                    elif section == "*NODES" and line.strip():
                        index, x, y, z = line.strip().split(",")
                        parsed[int(index)] = (float(x), float(y), float(z))
                        count += 1
            self.assertEqual(count, mesh.node_count)
            np.testing.assert_array_equal(parsed, mesh.nodes)
        metrics.update(nodes=mesh.node_count, elements=mesh.element_count,
                       material_volumes=measured, volume=float(volumes.sum()), export_roundtrip_exact=True)
        type(self).metrics = metrics
