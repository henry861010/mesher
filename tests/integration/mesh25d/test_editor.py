import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mesher.mesh25d import build_mesh
from mesher.mesh3d import EditorMesh3D
from mesher.process_flow.exporters.cdb import write_cdb_text
from mesher.tolerance import GeometryToleranceError
from tests.unit.mesh25d.test_centerlines import assert_chain


class BoundaryEditorIntegrationTests(unittest.TestCase):
    def test_generated_symmetry_meshes_select_and_export_boundary_sets(self):
        structure = {"root": {"bodies": [{
            "geometry": {
                "type": "BoxGeometry", "bottom_left": [10, 20, 0],
                "top_right": [19, 29, 0], "thk": 7,
            },
            "material": "Si",
        }]}}
        for mode, count in (("full", 9), ("upper_half", 7),
                            ("right_half", 7), ("upper_right_quarter", 5)):
            with self.subTest(mode=mode):
                mesh = build_mesh(structure, {
                    "schemaVersion": "1.0.0", "unitSystem": "um",
                    "mesher": "process_flow_2_5d", "globalElementSize": 3,
                    "symmetry": mode, "controls": [],
                })
                cross = mesh.sets["fix_z"].indices
                self.assertEqual(len(cross), count)
                self.assertTrue(np.all(mesh.nodes[cross, 2] == mesh.nodes[:, 2].min()))
                for name in ("fix_x", "fix_y"):
                    self.assertTrue(set(cross).issubset(mesh.sets[name].indices))
                for axis, modes, name in (
                    (0, ("right_half", "upper_right_quarter"), "fix_x"),
                    (1, ("upper_half", "upper_right_quarter"), "fix_y"),
                ):
                    if mode in modes:
                        face = np.flatnonzero(mesh.nodes[:, axis] == mesh.nodes[:, axis].min())
                        np.testing.assert_array_equal(mesh.sets[name].indices, np.union1d(cross, face))
                if mode in ("upper_half", "right_half"):
                    axis = 0 if mode == "upper_half" else 1
                    center = (14.5, 24.5)
                    faces = np.vstack((mesh.elements[:, :4], mesh.elements[:, 4:8]))
                    for z in np.unique(mesh.nodes[:, 2]):
                        layer = faces[np.all(mesh.nodes[faces, 2] == z, axis=1)]
                        assert_chain(self, mesh.nodes, layer, axis, center[axis],
                                     center[1 - axis], (19, 29)[1 - axis])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "boundary.cdb"
                    metadata = write_cdb_text(path, mesh=mesh)
                    self.assertEqual(metadata["setCount"], 3)
                    text = path.read_text()
                    for name in ("fix_x", "fix_y", "fix_z"):
                        self.assertIn(f'"{name}",node,', text)

    def test_boundary_uses_the_build_tolerance_and_runs_before_completion(self):
        structure = {"root": {"bodies": [{
            "geometry": {"type": "BoxGeometry", "bottom_left": [0, 0, 0],
                         "top_right": [9, 9, 0], "thk": 1}, "material": "Si",
        }]}}
        events = []
        with patch("mesher.mesh25d.pipeline.EditorMesh3D.set_boundary",
                   wraps=EditorMesh3D.set_boundary) as boundary:
            mesh = build_mesh(structure, {
                "schemaVersion": "1.1.0", "unitSystem": "um",
                "mesher": "process_flow_2_5d", "globalElementSize": 3,
                "symmetry": "upper_half", "controls": [], "geometryTolerance": 0.125,
            }, progress=lambda event: events.append((event, boundary.call_count)))
        self.assertIs(boundary.call_args.args[0], mesh)
        self.assertEqual(boundary.call_args.args[1], "upper_half")
        self.assertEqual(boundary.call_args.kwargs["tolerance"].absolute, 0.125)
        completed = [count for event, count in events
                     if event["event"] == "stage.completed" and event["stage"] == "building_3d_mesh"]
        self.assertEqual(completed, [1])

    def test_insufficient_cross_nodes_fail_without_completing_3d_stage(self):
        structure = {"root": {"bodies": [{
            "geometry": {"type": "BoxGeometry", "bottom_left": [0, 0, 0],
                         "top_right": [2, 2, 0], "thk": 1}, "material": "Si",
        }]}}
        for mode in ("full", "upper_half", "right_half", "upper_right_quarter"):
            with self.subTest(mode=mode):
                events = []
                with self.assertRaises(GeometryToleranceError) as caught:
                    build_mesh(structure, {
                        "schemaVersion": "1.0.0", "unitSystem": "um",
                        "mesher": "process_flow_2_5d", "globalElementSize": 1,
                        "symmetry": mode, "controls": [],
                    }, progress=events.append)
                self.assertEqual(caught.exception.context["stage"], "boundary_selection")
                self.assertEqual(caught.exception.code, "fixed_constraint_conflict")
                self.assertFalse(any(event["event"] == "stage.completed"
                                     and event["stage"] == "building_3d_mesh" for event in events))
