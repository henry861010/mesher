import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesher.mesh25d import build_mesh
from mesher.mesh3d import EditorMesh3D
from mesher.process_flow.exporters.cdb import write_cdb_text


class BoundaryEditorIntegrationTests(unittest.TestCase):
    def test_generated_symmetry_meshes_select_and_export_boundary_sets(self):
        structure = {"root": {"bodies": [{
            "geometry": {
                "type": "BoxGeometry", "bottom_left": [10, 20, 0],
                "top_right": [22, 32, 0], "thk": 2,
            },
            "material": "Si",
        }]}}
        for mode, count in (("full", 9), ("upper_half", 7),
                            ("right_half", 7), ("upper_right_quarter", 5)):
            with self.subTest(mode=mode):
                mesh = build_mesh(structure, {
                    "schemaVersion": "1.0.0", "unitSystem": "um",
                    "mesher": "process_flow_2_5d", "globalElementSize": 1,
                    "symmetry": mode, "controls": [],
                })
                EditorMesh3D.set_boundary(mesh, mode)
                cross = mesh.sets["fix_z"].indices
                self.assertEqual(len(cross), count)
                self.assertTrue(np.all(mesh.nodes[cross, 2] == mesh.nodes[:, 2].min()))
                for name in ("fix_x", "fix_y"):
                    self.assertTrue(set(cross).issubset(mesh.sets[name].indices))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "boundary.cdb"
                    metadata = write_cdb_text(path, mesh=mesh)
                    self.assertEqual(metadata["setCount"], 3)
                    text = path.read_text()
                    for name in ("fix_x", "fix_y", "fix_z"):
                        self.assertIn(f'"{name}",node,', text)
