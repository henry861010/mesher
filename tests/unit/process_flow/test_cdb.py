import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mesher import Mesh3D, MeshSet
from mesher.process_flow.exporters.cdb import write_cdb_text


class CdbWriteTests(unittest.TestCase):
    def _mesh_with_sets(self, sets):
        return Mesh3D(
            nodes=np.zeros((4, 3)),
            elements=np.zeros((2, 20), dtype=np.int32),
            element_comps=[1, 1],
            comps={"body": 1},
            element_types=[1, 1],
            types={1: 185},
            element_reals=[0, 0],
            reals={},
            element_sections=[0, 0],
            sections={},
            element_node_num=[8, 8],
            sets=sets,
        )

    def test_exports_sorted_sets_with_json_names_and_complete_progress(self):
        special_name = 'a,邊界"\\\n'
        mesh = self._mesh_with_sets({
            "z_elements": MeshSet(kind="element", indices=[1, 0]),
            "empty": MeshSet(kind="node", indices=[]),
            special_name: MeshSet(kind="node", indices=[3, 0, 3]),
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sets.cdb"
            baseline_path = Path(temp_dir) / "baseline.cdb"
            events = []
            metadata = write_cdb_text(path, mesh=mesh, progress=events.append)
            content = path.read_text(encoding="utf-8")
            baseline_events = []
            baseline_metadata = write_cdb_text(
                baseline_path, mesh=replace(mesh, sets={}),
                progress=baseline_events.append,
            )
            write_cdb_text(path, mesh=replace(mesh, sets=dict(reversed(list(mesh.sets.items())))))
            self.assertEqual(path.read_text(encoding="utf-8"), content)

        self.assertIn("\nset_count=3\n", content)
        section = content.split("\n*SETS,name_json,kind,indices_json\n")[1]
        decoded = []
        decoder = json.JSONDecoder()
        for row in section.splitlines():
            name, end = decoder.raw_decode(row)
            self.assertEqual(row[end], ",")
            kind, indices = row[end + 1:].split(",", 1)
            decoded.append((name, kind, json.loads(indices)))
        self.assertEqual(decoded, [
            (special_name, "node", [0, 3]),
            ("empty", "node", []),
            ("z_elements", "element", [0, 1]),
        ])
        self.assertEqual(metadata["setCount"], 3)
        self.assertEqual(baseline_metadata["setCount"], 0)
        expected_total = baseline_events[-1]["total"] + 3
        self.assertEqual(events[-1]["current"], expected_total)
        self.assertEqual(events[-1]["message"], "CDB output written.")
        self.assertTrue(all(event["total"] == expected_total for event in events))

    def test_export_normalizes_modified_indices_without_mutating_the_mesh(self):
        mesh = self._mesh_with_sets({"nodes": MeshSet(kind="node", indices=[0, 1, 2])})
        mesh.sets["nodes"].indices[:] = [2, 0, 2]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sets.cdb"
            write_cdb_text(path, mesh=mesh)
            self.assertIn('"nodes",node,[0,2]\n', path.read_text(encoding="utf-8"))
        np.testing.assert_array_equal(mesh.sets["nodes"].indices, [2, 0, 2])

    def test_revalidates_sets_before_opening_the_output(self):
        negative = MeshSet(kind="node", indices=[0])
        negative.indices[0] = -1
        non_vector = MeshSet(kind="node", indices=[0])
        non_vector.indices.resize((1, 1))
        cases = (
            ({"": MeshSet(kind="node", indices=[])}, ValueError),
            ({1: MeshSet(kind="node", indices=[])}, TypeError),
            ({"bad": {"kind": "node", "indices": [0]}}, TypeError),
            ({"bad": MeshSet(kind="node", indices=[4])}, IndexError),
            ({"bad": MeshSet(kind="element", indices=[2])}, IndexError),
            ({"bad": negative}, ValueError),
            ({"bad": non_vector}, ValueError),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sets.cdb"
            for sets, error in cases:
                with self.subTest(sets=sets):
                    mesh = self._mesh_with_sets({})
                    mesh.sets.update(sets)
                    events = []
                    with self.assertRaises(error):
                        write_cdb_text(path, mesh=mesh, progress=events.append)
                    self.assertFalse(path.exists())
                    self.assertEqual(events, [])
                    path.write_bytes(b"previous complete output")
                    with self.assertRaises(error):
                        write_cdb_text(path, mesh=mesh)
                    self.assertEqual(path.read_bytes(), b"previous complete output")
                    path.unlink()

    def test_write_exports_mesh_3d(self):
        mesh = Mesh3D(
            comps={"EMPTY": 0, "body": 1},
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.25, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            elements=np.array(
                [[0, 1, 1, 0, 0, 1, 1, 0] + [0] * 12],
                dtype=np.int32,
            ),
            element_comps=np.array([1], dtype=np.int32),
            element_types=np.array([1], dtype=np.int32),
            types={1: 185},
            element_reals=np.array([3], dtype=np.int32),
            reals={3: [1.5, "real"]},
            element_sections=np.array([4], dtype=np.int32),
            sections={4: [2, 3]},
            element_node_num=np.array([8], dtype=np.int32),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mesh.cdb"
            events = []
            metadata = write_cdb_text(output_path, mesh=mesh, progress=events.append)
            content = output_path.read_text(encoding="utf-8")

        self.assertEqual(metadata["nodeCount"], 2)
        self.assertEqual(metadata["elementCount"], 1)
        self.assertEqual(metadata["componentCount"], 2)
        self.assertEqual(metadata["typeCount"], 1)
        self.assertEqual(metadata["realCount"], 1)
        self.assertEqual(metadata["sectionCount"], 1)
        self.assertEqual(metadata["setCount"], 0)
        self.assertEqual(
            content,
            "# Process Flow CDB text export\n"
            "# Format: raw mesh array sections\n"
            "# unit_system=um\n"
            "# geometry_tolerance=0.001\n"
            "# coordinate_alignment=bounded_xy_equivalence\n"
            "# tolerance_policy_version=3\n"
            "node_count=2\n"
            "element_count=1\n"
            "component_count=2\n"
            "type_count=1\n"
            "real_count=1\n"
            "section_count=1\n"
            "\n*NODES,index,x,y,z\n"
            "0,0.0,0.0,0.0\n"
            "1,1.25,0.0,1.0\n"
            "\n*ELEMENTS,index,n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19\n"
            "0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
            "\n*ELEMENT_NODE_NUM,index,node_num\n"
            "0,8\n"
            "\n*ELEMENT_TYPE,index,type_id\n"
            "0,1\n"
            "\n*TYPES,type_id,ansys_element_type\n"
            "1,185\n"
            "\n*ELEMENT_REAL,index,real_id\n"
            "0,3\n"
            "\n*REALS,real_id,values_json\n"
            '3,[1.5,"real"]\n'
            "\n*ELEMENT_SECTION,index,section_id\n"
            "0,4\n"
            "\n*SECTIONS,section_id,values_json\n"
            "4,[2,3]\n"
            "\n*ELEMENT_COMP,index,component_id\n"
            "0,1\n"
            "\n*COMPS,component_id,name\n"
            '0,"EMPTY"\n'
            '1,"body"\n',
        )
        self.assertEqual(events[-1]["current"], 13)
        self.assertEqual(events[-1]["message"], "CDB output written.")
        self.assertTrue(all(event["total"] == 13 for event in events))


if __name__ == "__main__":
    unittest.main()
