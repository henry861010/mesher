import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesher import Mesh3D
from mesher.process_flow.exporters.cdb import write_cdb_text


class CdbWriteTests(unittest.TestCase):
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
        self.assertEqual(
            content,
            "# Process Flow CDB text export\n"
            "# Format: raw mesh array sections\n"
            "node_count=2\n"
            "element_count=1\n"
            "component_count=2\n"
            "type_count=1\n"
            "real_count=1\n"
            "section_count=1\n"
            "\n*NODES,index,x,y,z\n"
            "0,0,0,0\n"
            "1,1.25,0,1\n"
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
