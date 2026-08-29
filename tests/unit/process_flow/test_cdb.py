import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesher import Mesh3D
from mesher.process_flow.exporters.cdb import write_cdb_text


class CdbWriteTests(unittest.TestCase):
    def test_write_exports_mesh_3d(self):
        mesh = Mesh3D(
            component_ids_by_name={"EMPTY": 0, "body": 1},
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.25, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            elements=np.array(
                [[0, 1, 1, 0, 0, 1, 1, 0]],
                dtype=np.int32,
            ),
            element_component_ids=np.array([1], dtype=np.int32),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mesh.cdb"
            events = []
            metadata = write_cdb_text(output_path, mesh=mesh, progress=events.append)
            content = output_path.read_text(encoding="utf-8")

        self.assertEqual(metadata["nodeCount"], 2)
        self.assertEqual(metadata["elementCount"], 1)
        self.assertEqual(metadata["componentCount"], 2)
        self.assertEqual(
            content,
            "# Process Flow CDB text export\n"
            "# Format: raw mesh array sections\n"
            "node_count=2\n"
            "element_count=1\n"
            "component_count=2\n"
            "\n*NODES,index,x,y,z\n"
            "0,0,0,0\n"
            "1,1.25,0,1\n"
            "\n*ELEMENTS,index,n0,n1,n2,n3,n4,n5,n6,n7\n"
            "0,0,1,1,0,0,1,1,0\n"
            "\n*ELEMENT_COMP,index,component_id\n"
            "0,1\n"
            "\n*COMPS,component_id,name\n"
            '0,"EMPTY"\n'
            '1,"body"\n',
        )
        self.assertEqual(
            [(event["current"], event["message"]) for event in events],
            [
                (0, "Writing CDB nodes."),
                (1, "Writing CDB nodes."),
                (2, "Writing CDB nodes."),
                (2, "Writing CDB elements."),
                (3, "Writing CDB elements."),
                (3, "Writing CDB element components."),
                (4, "Writing CDB element components."),
                (4, "Writing CDB component table."),
                (5, "Writing CDB component table."),
                (6, "Writing CDB component table."),
                (6, "CDB output written."),
            ],
        )
        self.assertTrue(all(event["total"] == 6 for event in events))


if __name__ == "__main__":
    unittest.main()
