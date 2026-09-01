import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mesher import Mesh3D
from mesher.process_flow.worker import _write_cdb_atomically


class AtomicWorkerOutputTests(unittest.TestCase):
    def test_failed_export_preserves_existing_output_and_removes_temporary_file(self):
        mesh = Mesh3D(
            nodes=np.empty((0, 3)),
            elements=np.empty((0, 8), dtype=np.int32),
            element_comps=np.empty(0, dtype=np.int32),
            comps={"EMPTY": 0},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mesh.cdb"
            output_path.write_bytes(b"previous-complete-output")

            def fail_after_partial_write(path, **_kwargs):
                Path(path).write_bytes(b"partial")
                raise OSError("simulated exporter failure")

            with (
                patch(
                    "mesher.process_flow.worker.write_cdb_text",
                    side_effect=fail_after_partial_write,
                ),
                self.assertRaisesRegex(OSError, "simulated exporter failure"),
            ):
                _write_cdb_atomically(output_path, mesh=mesh, progress=None)

            self.assertEqual(output_path.read_bytes(), b"previous-complete-output")
            self.assertEqual(list(Path(temp_dir).glob(".mesh.cdb.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
