import unittest

import numpy as np

from mesher import Mesh3D, MeshSet


def _mesh(**overrides):
    values = {
        "nodes": np.zeros((4, 3)),
        "elements": np.zeros((2, 20), dtype=np.int32),
        "element_comps": [1, 1],
        "comps": {"body": 1},
        "element_types": [1, 1],
        "types": {1: 185},
        "element_reals": [0, 0],
        "reals": {},
        "element_sections": [0, 0],
        "sections": {},
        "element_node_num": [8, 8],
    }
    values.update(overrides)
    return Mesh3D(**values)


class MeshSetTests(unittest.TestCase):
    def test_normalizes_and_owns_indices(self):
        source = np.array([3, 99, 1, 99, 3, 99], dtype=np.int64)[::2]
        selection = MeshSet(kind="node", indices=source)
        source[:] = 0

        self.assertEqual(selection.kind, "node")
        np.testing.assert_array_equal(selection.indices, [1, 3])
        self.assertEqual(selection.indices.dtype, np.int32)
        self.assertTrue(selection.indices.flags.c_contiguous)
        self.assertTrue(selection.indices.flags.writeable)

    def test_accepts_integer_sequences_and_empty_sets(self):
        for values in ([2, 0, 2], (2, 0, 2), np.array([2, 0, 2], dtype=np.uint64)):
            with self.subTest(values=values):
                np.testing.assert_array_equal(
                    MeshSet(kind="element", indices=values).indices, [0, 2]
                )
        for values in ([], (), np.empty(0, dtype=np.int64)):
            with self.subTest(values=values):
                indices = MeshSet(kind="node", indices=values).indices
                self.assertEqual(indices.shape, (0,))
                self.assertEqual(indices.dtype, np.int32)

    def test_accepts_largest_int32_index_without_wrapping(self):
        largest = np.iinfo(np.int32).max
        selection = MeshSet(kind="node", indices=np.array([largest], dtype=np.uint64))
        np.testing.assert_array_equal(selection.indices, [largest])

    def test_rejects_invalid_kind(self):
        for kind in ("NODE", "face", "", None, 1, ["node"]):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "kind"):
                    MeshSet(kind=kind, indices=[])

    def test_rejects_non_integer_values_including_mixed_booleans(self):
        for values in (
            [0.0, 1.0], [True], [0, True], [0, np.bool_(True)], ["1"],
            np.array([1.0]), np.array([True]), np.array([1 + 0j]),
            np.array([1], dtype=object),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(TypeError, "integers"):
                    MeshSet(kind="node", indices=values)

    def test_rejects_non_vector_indices(self):
        for values in (0, None, [[0, 1]], np.empty((0, 1))):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    MeshSet(kind="node", indices=values)

    def test_rejects_negative_and_overflowing_indices_before_casting(self):
        for values in ([-1], [-2**100], np.array([-1], dtype=np.int64)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    MeshSet(kind="node", indices=values)
        for values in (
            [2**31], [2**100], np.array([2**64 - 1], dtype=np.uint64),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(OverflowError, "int32"):
                    MeshSet(kind="node", indices=values)


class Mesh3DSetTests(unittest.TestCase):
    def test_default_sets_are_empty_and_independent(self):
        first, second = _mesh(), _mesh()
        self.assertEqual(first.sets, {})
        first.sets["nodes"] = MeshSet(kind="node", indices=[0])
        self.assertEqual(second.sets, {})

    def test_owns_mapping_and_each_set_array(self):
        selection = MeshSet(kind="node", indices=[0, 3])
        source = {"boundary": selection}
        mesh = _mesh(sets=source)
        source.clear()
        selection.indices[:] = 1

        self.assertEqual(set(mesh.sets), {"boundary"})
        self.assertIsNot(mesh.sets["boundary"], selection)
        np.testing.assert_array_equal(mesh.sets["boundary"].indices, [0, 3])

    def test_supports_overlapping_and_case_sensitive_names(self):
        mesh = _mesh(sets={
            "body": MeshSet(kind="node", indices=[0, 3]),
            "Body": MeshSet(kind="node", indices=[1, 3]),
            "solid": MeshSet(kind="element", indices=[0, 1]),
            "empty": MeshSet(kind="element", indices=[]),
        })
        self.assertEqual(set(mesh.sets), {"body", "Body", "solid", "empty"})
        self.assertEqual(mesh.comps, {"body": 1})
        np.testing.assert_array_equal(mesh.element_comps, [1, 1])

    def test_rejects_invalid_names_and_values(self):
        valid = MeshSet(kind="node", indices=[])
        cases = (
            ({"": valid}, ValueError, "empty"),
            ({1: valid}, TypeError, "strings"),
            ({"bad": {"kind": "node", "indices": []}}, TypeError, "MeshSet"),
            ([], TypeError, "dictionary"),
        )
        for values, error, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(error, message):
                    _mesh(sets=values)

    def test_checks_bounds_against_the_selected_entity_type(self):
        for kind, count in (("node", 4), ("element", 2)):
            with self.subTest(kind=kind):
                _mesh(sets={"last": MeshSet(kind=kind, indices=[count - 1])})
                with self.assertRaisesRegex(IndexError, f"out-of-range {kind}"):
                    _mesh(sets={"past_end": MeshSet(kind=kind, indices=[count])})

    def test_revalidates_set_arrays_modified_before_mesh_construction(self):
        selection = MeshSet(kind="node", indices=[0])
        selection.indices[0] = -1
        with self.assertRaisesRegex(ValueError, "set 'boundary'.*nonnegative"):
            _mesh(sets={"boundary": selection})


if __name__ == "__main__":
    unittest.main()
