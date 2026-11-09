import unittest

import numpy as np

from mesher import GeometryTolerance, GeometryToleranceError, Mesh3D, MeshSet
from mesher.mesh3d import EditorMesh3D


def _mesh(xs=range(-3, 4), ys=range(-3, 4), zs=(-2, 0, 2), *, offset=(0, 0, 0)):
    nodes = np.array([(x, y, z) for z in zs for y in ys for x in xs], dtype=float)
    nodes = nodes.reshape(-1, 3) + offset
    return Mesh3D(
        nodes=nodes, elements=np.empty((0, 20), dtype=np.int32),
        element_comps=[], comps={}, element_types=[], types={},
        element_reals=[], reals={}, element_sections=[], sections={},
        element_node_num=[],
    )


def _ids_at(mesh, coordinates):
    lookup = {tuple(point): index for index, point in enumerate(mesh.nodes)}
    return np.sort([lookup[tuple(point)] for point in coordinates])


class EditorMesh3DTests(unittest.TestCase):
    def test_all_modes_fix_cross_in_every_direction_and_add_symmetry_faces(self):
        cases = (
            ("full", range(-3, 4), range(-3, 4),
             [(0, 0), (-2, 0), (-1, 0), (1, 0), (2, 0),
              (0, -2), (0, -1), (0, 1), (0, 2)], ()),
            ("upper_half", range(-3, 4), range(4),
             [(0, 0), (-2, 0), (-1, 0), (1, 0), (2, 0), (0, 1), (0, 2)], (1,)),
            ("right_half", range(4), range(-3, 4),
             [(0, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2)], (0,)),
            ("upper_right_quarter", range(4), range(4),
             [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)], (0, 1)),
        )
        for mode, xs, ys, cross_xy, symmetry_axes in cases:
            with self.subTest(mode=mode):
                mesh = _mesh(xs, ys, offset=(11, 17, 5))
                before = mesh.nodes, mesh.elements
                self.assertIs(EditorMesh3D.set_boundary(mesh, mode), mesh)
                cross = _ids_at(mesh, [(x + 11, y + 17, 3) for x, y in cross_xy])
                for axis, name in enumerate(("fix_x", "fix_y", "fix_z")):
                    expected = cross
                    if axis in symmetry_axes:
                        face = np.flatnonzero(mesh.nodes[:, axis] == (11, 17)[axis])
                        expected = np.union1d(cross, face)
                        self.assertTrue(np.any(mesh.nodes[face, 2] > 3))
                    selection = mesh.sets[name]
                    self.assertEqual(selection.kind, "node")
                    self.assertEqual(selection.indices.dtype, np.int32)
                    np.testing.assert_array_equal(selection.indices, expected)
                self.assertIs(mesh.nodes, before[0])
                self.assertIs(mesh.elements, before[1])
                self.assertFalse(np.shares_memory(
                    mesh.sets["fix_x"].indices, mesh.sets["fix_z"].indices,
                ))

    def test_nonuniform_shuffled_nodes_choose_nearest_two_per_arm(self):
        mesh = _mesh(
            [-9, -1, -0.25, 0, 0.5, 3, 9], [-8, -2, -0.75, 0, 1, 4, 8],
        )
        np.random.default_rng(42).shuffle(mesh.nodes)
        cross_xy = [(0, 0), (-0.25, 0), (-1, 0), (0.5, 0), (3, 0),
                    (0, -0.75), (0, -2), (0, 1), (0, 4)]
        EditorMesh3D.set_boundary(mesh, "full")
        expected = _ids_at(mesh, [(x, y, -2) for x, y in cross_xy])
        np.testing.assert_array_equal(mesh.sets["fix_z"].indices, expected)

    def test_bottom_center_ignores_wider_upper_layers(self):
        mesh = _mesh(zs=(-2,))
        # The upper footprint has a different center from the bottom footprint.
        from dataclasses import replace

        mesh = replace(mesh, nodes=np.vstack((mesh.nodes, [[20, 40, 2], [10, 20, 2]])))
        EditorMesh3D.set_boundary(mesh, "full")
        self.assertEqual(len(mesh.sets["fix_z"].indices), 9)
        self.assertTrue(np.all(mesh.nodes[mesh.sets["fix_z"].indices, 2] == -2))

    def test_replaces_boundary_sets_preserves_others_and_reads_current_coordinates(self):
        mesh = _mesh(range(4), range(4))
        other = MeshSet(kind="node", indices=[mesh.node_count - 1])
        mesh.sets.update({"load": other, "fix_z": MeshSet(kind="node", indices=[])})
        sets = mesh.sets
        EditorMesh3D.set_boundary(mesh, "upper_right_quarter")
        self.assertIs(mesh.sets, sets)
        self.assertIs(mesh.sets["load"], other)
        old_indices = mesh.sets["fix_z"].indices.copy()
        mesh.nodes[:, 0] *= -1
        mesh.nodes[:, 1] *= -1
        EditorMesh3D().set_boundary(mesh, "upper_right_quarter")
        self.assertIs(mesh.sets["load"], other)
        self.assertFalse(np.array_equal(mesh.sets["fix_z"].indices, old_indices))
        expected = _ids_at(mesh, [(-3, -3, -2), (-2, -3, -2), (-1, -3, -2),
                                  (-3, -2, -2), (-3, -1, -2)])
        np.testing.assert_array_equal(mesh.sets["fix_z"].indices, expected)

    def test_switching_modes_removes_previous_symmetry_face_membership(self):
        mesh = _mesh(range(7), range(7))
        EditorMesh3D.set_boundary(mesh, "upper_right_quarter")
        self.assertGreater(len(mesh.sets["fix_x"].indices), 9)
        EditorMesh3D.set_boundary(mesh, "full")
        for name in ("fix_x", "fix_y", "fix_z"):
            self.assertEqual(len(mesh.sets[name].indices), 9)

    def test_tolerance_is_absolute_inclusive_and_applies_to_whole_symmetry_face(self):
        mesh = _mesh(range(4), range(4), zs=(0, 1))
        # A top node exactly T from x=min is included; the next float is not.
        near = _ids_at(mesh, [(0, 3, 1)])[0]
        far = _ids_at(mesh, [(0, 2, 1)])[0]
        mesh.nodes[near, 0] = 0.125
        mesh.nodes[far, 0] = np.nextafter(0.125, np.inf)
        EditorMesh3D.set_boundary(mesh, "upper_right_quarter", tolerance=GeometryTolerance(0.125))
        self.assertIn(near, mesh.sets["fix_x"].indices)
        self.assertNotIn(far, mesh.sets["fix_x"].indices)

    def test_cross_arm_matching_uses_tolerance(self):
        mesh = _mesh(zs=(0,))
        near = _ids_at(mesh, [(1, 0, 0)])[0]
        mesh.nodes[near, 1] = 0.125
        EditorMesh3D.set_boundary(mesh, "full", tolerance=0.125)
        self.assertIn(near, mesh.sets["fix_z"].indices)
        mesh.nodes[near, 1] = np.nextafter(0.125, np.inf)
        EditorMesh3D.set_boundary(mesh, "full", tolerance=0.125)
        self.assertNotIn(near, mesh.sets["fix_z"].indices)
        far = _ids_at(mesh, [(3, 0, 0)])[0]
        self.assertIn(far, mesh.sets["fix_z"].indices)

    def test_missing_ambiguous_center_and_short_arms_fail_before_mutating_sets(self):
        meshes = (
            _mesh([-3, -2, -1, 1, 2, 3]),
            _mesh([-3, -2, -1, 0, 0, 1, 2, 3]),
            _mesh(range(-1, 2)),
        )
        for mesh in meshes:
            with self.subTest(nodes=mesh.node_count):
                previous = MeshSet(kind="node", indices=[0])
                mesh.sets.update({"fix_x": previous, "load": previous})
                with self.assertRaises(GeometryToleranceError) as caught:
                    EditorMesh3D.set_boundary(mesh, "full")
                self.assertEqual(caught.exception.context["stage"], "boundary_selection")
                self.assertEqual(mesh.sets, {"fix_x": previous, "load": previous})

    def test_invalid_arguments_and_coordinates_fail_before_mutating_sets(self):
        for mode in ("", "quarter", "FULL", None, [], 1):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "model_type"):
                EditorMesh3D.set_boundary(_mesh(), mode)
        for tolerance in (0, -1, True, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                EditorMesh3D.set_boundary(_mesh(), "full", tolerance=tolerance)
        for coordinate in (np.nan, np.inf):
            mesh = _mesh()
            mesh.nodes[-1, 2] = coordinate
            with self.subTest(coordinate=coordinate), self.assertRaisesRegex(ValueError, "finite"):
                EditorMesh3D.set_boundary(mesh, "full")
            self.assertEqual(mesh.sets, {})
        with self.assertRaisesRegex(ValueError, "contain nodes"):
            EditorMesh3D.set_boundary(_mesh([]), "full")
        with self.assertRaisesRegex(TypeError, "Mesh3D"):
            EditorMesh3D.set_boundary(None, "full")

    def test_numeric_resolution_failure_preserves_sets(self):
        mesh = _mesh(offset=(1e16, 0, 0))
        with self.assertRaises(GeometryToleranceError) as caught:
            EditorMesh3D.set_boundary(mesh, "full")
        self.assertEqual(caught.exception.code, "numeric_resolution_exceeded")
        self.assertEqual(mesh.sets, {})

    def test_public_stateless_api_accepts_existing_symmetry_enum(self):
        import mesher
        from mesher.mesh25d import SymmetryMode
        from mesher.mesh3d.editor import EditorMesh3D as EditorFromPackage

        self.assertIs(EditorMesh3D, EditorFromPackage)
        self.assertIsInstance(EditorMesh3D.__dict__["set_boundary"], staticmethod)
        self.assertNotIn("EditorMesh3D", mesher.__all__)
        with self.assertRaises(AttributeError):
            EditorMesh3D().mesh = _mesh()
        mesh = _mesh()
        EditorMesh3D.set_boundary(mesh, SymmetryMode.FULL)
        self.assertEqual(len(mesh.sets["fix_z"].indices), 9)
