"""Regression evidence for the absolute process-flow geometry contract."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesher.contracts.process_flow_2_5d import mesh_control_settings, mesh_control_tolerance
from mesher.mesh25d import build_mesh
from mesher.mesh25d.domain import ModelDomain, SymmetryMode, face_intersects_domain
from mesher.mesh25d.geometry.context import prepare_geometry
from mesher.mesh25d.plan2d import _section_avg_lines
from mesher.mesh3d.extrusion import Dragger
from mesher.process_flow.exporters.cdb import write_cdb_text
from mesher.tolerance import GeometryToleranceError


def settings(**values):
    return dict(schemaVersion="1.1.0", unitSystem="um", mesher="process_flow_2_5d",
                globalElementSize=1.0, symmetry="full", controls=[], **values)


def body(*, x=0.0, z=0.0, thickness=1.0, material="Si"):
    return {"geometry": {"type": "BoxGeometry", "bottom_left": [x, 0, z],
                         "top_right": [x + 1, 1, z], "thk": thickness},
            "material": material}


def structure(*bodies):
    return {"root": {"bodies": list(bodies), "vias": [], "circuits": [],
                     "bumps": [], "children": []}}


def point(z):
    return {"method": "Z_POINT", "coordinateMode": "absolute", "label": "cut", "z": z}


class AbsoluteTolerancePipelineTests(unittest.TestCase):
    def test_schema_versions_share_default_and_validate_explicit_policy(self):
        for version in ("1.0.0", "1.1.0"):
            value = settings()
            value["schemaVersion"] = version
            before = copy.deepcopy(value)
            mesh_control_settings(value)
            self.assertEqual(mesh_control_tolerance(value).absolute, .001)
            self.assertEqual(value, before)
            value["geometryTolerance"] = .005
            mesh_control_settings(value)
            self.assertEqual(mesh_control_tolerance(value).absolute, .005)
        for invalid in (None, True, 0, -1, float("nan"), float("inf"), "0.001"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mesh_control_settings(settings(geometryTolerance=invalid))

    def test_near_z_controls_coalesce_and_preserve_volume_in_either_order(self):
        for values in ((.5, .5004), (.5004, .5)):
            config = settings()
            config["controls"] = [point(z) for z in values]
            model = structure(body())
            before = copy.deepcopy((model, config))
            mesh = build_mesh(model, config)
            np.testing.assert_array_equal(np.unique(mesh.nodes[:, 2]), [0, .5, 1])
            self.assertEqual(mesh.element_count, 8)
            self.assertEqual((model, config), before)

    def test_point_control_snaps_to_material_boundary_instead_of_creating_sliver(self):
        model = structure(body(thickness=.5), body(z=.5, thickness=.5, material="Cu"))
        config = settings()
        config["controls"] = [point(.4996)]
        mesh = build_mesh(model, config)
        np.testing.assert_array_equal(np.unique(mesh.nodes[:, 2]), [0, .5, 1])
        self.assertEqual(mesh.element_count, 8)
        self.assertEqual(set(mesh.element_comps), {mesh.comps["Si"], mesh.comps["Cu"]})

    def test_geometry_is_not_rounded_to_a_fixed_decimal_grid(self):
        mesh = build_mesh(structure(body(z=1.000499, thickness=.5)), settings())
        np.testing.assert_array_equal(np.unique(mesh.nodes[:, 2]), [1.000499, 1.500499])

    def test_subresolution_material_or_koz_is_an_explicit_error(self):
        for thickness in (.0004, .001):
            with self.subTest(thickness=thickness), self.assertRaises(GeometryToleranceError) as caught:
                build_mesh(structure(body(thickness=thickness)), settings())
            self.assertEqual(caught.exception.code, "feature_below_resolution")
        model = structure(body())
        model["root"]["vias"] = [{**body(material="Cu"), "density": 100, "koz": .0001}]
        with self.assertRaisesRegex(GeometryToleranceError, "KOZ"):
            build_mesh(model, settings())

    def test_koz_does_not_reduce_global_tolerance(self):
        model = structure(body())
        model["root"]["vias"] = [{**body(material="Cu"), "density": 100, "koz": .05}]
        context = prepare_geometry(model["root"], SymmetryMode.FULL)
        self.assertEqual(context.pattern.tolerance.absolute, .001)

    def test_ambiguous_z_chain_is_rejected_before_extrusion(self):
        config = settings()
        config["controls"] = [point(z) for z in (.5, .5008, .5016)]
        with self.assertRaises(GeometryToleranceError) as caught:
            build_mesh(structure(body()), config)
        self.assertEqual(caught.exception.code, "ambiguous_geometry")

    def test_normalization_does_not_swallow_a_distinct_material_rim(self):
        model = structure(body(), body(x=.0004, material="Cu"))
        before = copy.deepcopy(model)
        with self.assertRaises(GeometryToleranceError) as caught:
            build_mesh(model, settings())
        self.assertEqual(caught.exception.code, "feature_below_resolution")
        self.assertEqual(caught.exception.context["reason"], "material_boundary_conflict")
        self.assertEqual(model, before)

    def test_polygon_z_projection_shares_the_total_displacement_budget(self):
        polygon = {"geometry": {"type": "PolygonGeometry", "thk": 1, "polys": [[
            [.0008, 2, .0008], [1, 2, 0], [1, 3, 0], [.0008, 3, 0],
        ]]}, "material": "Si"}
        model = structure(body(), polygon)
        before = copy.deepcopy(model)
        with self.assertRaises(GeometryToleranceError) as caught:
            prepare_geometry(model["root"], SymmetryMode.FULL)
        self.assertEqual(caught.exception.code, "ambiguous_geometry")
        self.assertGreater(caught.exception.context["displacement"], .001)
        self.assertEqual(model, before)

    def test_normalization_preserves_partial_material_rims_and_shared_interfaces(self):
        outer = body()
        outer["geometry"]["top_right"] = [2, 2, 0]
        inner = body(x=.0005, material="Cu")
        inner["geometry"]["top_right"] = [1, 2, 0]
        model = structure(outer, inner)
        before = copy.deepcopy(model)
        with self.assertRaises(GeometryToleranceError) as caught:
            prepare_geometry(model["root"], SymmetryMode.FULL)
        self.assertEqual(caught.exception.code, "feature_below_resolution")
        self.assertEqual(caught.exception.context["reason"], "material_boundary_conflict")
        self.assertEqual(model, before)
        # An interface that was already shared is valid, including duplicate
        # geometry carrying an independent material assignment.
        inner["geometry"]["bottom_left"][0] = 0
        context = prepare_geometry(structure(outer, inner, body(material="Cu"))["root"], SymmetryMode.FULL)
        self.assertEqual(len(context.container["bodies"]), 3)

    def test_circle_tangency_cannot_erase_a_material_gap(self):
        outer = body()
        outer["geometry"]["top_right"] = [3, 3, 0]
        circle = {"geometry": {"type": "CylinderGeometry", "center": [1.0005, 1.5, 0],
                                "bottom_radius": 1, "thk": 1}, "material": "Cu"}
        anchor = body(x=1, z=2)
        with self.assertRaises(GeometryToleranceError) as caught:
            prepare_geometry(structure(outer, circle, anchor)["root"], SymmetryMode.FULL)
        self.assertEqual(caught.exception.code, "feature_below_resolution")
        self.assertEqual(caught.exception.context["reason"], "material_boundary_conflict")

    def test_polygon_holes_and_separate_islands_keep_their_gaps(self):
        outer = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]
        for loops in (
            [outer, [[.0004, 1, 0], [1, 1, 0], [1, 2, 0], [.0004, 2, 0]]],
            [outer, [[4.0004, 1, 0], [5, 1, 0], [5, 2, 0], [4.0004, 2, 0]]],
        ):
            model = structure({"geometry": {"type": "PolygonGeometry", "thk": 1,
                                             "polys": loops}, "material": "Si"})
            before = copy.deepcopy(model)
            with self.subTest(loops=loops), self.assertRaises(GeometryToleranceError) as caught:
                prepare_geometry(model["root"], SymmetryMode.FULL)
            self.assertEqual(caught.exception.code, "feature_below_resolution")
            self.assertEqual(caught.exception.context["reason"], "polygon_loop_gap")
            self.assertEqual(model, before)

    def test_invalid_source_polygon_loops_are_reported_distinctly(self):
        model = structure({"geometry": {"type": "PolygonGeometry", "thk": 1, "polys": [
            [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
            [[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]],
        ]}, "material": "Si"})
        with self.assertRaises(GeometryToleranceError) as caught:
            prepare_geometry(model["root"], SymmetryMode.FULL)
        self.assertEqual(caught.exception.code, "invalid_geometry")
        self.assertEqual(caught.exception.context["reason"], "polygon_loop_contact")

    def test_normalized_polygon_gap_must_still_exceed_tolerance(self):
        # These loops start .0016/sqrt(2) apart. Source cuts shift the outer
        # diagonal inward, leaving a .0008/sqrt(2) material rim.
        polygon = {"geometry": {"type": "PolygonGeometry", "thk": 1, "polys": [
            [[0, 0, 0], [4, 0, 0], [0, 4, 0]],
            [[1, 2.9984, 0], [1, 2, 0], [.5, 2, 0]],
        ]}, "material": "Si"}
        anchor = body()
        anchor["geometry"].update(bottom_left=[3.9992, 3.9992, 0], top_right=[5, 5, 0])
        with self.assertRaises(GeometryToleranceError) as caught:
            prepare_geometry(structure(polygon, anchor)["root"], SymmetryMode.FULL)
        self.assertEqual(caught.exception.code, "feature_below_resolution")
        self.assertEqual(caught.exception.context["reason"], "polygon_loop_gap")
        self.assertLess(caught.exception.context["measured_distance"], .001)

    def test_near_integer_subdivision_has_stable_xy_and_z_counts(self):
        config = settings()
        config["globalElementSize"] = .1
        mesh = build_mesh(structure(body(thickness=.3)), config)
        np.testing.assert_allclose(np.unique(mesh.nodes[:, 2]), [0, .1, .2, .3], rtol=0, atol=1e-16)
        cuts = sorted(_section_avg_lines([0, .3], .1, (-1, 1)))
        np.testing.assert_allclose(cuts, [0, .1, .2, .3], rtol=0, atol=1e-16)

    def test_shape_membership_uses_the_same_euclidean_distance(self):
        # Side displacement .00075 is inside T; diagonal displacement has
        # length sqrt(2)*.0008 > T and must remain outside both shapes.
        box = {"type": "BOX", "dim": [0, 0, 1, 1]}
        polygon = {"type": "POLYGON", "dim": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}
        for right, top, expected in ((1.00075, .9, True), (1.0008, 1.0008, False),
                                     (1.004, .9, False)):
            dragger = Dragger()
            dragger.set_2D([[.5, .5], [right, .5], [right, top], [.5, top]], [[0, 1, 2, 3]])
            for face in (box, polygon):
                self.assertEqual(dragger._search_faces(face).tolist(), [expected])

    def test_polygon_retention_and_source_area_are_translation_invariant(self):
        for offset in (0.0, 1e7, 1e8):
            loop = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float) + offset
            face = {"type": "POLYGON", "dim": [loop.tolist()]}
            self.assertTrue(face_intersects_domain(
                face, ModelDomain(SymmetryMode.RIGHT_HALF, center_x=offset + .5)))
            dragger = Dragger()
            dragger.set_2D(loop, [[0, 1, 2, 3]])
            np.testing.assert_array_equal(dragger.element_2D_volume, [1.0])

    def test_export_roundtrips_float_coordinates_and_reports_policy(self):
        model = structure(body(x=1e9 + .004))
        mesh = build_mesh(model, settings())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.cdb"
            metadata = write_cdb_text(path, mesh=mesh)
            text = path.read_text()
            section = text.split("*NODES,index,x,y,z\n", 1)[1].split("\n\n", 1)[0]
            restored = np.array([[float(value) for value in row.split(",")[1:]]
                                 for row in section.splitlines()])
        np.testing.assert_array_equal(restored, mesh.nodes)
        self.assertEqual(metadata["geometryTolerance"], .001)
        self.assertEqual(metadata["unitSystem"], "um")

    def test_numeric_resolution_is_checked_before_meshing(self):
        with self.assertRaises(GeometryToleranceError) as caught:
            build_mesh(structure(body(x=1e14)), settings())
        self.assertEqual(caught.exception.code, "numeric_resolution_exceeded")
