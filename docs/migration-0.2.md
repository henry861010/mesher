# Migrating to mesher 0.2

> 本文件記錄 0.2 的模型／import 遷移。後續統一幾何容差已移除 `relative`；目前 API 請參考 [容差遷移說明](tolerance-migration.md)。

Version 0.2 is intentionally breaking. It does not provide compatibility
modules for the old flat package layout or the former `process_flow_mesher`
package.

## Import changes

| Before | After |
| --- | --- |
| `mesher.circular` | `mesher.mesh2d.circular` |
| `mesher.generators` | `mesher.mesh2d.generators` |
| `mesher.quality` | `mesher.mesh2d.quality` |
| `mesher.visualization` | `mesher.mesh2d.visualization` |
| `process_flow_mesher.Mesh3D` | `mesher.Mesh3D` |
| `process_flow_mesher.build_mesh_from_structure` | `mesher.mesh25d.build_mesh` |
| `process_flow_mesher.visualization.MeshViewer` | `mesher.mesh3d.visualization.MeshViewer` |
| `process_flow_mesher.exporters.cdb.write_cdb_text` | `mesher.process_flow.write_cdb_text` |

## API changes

- `process_flow_mesher.meshing.extrusion.Dragger` moves to
  `mesher.mesh3d.extrusion.Dragger`; its `set_2D` and `build` workflow is
  preserved.
- Geometry translation is internal to `mesher.mesh25d`; callers use
  `mesher.mesh25d.build_mesh` instead of `StandardV1Translator`.
- `Mesh3D` retains the original `element_comps` and `comps` field names; the
  temporary `element_component_ids` and `component_ids_by_name` names are not
  supported.
- `Mesh3D.elements` now uses fixed-width `(m, 20)` connectivity. Constructors
  must also provide `element_types`, `types`, `element_reals`, `reals`,
  `element_sections`, `sections`, and `element_node_num`; unused connectivity
  slots repeat the final effective node id.
- `model_type` is now `symmetry`.
- `Full_Model`, `Quarter_Model`, `Half_Model_X`, and `Half_Model_Y` become
  `full`, `upper_right_quarter`, `upper_half`, and `right_half` respectively.
- `MeshViewer.show(isRandomColor=...)` is now
  `MeshViewer.show(randomize_colors=...)`.
- Worker invocation moves from `python -m process_flow_mesher.worker` to
  `python -m mesher.process_flow.worker` or `mesher-process-flow`.

`Mesh2D` now owns its input and always stores three coordinate columns. Code
that retained references to constructor inputs or expected `(n, 2)` storage
must read back `mesh.nodes` and use `mesh.nodes[:, :2]` for planar work.

All Mesh2D Z values must now be exactly zero. Inputs on a nonzero constant plane
or with tiny Z noise are rejected, as are nonfinite Z values. If projection is
intended, construct `Mesh2D(xyz[:, :2], elements)` explicitly. The `nodes` setter
and `replace_data` enforce the same rule. In-place NumPy editing remains
supported, but any invalid Z must be repaired before calling a mesh operation,
including a no-op; failed validation does not change the mesh.

Remove the `planar` argument and `planarity` property from uses of
`GeometryTolerance`; only the XY `absolute` and `relative` tolerances remain.
`Dragger.set_2D` also requires zero Z when a third source column is provided.
3D layer heights, Mesh3D coordinates, and exported 3D geometry are unaffected.

## Mesh25D reorganization

Callers of the earlier process-flow Python API must update these imports:

| Before | After |
| --- | --- |
| `mesher.process_flow.build_mesh_from_structure` | `mesher.mesh25d.build_mesh` |
| `mesher.process_flow.SymmetryMode` | `mesher.mesh25d.SymmetryMode` |
| `mesher.process_flow.validate_mesh_control` | `mesher.contracts.process_flow_2_5d.validate_mesh_control` |
| `mesher.process_flow.translation.parse_meshcontrol.get_control_obj` | `mesher.meshcontrol.get_control_obj` |

The former translator and pipeline modules are removed. Geometry patterns,
planar plans, and layer plans are internal to Mesh25D. The worker command and
CDB/progress contracts remain unchanged. Controls use the current document
contract with required `label` and `coordinateMode`; legacy per-coordinate
`mode`/`value` objects are not accepted. Z controls are applied; `3D_REGION`
adds XY cuts by evenly subdividing each box/polygon vertex interval into
`max(1, floor(length / elementSize))` segments. The existing grid generator
fills remaining gaps using the global planar element size. These lines span
the model; the region's Z extent does not change Z layers.
See [Mesh25D design](mesh25d-design.md).
