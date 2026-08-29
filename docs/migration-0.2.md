# Migrating to mesher 0.2

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
| `process_flow_mesher.build_mesh_from_structure` | `mesher.process_flow.build_mesh_from_structure` |
| `process_flow_mesher.visualization.MeshViewer` | `mesher.mesh3d.visualization.MeshViewer` |
| `process_flow_mesher.exporters.cdb.write_cdb_text` | `mesher.process_flow.write_cdb_text` |

## API changes

- `Dragger` and `set_2D` are replaced by `ExtrusionLayer` and `extrude_mesh`.
- `Mesh3D.element_comps` is now `element_component_ids`.
- `Mesh3D.comps` is now `component_ids_by_name`.
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
