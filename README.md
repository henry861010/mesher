# mesher

`mesher` is a typed Python library for planar mixed Tri3/Quad4 meshes,
layered Wedge6/Hex8 extrusion, mesh quality checks, visualization, and the
process-flow geometry-to-CDB workflow.

The package is organized by responsibility:

- `mesher.mesh2d` owns planar models, generators, circular operations,
  quality checks, and 2D visualization.
- `mesher.mesh3d` owns solid models, process-flow extrusion, and 3D visualization.
- `mesher.meshcontrol` resolves Standard V1 geometry references and owns each
  control method's mesher-specific behavior.
- `mesher.mesh25d` translates Standard V1 geometry and coordinates the complete
  workflow through `plan2d.py` (2D meshing) and `plan3d.py` (3D meshing).
- `mesher.process_flow` owns the worker CLI, progress transport, and CDB export.

See [Architecture](docs/architecture.md) for the dependency rules,
[Mesh25D design](docs/mesh25d-design.md) for the architecture and data-flow diagrams,
and [Migrating to 0.2](docs/migration-0.2.md) for breaking import and field changes.

## Installation

Install the NumPy-only core:

```bash
python -m pip install .
```

Install optional capabilities independently:

```bash
python -m pip install '.[visualization]'
python -m pip install '.[gui]'
python -m pip install '.[process-flow]'
```

The basic `import mesher` path does not import PyVista, Matplotlib, PySide6,
or `process_flow_kernel`.

## Mesh models

The package root exports only the central mesh models and the planar topology enum:

```python
from mesher import ElementType2D, Mesh2D, Mesh3D
```

`Mesh2D` owns contiguous `float64` nodes with shape `(n, 3)` and `int32`
connectivity with shape `(m, 4)`. XY input is accepted and normalized with a
zero Z coordinate. Tri3 rows use `[n0, n1, n2, n2]`; `element_types` exposes
the inferred Tri3/Quad4 topology.

`Mesh3D` owns `(n, 3)` nodes and fixed-width `(m, 20)` ANSYS connectivity.
`element_node_num` identifies the effective leading slots in each row; every
remaining slot repeats the last effective node id. Per-element `int32` arrays
store component, element-type, real-constant, and section ids, while `comps`,
`types`, `reals`, and `sections` hold their definition tables. The process-flow
extruder currently emits SOLID185 connectivity with eight effective slots.

## 2D generation and circular features

```python
from mesher.mesh2d.circular import extend_circular_mesh, imprint_circle
from mesher.mesh2d.generators import generate_rectilinear_mesh

mesh = generate_rectilinear_mesh(
    target_edge_size=1.0,
    x_coordinates=[-10.0, 0.0, 10.0],
    y_coordinates=[-10.0, 0.0, 10.0],
)

imprint_circle(
    mesh,
    center=(0.0, 0.0),
    radius=5.0,
    band_width=1.0,
    target_edge_size=0.5,
)

extend_circular_mesh(
    mesh,
    element_size=1.0,
    center_x=0.0,
    center_y=0.0,
    inner_radius=5.0,
    outer_radius=10.0,
)
```

Both circular operations are transactional and in place: they return the same
`Mesh2D` after success and leave it unchanged on failure.

For the second-generation `circular.imprint_v2` implementation, see the
[current inspection cases and stage diagnostics](examples/failures/imprint_v2_holes/README.md).
It processes a conservative circle-band patch with shared-node and merge closure,
preserves remote geometry, and commits the stitched result atomically.
As a final step, local quads with an interior angle of at least `flat_quad_angle`
(default: 170°, valid range: `0 < flat_quad_angle <= 180`) are split into two
triangles using the diagonal that maximizes their smallest angle; the original
nodes and cell boundary are preserved.
The case collection includes a reproducible 100,000-element performance comparison.

## Process-flow 3D extrusion

```python
from mesher.mesh3d.extrusion import Dragger

dragger = Dragger()
dragger.set_2D(mesh.nodes, mesh.elements)
solid = dragger.build(layer_infos)
```

`layer_infos` is an ordered sequence of dictionaries with `z`, `element_size`,
and `assignments`. The final entry supplies the terminal Z boundary. The
Mesh25D workflow prepares this structure internally. Component id zero
represents an empty planar element, and adjacent layers reuse their shared
top/bottom nodes.

## Mesh25D pipeline

Install the `process-flow` extra, then build from a Standard V1 structure:

```python
from mesher.mesh25d import SymmetryMode, build_mesh
from mesher.contracts.process_flow_2_5d import validate_mesh_control

mesh_control = {
    "schemaVersion": "1.0.0",
    "unitSystem": "um",
    "mesher": "process_flow_2_5d",
    "globalElementSize": 100.0,
    "symmetry": SymmetryMode.UPPER_RIGHT_QUARTER,
    "controls": [],
}
validate_mesh_control(mesh_control)

mesh = build_mesh(
    structure,
    mesh_control,
    progress=optional_event_callback,
)
```

Supported symmetry values are `full`, `upper_half`, `right_half`, and
`upper_right_quarter`.

For bumps, circuits, and vias, `koz` is an inward XY keep-out distance. Both
the original footprint and the remaining eligible region contribute 2D mesh
boundaries; 3D metal allocation uses that same eligible region. Box, cylinder,
and polygon footprints are supported, including polygon holes and disconnected
regions. Polygon insets preserve straight corners, and their vertices contribute
X/Y grid lines; slanted edges retain the existing stair-step approximation.
Narrow circular KOZ boundaries automatically reduce the XY element size to
resolve both rings, which can increase mesh size without changing Z layer sizes.
If the inset removes the entire region, the feature allocates no metal while
its original footprint remains in the mesh pattern.

Run the worker through its console script or module path:

```bash
mesher-process-flow geometry.json mesh_control.json output.cdb
python -m mesher.process_flow.worker geometry.json mesh_control.json output.cdb
```

Success writes JSON metadata as the final stdout line. Progress events retain
the `PROCESS_FLOW_PROGRESS ` stderr prefix. `Z_SECTION_AVG`, `Z_SECTION_TOP`,
`Z_SECTION_BOT`, `Z_SECTION_CENTER`, and `Z_POINT` are applied through shared
control objects. Overlapping Z constraints use the smallest active size.
`3D_REGION` adds XY grid lines from its box or polygon region. Each interval
between consecutive vertex coordinates is split evenly into
`max(1, floor(length / elementSize))` segments. Controls contribute their cuts
independently; their union is clipped to the model bounds and symmetry domain.
The grid then fills remaining gaps using the global planar element size.
Lines span the model, so refinement also affects points outside the region
along the same X/Y projections. The region's Z extent adds no Z constraints.

## Optional tools

Launch the 2D quality GUI with:

```bash
mesher-quality-gui
```

Import visualization only after installing the corresponding extra:

```python
from mesher.mesh2d.visualization import view_mesh
from mesher.mesh3d.visualization import MeshViewer
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m build
```
