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

## 2D mesh editing

`EditorMesh2D` is a stateless NumPy-only API: every call receives a `Mesh2D`.

```python
from mesher.mesh2d import EditorMesh2D

# Select elements whose vertices are all inside this inclusive XY rectangle.
indices = EditorMesh2D.search_element(mesh, 0.0, 0.0, 1.0, 1.0)
EditorMesh2D.delete_element(mesh, indices)  # In place; keeps all node IDs.

combined = EditorMesh2D.merge(mesh, other_mesh)  # New mesh; sources unchanged.
EditorMesh2D.equivalence(combined, tol=0.001)  # Weld explicitly, in place.
outlines = EditorMesh2D.outline(combined)  # Ordered int64 node-ID arrays.
outlines_by_edge = EditorMesh2D.outlines_by_edge(combined)  # Loops -> edges -> node IDs.
```

Deletion and equivalence return the input mesh. Merge preserves source order,
owns independent arrays, and does not weld coincident nodes. Equivalence uses
transitive XY distance groups and retains the lowest original node ID's complete
XYZ coordinate. It converts adjacent-vertex Quad4 collapses to padded Tri3,
removes unusable or zero-area cells, and removes unreferenced nodes; winding,
concave cells, and duplicate elements are retained. Tolerance must be finite and
non-negative; zero merges only identical XY coordinates. Referenced XYZ values
must be finite. Deletion renumbers element rows, and equivalence can renumber
both nodes and elements, so obtain fresh selections after editing.

Outlines include exterior, hole, and disconnected-component loops with implicit
closure (the first node is not repeated). Winding is preserved; loops start at
their lowest directed edge and are sorted deterministically. Invalid topology,
non-manifold edges, or inconsistent shared-edge winding raise `ValueError`.

`outlines_by_edge(mesh, angle_tolerance=1e-8)` groups consecutive collinear XY
segments into logical edges, retaining every intermediate node and including
both endpoints on each edge. The tolerance is an angle in radians. It orients
exteriors clockwise and holes counterclockwise so existing mesh stays on the
right, including for clockwise input elements. Each loop starts at its lowest
directed corner edge. For near-axis Quad4 meshes, pass one loop directly to
`extend_quad_layer(mesh, outlines_by_edge[0], element_size)`.

Operations use batched masks, spatial-cell grouping, and index lookups without a
global node-pair distance matrix or cross-call caches. Run the 25k/50k/100k
performance checks, including 100k disconnected loops, with:

```bash
PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_editor.py
```

Each case runs in an isolated process and reports the median of three timed
calls after warm-up, plus peak process RSS (including fixtures and results).
The default time budget is 10 seconds per case.

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

## Single-layer direct Quad4 extension

```python
import numpy as np
from mesher import Mesh2D
from mesher.mesh2d.adapter import extend_quad_layer

mesh = Mesh2D(
    [[0, 0], [4, 0], [4, 3], [0, 3]],
    np.array([[0, 1, 2, 3]], dtype=np.int32),
)
result = extend_quad_layer(
    mesh,
    outline=[[0, 3], [3, 2], [2, 1], [1, 0]],  # clockwise exterior
    element_size=0.8,
    pattern_lines=[[[4.2, 3.4], [4.6, 3.4]], [[4.4, 3.2], [4.4, 3.6]]],
    expand_ratio=1.0,
)
assert result.mesh is mesh
# result.outline can be passed into a later, separate single-layer call.
```

This adapter requires a planar Quad4 input and produces strictly convex, CCW
Quad4 cells throughout construction, including temporary meshes. It does not
triangulate or convert triangles. It preserves all existing node values, node
IDs, elements, and source segments, then commits the entire layer atomically.

Each logical outline edge is an ordered node chain near global X or Y;
consecutive edges are perpendicular. Exterior loops are clockwise and holes
counterclockwise. Growth is to the left of each directed edge. A finite parallel
pattern with overlapping edge projection and nearest positive distance `d`
sets that whole edge's expansion when `d <= element_size * expand_ratio`;
otherwise the expansion is `element_size`. Coincident guides do not stop growth.
`axis_tolerance` (default `1e-3`) limits transverse deviation relative to each
logical edge's length. The new front is axis aligned; old coordinates never move.

Finite horizontal/vertical patterns support internal endpoints, T/X junctions,
overlaps, and closed loops. Only portions inside the resulting mesh must be
covered by mesh edges. A pattern crossing a fixed source segment or an existing
element is rejected unless it already follows existing edges. Closed pattern
loops remain filled. Auxiliary seams may extend beyond finite pattern endpoints.

Construction uses convex strips/corner blocks, or a thin source collar plus
locally partitioned rectangles inside the same layer. Shared interfaces are
noded once and their subdivision parity is coordinated before direct unit or
inner-ring quad filling. `result.diagnostics` records the path, section counts,
parity solver, builders, expansion distances, and added node/element counts.
Section names annotate blocks; a concave corner can use two incident strips.

This is a conservative direct-quad algorithm, not universal quadrangulation.
`element_size` is a target; topology, frozen boundaries, and finite constraints
take priority over size and aspect ratio. Input cells must be nondegenerate
convex Quad4 cells (either orientation), and all nodes must share one constant Z.
Topology changes, collisions, or incompatible old boundaries raise
`QuadLayerError`. If no certified source collar or valid direct quad fill can
be constructed, including almost-tangent near-axis configurations, it raises
`UnmeshableDirectQuadLayout`. Both exceptions leave the original arrays unchanged.

Run [the visual example](examples/extend_quad_layer.py) with
`python examples/extend_quad_layer.py`. It opens a Matplotlib window and saves
`examples/output/extend_quad_layer.png`, showing input, output, and enlarged
details for a cross, a cross with finite patterns, and a hole. Use `--case pattern`
to focus on finite guides, `--element-size 0.25` to change the target size, or
`--no-show --plot output.png` to save without opening a window. Matplotlib is
provided by the optional `visualization` dependency.

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
