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
connectivity with shape `(m, 4)`. XY input is padded with a zero Z coordinate;
XYZ input must have exactly zero Z at every node, including unused nodes.
Nonzero Z (even tiny noise), NaN, and infinity raise `ValueError` with the first
invalid node's index and Z value. No Z tolerance or automatic projection is
applied. To explicitly project source coordinates, pass `xyz[:, :2]`.
Tri3 rows use `[n0, n1, n2, n2]`; `element_types` exposes the inferred topology.

`mesh.nodes` remains writable. Direct edits can temporarily make a mesh invalid;
public mesh operations validate Z before work, including no-op operations.
Getters remain available for inspection and repair. Constructors, the `nodes`
setter, and `replace_data` validate new coordinates before assigning them.
All 2D geometry uses XY and generated nodes have Z=0. `GeometryTolerance` has
only `absolute` and `relative` settings; `planar` and `planarity` are removed.
The 3D extruder validates its 2D source and generates Z from the layer plan.

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
transitive XY distance groups and retains the lowest original node ID's
XY coordinate with Z=0. It converts adjacent-vertex Quad4 collapses to padded Tri3,
removes unusable or zero-area cells, and removes unreferenced nodes; winding,
concave cells, and duplicate elements are retained. Tolerance must be finite and
non-negative; zero merges only identical XY coordinates. Referenced XY values
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
outline = [[0, 3], [3, 2], [2, 1], [1, 0]]  # clockwise exterior
patterns = [[[0, 3.4], [4, 3.4]]]             # finite parallel guide
result = extend_quad_layer(
    mesh, outline, element_size=0.8,
    pattern_lines=patterns, expand_ratio=1.0,
)
assert result.mesh is mesh
# result.outline can be passed into a later, separate single-layer call.
```

The adapter requires a planar Quad4 input and adds strictly convex, CCW Quad4
cells. Existing coordinates, node IDs, elements, and source segments stay
unchanged. The complete layer is validated before an atomic commit; errors
leave the input mesh unchanged.

Each logical outline edge is an ordered node chain near global X or Y;
consecutive edges are perpendicular. Exterior loops are clockwise and holes
counterclockwise. Growth is to the left, with XY normal `(-dy, dx)`. A finite
parallel pattern with overlapping edge projection and nearest positive distance
`d` sets the whole edge's expansion when `d <= element_size * expand_ratio`;
otherwise the expansion is `element_size`. Coincident guides do not stop growth.
`axis_tolerance` (default `1e-3`) limits transverse deviation relative to each
logical edge's length. The new front is axis aligned; old coordinates never move.

`extend_quad_layer()` performs both levels internally in one call:

1. **Sections describe geometry.** Determine `OUTER`, `INNER`, `INNER_LEFT`,
   `INNER_RIGHT`, and `INNER_BOTH` corners first. Their reserved ranges and
   complete pattern chords delimit the remaining `LINE` sections. A long LINE
   stays intact; source-edge count and target size do not split it at this stage.
2. **Units build each section.** Coordinate left/right seam node counts around
   the entire loop, then subdivide each section into feasible `build_unit_*`
   calls. Ordinary seams use 2 or 3 nodes, meaning 1 or 2 segments. Mandatory
   seam points are preserved, and neighboring sections share the same node IDs.

For a `LINE`, the outer edge is sampled **before pairing**: endpoints and finite
pattern intersections stay fixed, and additional nodes fill the intervals as
close to `element_size` as parity permits. At most two nearest legal total
segment counts are considered, ordered by outer size error, then fewer segments.
The first feasible count is accepted. From alternating ends, each segment on
the shorter chain is paired with the nearest allowed node on the longer chain
along the main axis, reserving enough nodes for the remaining units. Internal
connectors contain one or two segments to keep each unit's boundary count even.

A valid unit is accepted immediately. Only an invalid unit tries the adjacent
`-1` and `+1` cut indices. If the center cannot close, the last committed cut
may move by one node, rebuilding only its two neighboring units. There is no
recursive backtracking or global LINE quality search. Exhausting these repairs
and the other legal outer count reports a layout error; this does not prove
that every possible quadrangulation is infeasible.

`plan_quad_sections()` is an optional read-only preview of the first level,
importable from `mesher.mesh2d.adapter`. It is not a required step before
`extend_quad_layer()`. Its `QuadSectionPlan` contains `sections`, `seams`, `front`, and `edge_expansion`.
Sections retain their source node chains, outer boundary, and left/right seam
IDs; seam coordinates run from source to front. This preview verifies geometric
partitioning; full unit feasibility is determined during `extend_quad_layer()`.
`result.diagnostics` records section details, seam states, units and their global
element ranges, construction methods, quality, expansion distances, and added
node/element counts. The construction path is `sections`.

Finite horizontal/vertical patterns must coincide with a source/front boundary
or form complete chords from an existing source node to the corresponding
front. Parallel guides may partially overlap the source edge's projection.
Overlapping or repeated guides are combined for geometry classification.
Patterns crossing frozen source segments or existing cell interiors are
rejected. Internal endpoints, interior T/X junctions, closed pattern loops,
and chords to an unrelated front are unsupported in this version and report
`unsupported_pattern_layout`. Finite guides are never silently extended.

Among the constructed section candidates, the seam solver minimizes the sum of
the sections' mean squared normalized
outer-edge size errors: each section averages `(edge_length / element_size - 1)²`.
For equal size error, it minimizes the worst `max(edge_aspect_ratio,
1 / corner_scaled_jacobian)` across new cells. Edge aspect ratio is a cell's
longest edge divided by its shortest; the corner scaled Jacobian is the
normalized cross product of consecutive edge vectors. A score closer to 1 is
better. This tie-break never triggers additional LINE trials. There is no new
aspect-ratio or scaled-Jacobian rejection threshold.
Large coarsening can therefore retain thin cells or multiple transition rows
inside a unit, while the separate whole-layer thin collar has been removed.

Input cells must be nondegenerate convex Quad4
cells (either orientation), and all nodes must have Z=0. Corner
reservations that overlap, a short/coarse source edge, incompatible seam counts,
collisions, or topology changes can make a layer infeasible. `QuadLayerError`
and `UnmeshableDirectQuadLayout` include the failure context. Expansion is fixed:
there is no thin-collar retry or automatic shrinking on failure.

For each sampled outer configuration and seam pair, LINE uses at most
`8 * min(source_segments, outer_segments)` unit trials; ordinary cases build
each unit once. This bounds trial construction, while runtime still includes
the generated cells and geometric validation. A long LINE remains one geometric
section. `diagnostics.elapsed_seconds` measures the extension call, including
validation, but excludes the separate preview call and rendering.
`line_planner` is `outer_first`; `line_unit_trial_count` and
`line_local_repair_count` include all attempted seam pairs and outer counts,
including rejected configurations. LINE section details expose
`unit_trial_count`, `local_repair_count`, and `outer_count_attempts`.
Repair counts measure attempted alternative cut positions, including failures.
The [LINE benchmark](benchmarks/mesh2d/benchmark_adapter_line.py) reports unit
trial counts, runtime, memory, and final quality for configurable source counts
and seam states. Use `--timeout-seconds 30` to bound exploratory measurements.
The [full-layer benchmark](benchmarks/mesh2d/benchmark_adapter_layer.py) repeats
the test2 L-shaped case with source spacing 5 and target size 50. It excludes
fixture preparation and rendering, performs one warmup and five measurements,
and requires the median full extension call to be below one second by default:
`venv/bin/python benchmarks/mesh2d/benchmark_adapter_layer.py`.

The implementation lives in `src/mesher/mesh2d/adapter/`; files in `examples/`
only demonstrate its public API. Each example calls `extend_quad_layer()` once,
then `mesher.mesh2d.adapter.visualization.plot_quad_layer()` draws the returned
section, unit, and element diagnostics without planning or generating another mesh.

Run [the visual example](examples/extend_quad_layer.py) with
`python examples/extend_quad_layer.py`. It opens a Matplotlib window and saves
`examples/output/extend_quad_layer.png`, comparing **sections, selected units,
and the final mesh** for a cross, finite pattern chords, and a hole. Use
`--case pattern` to focus on guides or `--no-show --plot output.png` to save
without opening a window. [The L-shaped regression example](examples/test2.py)
uses source spacing 5 and expansion 50: it shows 6 LINE, 5 OUTER, and 1 INNER
section, with `(0, 0) → (0, 100)` retained as a single LINE. Each example prints
elapsed time and the final quality summary, also shown in the mesh panel.
Use `--diagnostics output.json` to retain the section/unit records, chosen seam
counts, global element ranges, quality, and timing from the same run as the PNG.
Run `test2.py` with `--view-mesh` to also open the PyVista viewer. Matplotlib and PyVista are provided
by the optional `visualization` dependency.

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
