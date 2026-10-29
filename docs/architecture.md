# Package architecture

The dependency direction is one-way:

```text
Python callers / mesher.process_flow
                  │
                  ▼
           mesher.mesh25d ──────────► mesher.contracts
             │    │    │
             │    │    └───────────► mesher.mesh3d
             │    └────────────────► mesher.mesh2d
             └─────────────────────► mesher.meshcontrol
                                        │
                                        └──► mesh2d / mesh3d control data types
```

See [Mesh25D design](mesh25d-design.md) for the Mermaid dependency and data-flow
diagrams, design decisions, and XY refinement behavior.

`mesh2d` and `mesh3d` own reusable models, control data interfaces, and
dimension-specific algorithms. They never import `mesh25d`, `meshcontrol`, or
`process_flow`. Control methods decide their own mesher-specific constraints;
their shared factory resolves Standard V1 references without loading algorithms.

Within `mesh2d`, `editor/` exposes the stateless `EditorMesh2D` API and owns its
element cleanup rules. `utils/` holds NumPy-only node-equivalence and topology
algorithms shared by the editor and circular features. Shared utilities depend
on the mesh model, without importing feature packages. The editor remains
available from both `mesher.mesh2d` and `mesher.mesh2d.editor`; circular modules
retain their existing private helper imports. Editor and utility tests mirror
these packages under `tests/unit/mesh2d/`, and editor performance checks live in
`benchmarks/mesh2d/`.

The single-layer implementation lives in `mesh2d.adapter`. Its public
`extend_quad_layer()` entry point performs geometry planning and mesh construction
in one call; callers do not coordinate those internal stages. The optional
`plan_quad_sections()` API provides a read-only geometry preview: it calculates
edge expansion and the new front, reserves corner sections, then allocates the
remaining LINE ranges using complete finite pattern chords. Public `Section`,
`Seam`, and `QuadSectionPlan` objects use world XY coordinates and preserve the
source node chains. Sections need not themselves be convex. Left/right refer
to outline traversal, and seam points run from source to front.

`extend_quad_layer()` uses the same internal section planner. Section builders
produce and isolate-test unit plans for candidate seam node counts; a cyclic
solver chooses compatible states around the entire layer. Neighboring sections
share seam node IDs. Unit builders generate strictly convex Quad4 cells, and
final validation checks the layer before committing it to the existing mesh.
Diagnostics connect sections to units and units to global element ranges.
`mesh2d.adapter.visualization.plot_quad_layer()` reads the completed result
to show sections, units, and the final mesh, without invoking planning or meshing.
The [visual example](../examples/extend_quad_layer.py) only prepares inputs and
calls these package APIs; reusable implementation does not live in `examples/`.
The cyclic objective first sums the constructed sections' mean squared
normalized outer edge-size errors, then minimizes the worst `max(edge aspect
ratio, 1 / corner scaled Jacobian)` over new cells. Quality only breaks ties
among the existing candidates; it never requests more LINE constructions or
imposes an additional rejection threshold. Unit-local construction can still
produce layered transitions or thin cells under substantial coarsening.
Source geometry and requested expansion remain fixed, and failures are atomic.
There is no thin-collar fallback. Patterns with interior endpoints, T/X
junctions, or closed internal loops are deliberately outside this version's
supported section layouts.

LINE construction samples the outer boundary first, preserving mandatory
pattern contacts and distributing nodes between anchors for at most two nearest
parity-compatible counts. Counts are ordered by size error, then fewer segments;
the production path stops at the first feasible one. The shorter source/outer
chain advances one segment per unit. Alternating from the left and right ends,
projected geometric proximity chooses a node on the longer chain, retaining
enough segments to complete every remaining unit. Ties choose the smaller node
index. Connector midpoints supply the required boundary parity and are shared
by their two owners.

Only failed unit construction tries the chosen cut's adjacent indices. A failed
central closure can move only the last committed cut by one node and rebuild
its two owners. There is no recursive repair, cut enumeration, or LINE quality
optimization. Unit trials per outer configuration and seam pair are bounded by
`8 * min(source_segments, outer_segments)`; ordinary cases need one per unit.
Byte-identical normalized builder outputs remain cached. This bound concerns
unit trials, not the cost of generating or validating every output cell.

Diagnostics identify `line_planner="outer_first"` and record all LINE unit
trials and attempted local repairs, including unsuccessful seam pairs and
outer counts. Section details include their own totals and outer-count
attempts. Extension time, quality, section/unit records, and these counters
can be saved without repeating construction. The full-layer benchmark uses
one warmup and five timed test2 extension calls; fixture creation and rendering
are outside the timer, while validation and atomic commit are included.

`mesh25d` exposes `build_mesh(geometry_structure, mesh_control, progress=...)`.
Internally, geometry extraction supplies footprint patterns, KOZ boundaries,
source provenance, priorities, and material assignments. The 2D stage in
`plan2d.py` combines patterns and XY constraints into grid/circular operations
and builds `Mesh2D`. The 3D stage in `plan3d.py` combines assignments and resolved
Z constraints into `layer_infos`, converts 2D winding, and invokes the existing
Dragger to build `Mesh3D`.
`_get_assignments` was moved verbatim into geometry extraction.

`process_flow` owns worker transport and CDB serialization. It calls `mesh25d`
without owning meshing or geometry translation. The CDB writer remains here
because it implements the repository's text contract rather than a general
ANSYS CDB serializer.

PyVista, Matplotlib, PySide6, Shapely, and `process_flow_kernel` are not imported
by the core package, shared control methods, or control interfaces. Mesh25D's
public module loads its optional runtime on a build call and raises an
actionable import error when the `process-flow` extra is missing. Input JSON
contract validation remains independent of that runtime.
