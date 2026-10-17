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
diagrams, design decisions, and current XY refinement boundary.

`mesh2d` and `mesh3d` own reusable models, control data interfaces, and
dimension-specific algorithms. They never import `mesh25d`, `meshcontrol`, or
`process_flow`. Control methods decide their own mesher-specific constraints;
their shared factory resolves Standard V1 references without loading algorithms.

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
