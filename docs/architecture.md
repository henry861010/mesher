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

`Mesh2D` stores `(n, 3)` arrays for interface compatibility, but its valid
geometry is strictly XY with Z=0. The NumPy-only model owns the shared exact-Z
validator; model writes and public mesh operations reject nonzero or nonfinite
Z before mutation, including unused nodes and early no-ops. Writable arrays
remain supported, so the condition is checked at operation boundaries rather
than guaranteed between arbitrary in-place edits. Geometry helpers work on XY;
three-column outputs append zeros. `Dragger.set_2D` reuses the model validator
for raw 2D sources; actual Mesh3D heights come solely from layer definitions.

Within `mesh2d`, `editor/` exposes the stateless `EditorMesh2D` API and owns its
element cleanup rules. `utils/` holds NumPy-only node-equivalence and topology
algorithms shared by the editor and circular features. Shared utilities depend
on the mesh model, without importing feature packages. The editor remains
available from both `mesher.mesh2d` and `mesher.mesh2d.editor`; circular modules
retain their existing private helper imports. Editor and utility tests mirror
these packages under `tests/unit/mesh2d/`, and editor performance checks live in
`benchmarks/mesh2d/`.

Within `mesh3d`, `editor/` exposes the stateless NumPy-only `EditorMesh3D` API
from both `mesher.mesh3d` and `mesher.mesh3d.editor`. `set_boundary` selects the
bottom center cross and symmetry faces, then replaces the `fix_x`, `fix_y`,
and `fix_z` node sets only after validation. It depends on the mesh model and
shared absolute tolerance policy, without importing orchestration or solvers.

The single-layer implementation lives in `mesh2d.adapter`. Its files are grouped
by responsibility, with explicit public exports in package `__init__.py` files:

```text
mesh2d/adapter/
├── __init__.py          # Public layer APIs and result/error types
├── visualization.py     # Optional plotting and diagnostic export
├── unit/                # Reusable low-level Quad4 templates
└── layer/
    ├── __init__.py      # Same public layer API
    ├── _mesher.py       # Pipeline orchestration and atomic commit
    ├── _types.py        # Results, section plans, and exceptions
    ├── _planning/
    │   ├── source.py    # Input validation and coordinate normalization
    │   ├── front.py     # Automatic and explicit expansion fronts
    │   └── sections.py  # Geometric sections and seams
    ├── _construction/
    │   ├── units.py     # Unit plans, caching, and local assembly
    │   ├── line.py      # LINE section construction
    │   ├── corners.py   # Corner section construction
    │   ├── patch.py     # Four-sided patch construction
    │   └── sections.py  # Cyclic section solver and global assembly
    ├── _repair/
    │   ├── slots.py     # Slot/corner detection and staged source repair
    │   ├── patch.py     # Repair boundaries, patterns, and face filling
    │   └── adaptive.py  # Inward coarsening and core filling
    ├── _geometry.py     # Shared geometric predicates and spatial index
    ├── _old_mesh.py     # Lazy old-edge inventory and targeted owner queries
    ├── _patterns.py     # Shared axis-pattern operations
    ├── _registry.py     # Node identity and point lookup
    ├── _limits.py       # Element budgets and count-only preflight
    └── _validation.py   # Patch topology and geometry validation
```

`_mesher` coordinates planning, repair, construction, and final validation.
Planning depends on shared geometry and types, without importing repair or
construction. Repair may reuse construction primitives; construction never
imports repair or the pipeline. Unit templates depend on the mesh model and
other unit templates, without importing the layer implementation. Shared
geometry, patterns, types, registries, limits, and validators remain directly
under `layer/` because multiple stages use them.

The adapter also exports `generate_full_adaptive_mesh` and
`FullAdaptiveMeshResult`. Its private `_full` orchestration composes the layer
and rectilinear generators; `_full_geometry` classifies all original boundaries
into an alternating exterior/hole containment tree. Every boundary receives the
requested layers. The remaining domains are the global exterior and each hole
core, excluding their immediate focus children. They are meshed independently.
Global meshes determine final interface nodes before the last adaptive layers;
the fixed-boundary layer mode preserves that complete interface. Explicit
node mappings join those interfaces without renumbering source data, and final
topology/geometry validation precedes one caller commit. Errors are atomic
across the complete operation. This orchestration remains NumPy-only and does
not import Mesh25D or process-flow modules.
Both adaptive APIs validate `max_transition_count=20`, `min_size_ratio=0.5`, and `max_size_ratio=1.25`
through the shared `_line_policy` input contract before geometry preparation.
The full API forwards them only to ordinary intermediate layers; the last
fixed-interface layer remains exempt, including a one-layer full mesh. Result
diagnostics record the settings and `line_policy_active`.

Use `mesher.mesh2d.adapter` (or `mesher.mesh2d.adapter.layer`) for the layer API,
`mesher.mesh2d.adapter.unit` for unit builders, and
`mesher.mesh2d.adapter.visualization` for plotting and diagnostic export.
Modules and packages prefixed with `_` are implementation details; their paths
may change. Importing the adapter does not load optional plotting libraries.

The historical `old/` implementation and empty `comp/` placeholders are preserved
in [the adapter archive](../archive/mesh2d_adapter/README.md), outside `src/` and
the installed package. They are reference material, not supported runtime code.
This follows Python's [package/subpackage model](https://docs.python.org/3/tutorial/modules.html#packages)
and the [PyPA src-layout separation](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/);
the responsibility groups above are specific to this project.

Public layer inputs are flat integer node-ID loops: `outline_i` and an optional
`outline_o` target referencing the same mesh. Target IDs and exact coordinates
are retained, with additional front nodes allowed. Target nodes may already be
used by existing elements; shared edges must remain conforming and manifold.
Source normalization preserves every frozen node, segment and input direction,
and groups cyclic signed-cardinal runs into logical edges. Callers guarantee
complete, attachable source boundary edges and existing mesh on the right:
clockwise exteriors, counterclockwise holes. Both planning and construction
grow to the left in the supplied winding. The geometric preview assumes valid
source edges. Construction verifies their ownership before committing.
Staged repairs keep their semantic chains internally;
complete public result and prepared-source outlines are flat loops. The editor's
`outlines()` returns oriented loops without performing logical-edge grouping.

Old-edge geometry uses a lazy, sorted unique NumPy edge-ID inventory per
prepared mesh. The same sort identifies the single-owner edges for source
and target attachment checks. New layer and repair cells retain local
directed-incidence validation. Repair outlines combine surviving source edges
with boundaries extracted from actual new repair cells; they do not re-extract
the full mesh boundary. Original and repaired meshes have separate inventories.

The public `generate_layer_adaptive_mesh()` entry point performs geometry planning and mesh construction
in one call; callers do not coordinate those internal stages. The optional
`plan_quad_sections()` API provides a read-only geometry preview: it calculates
edge expansion and the new front, reserves corner sections, then allocates the
remaining LINE ranges using complete finite pattern chords. Public `Section`,
`Seam`, and `QuadSectionPlan` objects use world XY coordinates and preserve the
source node chains. Sections need not themselves be convex. Left/right refer
to outline traversal, and seam points run from source to front.

Both single-layer APIs expose `fixed_outer_boundary=False`. The flag reaches
planning before any sections are reserved. Fixed targets preserve the complete
node-ID cycle and directed segments, disable LINE/corner outer sampling and LINE
size/transition policies, and retain the existing transaction and budget checks.
`_planning/fixed_corners.py` maps free OUTER contacts to existing target nodes
without an angle cap. It excludes the outer corner itself and searches candidates
nearest first, with canonical-order ties and iterative backtracking for crossing
or overlapping assignments. INNER reservations and repaired shared seams stay
fixed. Pattern contacts must already be supplied target nodes.

All three adapter entry points accept `small_element_size=None` (positive world
length when enabled) and `dense_mapping_angle=30.0` (finite degrees in `[0, 90)`).
`_planning/dense.py` finds maximal consecutive source-edge runs strictly below
the size threshold within each corner/pattern-delimited LINE. Relative to the
outward normal, begin endpoints select the first usable contact in `[-angle, 0]`
and end endpoints in `[0, angle]`, ordered from zero outward. Endpoints map
independently. Ordinary mode falls back to a valid normal projection; fixed mode
skips an unmatched optional cut. Mandatory seams remain fixed. Shared geometry
checks in `_planning/mapping.py` reject crossing or out-of-band connectors.
Numerical geometric tolerance is retained; `dense_region_tolerance` is removed.
Diagnostics include mapped/skipped dense endpoints and chosen corner contacts.

Fully dense LINE sections use the transition cap for resampling and retain
`element_size` as a spacing preference. Their outer spacing may fall below or
above the ratio band. Ordinary sections and sections mixing dense and
non-dense source edges use both ratios. All share the
seam-parity solver, unit builders, and element-budget preflight. Diagnostics
record their section/source/seam IDs, without new section kinds or fake pattern
IDs. The full orchestrator exports dense seam contacts as global front anchors
and saves their canonical world coordinates, keyed by source edge and chain
position. Final construction reuses those contacts even if global meshing added
a closer point. Chain positions survive repair-node renumbering between boundary
builds. Dense splitting is opt-in; the default transition cap is relaxed to 20
independently. Internal thin-cell quality under large coarsening is unchanged.

Both APIs enable slot repair by default. Original-input and pattern validation
precede closure detection. Parallel-slot closure uses the gap between the actual
planned axis-aligned front supports, including near-axis source walls; fronts
that meet within tolerance or cross trigger repair. Supported slots are filled
in a temporary Quad4 mesh, preserving all old nodes and elements; the selected
boundary is then rebuilt and grouped into monotone logical X/Y chains. Only this prepared
source may contain slanted repair-mouth segments. Axial support expansion and
pattern snapping run on the rebuilt outline before the existing section planner.
Preview exposes this temporary source through `QuadSectionPlan.prepared_source`:
its node/element arrays are read-only, its outline is immutable, and section
source IDs reference the snapshot. Preview can therefore create slot-fill
elements without changing its caller's mesh, while full expansion feasibility
is still determined by construction.

Repair has a separate boundary target: the median segment length across the
three frozen source chains, capped by the layer's `element_size`. Mouth and
shared pattern chords are sampled once at this target with mandatory nodes
and even face-boundary parity preserved. `repair_max_element_size` is accepted
by both public entry points; `None` caps the interior target at four times the
boundary target or the layer target, whichever is smaller. Explicit caps must
be finite, positive, and no smaller than the boundary target, and are clipped
to the layer target. All lengths are normalized internally and restored to
world units in public diagnostics.

Each repair face coarsens inward using fixed-boundary LINE strips. A ring
doubles its predecessor's target, up to the interior cap, and uses that target
as its inward distance. The candidate must stay within half the first
boundary-collapse distance; its validated strips share inner-ring nodes and
corner seams. A positive-progress check and a finite bound derived from the
initial geometry and minimum step cap the number of rings. Once no ring fits,
four-sided cores use structured filling when opposite counts agree, otherwise
strips along the longer opposite sides. Other supported cores retain the
finite-candidate direct builder. Repair diagnostics expose the effective
boundary and interior targets plus per-face ring counts and stopping reasons.
There is one repair phase per public call: no inward builder re-enters layer
planning or slot detection. Core failure and topology changes after repair
propagate as atomic failures. Repair targets never replace the requested
outer expansion size.

`generate_layer_adaptive_mesh()` uses the same internal section planner. Section builders
produce and isolate-test unit plans for candidate seam node counts; a cyclic
solver chooses compatible states around the entire layer. Neighboring sections
share seam node IDs. Unit builders generate strictly convex Quad4 cells, and
final validation checks the layer before committing it to the existing mesh.
Diagnostics connect sections to units and units to global element ranges.
Repairs have their own diagnostics and absolute element ranges rather than
artificial ring sections. The final validator checks repair and expansion
together against the original source; the original mesh receives one commit
only after both stages succeed. The element budget and quality summary cover
both stages. `repair_slots=False` preserves the earlier closure rejection.
`mesh2d.adapter.visualization.plot_quad_layer()` reads the completed result
to show sections, units, and the final mesh, without invoking planning or meshing.
The [visual example](../examples/extend_quad_layer.py) only prepares inputs and
calls these package APIs; reusable implementation does not live in `examples/`.
The cyclic solver minimizes the peak across all internal LINE connector angles,
then the sum of sections' mean squared tangent-to-normal displacement ratios.
Remaining ties use the sum of relative outer-spacing errors and the worst cell
quality score. To optimize these objectives across a closed ring, a first pass
finds the achievable peak. A second pass compares the other costs among paths
within that peak.
Source geometry and requested expansion remain fixed, and failures are atomic.
There is no thin-collar fallback. Patterns with interior endpoints, T/X
junctions, or closed internal loops are deliberately outside this version's
supported section layouts.

Both layer entry points require existing Quad4 elements; boundary nodes alone
are not an initialization mode. This check does not affect the empty temporary
meshes used by unit builders. Pattern input has a fixed range: its length and
each endpoint's L-infinity distance from the source bounding box cannot exceed
one million times that box's largest XY span. The input stage checks this range
before pattern arithmetic or spatial queries and reports `numeric_range_exceeded`.
Shared axis-pattern primitives express contacts and occupied/covered intervals
in absolute axial coordinates in the normalized frame. Cell clipping and edge
intersections use local coordinates, retaining canonical source points and
original pattern IDs. Old-cell constraints cover the entire finite pattern,
including other components of the input mesh. Non-pattern geometry retains the
general segment predicates and its existing tolerances.

Spatial broad-phase queries use a static balanced AABB tree with at most 32
primitives per leaf. Splits use the widest centroid axis and stable original-ID
ties, so distant components do not change a global grid resolution. Queries
retain inclusive tolerance-expanded bounds and return sorted original IDs.
Repair-boundary self-intersection checks use the same index to select pairs
before applying the existing exact segment-contact predicate.

Layer construction accepts `max_added_elements` (default one million positive
integer elements), counting repair and expansion together. Repair candidates
consume only their selected output count; the section planner uses the remaining
budget. After repair geometry and size-cap validation, a count-only preflight
checks the boundary-segment lower bound before repair-local pattern processing
or sampling. A separate preflight computes the minimum section lower bounds
across seam/outer-count candidates before outer-layer sampling or unit
construction. Candidate guards exclude proven oversized boundaries and stop
assembly when actual counts exceed the limit. The selected
ring's actual count is checked before global node mapping and before commit.
Limit failures retain `element_limit_exceeded` rather than being swallowed by
geometry fallbacks. Candidate retries do not consume a cumulative budget. The
solver does not add a count-constrained optimization dimension: if its chosen
ring is too large, construction fails atomically. This guard does not promise
bounded execution time or memory consumption. The section preview has no
element-budget parameter and makes no budget-feasibility promise.

LINE construction pairs the ordered source and outer boundaries. It tries
placing movable outer points at source projections, with a bounded density
distribution available when required anchors or segment-count constraints
prevent exact projection. Ordinary and mixed LINE outer spacing lies in
`[min_size_ratio * element_size, max_size_ratio * element_size]`. Adjacent
mandatory anchors closer than the minimum form one unsplit interval. Each outer
segment has a source assignment bounded by `max_transition_count`.

Projection is computed once per section and reused for its seam states.
A regular row can be emitted directly. For coarsening, the planner reserves
enough source segments of the required parity for the remaining units.
On irregular chains each sweep compares a bounded neighborhood of the next
target and spacing limits. If both sweep directions fail to close, up to two
anchor-compatible counts supply fronts to fit against the actual paired
source nodes. These steps use coordinates and integer counts. Quad4
construction starts with the chosen layout, batching single cells and
rebuilding only the two owners of a locally adjusted cut.

All shared cuts between LINE units have two nodes. Extra section-end nodes
belong to the first or last unit. Boundary-segment parity restricts where cuts
can be made, so a transition can consume multiple segments on either side.
Unit assembly preserves the supplied coordinates and validates the actual
Quad4 cells. Candidate ranking accounts for the worst connector angle. The
section ring coordinates the endpoint seam states across neighboring sections.

The `projection` planner records `max_alignment_angle` in degrees and
the source segment assignments per outer segment. Full-layer benchmarks include
validation and atomic commit and exclude fixture creation and rendering.

`mesh25d` exposes `build_mesh(geometry_structure, mesh_control, progress=...)`.
Internally, geometry extraction supplies footprint patterns, KOZ boundaries,
source provenance, priorities, and material assignments. The 2D stage in
`plan2d.py` combines patterns and XY constraints into grid/adapter/circular operations
and builds `Mesh2D`. The 3D stage in `plan3d.py` combines assignments and resolved
Z constraints into `layer_infos`, converts 2D winding, and invokes the existing
Dragger to build `Mesh3D`.
The planar plan preserves full-model X/Y centerlines, or the missing interior
centerline for each half model, through grid cuts, adapter patterns and circular
guides. Center coordinates are fixed anchors; final planar cells are checked
for crossing the mandatory cuts. Off-center radial extensions that would lose
a cut are replanned as imprint-only, including in half models. Quarter models
retain their existing symmetry boundaries.
Before completing the 3D stage, the pipeline calls `EditorMesh3D.set_boundary`
with its normalized symmetry and shared tolerance. A successful result always
contains `fix_x`, `fix_y`, and `fix_z`. Boundary selection errors propagate before
worker serialization, preserving any existing output file; no automatic node
insertion or boundary-rule fallback occurs. Worker output progress includes
the set records already supported by the CDB writer.
`_get_assignments` was moved verbatim into geometry extraction.

The syntax-only input contract accepts optional top-level `regionMeshing`
(`grid` by default, or experimental `adapter`) and schema `1.1.0`
`geometryTolerance` (default `0.001`). Schema `1.0.0` remains readable. `region_adapter.py` owns the optional
XY strategy: normalize interior orthogonal regions with even/odd loop nesting,
generate seeds through the public holed rectilinear API, grow each region's
intermediate layers with fivefold target sizes, and merge them for one full
adaptive completion at the computed global planar size. All source meshes are
private. Boundary containment and cross-region clearance are checked between
stages; errors retain control provenance and adapter diagnostics. There is no
automatic grid fallback. Inactive regions keep the existing grid path.

Physical/KOZ segments and circular support cuts constrain seed generation,
intermediate layers, and final completion. The full adapter's optional
`tolerance` receives the same immutable `GeometryTolerance` as Mesh25D. KOZ
width never shrinks it; a required feature below resolution raises an explicit
conflict. All generated final fronts share canonical supports before global cuts. The meshing envelope remains the existing
symmetry-restricted grid envelope, including circular support margins. Circular
imprint/extension run after adaptive construction, followed by the unchanged
Z plan, winding conversion, material allocation, and extrusion. Shapely geometry
normalization remains confined to Mesh25D; the adapter itself stays NumPy-only.
When a region intersects a planned circular extension annulus, the circular
plan is recomputed as imprint-only with the corresponding grid envelope and
support cuts, preventing extension from replacing that region's refinement.

`process_flow` owns worker transport and CDB serialization. It calls `mesh25d`
without owning meshing or geometry translation. The CDB writer remains here
because it implements the repository's text contract rather than a general
ANSYS CDB serializer.

PyVista, Matplotlib, PySide6, Shapely, and `process_flow_kernel` are not imported
by the core package, shared control methods, or control interfaces. Mesh25D's
public module loads its optional runtime on a build call and raises an
actionable import error when the `process-flow` extra is missing. Input JSON
contract validation remains independent of that runtime.
