# Rectilinear global mesh around holes

`generate_rectilinear_mesh_optimal` builds a conforming Quad4 mesh for an
orthogonal domain with supplied holes. Its normal local strategy stops pattern
lines at the first outer or hole boundary, so the meshes on opposite sides of
a hole can use different grid coordinates. The holes remain empty. Their returned
boundary nodes can support a future connection to other meshes, including
adaptive meshes; that integration is independent of this generator.

## API

```python
import numpy as np

from mesher.mesh2d.generators import generate_rectilinear_mesh_optimal

result = generate_rectilinear_mesh_optimal(
    target_edge_size=2.0,
    outer_outline=np.array([[0, 0], [12, 0], [12, 8], [0, 8]]),
    holes=[np.array([[4, 2], [7, 2], [7, 6], [4, 6]])],
    pattern_lines=[[[1, 3], [3, 3]]],
    tolerance=1e-3,
    max_elements=1_000_000,
)

mesh = result.mesh
outer_node_ids = result.outer_outline
hole_node_ids = result.hole_outlines[0]
hole_xy = mesh.nodes[hole_node_ids, :2]
diagnostics = result.diagnostics
```

The outer outline is a simple, exactly axis-aligned polygon. Each hole is a simple
orthogonal polygon strictly inside it; holes cannot overlap, nest, or touch one
another or the outer boundary. Concave outlines and either input winding are
supported, with an optional repeated closing point. Pattern inputs are finite
horizontal or vertical segments, clipped to the domain; endpoints inside or on
the domain remain mesh nodes. The generator uses NumPy only.

Collinear input anchors remain boundary nodes. A pattern that spans a hole
activates the visible portions on both sides. In the local strategy, an extension
from a short pattern does not resume beyond a hole. Portions outside the outer
loop or inside holes are ignored.

`target_edge_size` is a positive maximum edge length, rather than a promise of
uniform spacing. Coordinates required by geometry and patterns can produce
smaller edges. Compatible geometry and pattern supports are canonicalized
jointly before visibility or intersections. Each point moves by at most the
absolute `tolerance` (default `0.001`); proximity chains wider than T, disappearing
holes/gaps, and incompatible fixed coordinates raise structured errors. This
policy is independent of the target size and numerical calculation epsilon.

`max_elements` is a positive integer, with a default of `1_000_000`; `None`,
booleans, and floating-point values are rejected. It limits the complete mesh,
including geometry, pattern, sizing, and parity refinement. A request exceeding
the limit raises `ValueError`. Raise the limit explicitly when a larger mesh is
intentional. The limit counts elements; it is not a byte-exact memory budget.

The returned `mesh` is a `Mesh2D` with counterclockwise Quad4 elements and zero
Z coordinates. The boundary arrays contain ordered mesh node IDs, without a
repeated closing node: `outer_outline` is clockwise, while every entry in
`hole_outlines` is counterclockwise. Hole ordering matches the input sequence.
Every returned boundary has an even number of nodes.

## Local cuts, sizing, and parity

In the local strategy, geometry corners generate the axis-aligned rays needed
to partition the domain into rectangles. Required collinear anchors also receive
a perpendicular cut. A pattern extends along its axis to the first geometry
boundary, with perpendicular cuts at its endpoints. Cuts cross existing grid
lines and stop at geometry boundaries, including a tangential corner contact.
They never stop at an interior grid intersection, which would leave a hanging node.

Sizing refines the current rectangular faces greedily, first by width and then
by height. An oversized face is divided into equal intervals no larger than
`target_edge_size`; the corresponding cuts extend through its visible domain.
The partition is updated as cuts are added. This keeps geometry and pattern
coordinates local to their visible portions of the domain. It does not first
construct a Cartesian product of all input X and Y coordinates.

Parity repair runs after sizing. Holes are processed by decreasing leftmost X
coordinate. If a hole has an odd boundary-node count, a horizontal repair cut
starts in a gap on its leftmost boundary and travels left to the first geometry
boundary. Candidate gaps avoid existing horizontal cut levels, and the widest
gap is preferred. Its destination is the outer boundary or a hole with a smaller
leftmost X coordinate. Processing in this order prevents later cuts from
changing an already repaired hole. At most one repair cut is needed per hole;
repair cuts only subdivide existing cells, preserving the size bound.

If an odd hole boundary has no representable local parity-repair cut, the
generator retries with a global rectilinear grid. This fallback projects
retained coordinates across the domain, so coordinates can propagate beyond a
hole and the mesh can contain more cells than the local result would have.
Canonical geometry and retained pattern endpoints stay exact, holes stay empty, the
maximum edge size is preserved, and every boundary still has even parity.
The same `max_elements` limit applies. Invalid geometry, numerical-range
failures, budget failures, and other consistency errors do not trigger fallback.

An even boundary count is necessary for a future all-quad fill, but does not
alone guarantee that an independently generated adaptive mesh will match the
boundary coordinates. Callers must use the returned boundary nodes when
preparing that integration.

## Diagnostics and limits

`diagnostics` includes the boundary-node counts before and after parity repair,
with the outer boundary first and holes in input order. `parity_repairs` records
each cut's endpoint coordinates, source `hole_index`, and
`target_hole_index` (`None` means the outer boundary). Cut counts and the quality
values `minimum_edge_length` and `maximum_aspect_ratio` help assess refinement.

`strategy` is `"local"` or `"global_grid_fallback"`. `fallback_reason` is `None`
for local construction and `"parity_repair_unavailable"` for the fallback.
Fallback diagnostics describe the returned mesh: `parity_repairs` is empty,
the before/after boundary counts are equal, and `cuts["parity"]` is zero.
Fallback cut counts count visible chords at global grid coordinates, assigning
each coordinate to its first applicable source in the order geometry, pattern,
then sizing. They do not include work from the discarded local attempt and are
not element counts.

Finite coordinates alone do not guarantee numerically safe area calculations.
Before calculating polygon areas, the generator checks the conservative bound
`sum(2 * N * W * H) <= float64_max / 2` across all loops. Here `N` is the number
of input vertices after removing optional repeated closure, including collinear
anchors; `W` and `H` are that loop's bounding-box spans. Inputs outside this
range raise `ValueError`, even when their mathematical net area is finite.
This is a calculation-safety range, not a physical-unit restriction. It uses
coordinate spans, so a large coordinate origin with small spans remains
supported. Unrepresentable cell areas or aspect ratios are also rejected.

The size limit does not imply a minimum edge length or an aspect-ratio bound.
Closely spaced mandatory geometry or patterns can still create thin rectangles.
Local propagation limits how far those constraints travel; it cannot eliminate
features that the input explicitly requires. Diagonal geometry, curved edges,
touching holes, and focus/adaptive mesh insertion are outside this API.

## Comparison example

From the repository root:

```sh
PYTHONPATH=src python examples/holed_rectilinear_mesh.py --no-show
PYTHONPATH=src python examples/holed_rectilinear_mesh.py --no-show \
    --plot /tmp/holed-grid.png --diagnostics /tmp/holed-grid.json
```

Omit `--no-show` to open an interactive Matplotlib plot. The non-plotting command
does not require Matplotlib. `--target-size` changes the default size of `2.0`.

The fixture contains a concave outer outline, two rectangular holes, and three
nearby short horizontal patterns to the left of the first hole. The plot shows
the input, the local mesh with hole-node counts and parity-repair lines, and a
global-projection reference. The reference passes every input coordinate to the
existing `generate_rectilinear_mesh`, then removes complete cells outside the
domain. It is a demo comparison, not the production process-flow pipeline.

The command reports element and node counts, minimum and maximum edge lengths,
maximum aspect ratios, and per-hole counts before and after repair. The JSON
output contains both comparisons and the generator diagnostics; the plot shows
where short patterns stop affecting the global region.

With the default fixture, the local mesh has 79 quads versus 90 in the reference.
Two repair cuts leave the hole boundaries with 14 and 8 nodes. Both meshes have
a minimum edge of 0.1 and a maximum aspect ratio of 20: this example demonstrates
reduced propagation and element count, while retaining the small input features.

## Performance benchmark

The benchmark covers a row of 100, 200, 400, and 800 holes, plus a rectangular
outline with 500, 1,000, and 2,000 collinear bottom-edge segments. Both families
have linear output size. Each repetition uses a fresh process; three repetitions
are the default. JSON output includes raw samples, median generation time, peak
RSS, mesh hashes, source hashes, and available geometry-work counters.

```sh
PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_holed_rectilinear.py \
    --output /tmp/holed-current.json
PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_holed_rectilinear.py \
    --source-dir /tmp/holed-baseline-sources --label baseline \
    --output /tmp/holed-baseline.json
```

For the second command, the source directory must contain saved versions of
`holed_rectilinear.py`, `_holed_geometry.py`, and any additional `_holed_*.py`
helpers used by that version. The benchmark overlays them in a temporary copy
of the package without modifying the checkout. Compare runs on
the same host and check mesh hashes as well as timing. Older implementations
without geometry counters report `null` for those counters. Use `--repeats`,
`--hole-counts`, or `--anchor-segments` to change the workload.

Logical edges and spatial indexes reduce work on sparse input while all
mandatory anchors remain mesh nodes. The generator does not promise linear or
`O(n log n)` end-to-end complexity: intersecting mandatory cuts can themselves
require a quadratic number of output cells, and spatial-index candidate counts
can be large for adverse geometry.

On 2026-11-04, the saved pre-change implementation and the updated generator
were compared on the same macOS 14.6 ARM64 host with Python 3.13.5 and NumPy
2.5.2. Each time below is the median of three fresh-process runs, in seconds.
All seven resulting mesh hashes matched the baseline exactly.

| Workload | Before | After | Speedup | Section edges read | Visibility edges read |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 holes | 0.4790 | 0.0264 | 18.2x | 1,006 | 1,616 |
| 200 holes | 1.8657 | 0.0513 | 36.4x | 2,006 | 3,216 |
| 400 holes | 7.4034 | 0.1104 | 67.0x | 4,006 | 6,416 |
| 800 holes | 29.7235 | 0.2524 | 117.8x | 8,006 | 12,816 |
| 500 collinear segments | 0.1024 | 0.0109 | 9.4x | 6 | 1,010 |
| 1,000 collinear segments | 0.3739 | 0.0174 | 21.5x | 6 | 2,010 |
| 2,000 collinear segments | 1.4482 | 0.0315 | 46.0x | 6 | 4,010 |

Edge-read counters describe the updated implementation; the saved baseline
has no counters. The spatial index eliminated all nonadjacent edge-distance
comparisons for these separated fixtures (`validation_pairs=0`). The 800-hole
case's median peak process RSS was 47.3 MiB. Timings are host- and
workload-specific measurements, not unit-test thresholds.
