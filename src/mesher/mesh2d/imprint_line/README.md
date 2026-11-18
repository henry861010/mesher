# Line-pair imprinting

This independent NumPy-only mesher does not change the existing adapter or
process-flow pipeline. It generates a new `Mesh2D` from a simple orthogonal
exterior, rectangular directional controls, and finite orthogonal patterns.

```python
from mesher.mesh2d.imprint_line import imprint_lines

mesh = imprint_lines(
    2.0,
    [{
        "direction": "x",
        "line1": [[4, 4], [4, 6]],
        "line2": [[6, 4], [6, 6]],
        "points": [[4, y] for y in (4, 4.4, 4.8, 5.2, 5.6, 6)],
        "line1_expand_size": 2.0,
        "line2_expand_size": 3.0,
    }],
    pattern_lines=[],
    outer_outline=[[-2, -2], [14, -2], [14, 12], [-2, 12]],
    min_transition_size=0.5,
    pattern_snap_distance=0.2,
)
```

For `direction="x"`, the two lines are vertical cross sections with identical
Y spans, and only the Y components of `points` define dense lines. Transpose
this definition for `"y"`. A rectangular focus uses both directions. Section
endpoints are included automatically. `line1_expand_size` and
`line2_expand_size` are both required finite, nonnegative scalar distances.
They follow `line1` and `line2` respectively, even when their ordering is
reversed. To use the same distance at both ends, explicitly set both fields to
that value. The former `expand_size` scalar/tuple input is no longer supported.

## Placement and topology

- E is the minimum straight extension before a transition. Obstacles may
  increase it. It does not include the transition's depth.
- Same-direction overlapping control/extension/transition envelopes merge.
  Their dense coordinates are united and carried across the common band.
  Crossing directions remain independent, with a rectangular grid at crossings.
- A single band never widens by itself. If it is too narrow, or an end has no
  legal transition before the exterior, its lines continue to that exterior.
- Transition width and depth are at least `min_transition_size` and at most
  `element_size` (apart from floating-point roundoff). A wide band has multiple
  adjacent transition cells. Both distance options use model units and are
  required; require `tolerance < min_transition_size <= element_size`.
- Patterns retain their canonical positions and endpoints. They can divide
  transition cells when each resulting width is admissible. Otherwise the
  transition moves beyond the obstruction. Nearby perpendicular patterns can
  attract a generated boundary outwards by `pattern_snap_distance`.
- Every transition cell has one dense side; the other three sides have only
  their endpoints. Even dense-node counts use only quads. Odd counts use
  exactly one triangle, stored as `[a, b, c, c]` in `Mesh2D`.
- The rectangular unit uses the adapter's nested topology with depth
  proportional to remaining dense-edge span. This avoids exponential depth
  collapse, but is not a universal bound on individual element aspect ratios.
- Input arrays are not mutated. Output nodes have exactly zero Z. Node and
  element ordering is deterministic. All inputs share the package's geometric
  tolerance policy (default absolute distance `1e-3`).

The background uses the existing generator's sparse visibility sweep and
rectangle index. Fine lines stop at reserved transitions instead of becoming
global grid coordinates. Generated sizing cuts respect locked unit edges;
hard contacts trigger replanning. Area and shared-edge checks run before return.

Invalid input raises `ValueError` or the package's `GeometryToleranceError`.
Planning/construction failures use `ImprintLineError` with `code` and `context`.
No partial result is returned. `max_elements` (default one million) bounds
intermediate rectangles and final cells. The planner has a finite retry budget;
if exhausted it reports `planning_limit`, rather than silently discarding
controls. Holes, diagonal constraints, and modifying an existing mesh are not
supported in this first version.

## Verification and previews

From the repository root:

```sh
PYTHONPATH=src venv/bin/python -m unittest tests.unit.mesh2d.imprint_line.test_imprint_lines
PYTHONPATH=src venv/bin/python examples/imprint_lines.py --output /tmp/imprint-lines
PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_imprint_lines.py --case separated --repeats 5
PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_imprint_lines.py --case overlap --repeats 5
```

The example saves four actual meshes as NPZ files, individual PNG previews,
an overview, and mesh counts. Both examples call the public `imprint_lines` API.
Their plots mark input controls, slanted quads, and triangles; they do not infer
planner region labels. The die example uses fixed zoom windows. Matplotlib is
only needed to render these examples.

Each benchmark has 100 pairs, 100 points per pair, 1000 finite patterns, and
a 102,400-cell unconstrained reference grid. Each repetition uses a fresh
process and times the complete public call, including validation. The script
reports time, output size, peak RSS, and a deterministic mesh hash. Ten seconds
is a same-host acceptance target for these fixtures, not a guarantee for
arbitrary geometry or output size.

The overlap fixture contains 25 aligned two-focus groups. Offsetting both
fine-coordinate sets can produce substantially more cells: a 25-group case
with 100 points per pair exceeds the default million-element budget. Increasing
that budget permits larger outputs but does not preserve the timing target.
