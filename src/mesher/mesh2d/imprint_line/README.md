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

## Controls from boxes

```python
from mesher.mesh2d.imprint_line import get_pair_line, imprint_lines

outer = [[0, 0], [12, 0], [12, 12], [0, 12]]
pairs = get_pair_line(
    [[1, 4, 3, 6]],
    expand_size=0.5,
    outer_outline=outer,
    distance=2.0,
    spacing=0.4,
)
mesh = imprint_lines(
    2.0, pairs, outer_outline=outer,
    min_transition_size=0.5, pattern_snap_distance=0.2,
)
```

`get_pair_line(boxes, expand_size, outer_outline=None, distance=None, *, spacing)`
returns two `LinePair` dictionaries per `[x1, y1, x2, y2]` box, preserving box
order: X (left/right sides), then Y (bottom/top sides). `line1` is always the
lower-coordinate side. Required positive `spacing` places points on `line1`
at fixed intervals from its lower endpoint and appends the exact upper endpoint
once. A length of 10 with spacing 3 gives positions 0, 3, 6, 9, 10. A short final
interval is retained; meshing resolution remains subject to `imprint_lines`.

Without an outline and threshold, all sides use `expand_size`. Supply both
`outer_outline` and `distance` to measure each side outward, perpendicular to
that side. Its expansion becomes the first boundary-contact distance `d` only
when `d < distance`; equality keeps `expand_size`. This replacement can increase
the default expansion. Concave outlines consider the entire side, including
endpoint contacts, rather than just its midpoint or the outline bounding box.
In the example, only the left expansion changes, from 0.5 to 1.0.

Boxes require finite coordinates, `x1 < x2`, and `y1 < y2`. With an outline,
each box must be fully contained; touching is allowed and has zero outward
distance on that side. Outlines use the existing simple orthogonal-loop format
and default geometry tolerance, accepting either winding and an optional
repeated closing vertex. Expansion and threshold must be finite and nonnegative.
Supplying only one of outline/threshold raises `ValueError`. Empty boxes return
an empty list; inputs are never modified.

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
- Patterns retain their **original float64 coordinates and endpoints**, even
  during input normalization. Only exactly horizontal or vertical nonzero
  patterns are accepted; nearly orthogonal segments are not projected.
  Distinct fixed coordinates that conflict under the tolerance policy raise
  `GeometryToleranceError(code="fixed_constraint_conflict")` instead of being
  merged. Exact duplicates and reversed copies are deduplicated. Patterns can divide
  transition cells when each resulting width is admissible. Otherwise the
  transition moves beyond the obstruction. Nearby perpendicular patterns can
  attract a generated boundary outwards by `pattern_snap_distance`. This option
  never moves patterns. Portions outside the exterior are clipped geometrically;
  retained endpoints, clipping intersections, and boundary-collinear parts remain
  exact constraints.
- Every transition cell has one dense side; the other three sides have only
  their endpoints. Even dense-node counts use only quads. Odd counts use
  exactly one triangle, stored as `[a, b, c, c]` in `Mesh2D`.
- The rectangular unit uses the adapter's nested topology with depth
  proportional to remaining dense-edge span. This avoids exponential depth
  collapse, but is not a universal bound on individual element aspect ratios.
  Units are constructed and checked during planning. An edge at or below the
  tolerance triggers outward replanning; an end without an admissible unit
  continues to the exterior. Exterior corners touching a unit are retained as
  mandatory contacts, including at concave corners.
- Input arrays are not mutated. Output nodes have exactly zero Z. Node and
  element ordering is deterministic. The package's geometric tolerance policy
  defaults to absolute distance `1e-3`. Exterior and control coordinates undergo
  one normalization with patterns protected, then remain fixed throughout planning
  and construction. Only generated coordinates can subsequently be adjusted.

The background uses the existing generator's sparse visibility sweep and
rectangle index. Fine lines stop at reserved transitions instead of becoming
global grid coordinates. Generated sizing cuts respect locked unit edges;
hard contacts trigger replanning. Every successful sizing iteration must split
its target rectangle and increase the rectangle count. Candidate cuts are tried
only finitely many times; a repeated planning state also terminates explicitly.
Area, shared-edge, exact constraint coverage, and retained endpoint checks run
before return. Constraint checks use sorted edge intervals rather than an
all-elements scan for each pattern.

Invalid input raises `ValueError` or the package's `GeometryToleranceError`.
Planning/construction failures use `ImprintLineError` with `code` and `context`.
No partial result is returned. `max_elements` (default one million) bounds
intermediate rectangles and final cells. The planner has a finite retry budget;
if exhausted it reports `planning_limit`, rather than silently discarding
controls. Holes, diagonal constraints, and modifying an existing mesh are not
supported in this first version.

Valid scalar parameters alone do not guarantee that the requested geometry can
be meshed at that resolution. For example, a fixed gap of `0.0021` with
`element_size=0.00104` requires at least three intervals, whose average length
is below the default `0.001` tolerance. This raises `feature_below_resolution`;
neither the fixed constraints nor the size limit are relaxed. Other diagnostics
include `sizing_no_progress` for a cut that fails to split its target,
`locked_sizing_conflict` for exhausted sizing candidates, `planning_limit`
with `reason="no_progress"` for repeated states, and `constraint_not_preserved`
if the final mesh loses a required segment or endpoint. Errors include the
stage and relevant bounds or source indices. Other mesh generators retain
their existing normalization and snapping behavior.

## Verification and previews

From the repository root:

```sh
PYTHONPATH=src venv/bin/python -m unittest discover -s tests/unit/mesh2d/imprint_line -t .
PYTHONPATH=src venv/bin/python examples/imprint_lines.py --output /tmp/imprint-lines
PYTHONPATH=src venv/bin/python examples/imprint_boxes_outline.py --output /tmp/box-outline
PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_imprint_lines.py --case separated --repeats 5
PYTHONPATH=src venv/bin/python benchmarks/mesh2d/benchmark_imprint_lines.py --case overlap --repeats 5
```

The example saves four actual meshes as NPZ files, individual PNG previews,
an overview, and mesh counts. Both examples call the public `imprint_lines` API.
Their plots mark input controls, slanted quads, and triangles; they do not infer
planner region labels. The die example uses fixed zoom windows. Matplotlib is
only needed to render these examples.

`imprint_boxes_outline.py` recreates the hand-drawn layout with 14 boxes in a
100-by-130 exterior (illustrative dimensions). It calls `get_pair_line` followed
by `imprint_lines`, with spacing 2, default expansion 2, and outline threshold 12.
It saves a layout/mesh overview, actual before/after mesh close-ups, both meshes
as NPZ, the generated controls, and a JSON report. Orange regions show the
requested extensions to the outline, not inferred planner region labels.
The example checks actual mesh-edge coverage for all 82 sampled lines across
the 15 selected sides before saving its previews. Interior controls retain the
default expansion; the mesher can still extend them further during planning.

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
