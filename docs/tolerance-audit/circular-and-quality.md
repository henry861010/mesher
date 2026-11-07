# Circular + mesh2d quality tolerance audit

Audit date: 2026-11-07. Source checkout: `/Users/henry/Desktop/code/mesher`. Read-only audit; no production or existing repository test files changed. Existing working-tree edits preserved. Reproducer: `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/circular_probe.py`, run with `PYTHONPATH=src venv/bin/python /Users/henry/Desktop/code/mesher/docs/tolerance-audit/circular_probe.py` from the repository root.

## Executive findings

The circular code does not implement a single model-space `0.001` tolerance. Legacy circular imprint and extension expose no geometry tolerance. V2 has projection/guide/merge distances but they are operation-specific and do not govern all predicates. Quality has another independent tolerance object/default. Some numerical epsilon checks should remain numerical; domain-equivalence must be established before those checks rather than replacing every epsilon with 0.001.

### C1 — High, reproduced: exact-flat geometry succeeds while a 1e-14 perturbation fails

- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/utils.py:2656`: cleanup decides a quad is strictly convex with `corner_cross > 0.0`.
- Same file `:2696`: such a quad is retained, bypassing repair. At exactly zero turn, the quad instead proceeds to the two-diagonal triangle repair (`:2701`).
- Same file `:1375`: `_split_circle` uses `2048*eps*max(element_scale,1)**2` for cross-product validity and rejects turns at or below this threshold (`:1391`).
- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/local.py:262`: cleanup precedes split. The dedicated near-flat-quad triangulation is only called after splitting, at `:296`.

Reproducer: Quad `[(0,0),(1,0),(2,h),(0,1)]`, circle center `(0,0)`, radius `0.5`; `projection_tolerance=guide_tolerance=merge_tolerance=0.001`, `minimum_area=1e-6`.

| h | Result |
|---|---|
| 0 | succeeds, 3 children |
| 1e-14 | ValueError: selected elements must be non-degenerate Tri3 or convex Quad4 |
| 1e-12 | same ValueError |
| 1e-9 | succeeds, 3 children |
| 0.0005 | succeeds, 3 children |

This is a demonstrated predicate inconsistency, not merely a small hardcoded constant. Use one dimensionally correct flatness classification across cleanup, split preconditions, repair and final validation. Decide repair before split if it is required to meet split preconditions. Never solve by weakening validation until folded cells pass.

### C2 — High, reproduced: circular extension rejects boundaries within 0.001

- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/extend/mesher.py:126`: boundary tolerance is `max(64*eps*coordinate_scale, 1e-9*inner_radius)`.
- `:148`: nodes farther than `inner_radius + tolerance` cause their elements to be discarded from the private working copy.
- `:234`: boundary membership requires radial residual <= that hardcoded tolerance.
- `:457`: public `extend_circular_mesh` offers no tolerance argument.

Eight-triangle fan with requested inner radius 1: radius 1 succeeds; radii 0.9995 and 1.0005 both fail with `the retained mesh must have exactly one exposed circular boundary or continuous arc on inner_radius`. Caller mesh remains transactional/unchanged on error. Thus a 0.0005 input deviation cannot be accepted by any explicit public 0.001 setting today.

### C3 — High, reproduced: guide normalization ignores the supplied guide tolerance

- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/local.py:197`: before projection, guide normalization receives only `128*eps*scale`, not `guide_tolerance`.
- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/utils.py:1047`: standalone projection repeats that choice.
- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/utils/pattern_segments.py:123`: axis alignment and nonzero-length classification use `max(minimum_tolerance,64*eps*axis_scale)`; invalid alignment raises at `:136`.

Public V2 imprint with all three distance tolerances 0.001 and guide `[(0.5,-2),(0.5000001,2)]` raises `each pattern segment must be non-zero and horizontal or vertical`. A 1e-7 deviation is rejected before the user tolerance is applied. Current documented guide_tolerance governs *membership*, not axis normalization, so this is an unmet global policy rather than proof that the existing narrow argument was implemented contrary to its docs. Define tolerant axis classification/canonicalization and maximum allowed movement; do not silently accept arbitrary diagonal guides.

### C4 — Medium/high, reproduced: legacy circle search has the opposite mismatch, a hidden 0.01

- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/utils/topology.py:23`: `_search_circle(..., tolerance=0.01)`.
- Same file `:84`: effective radius is `radius+tolerance`.
- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint/mesher.py:479` and `:486`: inner/outer band selection omit the tolerance parameter.

For radius 1, a small triangle at radial distances about 1.005 is included by the default ALL search but excluded when passing 0.001. This can change band membership/topology, and is 10x broader than the requested policy. Search margins (`band_width/10`) are distinct feature-selection parameters and must not be confused with tolerance.

### C5 — Medium, reproduced policy gap: sub-0.001 guide discrepancies can freeze projection silently

- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/utils.py:1117`: membership uses max(user guide_tolerance, numerical floor).
- Same file `:1159`: agreement among multiple guide targets uses only `256*eps*target_scale`.
- Same file `:1162`: a discrepancy above that floor marks the node conflicting and immovable; this is a silent no-op, not an exception.
- Same file `:853`: fixed-junction barrier resolver uses a similarly numerical-only agreement threshold.
- Same file `:166` and `:187`: circle/guide root existence and finite endpoint inclusion are numerical-only, not governed by guide_tolerance.

A node `(0.50005,0.8655)` on radius approximately 0.9995700338 with two vertical guides x=0.5 and x=0.5001 lies within both 0.001 corridors. With one guide it projects to `(0.5,0.8660254038)`; with both it remains unchanged because the two targets differ by about 0.0001155. Existing docs/tests intentionally preserve exact guide constraints and freeze conflicts. A new 0.001-equivalence policy must explicitly define guide canonicalization, multiple constraints, tangency/finite endpoint treatment and diagnostics rather than unconditionally merging incompatible guides.

### C6 — Medium/high, reproduced: quality validation has independent defaults and incomplete distance semantics

- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/quality/checker.py:62`: `GeometryTolerance` defaults absolute=relative=1e-12.
- `:298`: each checker creates this independent default unless explicitly supplied.
- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/quality/gui_model.py:98`: GUI uses the checker default; no policy propagation.
- Checker `:560`: length threshold = absolute + relative*longest_edge.
- Checker `:563`: Jacobian threshold = absolute**2 + relative*longest_edge**2.
- Checker `:661` and `:730`: zero-edge tests use length threshold; Jacobian near-zero tests use the area threshold.

A 1 by 0.0005 Quad4 is reported valid by the default checker; explicit `GeometryTolerance(absolute=.001,relative=0)` flags it degenerate. However a triangle `[(0,0),(1000,0),(500,0.0005)]` still has no invalid flags with absolute=.001 because all its edges are long and determinant 0.5 is far above tau²=1e-6. If global tolerance means point-to-line/altitude equivalence, tau² is insufficient; an area predicate proportional to tau*local_length (with the proper triangle/Jacobian factor) is needed. This is a semantics decision: current class explicitly documents its tau² formula. Raw Jacobian geometric validity is separate from scaled-Jacobian/aspect quality cutoffs.

### C7 — Medium, reproduced adjacent contract issue: minimum_area is not a final output guarantee

- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/local.py:186` silently caps `minimum_area` at 1.0, unlike the standalone cleanup helper `utils.py:2374`.
- Local `:262` passes minimum_area to cleanup only; `:270` calls split without it, and `:296` triangulates without it.
- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/utils.py:2198`: generated child area validity uses only `4096*eps*max(child_scale,1)**2`.
- `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/triangulation.py:58`: final triangulation checks only positive dimensionless corner sine.

Unit square, circle center corner `(0,0)`, radius .01, distance tolerances .001, minimum_area .001 produces areas `[0.5,0.49995,0.00005]`; the final child area violates the documented minimum retained element area. Length .001 and area .001 are different dimensions: do not conflate them. A policy must decide whether minimum_area is cleanup-only or an output postcondition, and handle undersized children without silently deleting coverage.

## Complete source-family inventory and migration relevance

All Python files under `mesh2d/circular` and `mesh2d/quality` were enumerated (circular roughly 10k lines plus checker 876 lines); searches covered eps/tolerance/isclose/allclose/exact comparisons and the predicate/control-flow implementations. Thin __init__ exports and GUI event/control boilerplate have no independent tolerance algorithm.

| Family | Exact representative refs | Status / migration concern |
|---|---|---|
| Legacy public imprint request and API | `circular/imprint/mesher.py:55`, `:748` | No geometry policy in request or public signature. Add propagation through selection, projection, pattern, strip, validation. |
| Legacy open-sector/ray classification | `circular/imprint/mesher.py:268`, `:379`, `:412`, `:429` | Ray membership uses max(1e-9*r,256eps*scale); angle threshold derives from radial tolerance/radius. Preserve dimensional conversion and define sector seam wrapping. |
| Legacy initial guide filtering | `circular/imprint/mesher.py:816` | Numeric-only guide tolerance; `.intersecting_circle` can discard a near circle/finite endpoint guide before later stages. |
| Legacy radial selection | `circular/utils/topology.py:23`, `:84` | Hidden 0.01, C4. Topological index equality and edge counts in same file are exact by design. |
| Circular extension | `circular/extend/mesher.py:126`, `:172`, `:225`, `:300`, `:402`, `:457` | C2; generated-layer quality length check is numerical only. Angular ordering/winding derived from radial threshold; do not set angles to .001 radians wholesale. |
| Prepared guides | `circular/utils/pattern_segments.py:123`, `:204`, `:235`, `:291`, `:359` | C3; cached `_PatternGuideSet` returned unchanged by `_coerce_pattern_guides` at :209, ignoring a later caller's minimum_tolerance. Policy metadata/revalidation required if a shared policy is introduced. Distinguish axis classification, endpoint membership, tangency and duplicate roots. |
| Legacy projection | `circular/utils/projection.py:105`, `:249`, `:449`, `:583`, `:647`, `:747`, `:823` | Numeric-only length tolerance controls chain edge validity, inside/outside status, circle fit, pattern anchor matching/conflicts. :105 correctly converts length tolerance to edge parameter. No public override in `_to_circle` at :1197. |
| Shared generated geometry | `circular/utils/geometry.py:35`, `:86`, `:123`, `:228`, `:250`, `:280`, `:394` | Helpers accept linear/angular tolerances from caller, but most callers supply numerical eps. Cross threshold `linear_tol * length_scale` is dimensionally correct. Minimum scaled Jacobian and smoothing quality epsilon are dimensionless and must remain separate. |
| Legacy pattern ring/arc | `circular/imprint/pattern.py:47`, `:70`, `:192`, `:535`, `:616` | Duplicate anchor tolerance ~eps; fit tolerance max(eps,1e-9*r); endpoint allclose explicitly rtol=0,atol=fit_tolerance (not an accidental NumPy default). Geometry policy must enter initialization and anchor combination. |
| Legacy strip fitted circles | `circular/imprint/strip_mesher.py:270`, `:274`, `:291`, `:302`, `:320` | Geometry tol max(ULP floor,1e-10*r), fit max(geometry,1e-9*r), angular=geometry/r. No shared caller policy. |
| Legacy strip predicates | `circular/imprint/strip_mesher.py:414`, `:469`, `:557`, `:662`, `:784`, `:977`, `:998` | One derived geometry threshold feeds area, intersections, on-segment, angular order, corner validity and guide matching; suitable consolidation point. |
| Legacy strip dimensional bug | `circular/imprint/strip_mesher.py:917` | `_axis_interval_is_active` compares dimensionless edge fraction to +/- linear tolerance. Must convert using edge length/fixed-axis delta as appropriate; directly increasing tolerance to .001 introduces unit/scale dependence. Found by inspection; no isolated outcome probe. |
| Legacy triangulation scoring | `circular/imprint/triangulation.py:51`, `:65`, `:83` | angular comparison epsilon, squared-angle score margin and dimensionless quality epsilon, respectively. Not all are spatial tolerances. Geometric turn/visibility thresholds at :343,:359,:411 already derive area from geometry tolerance. |
| Legacy quad merge | `circular/imprint/quad_merge.py:29`, `:42` | Uses supplied geometry and angular tolerances; user minimum scaled Jacobian remains dimensionless. |
| Legacy strip final checks | `circular/imprint/validation.py:361` | Area coverage tolerance `max(geometry_tol*boundary_length,1e-10*area_scale)`: dimensions correct, but derive from shared policy. Structural duplicate/connectivity checks should remain exact. |
| V2 public contract | `circular/imprint_v2/mesher.py:10`, `:39` | Projection reach, guide membership, node merge, area and angle are independent controls. Retain these knobs while adding a distinct common geometry policy. |
| V2 local patch/closure | `circular/imprint_v2/local.py:31`, `:78`, `:195`, `:262`, `:270`, `:296` | Broad phase width projection+merge+numerical floor; merge closure uses merge distance; no tolerance-consistent final geometry postcondition. Exact coordinate inequality at :276 detects any real moved node for interface closure and must stay exact. |
| V2 projection and constraints | `circular/imprint_v2/utils.py:122`, `:212`, `:332`, `:487`, `:672`, `:777`, `:1044`, `:1117`, `:1159`, `:1222` | C3/C5. Membership receives user distance but target agreement/root bounds/barrier endpoint exclusions do not. Projected-target merging uses projection_tolerance and deliberately keeps distinct guided targets (:530), then later cleanup merges using merge_tolerance. Two separate ownership rules need explicit policy. |
| V2 orientation normalization | `circular/imprint_v2/utils.py:590`, `:637` | Normalized signed area <0 decides winding; zero has no fuzzy geometric state. Numerical sign detection can remain strict after a shared geometric degeneracy classifier/canonicalization. |
| V2 splitting predicates | `circular/imprint_v2/utils.py:1375`, `:1400`, `:1459`, `:1483`, `:1502`, `:1558`, `:1617`, `:1698` | Convexity numerical area; stable discriminant and normalized parameter error numerical; endpoint snapping/angle merging receive merge_tolerance. :1698 correctly converts chord distance to angle `2*asin(tau/(2*r))`. |
| V2 generated geometry/coverage | `circular/imprint_v2/utils.py:2115`, `:2198`, `:2251`, `:2301` | Child validity/parent snap/coverage eps are numerical area or dimensionless quality. Need extra domain geometry postconditions, not blanket replacement. |
| V2 cleanup | `circular/imprint_v2/utils.py:2499`, `:2567`, `:2643`, `:2664` | C1/C7. Area gate and equivalent nodes independent; strict convexity differs from splitter. Exact dedup in warning classification :2438 is diagnostic, not merge policy. |
| V2 final flat-quad triangulation | `circular/imprint_v2/triangulation.py:36`, `:58` | Angle threshold degrees=170 is shape quality; cosine/sine comparisons dimensionless. Actual positive-area selection uses >0, with no minarea/domain geometry threshold. |
| Quality defaults/classification | `quality/checker.py:54`, `:294`, `:560`, `:661`, `:730` | C6. Distinguish length, determinant-area, dimensionless relative and quality thresholds. |
| Quality threshold roundoff | `quality/checker.py:815` | `32*eps*max(1,abs(threshold))` intentionally guards quality threshold comparisons; do not replace with .001 and weaken requested quality limits. |
| Quality GUI | `quality/gui_model.py:98`, `quality/gui.py` | GUI only consumes independent default checker and adds no policy controls. Wire policy into model/view context. |

Inventory table paths are relative to `/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/`; numbered finding paths above are absolute.

## Important numerical/semantic distinctions

1. Geometry distance tau=0.001 has units of model length. Node equivalence, coordinate coincidence, on-line distance, radius residual and endpoint distance should reference a common policy.
2. Projection reach and mesh feature dimensions are design controls, not tolerances; legacy band widths and target edge size should not be replaced by tau.
3. Oriented cross products / raw Jacobians / area have units L². Use domain classification based on tau times a relevant local length where the geometric meaning is near-collinearity/altitude; keep user `minimum_area` separate. Check multiplicative factors for triangle determinant versus quad parametric Jacobian.
4. Angular tolerances are dimensionless/radians and must be derived from lengths/radii if intended to represent spatial closeness. Scores and scaled Jacobians are also dimensionless but unrelated to angular tolerance.
5. Discriminant evaluation, normalized orientation arithmetic, polynomial root selection and area conservation residuals need numerical roundoff budgets. Keep eps/ULP guards, but separate them from domain acceptance. A numerical floor may exceed .001 for huge global coordinates: local.py:196 gives about .227 at scale 1e12; projection :1045 about .0284. Diagnose representability/coordinate recentering rather than silently enlarging the contractual geometry tolerance.
6. Exact topological node-ID equality, edge-use counts, dtype/shape checks, transaction no-change checks, and detecting any changed coordinate for interface closure are not tolerance bugs.
7. Global tolerance does not define the topology of several conflicting constraints or whether close features should merge. Define canonical representatives and ownership before snapping. Existing V2 guided constraints deliberately preserve exact features and freeze contradictions; raising all thresholds would silently change that model.

## Recommended migration and acceptance tests

- Add immutable geometry policy to the operation context (length tau=.001, explicit relative/numerical floor semantics); pass it to legacy and V2 requests, prepared guides, predicates, cleanup/repair and the quality checker. Prepared objects need policy identity or recomputation; avoid mixed cached tolerances.
- Centralize same_point, on_segment, radial classification, near-collinear turn, degenerate edge/element, orientation state, and conversion helpers (length->cross/area, length->edge-parameter, length->angle). Prefer tri-state sign with explicit repair for geometrically unresolved cases.
- Fix C1 first with a regression preserving exact-flat and near-flat outcomes while maintaining coverage and atomicity. Reconcile cleanup/split/triangulation preconditions; do not merely suppress the exception.
- Normalize near-axis guides once under an explicit snapping rule, detect duplicate/contradictory guides, and report conflicts with node/guide IDs and measured residuals.
- Audit final output with the same policy: node coincidence, smallest edge/altitude, positive valid elements, guide/radius residual, topology conformity, area conservation, output `minimum_area` semantics. Do not silently delete subthreshold children and create holes.
- Regression matrix: 0, one ULP, 1e-12, .0005, .000999, .001, nextafter(.001), .001001 deviations; positive/negative perturbations; same physical geometry with translation and consistent unit scaling; permutations of element and guide order; full-circle seam; tangency; nearly-axis-aligned/duplicate/conflicting guides; existing frozen boundaries; nodes near but not on shared edges.
- Include all probes C1–C7, checking both results and transactional failure (input arrays unchanged). Test query stage, canonicalization, repair, split and output quality agree. Explicitly retain numerical quality thresholds and ensure no self-intersection/fold is hidden by larger tau.

Case1 was subsequently traced to adaptive/global-grid interface coordinates, not a circular operation. The circular findings here are independently reproduced project risks.

## Existing-test check

`PYTHONPATH=src venv/bin/python -m unittest tests.unit.mesh2d.circular.test_extend tests.unit.mesh2d.circular.test_pattern_segments tests.unit.mesh2d.quality.test_checker` passed 59 tests. The new probes demonstrate policy gaps and a float discontinuity not covered by these existing tests.

`PYTHONPATH=src venv/bin/python -m unittest tests.unit.mesh2d.circular.test_imprint_v2 tests.unit.mesh2d.circular.test_imprint_v2_triangulation` also passed 74 tests (one expected invalid-cell cleanup RuntimeWarning from the existing fixture). Total selected existing checks: 133 tests passed. No repository source/test modifications were made.
