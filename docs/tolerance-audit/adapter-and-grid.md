# Tolerance audit: mesh2d adapter + rectilinear generators

Audit date: 2026-11-07. Read-only investigation; no production sources changed. Scope: all modules under `src/mesher/mesh2d/adapter`, plus `generators/rectilinear.py`, `holed_rectilinear.py`, `_holed_geometry.py`, `_holed_grid.py`. Paths below are relative to `/Users/henry/Desktop/code/mesher/`; use that absolute prefix in the consolidated report.

## Executive findings

The repository does **not** currently have a uniform physical tolerance contract of 0.001. Three incompatible meanings coexist:

1. Simple rectilinear `coordinate_tolerance=0.001`: merges input coordinate lines.
2. Holed/full `coordinate_tolerance=0.001`: rejects nearby distinct geometric boundaries, preserves input anchors/pattern coordinates exactly, and does not permit near-axis inputs.
3. Layer `axis_tolerance=0.001`: dimensionless off-axis drift divided by logical edge length; all other layer geometric matching uses a computed roundoff-scale epsilon, often ~2.84e-14 physical units.

The holed exact-axis and anchor-preservation behavior is explicitly documented; this is a contract incompatibility with the desired new guarantee, not an accidental failure to honor its current documented API. Conversely the dense threshold and simple rectilinear representability/count examples below demonstrate actual floating-point-dependent behavior under the current APIs.

## Reproducible evidence

Script: `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/adapter_probe.py`.
Command from repository root: `PYTHONPATH=src venv/bin/python /Users/henry/Desktop/code/mesher/docs/tolerance-audit/adapter_probe.py`.
Dependencies already available in the project's venv. Probes use tiny synthetic inputs except representability check (~40k cells); no input files or repo sources modified.

### L1 — P1: physical tolerance never reaches layer planning/building/repair/validation

- `src/mesher/mesh2d/adapter/layer/_planning/source.py:193-194`: `tol=max(128*eps, 4*spacing(max(abs(original)))/scale)`; no absolute input tolerance.
- Public layer APIs only expose `axis_tolerance`, `_mesher.py:172,232,367`; `_prepare` checks `axis_tol` in `[0,1)` at `source.py:152-154`, then tests drift `>axis_tol*length+tol` at `:223-224`.
- `adapter/_full.py:267-271,284-287,329-334` calls layer functions without propagating its public `coordinate_tolerance` (and layer API has nowhere to receive it). Full tol only reaches `discover_boundaries`, `validate_fronts`, and holed grid generation.
- Resulting `data['tol']` governs pattern normalization/coverage, corner contacts, dense seam mapping, source/target node lookup, coordinate pools, slot repairs, sampling, matching, final edge contact checks, and world-coordinate restoration.
- Unit square `_prepare(... axis_tolerance=.001)` returns normalized/physical tol `2.842170943040401e-14`.
- Public `plan_quad_sections` succeeds for pattern `[[-2,-1],[2,-1]]` but rejects `[[-2,-1],[2,-1+1e-10]]` as `patterns must be nonzero horizontal or vertical segments` (same for .0005 drift). Rejection originates `source.py:77-78`, stage `input_validation`.
- Source side drift .0005 with logical side length .01 fails near-axis validation, while drift .5 with length1000 passes. These are dimensionless-angle semantics, not absolute .001 semantics.
- Caller-specified target loop uses numerical tol only at `_planning/front.py:97-106`, so source and target alignment contracts also differ.

Impact: calling full meshing with .001 cannot guarantee .001 pattern matching or front/source contact behavior. Very small irrelevant geometry differences can create new seams, unsupported slots, unmatched fixed interfaces, or input errors. Do not replace all `data['tol']` consumers blindly; see migration warnings below.

### L2 — P1: dense classification changes topology after pure coordinate translation

- `src/mesher/mesh2d/adapter/layer/_planning/dense.py:40`: `norm(diff(points)) < small_element_size`, with no roundoff or physical comparison band.
- Probe: simple rectilinear source x `[0,.3,.6,.9]`, y `[0,1]`, target1; call public `plan_quad_sections(... small_element_size=.3, repair_slots=False)`.
- x offset0: lengths `[.3,.3,.30000000000000004]`; 0 dense sections / 0 dense seams.
- x offset1: lengths `[.30000000000000004,.30000000000000004,.2999999999999998]`; 2 dense sections / 2 dense seams.
- x offset10: lengths `[.3000000000000007,.29999999999999893,.3000000000000007]`; 2 dense sections / 4 dense seams.
- Equivalent geometry changes dense topology due solely to subtraction rounding. This is independent of case1 and demonstrates another concrete unstable branch.
- Decide threshold semantics for values within T of `small_element_size`; recommended classify as equal, then use a documented deterministic equality branch. Preserve nominal threshold and add error-budget tests around the boundary.

### L3 — P1 contractual mismatch: holed/full geometry rejects even 1e-10 axis drift

- `_holed_geometry.py:101-107`: exact coordinate equality for loop alignment; closure only exact at `:72-73`.
- `_holed_geometry.py:488-504`: pattern normalizer rejects any nonzero transverse component; exact tuple dedup at `:507-509`.
- `holed_rectilinear.py:468-469`: tol goes to domain validation, not pattern normalizer.
- `_full.py:91-92` invokes same exact pattern normalizer before full geometry validation.
- Public holed generator with coordinate_tolerance .001 accepts square `[[0,0],[2,0],[2,2],[0,2]]` but rejects changing top-right y by1e-10 or .0005. Similar pattern `[0,1]→[2,1+1e-10]` rejects.
- All downstream holed equality comparisons / dict keys / sets rely on the validated exact-axis invariant. Relaxing one validator without canonicalizing coordinates upstream breaks this invariant. Introduce one canonicalization stage and retain exact equality downstream where canonical IDs/supports are guaranteed.

### L4 — P1/P2 contractual mismatch: near-identical patterns create sub-tolerance sliver elements

- `_holed_geometry.py:507-509` dedups patterns by exact tuple only; grid stores coordinates exactly (`_holed_grid.py:74-100`, `holed_rectilinear.py:390-412`).
- Public holed generator on 2×2 square with target1, coordinate_tolerance .001, patterns y=1 and y=1+delta:
  - delta0: 4 cells, min edge1.
  - delta.0005: 6 cells, min edge .0004999999999999449.
  - delta1e-10: 6 cells, min edge1.000000082740371e-10.
- Boundary-separation tolerance does not extend to distinct pattern supports or consecutive collinear anchors. Current docs explicitly preserve all mandatory features; thus 0.001 does not currently mean minimum edge or geometric equivalence.
- This requires a design decision: near patterns that denote same logical constraint should canonicalize once; genuinely distinct required features closer thanT should receive an actionable conflict error (unless user explicitly permits sub-T feature mode), not produce surprise slivers. Do not discard mandatory/frozen anchors silently.

### L5 — P2: simple rectilinear merges chains farther than tolerance from retained representative

- `rectilinear.py:14-22` compares each sorted coordinate to its immediate original predecessor, not the last retained coordinate or bounded cluster representative.
- With tolerance .001, `[0,.00075,.0015]` becomes `[0]`; generator then errors needing≥2 x-lines.
- `[0,.00075,.0015,1]` becomes `[0,1]`; mandatory .0015 vanishes although .0015>.001 from its retained representative0.
- A sufficiently long chain can collapse an arbitrarily wide span. If a chained connected-component clustering policy is intended, it must be explicit; it is incompatible with a guarantee that every merged point moves≤T. Prefer a representative-distance/diameter-bounded deterministic cluster rule and preserve endpoint/domain-extent policy.

### L6 — P2: simple rectilinear subdivision count changes on one-ULP input perturbation

- `rectilinear.py:34`: raw `ceil(length / target_edge_size)`.
- target .1, x `[0,.3]`, y `[0,.1]`:3 cells.
- target .1, x `[0,.1+.2]`, y `[0,.1]`:4 cells (endpoint .30000000000000004).
- Current docs say hard maximum edge size, so the count is consistent with a literal real-value maximum; however it violates intended floating-error stability and differs from holed `_fits_size`, which already accepts a 32eps relative arithmetic margin. Add a dedicated size-comparison arithmetic policy, separate from node-equivalence T. The holed `_segment_count:63-70` also uses raw ceil for ratios>1; inspect it and adaptive `_inner_counts:63-71` for equivalent integer-boundary instability.

### L7 — P1 generic numeric robustness: simple rectilinear emits zero-length geometric edges at large origin

- `rectilinear.py:35` uses linspace without checking consecutive coordinates remain representably distinct; final assembly has no degeneracy validation.
- target1e-5, x `[1e12,1e12+.002]`, y `[0,.002]`, coordinate_tolerance .001: returns 39,597 nodes but only3,417 distinct XYZ positions, with72,000 zero-length cell edges. Same geometry at origin0 has40,401 unique coordinates and0 zero edges.
- This occurs when requested spacing is smaller than the local ULP and should fail early with numeric-resolution diagnostics, not return invalid elements. Holed `_densify_axis:140-142` already checks this specific failure.
- The example uses target<T intentionally to isolate representability; tolerance policy must explicitly state whether such targets are allowed. Even if future .001 profile disallows them, generic public API must not silently emit invalid connectivity.

## Tolerance-family inventory

| Family | Locations | Existing semantics | Required disposition |
|---|---|---|---|
| Simple coordinate merging | `generators/rectilinear.py:7,20,47,69-70` | absolute default.001; predecessor chaining | Canonical shared coordinate policy; bound max movement; distinguish mandatory anchors |
| Simple max size | `rectilinear.py:29,34-35` | exact zero / raw ceil / linspace | arithmetic stable integer decision; representability guard; physical size slack only if chosen |
| Holed boundary separation | `_holed_geometry.py:187-222,316,339` | reject nonadjacent/different boundary distance≤T; internal class default1e-9; public wrapper default.001 | central default; explicit conflict semantics; investigate default1e-9 callers (public wrappers do pass.001) |
| Holed axis/closure/duplicate keys | `_holed_geometry.py:72,101-117,488-509`; `holed_rectilinear.py:390-412`; `_holed_grid.py:74-100` | exact supplied coordinates, no snapping; retained anchors | canonicalize once at API boundary; retain exact internals after canonicalization |
| Holed max size | `_holed_grid.py:11-16,50,63-70,125-145`; `holed_rectilinear.py:418-420` |32eps relative arithmetic margin; ceil count | legitimate numeric margin; consistent count rounding; don't replace with.001 blindly |
| Holed area coverage | `holed_rectilinear.py:421-423` |relative1e-11 area | arithmetic invariant; if canonical geometry moves, compare canonical area with derived area budget, not raw lengthT |
| Full public tol propagation | `_full.py:208,229-231,257,267-287,307,329-334` | only forest/global calls receiveT | one policy object threaded into all layers/previews/repairs/final validation |
| Full initial boundary forest | `_full_geometry.py:113-150` | separationT/global scale; self-contact uses local128eps+ULP, skips same-owner near boundary edges | distinguish self-clearance policy from inter-boundary clearance; propagate physical self-feature policy if promised |
| Full source quad quality | `_full_geometry.py:195-205` | per-cell normalization,128eps length-scaled turns | numerical validity guard; add independent physical minimum-feature validation if desired |
| Full generated-front canonicalization | `_full_geometry.py:38-75,238-267` |128eps+ULP snap only numerical residue; exact section duplicate removal | preserve generated canonical support invariant; use policy from plan and ensure constrained support mapping |
| Full interface identities | `_full.py:101-115` | exact coordinates, topology and ID cycles | keep exact once global target reused; don't fuzzy merge unrelated original nodes |
| Full area/convexity/contact validator | `_full.py:160-196` |exact sign,rel1e-10 area,128eps+ULP contact tolerance | separate physical contacts from strict inversion and arithmetic coverage |
| Layer axis alignment | `_planning/source.py:152-154,223-224`; `_mesher.py:172,232,367` | dimensionless slope tolerance0.001 | rename/document angle tolerance; separate absolute alignment toleranceT |
| Layer common tol source | `_planning/source.py:180-200` | normalized128eps + origin ULP; h>32tol guard | physicalT/scale plus separately named numeric floor; guard must still use numeric epsilon |
| Pattern normalize / occupation / coverage | `_planning/source.py:44-83,256-275`; `_patterns.py:38-179`; `_validation.py:140-149` | geometry comparisons use layer numericaltol; mostly axis-world intervals | propagate physical match/constraint policy; preserve strict crossing classification |
| Front expansion / target validity | `_planning/front.py:11-35,48-106,112-159` | numericaltol for snapping, lengths, contacts; ratio+tol and area-vs-tol mixes units | use lengthT, normalized fractional toleranceT/h, and area budget separately |
| Section planning / matching | `_planning/sections.py:36-69,281-383`; `_planning/mapping.py:10-47`; `_planning/fixed_corners.py:14-108` | tol governs same point/seams, ordered chains, nearest candidates; area budget64tol×perimeter | semanticT for matching; exact canonical identities after mapping; bounded deterministic tie breaking |
| Dense classification / mapping | `_planning/dense.py:40,63-89` |strict raw world `<size`; tol matching and angular gate converted tolength | tolerance-aware classification; maintain proper dimensionless angle→distance logic |
| Boundary registries | `_registry.py:15-47,80-111`; `_construction/units.py:146-171` | normalizedtol spatial hash±1 cells, Euclidean match; original source collision rejected | central identity policy, no arbitrary unrelated-node merge, preserve bucket coverage and deterministic representatives |
| Unit chain validation / composition | `_construction/units.py:73-143,191-218`; `_construction/line.py:272-292`; `_construction/corners.py:22,103-111`; `_construction/patch.py:48` | caller numericaltol joins/extent, builders renormalize without passingtol | physical boundary validation before builder; builders maintain numerical validity contract |
| Sampling | `_construction/sampling.py:18-35,103-106,160-161,223-248`; `_construction/line.py:29,98,139` | normalizedtol in interval bounds and size; relative tol/h +32eps for count rounding; exact retained anchor identities | a good example of separating dimensional count tolerance; preserve exact copied anchors |
| Fixed outer boundary | `_fixed_boundary.py:15-27,31-47,51-66` | numericaltol coordinate correspondence then exact target IDs and exact restored world coordinates | physical matching during canonical preparation, retain exact frozen target identity afterward |
| Layer world restoration | `_mesher.py:32-44` | numerical round-trip check against commontol | retain independent numeric representability budget; avoid letting physicalT hide large reconstruction errors |
| Layer geometry predicates | `_geometry.py:37,94-218`; `_validation.py:29-114` | normalizedtol distances, point/segment membership, overlap depth; Fraction near-cancellation fallback | separate identity, proximity, strict topology; **do not** treat every shallow crossing as parallel usingT |
| Layer final coverage / convexity | `_validation.py:170-172,245-251`; `_repair/slots.py:434,478` |64eps×scale² convexity, max(8tol×perimeter,1e-10area) coverage | numerical strict quality; area-error budget derived fromlengthT if geometry canonicalized |
| Slot repair / patch | `_repair/slots.py:187-295,427-478`; `_repair/patch.py:46-92,114-188,303-312,423-434` | all physical-looking tests receive numericaltol; tol×perimeter area and tol×length turns mostly dimensional | propagate match/clearance policy; canonicalize once; protect source identities and parity |
| Adaptive repair rings | `_repair/adaptive.py:19-57,63-71,229-245` | tol×length corner turns;64eps normal determinant; slope<-tol; raw ceil; max_size relative64eps | keep dimensionless determinant epsilon; fix slope-vs-length semantics before T propagation; stable counts |
| Unit111x numeric checks | `unit/build_unit_1_1_1_x.py:25-55,90-120,137-147` |64eps×scale length and64eps×scale² area; topological IDs exact | legitimate numeric validity, but physical geometry policy must run upstream; area threshold based whole unit can reject tiny cell in huge unit, requires scale stress tests |
| Unit11xy numeric checks | `unit/build_unit_1_1_x_y.py:112,188-192,211-239` |128eps candidate margin; max(1e-10area,512eps×scale²×count) coverage;64eps intersection | do not replace numerical solver/area epsilon with.001; consistent upstream geometry contract |
| Unit1xyz visibility kernel | `unit/build_unit_1_x_y_z.py:190-222` | normalize unit;128eps area/clearance guard | numerical viability; separate physical clearance threshold and explicit unsupported error if required |

## Important exclusions: smaller than0.001 is not itself a defect

- `np.finfo(float).eps` in near-zero determinant uncertainty / exact Fraction fallback (`_geometry.py:105`, `_validation.py:78`) measures floating-point arithmetic uncertainty, not allowed physical point displacement. Replacing it with0.001 can classify real shallow crossings as parallel and permit invalid topology.
- Dimensionless solver determinant checks, normalized scaled-Jacobian positivity, angular gates and `rel_tol` area summation are not length tolerances.
- Exact integer node-ID/topology/parity comparisons must remain exact.
- Exact source-array preservation (`_validation.py:159`, `_full.py:154`) and exact canonical fixed interface IDs are essential invariants, not mistakes.
- Holed dictionary/set equality and exact canonical axes can remain exact after robust input canonicalization; changing all equals comparisons to isclose breaks key equivalence and transitivity.
- Narrow feature / zero-edge rejection should distinguish physical ambiguity from numeric unrepresentability. The user's T does not intrinsically define minimum element edge size, minimum area, minimum Jacobian or maximum size slack unless those semantics are deliberately adopted.

## Migration hazards if simply changing `data['tol']` to0.001/scale

1. `_prepare:197-200` would start rejecting any target size≤32*T (≤0.032), although it currently guards arithmetic resolution. Must keep a separate numeric_tol.
2. `_planning/front.py:31` compares dimensionless nearest/expand_h to ratio+tol; correct physical slack must be T/expand_h or compare lengths before division.
3. `_planning/source.py:235`, `_planning/front.py:59,92` compare areas to lengthtol; replace with a dimensionally correct budget (T×relevant perimeter plus numeric area error), with explicit policy for degenerate shapes.
4. `_repair/adaptive.py:57` compares a dimensionless inset velocity/slope to lengthtol; use dimensionless uncertainty.
5. `_cell_overlap` and `convex_overlap` ignore overlaps shallower thantol; if tol becomesT this could hide real mesh overlap. A closeness classification must lead to canonical shared topology or a clear conflict, not leave overlapping cells silently valid.
6. `_NodePool`/`CoordinatePool` start merging nodes withinT. Without provenance-aware identity policy and canonicalization-before-building, this can collapse fixed boundary segments, change parity, violate frozen-node IDs, or create repeated-node quads.
7. Unit templates independently apply machine tolerances after renormalization, so a relaxed input acceptance test can still fail later. Need canonical geometry passed through and explicit preflight validation, not merely loose downstream predicates.

## Recommended phases and interfaces

A. Freeze evidence: add case1 regression plus these synthetic examples; establish existing branch/test baseline. Decide world units and defaultT=.001. Document equivalence/movement, boundary clearance, size threshold, preserved anchors and genuine sub-T feature policy separately.

B. Introduce immutable `GeometryTolerance`/policy at public entry points and pass through a `GeometryContext` with origin/scale. Suggested named values/helpers: `coordinate_atol` in world units, `numeric_length_floor`, `axis_angle`/slope independent of length, `normalized_length_tolerance=T/scale`, `within_distance`, `classify_size`, `area_error_budget(perimeter,area)`, `parametric_tolerance(length)=T/length`. Default should be declared once; don't leave internal OrthogonalDomain default1e-9. Diagnostics expose requestedT, effective numeric resolution, units/frame, decision reason.

C. Canonicalize input constraints once, before topology planning: support-line grouping, near-axis projection, pattern endpoint matching and representative-bounded coordinate clustering; maintain original-to-canonical IDs and movement<=T. Preserve originals/frozen target IDs; fail explicitly when indistinguishable distinct fixed features cannot coexist. Define whether snapping occurs as an optional preprocessing operation or is part of accepted user contract.

D. Thread policy end-to-end: simple/holed/full→single layer→repair→partition→sample→unit assembly→fixed boundary→final validation. Keep numerical epsilon separate. First fix dense threshold, simple count and representability bugs; then align semantics. Validate with transaction rollback preserved.

E. Canonical downstream invariant: intern support/anchor IDs and use exact topology comparisons after canonicalization. Reuse same canonical target in full planning/global grid/fixed final construction. Never run independent tolerant merges at each stage.

F. Regression matrix and release gate: threshold neighbors `nextafter(T,±inf)`,0,T/2,T,2T; ±1e-10 noise; translations0,1,10,1e6 and representability boundary1e12; scale/unit conversions withT transformed; both source windings/start rotations; many chained close lines; near duplicated and crossing patterns; explicit fixed IDs/parity; compare plan vs build and layer vs full; strict no inverted/overlapping/repeated-node cells; diagnostic expected errors for true sub-T conflicts. Add property tests for bounded movement, deterministic order-independent representatives, translation invariance, unchanged originals on error, no stale plans/canonical mismatches. Numeric epsilon checks need their own shallow-crossing and large-scale stress regressions.

## Confirmed real case1 diagnosis (added after the user regenerated the case)

The actual case is now present at `/Users/henry/Downloads/case1/` with `model.geometry.json`, `model.mesh-control.json` and `cdb_75eb29e6d4e14728b283ac58b659d96c.log`. The input explicitly declares `unitSystem: um`; user confirmed absolute physical T=.001. Reproduction uses original files unchanged and only observation wrappers via `unittest.mock.patch` in a temporary script.

Observed matching failure after17.59s:

- `controls[7] ('Die top corners merged 3')`, `boundary_index=2`, `region_layer_indices=3`, full final layer index0.
- `_full.py:329` final fixed interface → `_mesher.py:93` normalize outline_o → `_planning/front.py:102-106` target edge rejection.
- Source normalization scale2600μm, origin[-175,-10800], layer normalized tol2.842170943040401e-14, physical equivalent7.389644451905042e-11μm.
- Target edge9 is `[2800.0000000000005,-8200] → [2800.0,-8200]`; true world gap4.547473508864641e-13μm, normalized difference[-2.220446049250313e-16,0].
- Target edge11 is `[1800.0000000000002,-8200] → [1800.0,-8200]`; true world gap2.2737367544323206e-13μm, normalized difference[-1.1102230246251565e-16,0].
- The rejection is **`abs(along)<=tol` (zero/too-short edge), not off-axis transverse error**. Error wording combines both possibilities. Increasing tolerance at this final check will not fix the case; it continues treating these edges as degenerate.
- Other returned interfaces are already contaminated: hole3 and hole5 have gaps1.1368683772161603e-13μm, so skipping only edge9 or boundary2 exposes subsequent failures.

Proven generating path:

1. Independently planned/restored hole2 front enters global mesher with support/anchors x1800.0000000000002 and2800.0000000000005, while hole5 front uses x1800.0 and2800.0. Each individual loop is exactly orthogonal and separated by thousands of μm. This is ordinary normalization/restoration residue, not user geometry containing sub-.001 features.
2. `_full_geometry.section_outline_points` canonicalizes each loop to its own front but does not canonicalize near-equal supports **across loops**.
3. Holed `_Builder.seed` at `holed_rectilinear.py:270-272` anchors every input point. `_Builder.anchor:260-267` shoots complete visibility chords using each exact coordinate. `_Builder.add:248-257` deduplicates by exact cut key and adds exact endpoints to Python sets.
4. Instrumented isolated grid replay proves a `kind='geometry'`, axis1 vertical chord x2800.0 with interval[-8200,7700] adds target endpoint[2800.0,-8200] beside existing[2800.0000000000005,-8200]. Same for1800 and-800 across other holes.
5. Global `_assemble:390-412` preserves both points and accepts the tiny positive rectangular elements. It returns22 anchors for hole2. `_full.py:311-313,327-334` stages these as the fixed target, which the layer correctly refuses to mesh as nonzero edges.

The root fault is **inconsistent canonical coordinate identity across independent fronts and grid-generated contacts**, plus insufficient global minimum-degeneracy/interface preflight. Missing physical policy propagation is the broader architectural issue, but simply replacing a machine epsilon with.001 does not remove the near-duplicate coordinates or restore correct parity/IDs.

Temporary evidence owned by this audit:

- `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/diagnose.py`: original case pipeline with read-only diagnostic wrappers.
- `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/diagnosis.jsonl`: structured plan, grid counts, failing edge coordinates/tol/context.
- `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/traceback.log`: reproduced exception stack.
- `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/grid_input.npz`: outer,16 patterns and8 planned front input loops.
- `/tmp/audit_layer_case1_grid_0.npz`: global grid nodes/elements and returned interface loops.
- `/tmp/audit_layer_case1_failing_target.npz`: failing staged source and target snapshot.
- `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/grid_replay.py` and `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/grid_provenance.json`: isolated replay proving which geometry cuts add near-duplicate endpoints.
- `/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/minimal_probe.py`: minimal no-pattern reproducer.

Minimal regression: outer box[0,5]×[0,5]; holeA[1,nextafter(2,+∞)]×[1,2]; holeB[1,2]×[3,4]; target1; coordinate_tolerance.001. Both inputs are exact-axis, well-separated valid rectangles. Holed generator returns a holeA edge length4.440892098500626e-16, and `_outline_o_directions(...,128eps)` raises the same class/message. This should become an end-to-end canonical-interface regression, not an exemption that ignores a zero-length edge.

Additional acceptance gate for the concrete fix: run the unmodified real case through final2D validation and complete2.5D/export; no source/frozen ID changes, no duplicate geometric edges, unchanged boundary parity, no new sliver cells, all8 region interfaces consistent, and total area/volume checks pass. At audit time no production fix has been applied, so success of the complete case remains future implementation validation.
