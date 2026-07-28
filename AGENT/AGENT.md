# AGENT.md — Prioritized (Fixed-Order) Multi-Robot Planning with Per-Region Cubic Bézier Trajectories

You are implementing a research prototype in multi-robot motion planning. Read this whole file before writing code. The companion design note `reservation-geometry-design-note.md` contains the full geometric derivations behind F1-F8 (originally worked out for a continuous B-spline / rolling-horizon representation — see the note on scope below for how that maps here); `bspline-migration-plan.md` contains the concrete file-by-file integration plan against the actual `stgcs/`/`trajopt/` codebase (some of its later phases, e.g. `pbs.py`/`windowed_coordination.py` wiring, are outside this phase's scope — see below).

## 1. Project Context

**System:** prioritized multi-robot planning on a Space-Time Graph of Convex Sets (ST-GCS), with a **fixed** priority order. Robots are planned one at a time in that order, each over its **full mission horizon in a single solve**; each planned robot's trajectory is *reserved* as a spatio-temporal obstacle, subtracted (permanently) from the space-time convex-set graph via Exact Convex Decomposition (ECD) so subsequent robots' problems remain clean convex programs.

**Baseline:** ST-GCS (Tang et al., IROS 2025, arXiv:2503.00583; code at github.com/reso1/stgcs, built on Drake + Mosek). ST-GCS reserves *piecewise-linear* space-time trajectories: inflated parallelepipeds per linear segment, subtracted from affected sets producing up to ~6 pieces each.

**Our delta over the baseline:** trajectories are **cubic Bézier, one segment per GCS vertex** — a GCS vertex already corresponds to exactly one region traversal with its own private decision variables, so raising the per-vertex control-point count from 2 (today's linear segment) to 4 makes each vertex *own* a cubic Bézier segment exclusively. Reservations are convex polytopes of that segment's space-time control points, certified by the convex hull property (curve ⊂ hull of its 4 control points, continuously over the segment — no sampling). Continuity between consecutive Bézier segments (within one robot's path, across GCS edges) is enforced directly as linear constraints — see F3.

**Explicitly out of scope for this phase (deferred, not designed away):**
- **Rolling horizon.** No windowed re-solving, no partial commit, no `DecompositionStore` versioning/expiry. Each robot's trajectory is solved once, over its entire mission horizon, in one shot. Reservations, once subtracted, are permanent for the rest of this phase's runs.
- **PBS-style dynamic priority search.** The priority order is fixed (given up front, or computed once, e.g. by release time) and never revised in response to conflicts. If the fixed order makes some robot's problem infeasible, that is a planning failure for this phase — not a trigger to search over alternative orderings.
- **Multi-span-per-region.** If a single region ever needs more shape freedom than one cubic segment offers, splitting it into multiple Bézier pieces *within one GCS vertex* would reintroduce the F2 interlock problem locally (those sub-spans would share control points). Flag before attempting.

**Why per-vertex Bézier, not a continuous multi-span B-spline.** An earlier pass of this design (`reservation-geometry-design-note.md` §2-4) considered a single B-spline crossing many knot spans, with per-span reservation hulls. That representation requires a knot-insertion "Bézier extraction" step at reservation time (design note §4, "Fix A") specifically *because* a continuous spline's per-span hulls share `p` control points and interlock — see F2 below. Once the trajectory is built directly as one Bézier segment per GCS vertex, that extraction step is a no-op: **GCS vertices already are the Bézier segments.** There is nothing to extract, and no interlock to guard against between vertices, because vertices were never a sliding window over a shared control polygon — they were always edge-disjoint region traversals. This is a real architectural simplification, not a naming change; see F2-F3 for exactly what it does and doesn't buy.

**Robot model:** known convex footprints. Footprint handling = Minkowski inflation of reservation polytopes (convex ⊕ convex = convex). Optional tracking-error-bound inflation (FaSTrack-style) is a later layer — leave a hook, don't implement now.

## 2. The Core Geometric Facts You Must Respect

These are settled results from the design phase. Do not re-derive or "simplify" them away; several naive shortcuts are provably wrong. F1-F3 are stated below against *both* the continuous-spline failure mode they rule out and the per-vertex construction actually used — read F2 especially as "why we build segments this way," not as a live runtime hazard for the current representation.

**F1 — Bézier hull property (per segment).** A degree-3 Bézier segment's curve lies in conv of its 4 control points, for the entire segment, continuously in the segment's parameter. This is the certificate the reservation pipeline relies on directly — no extraction needed to obtain it, because each GCS vertex's decision variables already *are* 4 joint control points.

**F2 — The interlock theorem (why we don't use a continuous multi-span spline).** For a degree-p B-spline with a shared knot vector, consecutive per-span hulls share p control points. When the control polygon bends (generic), consecutive hulls overlap volumetrically and **no separating hyperplane exists**. Per-span hulls of a raw continuous B-spline are therefore *structurally incompatible* with ECD-style segmentation — not fixable by better cut selection. This is exactly why segments here are private to a single GCS vertex from the start rather than a sliding window over shared control points: two segments belonging to different vertices share at most one C⁰ join point, never p of them, so this failure mode cannot arise between them. It *would* reapply within a vertex if multi-span-per-region were ever added (§1).

**F3 — Segments own their control points by construction, not by extraction.** Because each GCS vertex's control points are private decision variables, consecutive segments already share only a C⁰ join point ⇒ disjoint relative interiors ⇒ separable, with no knot-insertion step required to get there. C¹ (and optionally C²) continuity across the join is imposed instead as linear equality constraints between the neighboring vertices' boundary control points on the GCS edge — invisible to the decomposition, the same effect the extraction-based scheme gets from "continuity migrates into linear relations between neighbors' control points," just derived directly instead of via extraction. This is the concrete mechanism for "continuity between Bézier curves" along one robot's path.

**F4 — Time must be a control point, not an annotation.** The space-time certificate requires t to be one of the coordinates of each joint control point Qᵢ = (Pᵢ, Tᵢ), with monotonicity Tᵢ₊₁ ≥ Tᵢ along the segment. Tagging spatial control points with scalar timestamps certifies nothing.

**F5 — Monotone-time subdivision theorem.** De Casteljau subdivision is corner-cutting ⇒ preserves monotonicity of time controls ⇒ **every** post-hoc split at any s₀ within a segment yields left hull ⊂ {t ≤ t(s₀)}, right hull ⊂ {t ≥ t(s₀)}, touching at the single point (x(s₀), t(s₀)). t(s) is invertible per segment, so cuts can be placed at arbitrary target times t*, including globally synchronized slabs t = kΔ across all robots, after solving, at zero planning cost — useful for tightening the sequential reservation independent of any rolling-horizon window.

**F6 — Clamping semantics.** Planning-time control-point collapse is used ONLY at genuine waypoint constraints — most notably a robot's known initial state (position + velocity) at the start of its trajectory, and any true intermediate/goal waypoints. (The rolling-horizon "commit boundary" use case from the design note does not apply in this phase — see §1.) If ever collapsing at a waypoint: collapse space AND time together ((X, T*) repeated); spatial-only collapse creates a vertical hull segment with no time-normal cut, and with regular time forces dx/dt = 0 (a full stop). With joint collapse the limit velocity x″/t″ is finite and the velocity certificate passes to the limit. Interior reservation cuts within a segment are ALWAYS post-hoc splits (F5), never plan-time clamps.

**F7 — Dimensionality.** In 3D-space+time (4 ambient dims), a cubic segment's 4 control points are generically rank-deficient. Minkowski inflation is structurally required before subtraction. In 2D+time you get full-dimensional tetrahedra generically, but inflate anyway (footprint).

**F8 — Velocity certificate.** ‖x′(s)‖ ≤ v_max · t′(s) enforced per segment via second-order cone constraints on consecutive control-point differences (Bézier derivative controls are scaled consecutive differences of the degree-3 controls), linear in Tᵢ for the time side.

## 3. Trajectory Representation (implement exactly this)

```
CubicBezierPerVertex:
  degree p = 3, 4 control points per GCS vertex
  d = spatial dimension (start with 2; keep d generic)
  Per GCS vertex v (one region traversal):
    P: (4, d) spatial control points — decision variables local to v
    T: (4,)   time control points    — decision variables local to v
    joint control points: Q_i = (P_i, T_i) ∈ R^{d+1}, i = 0..3
  No shared knot vector across vertices — the graph structure (GCS vertices + edges)
  *is* the segmentation; there is no separate "spline" object spanning multiple regions.
```

**Per robot, solved once over its full mission horizon (no rolling window, no partial commit):**
- First vertex on the path: full control-point collapse at the trajectory start — first control point Q₀ = the robot's known initial state (position + velocity, per F6). T₀ = the robot's release time, fixed.
- Monotonicity Tᵢ₊₁ ≥ Tᵢ within each vertex; no artificial upper time bound beyond whatever the mission/goal constraint imposes.
- GCS containment: per vertex, all 4 (Pᵢ, Tᵢ) constrained inside that vertex's space-time region (H-rep linear constraints) — the existing per-vertex containment constraint, just applied to 4 points instead of 2. For a robot later in priority order, its regions already reflect the ECD subtraction of every higher-priority robot's reservation — this is what makes the planning "prioritized."
- Velocity: per vertex, SOC constraints ‖ΔP-based derivative controls‖ ≤ v_max · (ΔT-based derivative controls), one constraint per consecutive control-point pair (3 per cubic vertex).
- Edge continuity: C⁰ (and C¹) equalities between the tail vertex's last control point and the head vertex's first control point, per F3 — this is what makes consecutive Bézier segments join smoothly along one robot's path.

**Reservation pipeline (after each robot's full trajectory is solved, before planning the next robot in priority order), in order:**
1. ~~`extract_bezier`~~ — **not needed.** Each GCS vertex's control points already form a private Bézier segment (§1); this step, required for a continuous multi-span spline, is a no-op here.
2. `split(segments, policy) -> list[BezierSegment]` — post-hoc de Casteljau splits, still needed for sub-vertex resolution. Policies: `k1` (none — whole-vertex hull, one per region traversal; already what unmodified per-vertex reservation gives, since a vertex boundary already sits on a region boundary), `time_slabs(Δ)` (invert t(s) per segment to hit t = kΔ), `max_chord_deviation(k)`, `region_crossings` (degenerates to `k1` here, since segment boundaries already align with region boundaries by construction — this policy only does new work for a future multi-span-per-region extension). All policies must preserve exactness.
3. Optional `to_minvo(segment)` — fixed degree-3 change-of-basis matrix. NOTE: this is a change of basis, NOT reparametrization; MINVO controls pair only with MINVO basis functions and are not valid Bézier controls. MINVO simplices and Bézier hulls are not nested — compare, don't assume.
4. `inflate(hull, footprint, margin) -> HPolytope` — Minkowski sum. Use V-rep sum then H-rep conversion (cddlib/pycddlib or scipy.spatial + qhull), or direct H-rep support-function offset for polytopic footprints.
5. `subtract(region, obstacle) -> list[HPolytope]` — ECD-style: for each facet hyperplane of the obstacle clipped to the region, generate the piece on the far side; handle shallow join overlaps (from inflation) by subtracting the local union of adjacent inflated pieces within a region rather than sequentially. The result is written directly and permanently into the space-time region graph — no `DecompositionStore`, no versioning or expiry, since there is no rolling window to expire against in this phase (§1). This matches the baseline ST-GCS's `ecd.reserve()` semantics.

## 4. Suggested Stack & Layout

- Python 3.11+. `numpy`, `scipy` (Bézier machinery: De Casteljau evaluation/split — no knot insertion needed; qhull via `scipy.spatial`), `cvxpy` + a MISOCP-capable solver (Mosek if licensed; else SCIP/HiGHS through cvxpy, or solve the convex relaxation + rounding as ST-GCS does), `networkx` if a discrete path search over regions is needed to seed the GCS vertex sequence, `matplotlib` for space-time visualization. If integrating with the ST-GCS codebase (see `bspline-migration-plan.md` for the concrete file-by-file plan), Drake's `GcsTrajectoryOptimization` is a reference API for the transcription, but it differs from the scheme here: Drake treats duration as a separate rescaling variable, which makes duration-continuity bilinear; here time is a coordinate in the same ambient space as position, so edge continuity stays linear (F3, design note §4). Build the geometry layer standalone and solver-agnostic first.

```
src/
  bezier/        # per-segment control points, de casteljau eval/split, derivatives, t-inversion
  hulls/         # segment hulls, MINVO transform, inflation
  decomposition/ # HPolytope, ECD subtraction (permanent — no versioned store this phase)
  gcs/           # region graph, per-vertex 4-control-point transcription, edge continuity
  planner/       # fixed-priority-order loop: plan robot, reserve (permanent), next robot
tests/
benchmarks/
```

## 5. Implementation Order

1. **bezier/**: per-segment control points; evaluation; derivative segments; `decasteljau_split`; `invert_time(t*) -> s₀` (monotone ⇒ bisection per segment). No extraction step (F3). Tests first (Section 6, T1, T3).
2. **hulls/**: segment hulls (V-rep); MINVO matrix (hardcode the fixed degree-3 matrix from Tordesillas & How, arXiv:2010.10726, Section on A_MV); Minkowski inflation. Tests T2, T4-T5.
3. **decomposition/**: HPolytope with both reps; `subtract`, writing permanently into the region graph (no versioning this phase). Tests T7.
4. **gcs/**: minimal single-robot solve on a hand-built 2D+time region graph (a corridor map with 2-4 rectangular rooms, time-extruded), full mission horizon in one solve, 4 control points per vertex, C⁰/C¹ edge constraints. Fixed discrete vertex sequence first (skip MICP — just the convex program for a given vertex sequence); MICP/relaxation after. Verify: a bent 2-region corridor produces a curved, C¹-continuous path with no velocity discontinuity at the region boundary.
5. **planner/**: N-robot (start with 2) **fixed-priority-order** loop: plan robot 1 over its full mission, run the reservation pipeline (permanent subtraction), verify robot 2's program sees the fragmented sets, plan robot 2. No PBS search over orderings, no rolling-horizon loop (§1).
6. **benchmarks/** (Section 7) only after 1-5 are green.

## 6. Verification Tests (write these; they encode the theory)

- **T1 (hull certificate):** dense-sample x(s), t(s) over each vertex's segment; assert every sample inside that segment's control-point hull (tolerance 1e-9). Random control points, both 2D+t and 3D+t.
- **T2 (separability / non-interlock):** two parts.
  (a) *Historical guard for F2:* construct a bending control polygon for a continuous multi-span B-spline (not this codebase's representation); assert consecutive raw span hulls volumetrically intersect (interior point in both). Documents why we do not use a continuous spline and guards against anyone "optimizing" the pipeline back toward shared control points across vertices.
  (b) *Positive check for F3:* build two adjacent GCS-vertex Bézier segments (private control points, sharing only the boundary joint control point) with a bent control polygon; assert their hulls have disjoint relative interiors and are separable by a hyperplane through the shared point.
- **T3 (monotone subdivision / F5):** random monotone T within a segment; split at random s₀ and at inverted target times; assert left-piece time controls ≤ t(s₀) ≤ right-piece time controls, shared endpoint equality, and curve exactness after split.
- **T4 (MINVO):** basis-change round trip reproduces the curve when paired with MINVO basis functions; MINVO simplex contains dense curve samples; include a case where the MINVO simplex does NOT contain the Bézier hull and vice versa (non-nesting).
- **T5 (inflation):** inflated hull ⊇ hull ⊕ footprint on sampled sums; full-dimensionality check in 4D (rank of face-spanning set).
- **T6 (waypoint clamp / F6):** build a segment with joint (X,T*) collapse; assert curve passes through (X,T*); assert dx/dt limit ≈ x″/t″ and ≤ v_max; build the spatial-only collapse and assert the vertical-segment pathology (both adjacent hulls contain {X}×[Tᵢ,Tᵢ₊₂]).
- **T7 (subtraction soundness):** dense-sample the reserved trajectory ⊕ footprint; assert no sample lies in any post-subtraction free set. Completeness spot-check: points just outside the inflated hull remain covered.

## 7. Benchmarks (drive the open research questions)

- **B1:** set-count growth: per-vertex hulls (`k1`, one per region traversal — the default here) vs a merged hull across several consecutive vertices of one robot's path vs `time_slabs` policies, as a function of #robots. (The design note's original degree-2/3/5 sweep for overlap-depth scaling per F2 does not apply in this architecture — there is no cross-vertex control-point sharing to have overlap depth in the first place; that axis only becomes relevant again if multi-span-per-region is added.)
- **B2:** hull slack vs merged-hull span (arc length or vertex count) at fixed curvature (expect ~span²); corridor-following throughput (two robots, fixed priority order, same corridor, same interval) under `k1` (per-vertex) vs merged/coarser hulls — the case where coarser hulls forbid following that per-vertex hulls permit.
- **B3:** MINVO vs Bézier hulls: volume AND fixed-time cross-section area (volume minimization ≠ per-time-slice tightness — open question, measure both).
- **B4:** simplices vs boxes as subtracted obstacles: ECD piece counts (open question from prior work).
- **B5:** subtractions-per-region vs regions-touched for `k1` vs merged-hull granularity.
- **B6:** waypoint clamp cost: solution quality, single-point facet passage vs free facet crossing.

## 8. Pitfalls (each of these was a wrong turn during design — do not repeat)

1. Do not build a continuous multi-span spline with shared control points across GCS vertices (F2) — this is exactly why segments are private to a vertex from the start. If a future extension needs multi-span-per-region, extraction (design note §4) becomes necessary again inside that vertex.
2. Do not timestamp spatial control points instead of giving each control point its own time coordinate Tᵢ (F4). No certificate.
3. Do not clamp spatially without clamping time at waypoints (F6): no time-normal cut + forced stop.
4. Do not treat MINVO as a reparametrization or feed MINVO controls back as Bézier controls (change of basis only).
5. Do not assume MINVO simplex ⊂ Bézier hull or vice versa (non-nested; switching reservation basis can close previously traversable gaps).
6. Do not implement interior reservation cuts within a segment as plan-time clamps — post-hoc splits are strictly dominant (free, exact, tighter, no DOF cost).
7. Do not skip Minkowski inflation in 3D+t even for point robots "temporarily" — the raw hull is measure-zero there (F7).
8. Sequential subtraction of overlapping obstacles is *sound* (subtraction is associative) but fragments badly; batch or union locally where pieces overlap (inflated joins).
9. Global hull of ALL control points over a robot's whole mission is valid but useless — never reserve it; reserve per-vertex (or per split-policy piece).
10. Do not add multi-span-per-region as a quick shortcut for extra shape freedom — it reintroduces F2's interlock locally within a vertex; treat it as an out-of-scope extension requiring re-derivation, not a mechanical one (§1).
11. Do not implement PBS-style dynamic priority search (revising the order in response to detected conflicts) for this phase — the priority order is fixed; treat infeasibility under the fixed order as a planning failure, not a trigger to search over orderings (§1).
12. Do not build rolling-horizon commit/expire machinery (`DecompositionStore` versioning, window advance) for this phase — each robot solves once over its full mission horizon; reservations are permanent. Revisit only if/when rolling horizon is reintroduced (§1).

## 9. Definition of Done (phase 1)

N robots (start with 2), 2D corridor-with-junction map, each planned once in a **fixed** priority order (no PBS search, no rolling horizon): all T-tests green; robot 2's GCS program is correctly restricted by robot 1's permanently-subtracted reservation (T7); a space-time plot showing robot 2's trajectory threading behind robot 1's reservation under a coarser split policy (e.g. `time_slabs`) but blocked under per-vertex `k1` in the corridor-following scenario (B2's qualitative core, minus the window-length axis).
