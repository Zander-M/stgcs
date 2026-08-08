---
tags:
  - gcs
  - reservation-geometry
  - search-based
  - implementation-plan
---
# Implementation Plan: Per-Region Cubic Bézier Trajectories in *This* Search-Based STGCS Repo

> **Status (updated 2026-08-05):** Steps 1-5 below are implemented in `stgcs/` (branch
> `bezier`) — `bezier.py`, `hulls.py`, `spline_trajectory.py` exist; `graph.py`'s `STGCS` is
> `order`-aware end to end (constructor, `copy()`, `_reusable_gcs_cache` signature, L2/SOC
> velocity constraint, boundary/interior edge-constraint split, and — beyond this plan's
> original scope — C² continuity gated to `order>=4` and opt-in `energy_weight`/
> `uniform_time` regularization, see `AGENT.md` F3/F9); `gcs_solver.py`'s `solve` and
> `solve_convex_restriction` both branch on `order` to decode `STTrajectory` vs
> `STSplineTrajectory`; `ecd.py` has the full `order>2` reservation path
> (`_segment_ecd_pair`, `generate_all_ECD_pairs_spline`, `reserve_spline`,
> `bezier_segment_hull`, `bezier_decision_boundaries`, `reserve_region_bezier`). Steps 6-8
> are **not** done: `dominance_check.py`'s three raw-indexing sites (§3.1) are unfixed,
> `heuristics.py:619`'s `InterfaceToSetCostTableHeuristic` still hardcodes
> `make_Cartesian_power_hpoly(interface, 2)` and an unconditional `STTrajectory` (Step 7),
> and there is still no `tests/` directory anywhere in this repo (Step 8) — none of T1-T7 nor
> the search-specific acceptance test exist yet. See the per-step status notes in §2 and the
> updated §4 for what's left.
>
> Companion to `AGENT.md` (geometric theory, F1-F9) and
> `porting-bezier-to-search-based-stgcs.md` (a *speculative* port plan written without the
> target repo in hand — "I do not have the target repo in front of me"). This document
> replaces that speculation with concrete file/line references from `stgcs/*.py` and
> `stgcs/bfs/*.py`, and surfaces several non-trivial issues the speculative plan could not
> have seen.
>
> **Resolved decisions (fold into every section below):** (1) new code stays inside the
> existing `stgcs/` package — extend `graph.py`/`gcs_solver.py`/`ecd.py`/`trajectory.py` in
> place; only genuinely new types (`bezier.py`, `hulls.py`, `spline_trajectory.py`) get new
> files, under `stgcs/`, not a parallel `src/` tree. (2) §3.2's velocity-model question is
> resolved to the **L2/SOC** form (`AGENT.md` F8) — it's the constraint that's actually
> compatible with a Bézier segment's derivative-control certificate, not an arbitrary choice;
> `order=2` keeps its existing L∞ box unchanged, `order>2` gets the SOC constraint. (3)
> Dominance-check work (§3.5 of the porting doc, and the cone-based checks in this repo's
> `dominance_check.py`) is **descoped for this pass** — run the search with a plain
> duration-based dominance check only (this repo's `AStarDominanceCheck`, unmodified); see
> the updated §2 Step 6 and §3.1 below for what that does and doesn't let you skip.

---

## 0. What this repo already is

The porting doc guessed at "the shape a search-based ST-GCS implementation almost
certainly has." That guess is correct, and the concrete pieces are:

| Assumed-shape item (porting doc §1) | Actual file:line in this repo |
|---|---|
| Region graph (`HPolyhedron` vertices, `STVertex`) | `stgcs/graph.py:30-70` (`STVertex`), `:72-` (`STGCS`) |
| Discrete search (A*/best-first) | `stgcs/bfs/best_first_search.py:378-639` (`SearchAlgorithm`) |
| "Evaluate this candidate path" primitive | `stgcs/gcs_solver.py:296-324` (`solve_convex_restriction`) |
| Trajectory decode/eval type | `stgcs/trajectory.py` (`STTrajectory`) |
| Dominance/pruning checks | `stgcs/bfs/dominance_check.py` (5 concrete checks) |

This repo also has machinery the porting doc did not anticipate because it is specific
to this codebase, not generic to "a search-based GCS solver": a heuristic library
(`stgcs/bfs/heuristics.py`: `h_mot`, `h_tri`, `h_tab`), a whole-graph MICP baseline
planner living side-by-side with the search planner (`stgcs/st_planner.py`:
`MICPPlanner` vs `SearchPlanner`, both built on the *same* `STGCS.get_gcs_instance`),
and an already-built multi-robot layer (`stgcs/pbs.py`, `stgcs/windowed_coordination.py`,
`stgcs/mrmp_planner.py`) that consumes `STTrajectory` and the `order=2` `ecd.reserve()`
path today. None of that multi-robot layer is in scope for this phase (§1), but it exists
and its assumptions about `STTrajectory` need to be respected, not just the search core's.

**Reference material already staged, not yet wired in:**
- `AGENT/stgcs.py`, `AGENT/ecd.py` — a *finished* reference implementation of the
  `order`-generic Bézier path, from the sibling relaxation+rounding repo
  (`stgcs-devel`) the porting doc calls "this repo." Diffing them against
  `stgcs/graph.py` / `stgcs/ecd.py` here shows exactly what's missing (§2 below).
- `notebooks/ecd_bezier_visualization.ipynb` — an untracked, standalone scratch
  notebook already walking through the single-region ECD primitives
  (`bezier_segment_hull`, `bezier_decision_boundaries`, `reserve_region_bezier`)
  against a copied `ecd.py`. It is *not* wired into `stgcs/ecd.py` and doesn't
  demonstrate anything at the graph/search level — useful as a scratch reference,
  not a starting point to build from directly.

Confirmed by diff: `stgcs/graph.py` and `stgcs/ecd.py` here are exactly the
`order==2`-only code paths that `AGENT/stgcs.py` / `AGENT/ecd.py` keep alive
side-by-side with their new `order>2` branches — i.e. this repo is genuinely at the
"before" state the porting doc describes, not further along.

---

## 1. Scope (inherited from `AGENT.md` §1 — do not relitigate here)

Same non-goals apply: no rolling horizon / windowed re-solve changes, no PBS dynamic
reordering, no multi-span-per-region, no MINVO basis change. (C² continuity was listed
as a non-goal in the original draft of this section; it has since been implemented,
gated to `order>=4`, per `AGENT.md` F3 — updated 2026-08-05.) The
existing `stgcs/pbs.py` / `windowed_coordination.py` / `mrmp_planner.py` stay on
`order=2` this phase; touch them only enough that they keep working unmodified (they
should — see §4.6).

---

## 2. Implementation steps

### Step 1 — Tier-1 geometry, standalone (`stgcs/bezier.py`, `stgcs/hulls.py`, `stgcs/spline_trajectory.py`) — **done**

Port `AGENT`'s reference `bezier.py`/`hulls.py`/`spline_trajectory.py` (named in the
porting doc §2, not present verbatim in `AGENT/` here but fully specified) against
this repo's actual utilities:

- `de_casteljau_eval`, `de_casteljau_split`, `invert_time`: pure numpy, no repo
  dependency. Copy/write directly.
- Hull construction must use **this repo's** `stgcs.geometry_utils.make_hpolytope`
  (`geometry_utils.py:78-84`, a thin `scipy.spatial.ConvexHull` wrapper) — confirmed
  present and API-compatible with what the porting doc assumed.
- `inflate_hull` must call `.ReduceInequalities()` on the summed hull, per the
  porting doc's explicit warning (§2 table) — `ConvexHull` triangulates planar facets
  into simplices and roughly doubles row count with true duplicates otherwise. This
  repo doesn't have `remove_hpoly_redundancies`'s cdd-based canonicalization wired
  into the Bézier path yet; decide which one `inflate_hull` uses (cdd canonicalize
  vs Drake's `ReduceInequalities()` — the reference implementation uses the latter
  for hulls specifically, `remove_hpoly_redundancies`/cdd for graph vertices; keep
  that split, don't unify them).
- `STSplineTrajectory` needs `.xA(idx)`/`.xB(idx)` parity helpers matching
  `STTrajectory`'s (see §3.1 — this is not optional polish, it's required for every
  caller in `stgcs/bfs/*.py` to keep working).

Verify against `AGENT.md` T1 (hull certificate), T3 (monotone subdivision), T5
(inflation) before touching anything solver-shaped. **Not done: no such verification
exists yet — no `tests/` directory anywhere in this repo (see Step 8).**

### Step 2 — Order-aware `STGCS` (`stgcs/graph.py`) — **done**

*(Line numbers below are as of this doc's original drafting, before Step 2 landed;
kept for audit trail, not current — read `stgcs/graph.py` directly for today's
line numbers.)* Port the `order` parameter and the `order > 2` branch of `_init_constraints_costs`
from `AGENT/stgcs.py:78-88, 417-502` into `stgcs/graph.py`. Concretely:

- Add `order:int=2` to `STGCS.__init__` (`graph.py:75-82`), threaded through `copy()`
  (`:115-131`).
- Replace every hardcoded `make_Cartesian_power_hpoly(x, 2)` call in `graph.py`
  (`:258, 267, 270, 277, 336, 354, 358, 360, 376, 382, 386`) with
  `make_Cartesian_power_hpoly(x, self.order)`.
- Add the `order > 2` branch to `_init_constraints_costs` (`graph.py:395-422`) —
  monotone-time, velocity, and C⁰/C¹ edge constraints per `AGENT.md` F4/F5/F8 and
  `AGENT/stgcs.py:447-502`. **Velocity is resolved (§3.2): port `AGENT/stgcs.py:478-486`'s
  L2/SOC `LorentzConeConstraint` verbatim for `order>2`**, not this repo's `order==2` L∞
  box (`A_vmax`/`A_vmin`, `graph.py:400-410`) — F8's certificate is inherently isotropic,
  so the box is the wrong shape to carry over, not just a different valid encoding of the
  same limit. `order==2` keeps its existing box unchanged; the two `order` branches
  deliberately use different constraint shapes for the same `vlimit`, and that's correct,
  not an inconsistency to reconcile.
- Boundary-vs-interior edge constraint split (`AGENT/stgcs.py:317-327`,
  `edge_constraints_boundary` vs `edge_constraints`) doesn't exist in this repo's
  `graph.py` at all today (`:296-305` — every edge gets the same
  `self.edge_constraints`) because C¹ was never a concern for `order==2`. This needs
  adding as new machinery, not a straight port, since `order==2`'s edge constraint
  list must stay exactly what it is today (C⁰ only, no behavior change) while
  `order>2` edges need the boundary/interior split.

### Step 3 — Fix the class-level GCS cache signature (`graph.py:76, 307-370`) — **done**

`_reusable_gcs_cache` is a **class variable**, shared across every `STGCS` instance
regardless of `order`, keyed by `_base_gcs_signature()` (`graph.py:307-319`), which
does not include `order`. See §4.1 — this must be fixed as part of Step 2, not
deferred, or two `STGCS` instances of different order can silently share a
wrong-`CartesianPower` cached `GCS`. `_base_gcs_signature` now returns
`(self.order, planning_vertices, planning_edges)`, with `self.order` first in the
tuple and an inline comment explaining why (`graph.py:332-349`).

### Step 4 — Order-aware decode in `gcs_solver.py` — **done**

`solve_convex_restriction` (`gcs_solver.py:296-324`) and `solve` (`:250-293`)
currently hardcode the `order==2` decode: `space_dim = points[0].shape[0] // 2 - 1`
and construct `STTrajectory` unconditionally. Branch on `order` (available from the
`STGCS` instance, or pass it through explicitly) and construct `STSplineTrajectory`
for `order > 2`, per the porting doc §3 "Decode" — this repo's `solver.py` (i.e.
`gcs_solver.py`) is, as the porting doc says, "already dual-path" in spirit even
though today it only has the one path.

### Step 5 — ECD reservation pipeline (`stgcs/ecd.py`) — **done**

Port `AGENT/ecd.py`'s `order>2` section (`:333-539`: `_segment_ecd_pair`,
`generate_all_ECD_pairs_spline`, `reserve_spline`, `bezier_segment_hull`,
`bezier_decision_boundaries`, `reserve_region_bezier`) near-verbatim. `_apply_ecd_pairs`,
`update_edge`, `slice`, `time_cropping_bot_top`/`time_cropping_mid` are already
representation-agnostic in this repo's `ecd.py` (confirmed identical to
`AGENT/ecd.py`'s copies of the same functions in the diff) — no changes needed there.
Confirmed: this repo's multi-robot layer already uses the `safe_radius = 2 *
robot_radius` convention (`pbs.py:284,474`, `windowed_coordination.py:169,343`,
`mrmp_planner.py:101`) that `reserve_spline` assumes, so the reservation margin
convention carries over with no adapter needed.

### Step 6 — Audit the `STTrajectory` duck-typing surface actually exercised this pass — **still open**

Not done: the three raw-indexing sites below are still unfixed in `dominance_check.py`
as of 2026-08-05 (confirmed by grep: `:307, 462, 500`, unchanged). Nothing in the
current code *prevents* `PositionBasedDominanceCheck`/`ArrivalStateContainmentDominanceCheck`/
`restrict_final_position` from being wired into a `SearchAlgorithm` run against an
`order>2` `STGCS` — the "run with `AStarDominanceCheck` only" discipline below is a
caller convention enforced by nobody, not a guard in the code. Narrower than it would
otherwise be, because dominance-check work is descoped (see
§3.1): with the search configured to use only `AStarDominanceCheck` — which touches
`n.sol.duration`/`n.g` only, never `.points[...]` (`dominance_check.py:31-67`) — none
of the three raw-indexing sites in `dominance_check.py` (`:307, 462, 500`, inside
`PositionBasedDominanceCheck`/`ArrivalStateContainmentDominanceCheck`/
`restrict_final_position`) are on the code path at all. Confirm `best_first_search.py`'s
own accesses (`.xT`, `.duration`, `SearchTraceNode._trajectory_points`'s
`node.sol.points`) are accessor-based or tolerant of `STSplineTrajectory`'s shape —
they already are, per §3.1 — and leave `dominance_check.py`'s other checks alone
entirely (untouched, unused, not audited) until dominance-check work is picked back
up later.

### Step 7 — Heuristic updates (`stgcs/bfs/heuristics.py`) — **partially done**

- `MotionOnlyHeuristic`/`ZeroHeuristic`: no change needed. `h_mot`'s bound
  (`find_min_travel_time_state_to_spatial_point`, `geometry_utils.py:139-197`) is an
  L∞-distance-over-`vlimit` lower bound on travel time between two *states* —
  independent of segment shape, still admissible for Bézier segments under the same
  velocity limit (see §4.5).
- `TripletRelaxationHeuristic._calc_lower_bound_cost` (`heuristics.py:360-366`) calls
  `solve_convex_restriction` generically — becomes order-correct automatically once
  Step 4 lands, no direct change needed.
- `InterfaceToSetCostTableHeuristic._solve_convex_restriction_from_interface`
  (`heuristics.py:594-643`) hardcodes `make_Cartesian_power_hpoly(interface, 2)`
  (line 619) and an `STTrajectory` construction (line 641) — **these do need direct
  fixes**, order-aware, unlike the triplet heuristic. Confirmed still unfixed as of
  2026-08-05 (both lines unchanged); this is the one concrete remaining code fix
  from this step.

### Step 8 — Tests — **not started**

No `tests/` directory exists in this repo yet. Create one and port `AGENT.md`'s T1-T7
against this repo's actual `HPolyhedron`/`Interval`/`STGCS` types (not a standalone
harness) — particularly T2(b), T3, T7 need the real `stgcs/graph.py` and `stgcs/ecd.py`
after Steps 2 and 5 land. Add a search-specific acceptance test beyond the T-series:
a 2-3-region bent corridor, confirm `SearchAlgorithm.run` returns a C¹-continuous
`STSplineTrajectory` and that `solve_convex_restriction`'s `order==2` output is
bit-for-bit unchanged on a collinear-regions case (regression guard for Step 2/4).

---

## 3. Challenges without a mechanical/trivial solution

### 3.1 `STTrajectory`'s array layout is baked into caller code, not just its own methods

**Descoped for this pass, but read this before assuming Step 6 is a no-op.** The three
raw-indexing sites below live entirely inside checks that are out of scope now
(`PositionBasedDominanceCheck`, `ArrivalStateContainmentDominanceCheck`,
`restrict_final_position`) — running the search with `AStarDominanceCheck` only avoids
all three. The finding is retained here so it isn't rediscovered the hard way when
dominance-check work resumes: fix these three sites *before* enabling any dominance
check beyond `AStarDominanceCheck` for `order>2`, not after something silently mispruned.


`STTrajectory.points` is a list of flat `(2*(dim+1),)` vectors — `[xA, xB]`
concatenated per vertex (`trajectory.py:10-14`, `.xA`/`.xB` at `:121-125` just slice
`self.dim` in). `STSplineTrajectory` (per the porting doc's description) stores
`(order, dim+1)` control-point arrays per vertex — a genuinely different shape, not
a special case of the same one. `.xA(idx)`/`.xB(idx)` can be made to mean the same
thing on both types (entry/exit point of segment `idx`), but **`.points[idx]` cannot**
— there is no reshaping that makes a flat 2-point vector and an `(order, dim+1)`
matrix interchangeable under raw indexing.

This matters because callers don't consistently go through the accessors. Confirmed
by grep, three sites bypass them entirely:

- `dominance_check.py:307` — `p = n.sol.points[-1][n.sol.dim:]`
- `dominance_check.py:462` — same pattern, inside `PositionBasedDominanceCheck.check`
- `dominance_check.py:500` — same pattern, inside `restrict_final_position`

Each of these assumes `points[-1]` is a flat `2*(dim+1)` vector and slices `[dim:]`
to get the exit point. Under `STSplineTrajectory`, `points[-1]` is an `(order,
dim+1)` array — `[dim:]` on it either raises (shape mismatch on a 2D array slice
along axis 0 when `dim` isn't a valid row count) or, worse, silently returns some
wrong sub-array rather than the exit point, depending on exact shapes. This is
exactly the kind of thing that will not show up as a crash during development on a
toy example, only as silently-wrong dominance pruning later. **Fix: replace all
three with `n.sol.xB(-1)` (or equivalent), and grep for `.points[` across
`stgcs/bfs/` and `stgcs/*.py` once more after Step 6 to confirm nothing was missed**
— `best_first_search.py`'s own uses (`.xT`, `.duration`) are already accessor-based
and need no change; `windowed_coordination.py:468-469`'s `.xA(j)[-1] += t_offset`
is accessor-based too, but relies on `.xA()` returning a *view*, not a copy, into
the backing array (true for `STTrajectory` today because `self.points[idx][:self.dim]`
is a numpy view) — if `STSplineTrajectory.xA()` is ever asked to support this same
in-place-mutation idiom (only relevant if rolling-horizon/windowing is revisited,
out of scope now per §1), it needs to preserve that view semantics too, not just
the same return value.

### 3.2 The velocity-set shape used by `order==2` today doesn't match `AGENT.md`'s F8 — resolved

This repo's existing `order==2` velocity constraint is an **L∞ box**: independent
per-axis bounds `-vlimit <= dP_d/dT <= vlimit` for each axis `d`
(`graph.py:400-410`, `A_vmax`/`A_vmin`). `AGENT.md` F8 and `AGENT/stgcs.py:478-486`
specify an **L2 ball** (isotropic `‖ΔP‖ <= vlimit·ΔT`, via `LorentzConeConstraint`)
for `order>2`. These are different constraint sets (a box strictly contains the
inscribed ball of the same radius), so this was a real modeling decision, not a
mechanical translation step — **resolved: use the L2/SOC form for `order>2`.** F8's
derivation is inherently isotropic (a Bézier segment's derivative controls are scaled
consecutive differences of the degree-3 controls; the hull/bound argument doesn't
decompose per-axis), so the SOC constraint is the one that's actually *compatible
with the Bézier representation* — the L∞ box would under-certify true worst-case
speed between the directions it happens to check. `order==2` keeps its box
unchanged (a straight segment has no interior control points to certify between);
`order>2` gets `LorentzConeConstraint`s, which the target's Mosek-backed solver
already handles natively (`MPGCSInstance.default_solver_options`), so this adds no
new solver-capability requirement.

`stgcs.geometry_utils`, `dominance_check.py`'s `earliest_cone_containment`/
`exact_cone_containment`, and `heuristics.py`'s
`find_min_travel_time_state_to_spatial_point` all take `vmin`/`vmax` as an **outer
bounding box** on velocity, not the exact constraint set — an L2 ball is
conservatively enclosed by its circumscribing box, so those functions stay sound
(if loose) under the SOC constraint with no changes needed there.

### 3.3 Class-level GCS cache doesn't key on `order` — a latent correctness bug once order varies — **fixed**

`STGCS._reusable_gcs_cache` (`graph.py:76`) is declared as a **class attribute**
(`Dict[...] = {}`), shared across every `STGCS` instance ever constructed in the
process — not just within one instance's lifetime. `_base_gcs_signature()`
(`graph.py:307-319`) builds its cache key from vertex names, `id(hpoly)` per vertex,
and edge tuples — **`self.order` is not part of the signature**. Today this is
harmless because `order` is always `2`. The moment `order` becomes a real parameter,
two different `STGCS` instances (e.g. one `order=2` baseline run and one `order=4`
Bézier run in the same process, such as a benchmark script that constructs both for
comparison) that happen to produce the same `(vertex_name, id(hpoly))` signature —
plausible if a benchmark rebuilds "the same" region graph for both configurations —
would retrieve **each other's cached `GCS` object built with the wrong
`CartesianPower`**, producing silently wrong solves rather than a crash. This must be
fixed as part of Step 2/3 by adding `self.order` into the signature tuple; it is not
optional cleanup. **Confirmed fixed:** `_base_gcs_signature` (`graph.py:332-349`)
returns `(self.order, planning_vertices, planning_edges)`, with an inline comment
explaining exactly this hazard.

### 3.4 `h_tab`/`h_tri` precomputation cost scales with per-solve cost, which goes up

`TripletRelaxationHeuristic` and `InterfaceToSetCostTableHeuristic` both do
exhaustive backward search over **every** edge/triplet in the *unfragmented* base
graph at setup time (`heuristics.py:267-281`, `508-543`), each backward-search step
calling `solve_convex_restriction`. Today each such solve is a cheap `order==2` QP.
Under `order>2` each becomes an SOCP (§4.2) or a larger LP over more decision
variables per vertex — the precomputation cost of these two heuristics scales with
however much more expensive one `solve_convex_restriction` call gets, and that
multiplies against "every edge/triplet in the graph," which is already the stated
dominant cost in the `order==2` regime. This is not a correctness problem and has no
code-level fix — it is a benchmarking/scoping question (do `h_tri`/`h_tab`
precomputation remain worth it under Bézier, or does `h_mot` become the only
practical heuristic at graph sizes where fragmentation is heavy?) that should be
measured, not assumed, once Steps 2-5 land.

### 3.5 `MICPPlanner` and `SearchPlanner` share `STGCS.get_gcs_instance` — order-awareness isn't search-only — **resolved (as "allow")**

Unlike the porting doc's generic framing (which explicitly excludes the whole-graph
relaxation+rounding path, `gcs_solver.py:250-293`'s `solve`, from the port, since "the
target repo doesn't have an analog of it and doesn't need one" — true in the
abstract), **this repo does have that path, as `MICPPlanner`** (`st_planner.py:58-76`),
and it calls the *same* `STGCS.get_gcs_instance` (`st_planner.py:67`) that
`SearchPlanner` does (`st_planner.py:121`, with `reuse_base=True`). Making `STGCS`
order-aware in Step 2 makes `MICPPlanner` order-aware too, for free, whether or not
that's desired — `solve()`'s `MipPathExtraction`/decode (`gcs_solver.py:268-293`) would
need the same order-branch as `solve_convex_restriction` (Step 4) to not crash on an
`order>2` graph. Decide explicitly whether `MICPPlanner` is meant to be a live
`order>2` baseline for comparison (useful for benchmarking search vs. relaxation+
rounding on identical Bézier instances) or should assert/reject `order>2` — don't let
it silently half-work because the underlying `STGCS` changed out from under it.
**Resolved as "allow":** `MICPPlanner.plan` (`st_planner.py:71-76`) passes
`order=stgcs.order` into `solve()` explicitly, and `solve`'s decode branches on
`order` the same way `solve_convex_restriction` does (Step 4) — so `MICPPlanner` is a
live `order>2` baseline today, not a reject/assert path. No test exercises this
combination yet (Step 8).

### 3.6 Reservation fragmentation's cost driver doesn't change, only its magnitude

Per the porting doc §6: `add_edges_bidir`'s reconnection cost
(`graph.py:176-190`'s `IntersectsWith` calls, gated by the `AABB` pre-filter) is
already the dominant cost of one reservation call in the `order==2` regime, not the
ECD splitting itself. Bézier reservation hulls are generically less axis-aligned and
less tightly bound to a "home" region than the `order==2` parallelotope tubes (per
`AGENT/ecd.py`'s own composability-bug note on `reserve_region_bezier`, §4 of the
porting doc) — so this is very likely to get *worse*, not better, under Bézier, but
by how much is an open empirical question (this repo's `B1`/`B5`-style benchmarks
from `AGENT.md` §7 are the way to measure it, not something to reason about a
priori).

---

## 4. Suggested execution order (concrete, this-repo version of the porting doc §7)

**Updated 2026-08-05 — remaining work only.** Items 1-3 and 5 below (Steps 1-5, §3.3)
are done; kept here, marked, so the original ordering rationale isn't lost. Items 4,
6-8 are the actual remaining roadmap.

1. ~~Step 1 (Tier-1 geometry) — standalone, testable against T1/T3/T5, zero solver
   risk.~~ **Done**, T1/T3/T5 themselves still don't exist (folds into Step 8 below).
2. ~~Step 2 + Step 3 together (`order`-aware `STGCS`, cache-key fix) — velocity
   constraint is the L2/SOC form (§3.2, resolved).~~ **Done**, including the
   cache-key fix (§3.3) and, beyond original scope, C² continuity (`order>=4`) and
   opt-in `energy_weight`/`uniform_time` (`AGENT.md` F9). Regression test for
   "`order=2` behavior bit-identical before/after" from the original plan still
   doesn't exist (folds into Step 8).
3. ~~Step 4 (decode branch in `gcs_solver.py`).~~ **Done**, and covers `MICPPlanner`'s
   `solve()` path too (§3.5), not just `solve_convex_restriction`.
4. **Step 6 (still open)** — confirm the search loop, run with `AStarDominanceCheck`
   only, never touches the three descoped raw-indexing sites in `dominance_check.py`
   (§3.1) before wiring `STSplineTrajectory` into `SearchNode.sol` for real. Nothing
   in code enforces the `AStarDominanceCheck`-only discipline yet — this is still a
   caller convention, not a guard.
5. ~~Step 5 (ECD reservation) — verify against T7 using the real graph.~~ **Done**
   (the `order>2` path in `stgcs/ecd.py`); T7 itself still doesn't exist.
6. **Step 7 (partially open)** — `h_mot` needed nothing and got nothing (confirmed).
   `h_tab`'s hardcoded `CartesianPower(interface, 2)` (`heuristics.py:619`) and
   unconditional `STTrajectory` construction (`:641`) are still unfixed as of
   2026-08-05 — this is the one concrete remaining code change from the original
   8-step plan (excluding tests).
7. **Step 8 (not started)** — no `tests/` directory exists anywhere in this repo.
   T1-T7 from `AGENT.md` §6, plus the search-specific acceptance test (Step 8's last
   paragraph), all remain to be written against the now-real `stgcs/graph.py` and
   `stgcs/ecd.py` — nothing blocks starting this now; Steps 2 and 5 it depended on
   are both done.
8. Resolve §3.4 as an explicit scoping decision (not code) before running any
   `order>2` benchmark that includes `h_tri`/`h_tab`. §3.5 is resolved (`MICPPlanner`
   is a live `order>2` baseline, untested). Dominance-check re-derivation (§3.1's
   deferred remainder) is out of scope for this pass entirely — revisit only when
   picked back up explicitly, and only after Step 6/7's fixes land first.
