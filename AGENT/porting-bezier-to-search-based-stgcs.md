---
tags:
  - gcs
  - reservation-geometry
  - migration-plan
  - search-based
---
# Porting Plan: Per-Region Cubic Bézier Trajectories into a Search-Based ST-GCS Repo

> **Status:** Draft. Companion to `AGENT.md` (the geometric theory, F1-F8 — read that first, this
> doc does not re-derive it) and `bspline-migration-plan.md` (the file-by-file plan that ported the
> same theory *into this* codebase, which uses Drake's `GraphOfConvexSets` relaxation+rounding
> solve). **This doc is the analogous plan for porting the same trajectory representation and
> reservation pipeline into a *different* target repo — one that solves ST-GCS by discrete search
> over regions plus an exact continuous solve per fixed candidate path, for piecewise-linear
> (order=2) trajectories today.**
>
> I do not have the target repo in front of me. Everything below is written against the *shape*
> a search-based ST-GCS implementation almost certainly has (see §1), with the exact source
> locations in *this* repo (`stgcs-devel`) it should be ported from. Wherever the target repo's
> actual structure differs from the assumed shape, adapt the mapping, not the underlying theory —
> F1-F8 and the constraint math in §3 do not change based on solve strategy.

---

## 0. Read this first: what changes and what doesn't, moving from relaxation+rounding to search

The representation being ported (`AGENT.md` F1-F8: cubic Bézier per GCS vertex, joint space-time
control points, C⁰+C¹ edge continuity as linear/SOC constraints, ECD reservation via Minkowski-
inflated control-point hulls) is **entirely solve-strategy-agnostic**. Nothing about a Bézier
segment's convex-hull certificate, its monotone-time subdivision property, or how ECD subtracts an
obstacle from a region graph depends on whether the discrete region sequence was chosen by MICP
relaxation+rounding or by a search algorithm.

What *is* solve-strategy-specific, and the only thing that needs re-targeting rather than copying,
is **where the per-vertex decision variables and constraints get attached**. In this repo that
happens in two places:

- `STGCS._init_constraints_costs` / `_add_gcs_vertex_costs_constraints` / `_add_gcs_edge_costs_constraints`
  (`stgcs/stgcs.py:417-503`, `302-327`) — builds the constraint *objects* once, generically over
  `order`.
- `stgcs/gcs/solver.py:solve` (whole-graph relaxation+rounding, lines 20-77) and
  `solve_convex_restriction` (lines 79-109) — **the second one is the important one for this port.**
  `solve_convex_restriction` takes an *already-fixed* `vertex_path` (a specific ordered sequence of
  regions — exactly what a search algorithm produces) and calls `gcs.SolveConvexRestriction(E,
  options)`, no relaxation, no rounding, no MICP. This is already, in this codebase, the "solve the
  exact continuous sub-problem for one candidate discrete path" primitive — which is *exactly* the
  primitive a search-based solver needs at each node/goal it evaluates.

**Practical consequence:** if the target repo's search algorithm already has some function shaped
like "given this candidate sequence of regions, produce the trajectory and its cost" (it must have
*something* like this — that's how a search-based GCS solver evaluates candidates at all), that
function is the direct port target for the `order>2` constraint-construction logic in §3 below.
`solve()`'s whole-graph relaxation path (lines 20-77 here) is **not** part of this port — the target
repo doesn't have an analog of it and doesn't need one.

---

## 1. Assumed target-repo shape (verify and adjust before starting)

A search-based ST-GCS solver for piecewise-linear trajectories almost certainly has:

1. **A region graph** — vertices are `HPolyhedron`-like convex sets extruded/joint over space+time,
   edges are region adjacency (or reachability). Likely close in shape to this repo's `STGCS`/
   `STVertex` (`stgcs/stgcs.py:33-104`): `add_vertex`, `remove_vertex_from_graph`
   (`stgcs.py:152-183, 204-`), `add_edges_bidir` (`stgcs.py:188-`), each vertex carrying an
   `HPolyhedron` (`st_hpoly`), a time `Interval`, and per-axis space `Interval`s.
2. **A discrete search** (A*/Dijkstra/best-first) over that graph, using *some* admissible or
   heuristic cost estimate per edge/region to decide expansion order.
3. **An exact continuous solve for one fixed candidate path** — the search's "evaluate this
   candidate" step. Per-vertex today: 2 space-time points (entry, exit) — this repo's `order=2`
   representation, `STTrajectory` (`stgcs/trajectory.py`).
4. **A trajectory decode/eval type** for the result — `STTrajectory`-shaped: `.lerp(t)`, `.size`,
   `.x0`/`.xT`, `.duration`, `.xA(idx)`/`.xB(idx)`.
5. **Possibly a dominance/pruning check** comparing whether one partial search node's reachable set
   dominates another's — if present, this is the piece that needs genuine re-derivation, not
   mechanical porting (§5).

If any of 1-4 don't exist in recognizable form, the corresponding section below still describes
*what* needs to exist and *why* (citing F1-F8), even if it can't say exactly *where*.

---

## 2. Tier 1 — copy near-verbatim, zero solver dependency

These files only depend on `numpy` (+ `scipy.spatial.ConvexHull` / `pydrake`'s `HPolyhedron` for
geometry, no GCS/solver types at all). Port them first; they're testable completely standalone
against `AGENT.md`'s T1, T3, T5.

| File | What it is | Port notes |
|---|---|---|
| `stgcs/bezier.py` (67 lines) | `de_casteljau_eval`, `de_casteljau_split`, `invert_time` — pure math on `(order, dim)` numpy arrays. | Copy verbatim. No dependencies beyond numpy. |
| `stgcs/hulls.py` (39 lines) | `minkowski_sum_vertices`, `inflate_hull` — V-rep Minkowski sum + re-hull (`conv(A)+conv(B) == conv(A+B)`), then `.ReduceInequalities()`. | Copy verbatim if the target has (or can borrow) an equivalent `make_hpolytope(V) -> HPolyhedron` (this repo's is `geometry_utils.py:378-384`, a thin wrapper over `scipy.spatial.ConvexHull`). **Do not skip `.ReduceInequalities()`** — `ConvexHull` triangulates every planar facet into simplices, roughly doubling the row count with true duplicates; see `BLACKMAGIC/inflated-hull-facet-bound-proof.md` for the exact bound (≤14 true facets for a non-degenerate cubic segment) and this session's own finding that skipping it is a real, not cosmetic, cost multiplier once pieces accumulate. |
| `stgcs/spline_trajectory.py` (85 lines) | `STSplineTrajectory` — decode/eval type: `.control_points(idx)`, `.lerp(t)` (de Casteljau on inverted time, F5), `.find_segment_index(t)`, `.x0`/`.xT`/`.duration`, `.xA(idx)`/`.xB(idx)` (parity helpers for callers that only need segment endpoints). | Copy near-verbatim; only needs `stgcs/bezier.py`. If the target's linear-trajectory type has additional methods callers rely on (e.g. `get_chunk`/`append_chunk` for windowing — **not implemented in this repo either**, since rolling-horizon is out of scope here too, see `AGENT.md` §1), those need new de-Casteljau-split-based implementations, not a port — nothing to copy from this repo for that specific gap. |

---

## 3. Tier 2 — re-target, don't copy: per-vertex/per-edge constraint construction

This is the one genuinely solver-specific piece, and it needs to land wherever the target repo
builds the continuous convex program for **one fixed candidate region sequence** (§1 item 3) — the
functional analog of this repo's `solve_convex_restriction` / `_init_constraints_costs`.

**Per-vertex decision variables:** raise from 2 space-time points to `order` joint control points
`Q_i = (P_i, T_i)`, flat layout `[P_0,T_0, P_1,T_1, ..., P_{order-1},T_{order-1}]`
(`stgcs.py:448-457`, `p_slice`/`t_index` helpers). Container containment (`Q_i` inside the region's
`HPolyhedron`) is the existing per-vertex containment constraint, just applied to `order` points
instead of 2 — this repo does it via `HPolyhedron.CartesianPower(order)`
(`gcs/utils.py:75`, `make_Cartesian_power_hpoly`); use whatever the target's equivalent of "N copies
of this containment constraint" is.

**Per-vertex constraints** (`stgcs.py:465-486`, the `order > 2` branch of `_init_constraints_costs`):
- **Monotone time** (F4/F5): `T_{i+1} - T_i >= dt` for every consecutive pair, not just an overall
  `T_{order-1} - T_0 >= dt` — a per-span floor rules out the solver collapsing one interior spacing
  to ~0 while satisfying only the overall bound (leaves that span's implied velocity
  ill-conditioned even though the constraint as written is satisfied).
- **Velocity** (F8): replace the `order=2` case's single L∞ box constraint
  (`stgcs.py:425-433`) with `order-1` second-order-cone constraints, one per consecutive
  control-point pair: `‖P_{i+1}-P_i‖ <= vlimit·(T_{i+1}-T_i)`. This is the numerically real change —
  a box constraint becomes a Lorentz cone constraint per span.
- **Cost:** total segment duration only, `T_{order-1} - T_0` — interior time allocation between
  control points stays a free (uncosted) decision variable; this is unchanged in *kind* from
  `order=2`, only the endpoint indices move.

**Per-edge constraints** (`stgcs.py:488-502`, plus the boundary-vs-interior split at
`stgcs.py:317-327`):
- **C⁰** (always): tail's last control point == head's first control point, position *and* time
  jointly (F4) — same as today's `order=2` edge continuity, just at control point `order-1`/`0`
  instead of the segment's single exit/entry point.
- **C¹** (only between two real region-vertices, *not* at source/target/boundary edges — see
  `stgcs.py:317-324` for why: a source/target vertex has no internal velocity to match, so forcing
  C¹ there would spuriously demand zero velocity at the trajectory's start/end): tail's exit tangent
  equals head's entry tangent, `Q_tail[order-1] - Q_tail[order-2] == Q_head[1] - Q_head[0]` — a
  **linear** equality on raw control-point differences (`stgcs.py:497-502`). Worth restating why
  this stays linear (not bilinear) here, since it's easy to assume otherwise coming from
  duration-as-a-separate-variable formulations (e.g. Drake's `GcsTrajectoryOptimization`): because
  time is a coordinate *in* each joint control point rather than a separate rescaling variable,
  matching the raw space-time tangent vector on both sides is sufficient for physical `dx/dt`
  continuity without ever dividing by `ΔT`.

**Decode:** wherever the target reads back `points = [result.GetSolution(...) for e in E]` (this
repo's `solve_convex_restriction`, `solver.py:105`) and builds a trajectory object, branch on
`order` and construct `STSplineTrajectory(vertex_path, points, space_dim, order)` instead of the
linear type when `order > 2` — `solver.py:79-109` is the exact template, already dual-path.

---

## 4. Tier 2 — copy near-verbatim, needs an HPolyhedron+Interval region-graph API

The ECD reservation pipeline is **generic over `HPolyhedron`/`Interval`** and never inspects how
many halfspaces are in an obstacle or how bounds were derived — confirmed by reading it, not
assumed (this was re-verified multiple times this session while investigating a `get_hpoly_bounds`
performance question, and held at every scale tested). It only needs:

- A region-graph type with `add_vertex(hpoly, time_interval, ...) -> vertex_or_None`,
  `remove_vertex_from_graph(name)`, `add_edges_bidir(u, v)` (adjacency re-derived via
  `HPolyhedron.IntersectsWith`, gated by a cheap AABB pre-filter — `stgcs.py:188-200`).
- A trajectory object with `.control_points(idx) -> (order, dim+1) array` and `.size`
  (`STSplineTrajectory`, already ported in Tier 1).

| File/function | Role | Port notes |
|---|---|---|
| `ecd.py:time_cropping_bot_top`, `time_cropping_mid` (`ecd.py:263-291`) | Time-axis halfspace cropping, representation-agnostic. | Copy verbatim. |
| `ecd.py:slice` (`ecd.py:226-260`) | Peels a region against an `ECDPair`'s facet halfspaces, one at a time, restricting the remainder each step. Already generic — used identically by both the `order=2` parallelotope-tube path and the Bézier path in this repo. | Copy verbatim. |
| `ecd.py:update_edge`, `_apply_ecd_pairs` (`ecd.py:55-98, 147-168`) | Rebuilds graph adjacency around split pieces; batches "which vertex touches which obstacle" via AABB before doing any splitting, to avoid the composability bug documented in `reserve_region_bezier`'s docstring (a segment's Minkowski-inflated hull can reach past its "home" region into a neighbor — checked and fixed here, not optional). | Copy verbatim (needs your `add_vertex`/`remove_vertex_from_graph`/`add_edges_bidir`). |
| `stgcs/hulls.py:inflate_hull` (Tier 1) + `ecd.py:_segment_ecd_pair` (`ecd.py:346-382`) | Builds one segment's `ECDPair`: Minkowski-inflate the segment's control-point hull by the footprint, negate its H-rep rows to "outside this facet" halfspaces (`mid_halfspaces`), time-crop, compute bounds from the *raw* Minkowski-sum point cloud (not a second hull-to-vertex conversion — a set's per-axis extent is always achieved at an extreme point already present in that raw cloud). | Copy verbatim. |
| `ecd.py:generate_all_ECD_pairs_spline`, `reserve_spline` (`ecd.py:385-442`) | Whole-trajectory reservation: one `ECDPair` per segment (+ optional parked start/end pairs), then `_apply_ecd_pairs` graph-wide. | Copy verbatim — this is the function your search-based planner's "reserve this robot's solved trajectory before planning the next one" step calls. |
| `ecd.py:bezier_segment_hull`, `bezier_decision_boundaries`, `reserve_region_bezier` (`ecd.py:462-538`) | Single-region/single-segment primitives — useful if your search wants to check/reserve against one named region without a whole-graph AABB scan. **Two composability rules if you use these directly instead of `reserve_spline`:** (1) check every region's *original* extent for an AABB touch, not just a segment's nominal "home" region — Minkowski inflation routinely reaches past a shared region boundary; (2) reserve against a region's *current* surviving pieces (via `root_name`, since splitting is recursive), not a stale/now-removed vertex name. Both measured and fixed in this repo's own test suite (`tests/unit/test_reserve_region_bezier_unittest.py`) and notebook (`notebooks/ecd_bezier_visualization.ipynb`, Part 7). | Copy verbatim; keep the two rules in mind if composing calls yourself. |

**Do not port** the `order=2` linear-tube path (`generate_all_ECD_pairs`,
`parallelotope_side_halfspace_kd`) unless the target repo also wants to keep its existing
piecewise-linear reservation working side by side — this repo keeps both paths live specifically so
`order=2` never regresses (see `bspline-migration-plan.md` §1's dual-path rationale, which applies
here too: don't hard-cutover the target's existing linear reservation in place).

---

## 5. Tier 3 — do NOT mechanically port: search/dominance/heuristic pruning

If the target's search algorithm has anything shaped like this repo's (hypothetical, not present
here, but the concern is general) domination check — "does the reachable set from region-state `q`
contain the reachable set from `p`, so `p` can be pruned" — **that logic needs re-derivation, not
transliteration**, if it currently assumes a single space-time point's reachable set under a
velocity bound is a cone (a common, cheap, closed-form admissibility argument for point-to-point
kinodynamic search). A cubic Bézier segment's reachable set from a *hull of control points* is not
a cone in the same closed form. Concretely, from `bspline-migration-plan.md` §5 (written for this
exact situation against this repo's own low-level search, and directly applicable to any
cone-based dominance check in the target):

- **Conservative fallback** (minimal change): restrict dominance/pruning to compare entry points
  only, dropping any shape-aware tightening that intersects a candidate's actual region hull —
  correct but weaker pruning (a search-quality regression, not a correctness one).
- **New research** (if pruning strength matters): derive a genuinely hull-aware admissibility
  criterion. Don't assume this is a quick fix — it's an open question, not a mechanical step, and
  should be scoped/decided explicitly before writing code, same as it was flagged (and deferred)
  here.

Anything in the target's heuristic that only ever compares scalar costs or lower bounds derived
from pure distance/vlimit (independent of trajectory shape) needs no change at all — this repo's
own `HeurShortCut`/`HeurLowerBoundGraph`-style heuristics (per `bspline-migration-plan.md` §5) fall
into this category and required zero modification for the Bézier port.

---

## 6. Known tradeoffs to inherit knowingly (this session's findings)

Porting the ECD/reservation pipeline (§4) means inheriting its known cost/scaling properties,
independent of solve strategy — worth carrying forward rather than rediscovering:

1. **Keep bounds computation LP-based.** `geometry_utils.py:get_hpoly_bounds` solves two
   support-function LPs per axis rather than enumerating vertices. This was tested extensively
   (single split, then repeated reservations at increasing depth) against three vertex-enumeration
   alternatives (raw `cddlib`, `cddlib` + `ReduceInequalities()`, Qhull-backed `VPolytope`) — all
   three were eventually either slower, occasionally wrong, or crashed outright as reservations
   accumulated redundant facets. The LP-based approach never has this failure mode: a
   support-function LP is well-defined for any nonempty bounded polytope regardless of how
   degenerate its H-representation gets, since it never needs to enumerate a vertex set. Do not
   "optimize" this during the port.
2. **Reservation fragmentation is real and representation-independent.** Repeated `reserve_spline`
   calls (one per already-planned robot) genuinely fragment the graph — vertex/halfspace counts
   grow, and `add_edges_bidir`'s reconnection cost (an `IntersectsWith` LP per newly-adjacent-piece
   candidate pair, AABB-gated) is typically the single largest cost component of one reservation
   call, not the ECD splitting itself. A search-based solver changes how the *whole-graph solve*
   scales with graph size (favorably — see §0), but does not shrink the graph itself; this cost
   still needs its own accounting in the target.
3. **Base region overlap (if using IRIS or similar) compounds the above.** Volumetrically
   overlapping base regions mean a single reservation can touch multiple regions covering the same
   physical space, each needing its own split — this is a property of the *base decomposition*,
   separate from the Bézier/ECD port, and worth a deliberate choice (disjoint/facet-adjacent regions
   vs. overlapping ones) independent of this port.

---

## 7. Suggested execution order

1. **Tier 1** (§2): port `bezier.py`, `hulls.py`, `spline_trajectory.py` standalone. Verify against
   `AGENT.md` T1 (hull certificate), T3 (monotone subdivision), T5 (inflation) — none of these need
   any GCS/solver integration.
2. **Tier 3 constraint math** (§3): wire `order` through wherever the target builds one fixed
   candidate path's continuous program. Verify: a straight 2-region case reproduces the `order=2`
   answer exactly (regions collinear); a bent 2-region corridor produces a curved, C¹-continuous
   path with no velocity discontinuity at the boundary — same acceptance test
   `bspline-migration-plan.md` §2 used here.
3. **Tier 2 reservation pipeline** (§4): port `ecd.py`'s spline path + `hulls.py`. Verify against
   `AGENT.md` T7 (dense-sample the reserved trajectory ⊕ footprint, assert no sample lies in any
   post-subtraction free set) using the target's actual region-graph type, not a standalone harness.
4. **Tier 5** (§5): decide conservative-fallback vs. new-research for any dominance/pruning logic,
   explicitly, before writing code.
5. **Multi-robot loop**: wire `reserve_spline` into the target's fixed-priority-order planning loop
   (plan robot, reserve permanently, next robot) — structurally identical regardless of solve
   strategy.

## 8. Explicit non-goals (carried from `AGENT.md` §1 and `bspline-migration-plan.md` §9)

Same scope boundary applies to this port: no rolling horizon/windowed re-solving, no PBS-style
dynamic priority reordering, no multi-span-per-region (would reintroduce the F2 interlock *within*
a vertex), no MINVO basis change, no C² continuity — all explicitly deferred, not designed away.
Flag any of these to whoever owns the target repo before adding them; they're each a real
re-derivation, not a mechanical extension, exactly as in the source design.
