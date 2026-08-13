---
tags:
  - gcs
  - reservation-geometry
  - search-based
  - implementation-plan
  - working-document
---
# Implementation Plan: Fixed-Duration Segments, Prism Reservation, No Product Graph

> **Status:** Not started. This is the narrow, concrete-steps companion to
> `global-time-grid-idea.md` -- that doc has the full reasoning, alternatives
> considered, and open risks; this doc exists so implementation can proceed
> from a single focused spec instead of that doc's full discussion history.
> Where a step depends on an argument made there, it's cited by section
> number rather than re-derived. Edit in place as steps are attempted --
> mark done/blocked, don't just append.

## 0. The idea, in one paragraph

Every GCS-vertex visit (one Bézier segment) gets a **fixed duration `Δ`**:
its two endpoints `T_0`, `T_{order-1}` are locked to `k·Δ`, `(k+1)·Δ` for
whichever step `k` this particular visit is along the current candidate
path; interior control points stay free and monotonic (endpoint-lock,
`global-time-grid-idea.md` §3 variant (a-i)). There is **no `(region,k)`
product graph** -- the discrete search runs over the existing,
un-duplicated region-adjacency graph and is allowed to **revisit the same
named vertex** (consecutively, via a new "wait one `Δ`" self-loop, or
non-consecutively), exactly as in Morozov et al. 2025's Shortest-Walk
Problem in GCS (`global-time-grid-idea.md` §14) -- `k` for a given visit is
just that visit's position along the walk, already available for free from
existing search-node bookkeeping. Reservation geometry for a fixed-Δ segment
is exact, not approximate (F1): project the segment's control points to
space, Minkowski-inflate by the footprint, and the segment already occupies
exactly the known window `[kΔ,(k+1)Δ]` -- so the reservation is a spatial
prism, and the prism's lateral faces (zero time-coefficient, not slanted)
are the ECD decision-boundary half-spaces (`global-time-grid-idea.md` §13,
§13.1). Everything else in the existing ECD pipeline -- `slice()`,
`_apply_ecd_pairs`, the `SliceMethod` extension point, co-temporal edge
wiring -- is unchanged.

## 1. Concrete design decisions (no alternatives discussion here)

- Variant: **(a-i) endpoint-lock only.** Not (a-ii) full-lock, despite
  external evidence for it (`global-time-grid-idea.md` §14 finding 1) --
  deferred, not rejected.
- Graph: the **existing** `STGCS` region-adjacency graph, unmodified in
  vertex/edge count, plus one new self-loop edge per vertex (see Step 1).
- Search: **walks, not paths** -- revisits allowed, `k` = visit's depth
  along the current search node's path.
- Reservation: **prism**, exact under F1, per `global-time-grid-idea.md` §13.
- Scope: single-agent first (inherits `global-time-grid-idea.md` §8's
  scope decision). Multi-agent reservation is where the prism construction
  actually gets exercised, but graph/search changes (Steps 1-4, 6-8) don't
  need a second agent to validate.

## 2. Implementation steps against this repo's actual code

Each step names the file(s)/function(s)/line ranges it touches, as read
2026-08-11. Line numbers will drift as the code changes -- treat them as
"where to start looking," not a frozen contract.

### Step 1 -- allow a "wait `Δ`" self-loop

`stgcs/graph.py:189-204`, `add_edges_bidir`. Line 190 currently
early-returns unconditionally on `u_name == v_name`, forbidding self-loops
outright. Needs a dedicated path (either relax this guard when `u_name ==
v_name`, or add a separate `add_wait_edge(v_name)`) that adds a `(v, v)`
edge, reusing the existing `edge_constraints` (C0/C1 continuity,
`graph.py:446-452` for `order==2`, the analogous order>2 block) unchanged --
a self-loop is not geometrically special to Drake's GCS formulation (each
edge gets independent copies of its endpoint vertex's variables; see Step 3
for why this repo already relies on that).

### Step 2 -- allow revisits in the discrete search

`stgcs/bfs/best_first_search.py:582-584`, inside `expand_node`:
```python
if n.has_visited(successor):
    logger.debug(f"Cycle detected: {n.vertex_path} -> {successor}")
    continue
```
This unconditionally forbids any revisit and must be relaxed for walk
semantics. `SearchNode.has_visited`/`_visited_vertices`
(`best_first_search.py:126-130,172-173`) already track full visited-vertex
history for this check -- once it's relaxed, that bookkeeping is no longer
needed for correctness but may still be useful for the cycle-elimination
pass in Step 8.

**Risk, not a detail to skip:** with Step 1's self-loop always available,
"wait" becomes a legal successor at every node, which can blow up branching
factor (an agent can always choose to wait instead of progressing). Needs
one of: a cap on consecutive self-loop count, reliance on `f`-ordering plus
Steps 6-7's pruning to deprioritize unproductive waits (verify empirically),
or Step 8's post-hoc cleanup. Likely needs more than one of these --
Morozov et al. needed the post-hoc pass *in addition to* their guided
search (their §IV-D: "we attempt to eliminate unnecessary cycles along the
walk").

### Step 3 -- thread visit-depth `k` into the per-visit `T`-lock

**This is the step with the most engineering risk and the one piece of
Drake-internals behavior this plan has not empirically verified --
confirm before investing further engineering time (see §3 below).**

Today, `get_gcs_instance`/`_ensure_reusable_base_gcs`
(`graph.py:245-303`, `:351-378`) build **one persistent Drake `GCS.Vertex`
per named region**, with `_add_gcs_vertex_costs_constraints`
(`graph.py:305-309`) attaching the shared `self.vertex_constraints`/
`self.vertex_costs` template **once, before any search runs** -- i.e.
before any particular visit's depth `k` is known. `T_0 = kΔ`,
`T_{order-1} = (k+1)Δ` is depth-dependent, not region-dependent, so it
cannot be baked into that shared per-region template as currently
structured.

The load-bearing question: are a `GCS.Edge`'s `xu()`/`xv()`
(`gcs_solver.py:328`, `graph.py:447-452`) independent per-edge copies of
their endpoint vertices' variables, or one variable shared across every
incident edge? This repo's own C0/C1 continuity constraint
(`graph.py:446-452`, `A_cont`) is only *necessary* if they're independent
copies -- if Drake auto-shared one variable per vertex, continuity across a
join would be automatic and that block wouldn't need to exist. Reading the
code, independence looks like the right model, which would mean: a
per-visit `T_0 == kΔ` constraint can be added as an **extra equality
constraint scoped to that specific edge's own `xu()`/`xv()`**, at the point
a specific candidate walk's edges are solved -- not baked into the shared
per-region template. Concretely this likely means `solve_convex_restriction`
(`gcs_solver.py:302-333`) grows an optional per-position extra-constraint
argument (keyed by `k = index along vertex_path`), attached to the looked-up
`GCS.Edge` objects just before `gcs.SolveConvexRestriction(E, options)`.

**Before writing this: build a 5-minute standalone Drake repro** -- two
edges through one named vertex, add an extra equality constraint to one
edge's `xu()` only, solve, and check the other edge's corresponding copy is
unaffected. If this holds, Step 3 is a moderate, well-scoped change. If it
doesn't, the whole persistent-single-Drake-vertex-per-region architecture
(including its cross-query `_reusable_gcs_cache`, `graph.py:355-378`) needs
rethinking for this idea -- likely toward building a small, fresh set of
Drake vertices per candidate walk instead of reusing named ones, which is a
substantially bigger change. Do not proceed past this step on an assumption.

### Step 4 -- `graph.py`'s per-vertex constraint block, the (a-i) patch

`_init_constraints_costs` (`graph.py:429-523`), pending Step 3's resolution:
- `t_index(0)`, `t_index(order-1)` (`graph.py:465-466`) become the targets
  of the per-visit equality constraints from Step 3, rather than free
  entries in the shared decision vector.
- `A_mono` (`474-479`) and the per-span Lorentz cone SOC constraints
  (`517-523`) stay exactly as-is -- under (a-i) they remain genuine SOC
  constraints, not collapsible to fixed-radius balls (`global-time-grid-
  idea.md` §4).
- `A_dt_total` (`505-508`) becomes vestigial (always evaluates to `Δ` once
  `T_0`/`T_{order-1}` are locked) -- leave in place as a zero-effect no-op
  rather than special-casing it out, unless profiling says otherwise.
- `uniform_time`/`A_uniform` (`481-500`) stay `False` -- that's the (a-ii)
  knob, out of scope per §1's variant decision.

### Step 5 -- prism reservation in `ecd.py`

Per `global-time-grid-idea.md` §13.1, made concrete:
- `_segment_ecd_pair` (`ecd.py:404-437`): replace the `(d+1)`-dim
  `inflate_hull` call with one over `control_points[:, :-1]` (space only,
  time column dropped) against a genuinely `d`-dim, zero-time-extent
  footprint. The resulting hull's own H-representation rows become
  `mid_hs_list` directly (each has zero coefficient on the time axis by
  construction -- no explicit "extrude" step needed beyond leaving time
  unconstrained by these rows). Drop the `time_cropping_bot_top` call
  (line 434) and its `bot_hs`/`top_hs` outputs entirely -- the segment's
  time window already equals `[kΔ,(k+1)Δ]` exactly, nothing to crop against.
  `bounds` (`428-432`) is unaffected -- it's already set independently of
  `bot_hs`/`top_hs`.
- `parallelotope_verts_offset` (`ecd.py:352-354`) cannot just be called
  with `dim = stgcs.dimension` for this -- it structurally assumes its last
  axis is a zero-extent *time* axis appended to a `(dim-1)`-spatial box
  (`np.hstack([signs, zeros])`), so calling it with `d` would zero out a
  real spatial axis, not add a time one. Needs a genuine `d`-dim footprint
  constructor (or a parameter distinguishing "spatial dims" from "has a
  zero-extent trailing axis"), used at the `reserve_spline` call site
  (`ecd.py:492`).
- `ECDPair` (`ecd.py:18`, constructed at `195` and `437`) currently takes
  `bot_hs`/`top_hs` as required fields -- make them optional (or pass a
  benign always-satisfied placeholder) so `slice()`'s downstream cropping
  step is a no-op for prism-built pairs, without changing `slice()`'s own
  logic.
- `bezier_segment_hull`/`bezier_decision_boundaries` (`ecd.py:517-534`),
  used by `reserve_region_bezier` (`ecd.py:537+`), need a parallel `d`-dim
  variant for callers that invoke them directly rather than through
  `_segment_ecd_pair`.
- Everything else -- `slice()`, `_apply_ecd_pairs`, the `SliceMethod`
  extension point (`"fixed"`/`"greedy"`, `ecd.py:32`), and
  `add_edges_bidir`'s co-temporal/sequential edge-wiring branches (Step 1
  covers the self-loop addition, not this) -- is unchanged.

### Step 6 -- heuristics: add hop-count, re-examine `h_tri`/`h_tab` under revisits

Per `global-time-grid-idea.md` §11, in `stgcs/bfs/heuristics.py`:
- Add a hop-count-based admissible heuristic (`hop_count × Δ`), combined
  with the existing `MotionOnlyHeuristic` (`heuristics.py:26-38`'s
  `travel_time_lower_bound`) via `MaxHeuristic` (`heuristics.py:183-229`).
  Hop-count on the un-duplicated graph stays a valid lower bound even with
  self-loops available: a self-loop can only add hops, never reduce the
  true shortest hop-count to the goal, so a correct hop-count computation
  should exclude/never prefer them.
- `TripletRelaxationHeuristic` (`heuristics.py:232`) and
  `InterfaceToSetCostTableHeuristic` (`heuristics.py:491`) precompute
  root-level tables over the static graph structure -- verify (don't
  assume) whether those precomputations need to model self-loops as legal
  single-hop transitions, or whether excluding them from precomputation is
  safe (waiting never helps reach a *different* target faster, but their
  exact recursions should be checked against this claim, not just this
  plan's prose).

### Step 7 -- dominance checking under revisits

`AStarDominanceCheck` (`dominance_check.py:31`) and
`SetContainmentDominanceCheck` (`dominance_check.py:365`) were built
against a search that, before Step 2, could never produce two nodes with
the same `vertex_name`. Once revisits are legal, two different search
nodes can share a `vertex_name` at different depths `k` -- with different,
mutually incompatible `T_0`/`T_{order-1}` locks (Step 3) -- so any
dominance comparison keyed on raw vertex identity risks conflating "same
region" with "same state" and incorrectly pruning a genuinely different
node. `global-time-grid-idea.md` §14's secondary implication (dominance
might not need `(region,k)` keying at all, since cost-to-go can be defined
per named vertex if duration accumulates along the walk rather than being
encoded into vertex identity) is a plausible resolution but is **not**
verified against this repo's actual dominance-check implementations --
read them against this question before relying on either outcome.

### Step 8 -- cycle-elimination post-processing

New code; closest existing hook is `goal_condition`/`_solve_convex_restriction`
(`best_first_search.py:564-568,452-459`), which this would wrap once a
goal-reaching walk is found: look for repeated-subsequence cycles with no
net spatial progress, remove them, re-solve the shortened path's convex
restriction, and keep the shortened version if feasible and cheaper.
Mirrors Morozov et al.'s own post-processing step (`global-time-grid-idea.md`
§14) -- build this only after Step 2's revisit-branching risk is actually
observed to produce cycles worth eliminating in practice, not speculatively.

### Out of scope for this pass

Sizing `Δ` against `vlimit`/region extents (`global-time-grid-idea.md` §5
risk #1, §9 step 1), multi-agent reservation interaction and §12's
still-open ECD-fragmentation-cost question, and reconsidering (a-ii) per
§14 finding 1 -- all explicitly deferred, matching the parent doc's own
single-agent-first scope (§8/§9).

## 3. Suggested build/verification order

1. **Step 3's Drake repro first**, standalone, before any other code
   changes -- it gates whether Step 4 is a moderate patch or a much larger
   rework. Do not start Step 4 on an unverified assumption about `xu()`/
   `xv()` independence.
2. Steps 3+4 together, against a single hand-fixed segment (`T_0`,
   `T_{order-1}` set to specific numbers, no search involved yet) -- call
   `solve_convex_restriction` directly and confirm it solves correctly,
   before touching search or graph construction at all.
3. Steps 1+2 together, on the same hand-built corridor
   (`AGENT.md` §5 item 4's map) -- confirm a trivial "wait one `Δ` then
   proceed" plan is found correctly before anything harder.
4. Step 5 (prism reservation), validated standalone against `ecd.py`'s
   existing usage/tests -- it's a pure geometry change, independent of the
   search-graph changes, testable in isolation.
5. Steps 6-7 (heuristics/dominance) only after 1-4 are solid -- they're
   pruning/efficiency layers, not correctness-critical for a first working
   version (uninformed search still finds correct results, just slower).
6. Step 8 only once revisits are observed to actually produce
   cycles worth eliminating.

This mirrors this repo's own "measure before building" discipline
(`lazy-ecd-coarse-search-idea.md` §9.6, `global-time-grid-idea.md` §6).
