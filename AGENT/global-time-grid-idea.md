---
tags:
  - gcs
  - reservation-geometry
  - search-based
  - proposal
  - working-document
---
# Idea: lock every agent's control points to a shared global time grid (fixed Δ per edge)

> **Status:** Proposal / working document. Nothing here has been implemented or
> benchmarked yet. Edit in place as the idea gets refined -- add findings, revise
> the design, cross out rejected variants, don't just append.
> Companion to `AGENT.md` (F4, F5, F9 particularly -- the trajectory
> representation and time-control-point theory this idea sits on top of,
> unchanged) and `lazy-ecd-coarse-search-idea.md` (a different mechanism aimed at
> the same underlying symptom: ECD-driven vertex/edge explosion under
> reservation).

## 1. The problem this is trying to solve

`lazy-ecd-coarse-search-idea.md` §1 measures reservation-driven fragmentation
growing the graph from 1 vertex / 0 edges to thousands, one agent at a time. That
doc's fix is *scheduling* (defer/batch when splitting happens). This idea
attacks a different lever: *how many distinct cutting hyperplanes a reservation
generates in the first place*.

`ecd.py`'s `bezier_decision_boundaries` (lines ~425-435) places each cut exactly
on a facet of the actual, Minkowski-inflated reservation hull -- which is correct
and exact, but means the cut's time-location is wherever that particular agent's
control points actually happened to be. Under F4, interior time allocation within
a segment is free (monotonicity-only), and nothing at all ties one segment's
`T_0`/`T_{order-1}` to any other agent's or region's timing. So N agents crossing
near the same region generically produce N sets of *differently-timed* cuts
instead of shared ones -- every agent re-fragments the region at its own,
generically-irrational times.

## 2. Why F9 (`uniform_time`) doesn't already solve this

F9 forces one segment's own interior `T_i` into an arithmetic sequence between
that segment's own two endpoints (`graph.py:494-500`). That's a *local* shape
constraint -- it makes control points evenly spaced *within* a segment, but says
nothing about where that segment's endpoints sit relative to any other agent's
schedule or any other segment's duration. Two agents' segments can still occupy
completely unrelated, arbitrary real-time windows even with `uniform_time=True`
everywhere. Global alignment requires a mechanism external to F9.

## 3. The proposal (decided 2026-08-07): fix Δ, one graph edge = one Δ

Two top-level variants were on the table:

- **(a) Fixed: every edge spans exactly one Δ.** [chosen]
- (b) Variable: edge duration is `k·Δ` for a nonnegative integer `k` chosen by
  the discrete search layer. [rejected/deferred -- reintroduces a combinatorial
  choice into the search, closer to space-time A*/MAPF than a clean
  simplification]

Under (a), a path's elapsed time is exactly `(edge count) × Δ` -- time is fully
determined by graph depth and is never a quantity the continuous solver needs to
find.

**(a) itself splits into two sub-variants, both kept as of 2026-08-08 --
distinguished explicitly everywhere below rather than conflated (they were not
separated cleanly before this date; §4/§10 originally described (a-ii) only,
now corrected):**

- **(a-i) endpoint-lock.** Only `T_0` and `T_{order-1}` -- the GCS-vertex joints
  -- are pinned to `k·Δ`, `(k+1)·Δ`. Interior `T_1...T_{order-2}` remain free,
  monotonic decision variables within `[T_0, T_0+Δ]` (`uniform_time` off).
- **(a-ii) full-lock.** Endpoints pinned as in (a-i), *and* interior `T_i` are
  additionally forced into an arithmetic sequence (`uniform_time` bundled in),
  so every control point -- not just segment boundaries -- lands on a shared
  fine grid (`Δ/(order-1)` spacing).

Only segment *endpoints* landing on the shared grid is what §1's motivation
(agents' reservation cuts landing at shared times) strictly needs, since
`ecd.py`'s reservation-hull cuts are facets of the whole segment's
control-point hull, not just its endpoints -- (a-i) gives weaker cross-agent
hull-shape alignment than (a-ii) even though both give identical segment-
boundary alignment. See §4 and §10 for how the two diverge in cost.

## 4. What this changes, concretely, against the actual code

**Corrected 2026-08-08: this section originally described (a-ii) [full-lock]
only, without saying so. (a-i) [endpoint-lock] is a much smaller change --
split out below.**

- **`graph.py` under (a-ii) [full-lock]:** the per-vertex control-point block
  shrinks from `order*(d+1)` to `order*d` -- `T` is no longer part of the
  decision vector at all, so `t_index`, `A_mono`, `A_uniform` (lines 465-500)
  go away entirely, not just their `order-2` uniform-time rows. F8's Lorentz
  cone (lines 517-523), currently `||P_{i+1}-P_i|| <= vlimit*(T_{i+1}-T_i)`
  with `T` a decision variable, becomes a fixed-radius norm ball
  `||P_{i+1}-P_i|| <= vlimit*Δ/(order-1)` -- a real reduction in constraint
  class (no more space-time coupling), not just fewer rows.
- **`graph.py` under (a-i) [endpoint-lock]:** a much lighter patch. Only 2 of
  the `order` per-vertex `T` scalars (`T_0`, `T_{order-1}`) become known
  constants (substituted from graph depth × Δ) instead of decision variables;
  the remaining `order-2` interior `T_i` stay exactly as free and
  monotonicity-constrained as they are today. F8's Lorentz cones
  (lines 517-523) mostly **do not collapse** to fixed-radius balls under
  (a-i): for `order > 2`, every consecutive-pair constraint except possibly
  the very first/last still couples a free interior `T_i`, so it remains a
  genuine SOC constraint, not a constant bound. (a-i) does *not* buy the
  dimension-reduction/constraint-simplification payoff that motivated this
  idea's very first framing ("shouldn't I just remove one dimension...") --
  that payoff is specific to (a-ii).
- **Graph construction, not just the per-vertex cost/constraint code, has to
  change.** Today an `STVertex` is a continuous-time space-time region, carved by
  reservations at whichever arbitrary times other agents' trajectories actually
  passed through. Fixed-Δ edges only make sense over a graph that is already a
  space×time product (vertices indexed by `(region, k)` for discrete step `k`),
  built that way from the start -- not a graph that becomes discretized as a side
  effect of reservations. This almost certainly needs a same-region "wait one Δ"
  self-loop, which `add_edges_bidir` currently forbids outright (early-returns on
  `u_name == v_name`; see `lazy-ecd-coarse-search-idea.md` §9.2.1).
- **`ecd.py`'s reservation pipeline** could potentially be replaced by
  marking/removing `(region, k)` cells rather than computing
  `bezier_decision_boundaries`/`slice()` hyperplane geometry -- a much simpler
  mechanism *if the rest of this holds up*, not merely a less-fragmented version
  of the current one. **Corrected 2026-08-09: this is no longer just
  "unverified" -- §12 confirms the current ECD pipeline actively breaks the
  edge=Δ invariant this idea depends on, via a specific, identified mechanism.
  Not a matter of degree; read §12 before treating this bullet as optional
  polish.**

## 5. Open questions / risks (unverified -- do not assume)

1. **Per-Δ feasibility.** Any segment whose true minimum-time traversal exceeds
   Δ is infeasible outright under fixed one-edge-one-Δ -- there is no fallback
   (variant (b) was the fallback, and was declined). Needs a sizing analysis of
   Δ against `vlimit` and the benchmark environments' region sizes, or a
   pre-processing pass that splits long regions into multiple graph edges before
   solving so no single edge ever needs more than one Δ's worth of travel.
2. **Relationship to ECD's "exact cuts, not blind grid" argument.**
   `lazy-ecd-coarse-search-idea.md` §7 argues at length that ECD's
   obstacle-derived, exact cuts are not optional polish versus a blind
   octree-style grid, because a blind grid either under- or over-approximates
   free space near a curved boundary. That argument is about the *spatial* axes.
   Discretizing only the *time* axis, while keeping ECD's spatial cuts exact,
   does not obviously reintroduce that failure mode -- but this has only been
   argued informally here, not checked rigorously.
3. **Scope.** This is a graph-construction and reservation-backend rewrite, not a
   `graph.py` parameter flip. Comparable in scope to, or larger than, the
   coarse-graph proposal in `lazy-ecd-coarse-search-idea.md` -- and that proposal
   turned out to have a fatal flaw only after a careful read against the actual
   code (see its §9). This idea has not yet had that pass.
4. **Search-machinery interaction.** `h_mot`'s admissibility,
   `AStarDominanceCheck`'s vertex-identity keying, and
   `TripletRelaxationHeuristic`'s root-level tables were all derived against the
   current continuous-time, reservation-carved graph. Their behavior under a
   `(region, k)` product graph (does dominance now need to key on `(region, k)`
   instead of vertex name? does `h_mot` need a per-step admissible bound instead
   of a continuous travel-time one?) is unexamined.

## 6. Next steps

Not started. Before writing any code: size Δ against the benchmark environments
(`benchmark/environment/uav_village.py`, `maze/maze_env.py`, `grid.py`,
`iris.py`) and current `vlimit`/region-size distributions, to check variant
(a)'s fixed-Δ feasibility isn't degenerate in either direction (Δ too small →
huge `(region, k)` product graph; Δ too large → can't fit through tight
regions/corridors). Same "measure before building" discipline
`lazy-ecd-coarse-search-idea.md` §9.6 used for the coarse-search proposal.

**Superseded in priority by §7.1 below — do Step 0 there first.** It answers the
same motivating question (§1) far more cheaply and may make the rest of this
proposal unnecessary.

## 7. Discussion: cross-check against `AGENT.md` and `lazy-ecd-coarse-search-idea.md` (2026-08-07)

Written after reading `AGENT.md` in full (F1-F9, §8 pitfalls, definition of
done) and `search-based-bezier-implementation-plan.md`, specifically to check
this idea for conflicts with settled facts or with the other open proposal. No
code changed to produce this section.

### 7.1 Likely conflict: F5 already solves §1's motivating problem, more cheaply, without giving up optimality

`AGENT.md` F5 states the post-hoc split can hit "globally synchronized slabs
t = kΔ across all robots, after solving, **at zero planning cost**"
([AGENT.md:34](AGENT.md#L34)), and §3 item 2 names this a `time_slabs(Δ)` split
policy alongside the currently-default `k1` (whole-vertex-hull, no split). This
is not implemented yet as a named policy, but its primitives are:
`de_casteljau_split` and `invert_time` already exist in `stgcs/bezier.py` and
`invert_time` is already called from `stgcs/spline_trajectory.py:89`.

If reservations are split at t = kΔ post-hoc (same Δ for every robot) before
being subtracted, reservation-hull cuts from different agents should already
tend to land at shared times, addressing §1's actual complaint (agents
re-fragmenting regions at mutually arbitrary times) — **without fixing edge
duration, without a product-graph rewrite, and without giving up the free-time
solve's optimality** (F9's cost of `uniform_time` and this idea's cost of fixed-Δ
both don't apply; the underlying solve stays exactly as free as it is today,
only the *reservation* geometry gets sliced at shared boundaries after the
fact).

**This should be measured before any of §3-§5 here is built.** If
`time_slabs(Δ)` alone recovers most of the fragmentation reduction this idea is
chasing, the fixed-Δ-per-edge product-graph rewrite (§3-§4) is not justified by
§1's stated motivation and this proposal should be scoped down to "add the
`time_slabs` split policy" — a small, additive change fitting entirely inside
the existing free-time architecture. If it does not (e.g. because whole-segment
durations still vary too much for slab cuts to reliably coincide across agents,
or because unsliced regions between slabs still fragment differently per
agent), that's the actual, narrower justification for pursuing fixed-Δ edges,
and it should be stated as such rather than assumed.

### 7.2 Tension with `lazy-ecd-coarse-search-idea.md`'s core goal

That proposal's entire premise (§2, §5) is keeping the discrete search graph
**small and ~constant-size**, independent of reservation count. §4 of this doc
currently describes the `(region, k)` product graph as "built that way from the
start" — i.e. materializing a vertex for every `(region, k)` pair up front. That
multiplies the *base* graph size by `tmax/Δ` steps before any reservation or
search even happens, which is the opposite direction from what the other
proposal is trying to achieve, and could plausibly make the vertex/edge
explosion problem both docs are ultimately about *worse*, not better.

If §3-§5 survive the §7.1 check, this should be corrected: `(region, k)`
identity should be generated lazily during search (`k` carried as part of
`SearchNode` state / cost, successors expanded on demand — standard practice
for space-time A*), not pre-built as static Drake GCS vertices in `graph.py`.
§4's "built that way from the start" wording should be read as describing
*logical* structure, not literal upfront materialization, and should be
corrected if this idea proceeds.

### 7.3 Clarification needed: §5 risk #1's region pre-splitting must not become the F2 interlock pitfall

§5 risk #1 mentions a possible mitigation: "a pre-processing pass that splits
long regions into multiple graph edges before solving so no single edge ever
needs more than one Δ's worth of travel." This must be implemented as
additional, edge-disjoint **GCS vertices** (private control points each, joined
only at a C⁰/C¹ boundary per F3) — never as multiple Bézier sub-spans sharing
control points *within* one vertex. The latter is explicitly the out-of-scope,
provably-wrong shortcut F2 rules out and `AGENT.md` §8 pitfall 10 names
directly ("do not add multi-span-per-region as a quick shortcut for extra shape
freedom — it reintroduces F2's interlock locally within a vertex"). Read as
written, §5 risk #1 doesn't specify which of these it means; if pursued, it
must be the former.

### 7.4 Not a conflict, but easy to misread as one: this is not rolling horizon

Explicit discrete time steps might look, superficially, like the rolling-horizon
windowed re-solving `AGENT.md` §1 and pitfall 12 rule out of scope. It isn't:
each robot still solves once, over its full mission horizon, with no partial
commit or window expiry — `k` here indexes graph depth within one single solve,
not a re-planned window boundary. Worth stating explicitly if this idea is
ever presented for review, to preempt exactly that misreading.

## 8. Scope decision (2026-08-08): single-agent first, `lazy-ecd-coarse-search-idea.md` deferred

Current working scope is **single agent only**. §7.2's tension (this idea's
`(region, k)` product graph growing the *base* graph, vs.
`lazy-ecd-coarse-search-idea.md`'s goal of keeping it small) only bites once
multiple agents are reserving against each other -- with one agent there is no
reservation-driven fragmentation to compare against yet, so §7.2 is **deferred,
not resolved**, and does not block single-agent work. Likewise §4's third bullet
(replacing `ecd.py`'s reservation pipeline with `(region, k)` cell
marking/removal) is multi-agent-only and out of scope for now. §7.1's
`time_slabs(Δ)` cross-check is also multi-agent-motivated (it's about *shared*
cut times across robots) and doesn't need to be settled before single-agent
work either -- it only bears on whether §3-§4 are worth pursuing *beyond* the
single-agent case.

This narrowing matches `AGENT.md` §5's own implementation order: single-robot
solve (§5 item 4) before the N-robot fixed-priority loop (§5 item 5). Same
staging here.

**What "single agent" actually means for this idea.** With no other agent to
synchronize against, the payoff isn't cut-time sharing (§1) -- it's validating
the mechanism itself: does a `(region, k)` product graph with fixed Δ per edge
correctly build, does `graph.py`'s per-vertex block shrink as §4 describes,
does a feasible curve exist end to end for a hand-built corridor, and does the
Δ-feasibility risk (§5 #1) actually bite on realistic region sizes. That's a
self-contained, single-robot question, answerable with the existing §5
Implementation Order-style hand-built 2-4-room corridor map (`AGENT.md` §5 item
4), no reservation/ECD code involved at all.

**One consequence worth recording here (raised in conversation, not yet in the
doc): the minimum-time objective becomes discrete even for a single agent.**
Under fixed one-edge-one-Δ, total duration for any solution is exactly `(path
length in edges) × Δ` -- an edge that geometrically needs less than a full Δ at
`vlimit` still consumes the whole Δ, so the achievable time set is `{Δ, 2Δ,
3Δ, ...}`, not the reals. This means:
- `A_dt_total` (`graph.py:505-508`, the continuous L2Norm cost on
  `T_{order-1}-T_0`) becomes vestigial once `T` is fixed per edge -- there is
  nothing left for the continuous solver to optimize about time.
- Minimum-time planning moves entirely into the discrete search as "minimize
  edge count," i.e. uniform-edge-weight shortest path over the `(region, k)`
  graph. The continuous solve per candidate path only needs to check
  *feasibility* within the fixed Δ-per-edge budget.
- `h_mot` and the search heuristics generally need attention -- **see §11,
  which corrects this bullet as originally written.** `h_mot` does not
  strictly need to *become* an edge-count bound for correctness (it stays
  admissible either way); the actual finding is narrower and different, and
  involves `h_tri`/`h_tab` more than `h_mot` itself.

## 9. Next steps (revised for single-agent scope)

Supersedes §6. In order:

1. Hand-build a small 2-4 room corridor map (matching `AGENT.md` §5 item 4) and
   size Δ against `vlimit` and that map's region extents -- check §5 #1's
   feasibility risk isn't degenerate before building anything else.
2. Build the `(region, k)` product graph for that one map, single agent, no
   reservations, for **both (a-i) and (a-ii) separately** (§3) -- they diverge
   materially in implementation size (§4) and in joint-continuity risk (§10).
   Confirm each produces a feasible curve through a bent 2-region corridor
   (same acceptance criterion `AGENT.md` §5 item 4 already uses for the
   free-time case), *and* through a tapering-scale corridor (differently-sized
   adjacent regions, not just a bend) to specifically exercise §10's risk --
   expect (a-i) to pass more easily than (a-ii); confirm rather than assume.
3. Confirm the discrete min-time objective (§8, this section) works end to
   end: discrete search minimizes edge count, continuous solve only checks
   feasibility per candidate path.
4. Only after 1-3: revisit §7.1 (`time_slabs(Δ)` as a cheaper alternative) and
   §7.2 (the `lazy-ecd-coarse-search-idea.md` tension) once a second agent and
   reservations enter the picture.

## 10. Risk: fixed Δ may reduce joint C1 slack, threatening F3's own feasibility margin

Raised in conversation, 2026-08-08. **Corrected the same day: this risk is
specific to (a-ii) [full-lock]. Under (a-i) [endpoint-lock] it largely does not
apply** -- see the split below. §3 did not originally distinguish the two
variants, so this section, as first written, implicitly assumed (a-ii)
throughout.

Physical C1 continuity at a GCS-edge joint is a *ratio* condition,
`ΔP_A/ΔT_A = ΔP_B/ΔT_B` (F3's linear joint constraint on `(P,T)`). Under free
time, `ΔT_A` and `ΔT_B` are decision variables -- an extra knob the solver can
use to reconcile adjacent regions whose natural control-polygon scale differs
(e.g. a wide room feeding a narrow doorway -- tapering, not just bending).

**Under (a-ii) [full-lock]:** interior `T_i` are forced uniform, so
`ΔT_A = ΔT_B = Δ/(order-1)` becomes a constant on both sides of every joint,
and the ratio condition collapses to a literal equality `ΔP_A = ΔP_B` -- same
direction *and* magnitude, not just matching ratio. This is strictly more
constraining and removes exactly the freedom that reconciles differently-
scaled adjacent regions today. F3's own cited validation ("verified
non-empty... down to a 0.1-wide bent corridor and a 3-bend zigzag chain") was
measured under free time, and there is no reason to assume that feasibility
margin survives once `ΔP_A = ΔP_B` is forced literally -- this must be
rechecked, not assumed to carry over.

**Under (a-i) [endpoint-lock]:** the local `ΔT_A` (gap between segment A's
last two control points, `T_{order-1}` fixed but `T_{order-2}` free) and
`ΔT_B` (gap between segment B's first two, `T_0` fixed but `T_1` free) are
*both still free decision variables*. The joint ratio-matching condition keeps
essentially the same slack it has under fully-free time -- the only
difference from today's baseline is that each segment's *total* duration is
capped at Δ rather than freely chosen, which is a separate concern (§5 risk
#1's per-Δ feasibility), not a joint-continuity one. So (a-i) is expected to
preserve F3's cited feasibility margin much more closely than (a-ii); this
should still be verified, not assumed, but the *a priori* risk is
substantially lower.

**Action:** step 2 of §9's next steps should test both variants separately on
a tapering-scale joint (not just a bend), since they're expected to diverge
here -- (a-ii) is the one actually at risk.

**Secondary effect (applies to both variants):** `A_dt_total`
(`graph.py:505-508`) becomes vestigial either way, since it only depends on
`T_{order-1}-T_0`, which is fixed to Δ under both (a-i) and (a-ii) -- nothing
is left for the continuous solver to minimize about time. Under (a-ii),
whatever spatial slack remains has no default pressure toward smoothness at
all, so `energy_weight` likely needs to be on by default. Under (a-i), the
interior `T_i` are still free, so there is at least some residual solver
freedom beyond pure feasibility-seeking, though whether that freedom actually
produces smooth/rounded shapes without `energy_weight` is unverified either
way.

## 11. Correction: what actually needs to change in `stgcs/bfs/heuristics.py`

Raised in conversation, 2026-08-08, after reading `heuristics.py` directly
(previously reasoned about only informally). **Corrects §8's heuristic bullet,
which claimed `h_mot` "would need to become an edge-count lower bound" -- too
strong and not quite the right target.**

**`h_mot` remains admissible without modification.**
`MotionOnlyHeuristic.travel_time_lower_bound`
(`heuristics.py:35-38`) computes a physical bound,
`max(|start-goal|)/vlimit`, that holds for *any* trajectory obeying the
velocity limit -- independent of how time is parametrized. Fixing Δ per edge
only adds constraints on top of the same physical velocity limit (§4), so
true discretized-optimal cost is always `>=` continuous-optimal cost, which
`h_mot` already lower-bounds. Nothing here needs fixing for correctness.

**What actually degrades is tightness, and the fix is a cheap *additional*
heuristic, not a repaired `h_mot`.** `g` (`n_next.f = n_next.sol.duration`,
e.g. `heuristics.py:685`) now only takes values in `{Δ, 2Δ, ...}` (§8), while
`h_mot` reports a continuous number -- a persistent slack of up to ~Δ almost
everywhere, weakening pruning without breaking correctness. Since the
objective is edge-count (§8), plain topological hop-distance over the
`(region, k)` graph (ignoring geometry) is *also* a valid admissible bound --
any real feasible path is a subset of graph paths, so hop-count is always
`<=` true edges-needed -- and it is a single backward BFS, cheaper than
`h_mot`'s per-query geometric computation. `hop_count × Δ` combines with
`h_mot` via the `MaxHeuristic` class already in this file
(`heuristics.py:183-229`) -- no new composition machinery needed, just a new
`Heuristic` implementation to add to the max.

**Bigger finding: `h_tri`/`h_tab` mostly lose their reason to exist under this
scheme.** Both spend real solver time calling `solve_convex_restriction` to
get `sol.duration` as a lower-bound cost for small sub-paths
(`TripletRelaxationHeuristic._calc_lower_bound_cost`, `heuristics.py:361-366`;
`InterfaceToSetCostTableHeuristic._backward_search_dijkstra`,
`heuristics.py:645-690`, same pattern). Under fixed-Δ, once a sub-path is
known *feasible*, its cost is deterministic -- `(edge count) × Δ` -- no SOCP
needed to learn it. Their remaining value shifts entirely to *feasibility
checking* (is this triplet/interface realizable at all under
containment/velocity constraints), not cost estimation -- a different,
narrower job than either is currently built around. Whether to keep them
(as feasibility-only pruners, likely cheaper to run that way too since a
feasibility check can short-circuit where a full duration-optimizing solve
cannot) or retire them in favor of the hop-count heuristic above is open and
should be measured, not assumed.

**Separate, still-open concern -- not resolved by this section:** dominance
checking (`AStarDominanceCheck`, `SetContainmentDominanceCheck` used inside
`h_tab`'s own precomputation at `heuristics.py:657`) still needs `(region, k)`
identity keying under a product graph, per §7.2/§7.4 item 4 -- this section is
about heuristic *cost estimation*, a different piece of the search machinery.

## 12. Confirmed: ECD fragmentation breaks the edge=Δ invariant this idea depends on

Raised in conversation, 2026-08-09 ("if we adopt locked timestep, ECD won't
work as is"). Verified by reading `ecd.py` and `add_edges_bidir` directly
(previously only referenced secondhand via `lazy-ecd-coarse-search-idea.md`).
This is stronger than a scope/tension note (§7.2) -- it's a specific,
identified correctness mechanism, confirmed against the actual code, not a
concern awaiting verification.

**The mechanism.** `add_edges_bidir` (`graph.py:189-204`) has three cases
depending on how two vertices' `time_itvl`s relate:
```
u.time_itvl.end == v.time_itvl.start  -> edge u -> v      (sequential)
v.time_itvl.end == u.time_itvl.start  -> edge v -> u      (sequential, other way)
otherwise                             -> edges both ways   (co-temporal / overlapping)
```
The third branch exists specifically for what ECD splitting routinely
produces: `_apply_ecd_pairs`/`slice()` (`ecd.py:56-94`, `286-318`) carving one
obstacle out of a region generally yields convex fragments that share
*overlapping*, not sequential, time ranges -- pieces of the same time window,
spatially separated only because the freed area stopped being convex.
`add_edges_bidir` correctly treats this as "same moment, different convex
piece" today, wiring both directions with no notion of elapsed time between
them (and, under the current free-time architecture, F4's `dt` floor lets the
solver cross such an edge for an arbitrarily small time cost -- effectively
free).

**Why this is fatal to the fixed-Δ premise specifically, not just untidy.**
§8/§9/§11 all rest on "graph edge = Δ, so path duration = edge count × Δ."
Nothing distinguishes a same-time-window fragment edge from a genuine
`(region,k) -> (region,k+1)` transition edge once both exist as ordinary graph
edges -- traversing one under the naive rule charges a full Δ for moving
between two pieces that are, by construction, still in the *same* Δ-slab.
Worse than a bookkeeping error: since a GCS vertex is a full private Bézier
segment (F1-F3), hopping `piece_free1 -> piece_free2` requires solving an
*entire second segment* with its own `[T_0, T_{order-1}]` window -- there is
no existing mechanism for that segment to legitimately claim "zero additional
time, I'm describing more of the same slab." The implicit bijection "one GCS
vertex ≈ one Δ-slab," which §4's product-graph description assumes throughout,
is exactly what ECD fragmentation breaks: a single conceptual time-slab can
require *multiple* GCS vertices once an obstacle carves through it
non-convexly, and none of §4/§8/§9/§11's edge-count bookkeeping currently
distinguishes those from real transitions.

**Applies equally to (a-i) and (a-ii)** -- this is about edge *cost semantics*
generally, independent of how much of the per-segment `T` is locked.

**Scope, per §8:** does not block current single-agent, no-reservation work --
ECD never runs there. It is, however, a *confirmed* blocker (not merely a
deferred tension) for the moment a second agent and reservations enter,
sharpening §7.2's more general concern into a specific mechanism with a known
location in the code.

**Not resolved here -- two directions worth weighing later, neither
committed to:**
1. Give ECD-fragment edges a distinct type from real transition edges, and
   have all edge-count bookkeeping (path cost, `h_mot`/hop-count from §11,
   `T` assignment by graph depth) count only real transitions -- i.e. `k` is
   no longer literally "graph depth," it's "count of real-transition edges in
   the path prefix." Threads through everything built on "graph depth = k"
   so far.
2. Avoid creating multiple GCS vertices for one time-slab in the first place
   -- e.g. handle same-slab non-convexity some other way than ECD's normal
   split-and-reconnect. Unclear what this would look like without breaking
   GCS's convexity requirement per vertex, and not sketched further here.

## 13. Reservation geometry simplifies to a prism (spatial hull × fixed Δ-window) -- exact, not approximate, but more conservative than today

Raised in conversation, 2026-08-10 ("the problem degrades to planning with
trajectory projected as a prism where the xy vertices are the control points
projected onto the space dimension"). Confirmed: this follows directly from
F1, not as a new approximation.

**Why it's exact.** F1: the curve `(x(s),t(s))` lies in
`conv(Q_0,...,Q_{order-1})` for the whole segment. Convex hull commutes
exactly with any linear map: `π(conv(S)) = conv(π(S))`. Dropping the time
coordinate is one such map, so the spatial curve `x(s)` alone lies in
`conv(P_0,...,P_{order-1})` -- the hull of the control points' spatial
coordinates only. No new theorem, F1 applied after projecting away a
coordinate.

**Why fixed-Δ is what makes this the natural representation.** Under free
time, a segment's time extent isn't known until after solving, so the hull
has to be built jointly in `(d+1)` dimensions with slanted facets
(`bezier_decision_boundaries`, `parallelotope_side_halfspace_kd`). Under
fixed-Δ, every segment's time extent is `[kΔ,(k+1)Δ]`, known in advance --
so the reservation is exactly `hull({P_i}) × [kΔ,(k+1)Δ]`, a prism. This also
means `slice()`'s bottom/top temporal-cropping machinery (`ecd.py:299-304`)
isn't merely degenerate here (as traced for §12) -- it's unnecessary outright,
since there's no "before/after" split left once every region's time extent
already matches the slab it lives in. Minkowski inflation collapses from a
`(d+1)`-dim sum to a plain `d`-dim spatial one for the same reason (the
footprint already has zero time-extent per `reserve_spline`'s own docstring,
so inflation and projection commute).

**What this simplifies, concretely:** subtraction reduces from general
`(d+1)`-dimensional H-polyhedron facet-cutting to plain `d`-dimensional
spatial polygon subtraction, re-extruded by the slab's known, fixed Δ. Cheaper
to compute, and probably easier to reason about correctness of, than the
current general-hull machinery.

**What this does NOT fix: §12 is untouched by this.** Prism-izing simplifies
the *geometry* of subtraction; it says nothing about how many GCS
vertices/edges a fragmented slab needs or what each edge should cost. Two
spatial pieces of the same slab, both re-extruded to the identical
`[kΔ,(k+1)Δ]`, are still two separate GCS vertices needing two separate
Bézier segments (F1-F3). §12's open question -- should hopping between them
cost `0` or `Δ` -- is exactly as open after this section as before it.

**Trade-off worth naming explicitly, not glossing over:** this is strictly
more conservative than the current exact scheme, not a free simplification.
`hull({P_i}) × [kΔ,(k+1)Δ]` can reserve more space-time volume than the agent
actually sweeps -- it assumes the agent could be anywhere in its spatial hull
at any instant of the slab, when the real curve only visits specific points
at specific times within it. That is a real departure from this repo's
"exact, not approximate" stance (F1's own framing; `lazy-ecd-coarse-search-
idea.md` §7's explicit argument that ECD's exactness is "not optional
polish"). Recorded here as a conscious trade this idea makes, not an
accidental regression -- whether the resulting looseness is acceptable is
unmeasured and should be checked against benchmark environments once
single-agent validation (§9) is further along.

### 13.1 Scope narrowed to (a-i) only, and the construction made concrete against `ecd.py` (2026-08-11)

Raised in conversation, 2026-08-11. Two refinements to §13, which was written
generically without pinning down which variant it assumed.

**Scope: endpoint-lock (a-i) only, for now.** §9 step 2 had (a-i) and (a-ii)
proceeding in parallel as separate things to build and compare. That is
narrowed here -- (a-i) is the one being built; (a-ii) is not pursued
alongside it for now. Worth noting this doesn't change §13's own derivation:
the prism argument only needs the segment's *time extent* `[kΔ,(k+1)Δ]`
known in advance, which (a-i) alone already guarantees (`T_0`, `T_{order-1}`
are the locked endpoints under both variants, per §3) -- (a-ii)'s additional
interior-`T_i` locking was never load-bearing for §13, only for the separate
dimension-reduction payoff §4 describes. So narrowing to (a-i) costs §13
nothing.

**The construction, concretely:**
1. Take the segment's control points and drop the time coordinate --
   project each `Q_i = (P_i, T_i)` down to its spatial part `P_i` alone.
   Under (a-i) the interior `P_i` (between the two locked-time endpoints)
   are exactly the free decision variables the solver still has -- "inner
   control points" here means these, though the endpoints' spatial parts
   `P_0`, `P_{order-1}` are part of the same hull and not dropped, only
   their time coordinate is.
2. Minkowski-inflate `conv(P_0,...,P_{order-1})` by the (already
   zero-time-extent, per `reserve_spline`'s docstring) spatial footprint --
   a `d`-dimensional sum, not `(d+1)`-dimensional, per §13.
3. Extrude the inflated spatial hull along time over `[kΔ,(k+1)Δ]` -- the
   slab boundary is already known from the locked endpoints, nothing to
   solve for.

**Why "sides as half-space constraints, no more slanted time dim" is the
actual code-level change.** Today, `parallelotope_side_halfspace_kd`
(`ecd.py:357-388`) builds each side half-space with a nonzero time
coefficient by construction -- `slope = (xq[:space_dim]-xp[:space_dim]) / dt`
folded into `A[..., -1]` (`ecd.py:371,378,384`) -- because a general
(d+1)-dim control-point hull's facets are genuinely slanted in space-time
(this is also what `bezier_decision_boundaries`/`_segment_ecd_pair`'s
`mid_hs_list`, `ecd.py:525-534` and `410-436`, inherit from the hull's raw
H-representation for `order > 2`). A prism's lateral faces are the extrusion
of the spatial hull's own facets -- each one has **zero coefficient on the
time axis**, by construction, since extruding along time only ever adds
`0·t` to a purely-spatial supporting hyperplane. So the "sides" (lateral
faces) directly replace the current `mid_hs_list`/side-halfspace
construction, just computed from a `d`-dim hull instead of a `(d+1)`-dim
one, and the top/bottom caps (`time_cropping_bot_top`, `ecd.py:299-304` via
`bot_hs, top_hs`) drop out entirely as already argued in §13 (the slab
already matches `[kΔ,(k+1)Δ]` exactly, nothing to crop). Net effect on
`_segment_ecd_pair`: no `bot_hs`/`top_hs`, and `mid_hs_list` built from a
`d`-dim `inflate_hull` call instead of the current `(d+1)`-dim one --
smaller change than it sounds, since `slice()` and `_apply_ecd_pairs`
downstream are untouched (§7.1 already noted this shape of reuse).

**Still not resolved here:** §12's fragmentation-cost question (does hopping
between same-slab ECD fragments cost `0` or `Δ`) and the §13 conservativeness
trade-off are both unaffected by pinning down the construction -- this
section is only about *how* the prism's half-spaces get built, not whether
the reservation is the right amount of space to reserve or how fragments are
costed.

## 14. External prior art: Morozov et al. 2025 (SWP in GCS) validates fixed-duration
segments and offers a fix for §7.2's product-graph blowup

Raised in conversation, 2026-08-11, re: "Mixed Discrete and Continuous
Planning using Shortest Walks in Graphs of Convex Sets" (Morozov, Marcucci,
Graesdal, Amice, Parrilo, Tedrake; MIT/UCSB/TRI; arXiv:2507.10878). Read in
full (not secondhand) before writing this section.

**What the paper actually proposes (correcting an initial mischaracterization
as "two separate objectives, weighted or lexicographic").** The Shortest-Walk
Problem in GCS is a nested optimization, eq. (2): `inf_K min_{w,τ} l(w,τ)` --
an outer infimum over discrete step count `K`, an inner minimization of one
combined cost `l(w,τ)` (sum of per-vertex + per-edge costs) for that `K`. It
is not two weighted objectives. Where a literal step-count-penalty +
continuous-cost sum does appear is inside individual edge/vertex costs (their
toy example, §II-C: `l_e(x_u,x_v) = 1 + ||x_u-x_v||^2`, "the first term
penalizes the number of steps taken, while the squared displacement term
penalizes the size of each step"; similarly skill-chaining, §III-B: "cost is
defined as 1 plus the arm's horizontal displacement"). The mechanism that
makes this tractable is allowing **vertex revisits** ("walks" rather than
"paths") -- see below.

**Finding 1: fixed-duration-per-vertex-visit is externally validated, not
speculative, but the demonstrated payoff is (a-ii)'s, not (a-i)'s.** §IV-A,
their collision-free motion-planning experiment: *"we search for a cubic
Bézier curve of fixed duration Δt = 125ms at each vertex visit... Fixing the
duration allows us to explicitly enforce acceleration limits in a convex
manner during incremental search."* (Fig. 3b caption: "fixed-duration Bézier
curve per vertex visit, resulting in convex acceleration constraints.")
Results (Fig. 7, §V-A): their fixed-duration SWP is 1.5x faster to solve than
SPP-with-TOPP-postprocessing and 2.3x faster than SPP-with-non-convex
postprocessing, with trajectory duration within 1-2% of both -- i.e. fixing
duration bought a large solve-time win at a small optimality cost, in a real
implementation, not just in principle. **But** the specific benefit they
report -- acceleration constraints becoming convex during search -- is this
doc's **(a-ii) full-lock** payoff per §4 ("F8's Lorentz cone... becomes a
fixed-radius norm ball... a real reduction in constraint class"), not
(a-i)'s (§4: "(a-i) does *not* buy the dimension-reduction/
constraint-simplification payoff... that payoff is specific to (a-ii)").
§13.1 narrowed current scope to (a-i)-only "for now" on C1-joint-slack
grounds (§10); this paper is independent, published evidence that (a-ii)
works well in practice for a real robot arm, which is worth weighing
against that narrowing once (a-i) validation (§9) is done -- (a-ii)'s §10
joint-slack risk is still real and this paper doesn't address it (no
adjacent differently-scaled-region joints in their benchmarks), so this is
not a reason to abandon (a-i) first, just a reason not to let (a-ii) drop
out of consideration permanently.

**Finding 2: their vertex-revisit "walk" mechanism is a direct, existing fix
for §7.2's tension, more concrete than anything currently in this doc or in
`lazy-ecd-coarse-search-idea.md`.** Fig. 5 draws precisely the
`(region,k)`-style layered/duplicated graph this doc's §4 proposes ("built
that way from the start") and states plainly: *"This layered construction
was considered [Marcucci's thesis, §10.2.3], where it was shown to be
computationally expensive. Moreover, for the shortest walks, this approach
is intractable as it requires solving SPP queries over increasingly larger
GCS instances."* Their fix is not lazy/on-demand refinement (the mechanism
`lazy-ecd-coarse-search-idea.md` proposed and whose own §9 found fatally
flawed) -- it's allowing the discrete search to **revisit the same GCS
vertex** multiple times, each visit contributing its own fixed-duration
segment, with `K` (and hence total duration `K×Δt`) emerging from an
incremental greedy search guided by a semidefinite-programming-derived
cost-to-go lower bound (§IV-B-D), never materialized as `(region,k)`
vertices. Quoting their own framing directly against this doc's concern:
*"the shortest-walk formulation naturally resolves this trade-off by
allowing vertex revisits. Vertex duplication thus becomes unnecessary, and
the problem's complexity is not artificially inflated."* This is a strictly
more relevant precedent than either of this doc's two prior attempts at the
same problem (§4's "built that way from the start" product graph, §7.2's own
correction toward lazy `SearchNode`-carried `k`) -- it suggests the right
target architecture is neither, but a walk-based search over the *existing*,
un-duplicated GCS, with per-vertex-visit fixed duration standing in for this
doc's per-edge Δ.

**Secondary implication, not yet acted on:** if this repo's search moves to
a walk-based formulation, §7.2/§7.4's question of whether dominance checking
needs `(region,k)` identity keying may be the wrong question -- their
cost-to-go `J*_v(x_v)` is defined per graph vertex `v` alone, independent of
visit count, precisely because duration accumulates along the walk rather
than being encoded into vertex identity. Not confirmed against this repo's
`AStarDominanceCheck`/`SetContainmentDominanceCheck` machinery (`heuristics.py`,
`dominance_check.py`) -- flagged here as a promising direction, not verified.

**What this paper does NOT address, so does not resolve here:** §12's ECD
fragmentation problem. Their GCS is built once via IRIS-NP and never
re-fragmented by other agents' reservations mid-search -- they have no
reservation/multi-agent-carving mechanism analogous to `ecd.py` at all. §12's
question (does hopping between same-slab ECD fragments cost `0` or `Δ`, and
does a walk-based search change that) is untouched by this section and
remains exactly as open as §12 left it.
