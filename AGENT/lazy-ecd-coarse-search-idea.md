---
tags:
  - gcs
  - reservation-geometry
  - search-based
  - proposal
  - working-document
---
# Idea: decouple discrete-search cost from ECD fragmentation via a coarse graph + lazy, on-demand region refinement

> **Status:** Proposal / working document. Nothing here has been implemented or
> benchmarked yet. This is meant to be edited in place as the idea gets refined --
> add findings, revise the design, cross out rejected variants, don't just append.
> Companion to `AGENT.md` (F1-F8, the trajectory representation and reservation
> theory this idea builds on top of, unchanged) and
> `search-based-bezier-implementation-plan.md` (the concrete state of this repo's
> search-based solver this idea would modify).

---

## 1. The problem this is trying to solve

`notebooks/search_based_stgcs.ipynb` Part 8 runs 10 agents, fixed priority order,
planning sequentially into a single initially-empty region via the search-based
planner (`SearchPlanner` + `AStarDominanceCheck` + `MotionOnlyHeuristic`), each
agent's solved trajectory permanently reserved (`ecd.reserve_spline`) before the
next agent plans. Measured on one run (`STRESS_SEED=7`, `STRESS_W=10`):

| agent | search (s) | reserve (s) | vertices after | edges after |
|---|---|---|---|---|
| 0 | 0.001 | 0.041 | 17 | 123 |
| 1 | 0.103 | 0.576 | 163 | 2,081 |
| 4 | 2.182 | 1.696 | 1,045 | 13,799 |
| 7 | 16.206 | 3.676 | 2,460 | 33,495 |
| 9 | 45.141 | 4.568 | 4,248 | 58,663 |

Two things this data established (see the notebook's own Part 8 discussion for
the full table and reasoning):

1. **Search time, not reservation time, dominates the wall clock overall**
   (~118s vs. ~20s summed across the run), and dominates increasingly sharply as
   agent index grows (ratio search/reserve goes from ~0.02 at agent 0 to ~9.9 at
   agent 9) -- even though reservation cost is what's *driving* the graph size
   that then makes search expensive.
2. **The root cause is that `reserve_spline` -> `_apply_ecd_pairs` eagerly
   fragments the *entire graph*, permanently, on every reservation** -- every
   vertex whose AABB overlaps any part of the reserved obstacle gets split,
   regardless of whether any future query will ever traverse it. Vertex count
   grew 1 -> 4,248 and edge count 0 -> 58,663 over 10 reservations. The discrete
   search (`SearchAlgorithm.run`/`expand_node`,
   `stgcs/bfs/best_first_search.py:570-609`) then has to explore an
   ever-larger candidate space every subsequent query, with only
   `AStarDominanceCheck` (a plain same-named-vertex cost dedupe, no
   cross-vertex/reachable-set pruning -- cone-aware dominance checks are
   explicitly descoped, see `search-based-bezier-implementation-plan.md` §3.1/
   §3.5) and `MotionOnlyHeuristic`/`h_mot` (an edge-local, 1-step-lookback
   travel-time bound, not a full-history-aware one -- see its
   `motion_only_domain_with_cache_policy`, `stgcs/bfs/heuristics.py:82-119`) to
   cut down the branching. Neither of those was ever going to fix the underlying
   issue: the graph itself is bigger than it needs to be for almost every query.

## 2. The core idea

Fragmentation of a region is only *required* at the point where the continuous
convex program (`solve_convex_restriction`) actually needs a convex domain for
that region -- **not** at the point where some other, unrelated region gets
reserved against. The discrete search step (choosing *which sequence of regions*
to traverse) never needs the fragmented graph at all: region *adjacency* doesn't
change when a region's *interior* gets an obstacle carved out of it, only which
sub-volume of that region is actually free.

So: keep the discrete search operating over the small, fixed-size **original
(coarse, unfragmented) region adjacency graph**, always. Only compute a region's
actual current convex pieces **lazily, on demand, cached per region** -- the
first time some candidate path from the coarse search actually needs to solve
`solve_convex_restriction` through it. If two agents' candidate paths never touch
the same coarse region, neither ever pays fragmentation cost for it, no matter how
many reservations have piled up elsewhere in the graph.

This is not new math -- it's a lazy/incremental scheduling of the *same* ECD
subtraction this repo already does eagerly. It doesn't change what a valid
solution looks like (F1-F8 are solve-strategy- and schedule-agnostic, per
`porting-bezier-to-search-based-stgcs.md` §0), only *when* the splitting work
happens and *how much of the graph* it's spent on per reservation.

## 3. Why this is sound, not just plausible

**Heuristic admissibility survives unchanged.** `MotionOnlyHeuristic`'s bound is
a function of a region's spatial extent (`find_min_travel_time_state_to_spatial_point`
over the region's vertex set). The original, unfragmented region is a superset of
any of its post-ECD-split remains -- minimizing travel time over a superset can
only produce a value `<=` the true minimum over the (tighter) true remaining free
volume, so it's still a valid lower bound, just a looser one in general (looser
exactly to the extent that the region has actually been carved into). No
Bezier-specific or fragmentation-specific rederivation needed here, matching how
`h_mot` already needed zero changes for `order>2` (`search-based-bezier-
implementation-plan.md` §7).

**The convexity requirement doesn't go away -- it just gets deferred.** A region
with a hole in it is not convex, and `solve_convex_restriction`'s GCS/SOCP
transcription fundamentally requires one convex set per vertex (that's the reason
ECD exists at all). So "search the coarse graph" cannot mean "solve the continuous
program over the coarse region directly" once that region has *any* reservation
inside it -- the moment a candidate path from the coarse search actually needs to
traverse a region, that region's *current, real* convex pieces are required for
the solve, same as today. The only thing this idea changes is *eagerness*: split
now, for the whole graph, on every reservation (today) vs. split only the specific
region a query actually needs, the first time it's needed, and cache the result
(proposed).

**This repo already has the single-region primitive this needs.**
`stgcs/ecd.py:reserve_region_bezier` (lines ~437-478) already does exactly "slice
one named region against one segment's inflated hull, reconnect its neighbors,
leave everything else untouched" -- it's currently only used for
testing/demonstration (`notebooks/ecd_bezier_visualization.ipynb` Parts 5-7,
`tests/unit/test_reserve_region_bezier_unittest.py`), not wired into the search
loop. Its own docstring and the porting doc (§4, table entry for
`reserve_region_bezier`) already document the two composability rules a lazy
per-region scheme would need to respect:
1. Check every region's *original* extent for an AABB touch when deciding whether
   a given reservation affects it -- not just a segment's nominal "home" region --
   since Minkowski inflation routinely reaches past a shared region boundary.
2. Reserve against a region's *current* surviving pieces (via `STVertex.root_name`/
   `.root`, already implemented in `stgcs/graph.py`), not a stale/removed vertex
   name, since splitting is recursive.
`STVertex.root`/`.root_name` (`stgcs/graph.py:52-68`) already track which coarse
region a fragment descends from -- exactly the bookkeeping a lazy scheme needs to
answer "what coarse region does this piece belong to" and "has this coarse region
already been (partially) refined."

## 4. Proposed architecture (sketch, not final)

```
LazyRegionState (per coarse region name):
    original: HPolyhedron            # the untouched region, as built at graph creation
    applied: List[ECDPair]           # reservations already folded into `pieces`
    pieces: List[str]                # current live vertex names descending from this region
                                      # (== [original_name] until first refinement)

get_current_pieces(region_name, all_reservations) -> List[str]:
    state = region_states[region_name]
    pending = [r for r in all_reservations if r not in state.applied
                                            and AABB(r.bounds, original_extent(region_name))]
    for r in pending:
        stgcs = reserve_region_bezier(stgcs, <current piece(s) of region_name>, r.control_points, r.footprint)
        state.applied.append(r)
    state.pieces = [v for v in stgcs.G.nodes if stgcs.get_vertex(v).root_name == [region_name]]
    return state.pieces
```

Discrete search (`SearchAlgorithm.run`/`expand_node`) runs over the **coarse**
adjacency graph, unchanged in size regardless of reservation count. When it needs
to actually evaluate a candidate coarse-vertex sequence (today: build `E` and call
`solve_convex_restriction`), it first resolves each coarse region in the candidate
to its current piece(s) via `get_current_pieces` (lazy, cached, see above), then:

- If a consistent, connected sequence of *actual pieces* exists that realizes the
  coarse-level candidate (i.e. consecutive pieces really are adjacent in the
  now-fragmented graph -- not guaranteed, since a reservation can sever exactly
  the sub-area that used to connect two coarse-adjacent regions), solve
  `solve_convex_restriction` over that concrete piece sequence, same as today.
- If no such connected piece sequence exists, that coarse-level edge is
  (currently) broken by prior reservations -- **this is the "conflict" case**.
  Two options, matching the two branches from the conversation this doc is
  recording:
  - **"escalate fidelity":** nothing further to escalate here, actually --
    `get_current_pieces` already computed the exact, fully-resolved current
    fragmentation for both regions; "escalate" really means: try the other
    surviving piece(s) of the same coarse regions (there may be several after
    fragmentation) before giving up on this coarse edge.
  - **"find a nearby state":** if no piece-level connection exists at all, mark
    this coarse edge infeasible *for now* and let the search continue with its
    next-best coarse candidate (ordinary A*/best-first backtracking -- no new
    algorithm needed here, just don't treat a piece-level dead-end as a reason to
    abort the whole coarse-level search).

## 5. What this buys, concretely

- Discrete search graph size stays ~constant (bounded by the number of original,
  hand-built or IRIS-decomposed regions), instead of tracking total accumulated
  fragmentation across every prior agent. This directly targets the dominant,
  fastest-growing cost measured in Part 8 (search: 0.001s -> 45s over 10 agents).
- Reservation/fragmentation cost is paid only for regions some agent's candidate
  path actually needs, not for every region within AABB reach of an obstacle.
  `add_edges_bidir`'s reconnection cost (already identified as the dominant
  component of a *single* reservation call, `search-based-bezier-implementation-
  plan.md` §3.6, `ecd_bezier_visualization.ipynb` Parts 11-12) should shrink
  correspondingly, since it only ever needs to reconnect the pieces of regions
  actually touched by a live query, not the whole graph.
- No change to F1-F8, to the trajectory representation, or to what counts as a
  correct/optimal solution -- this is purely a scheduling change for when/how much
  ECD work happens, built on primitives (`reserve_region_bezier`, `root_name`)
  already implemented and already unit-tested.

## 6. Open questions / risks to resolve before implementing

1. **Piece-level adjacency bookkeeping.** Today `add_edges_bidir` derives adjacency
   generically via `HPolyhedron.IntersectsWith` (AABB-gated). A lazy scheme needs
   an efficient way to answer "do piece P (of coarse region A, freshly resolved)
   and piece Q (of coarse region B, freshly resolved) actually connect" without
   re-deriving the whole graph's adjacency -- likely just the existing
   `add_edges_bidir` check, scoped to the specific new pieces, but this needs
   confirming it doesn't secretly require a wider rescan.
2. **Cache invalidation ordering.** `LazyRegionState.applied` assumes reservations
   can be folded in incrementally, one at a time, regardless of order, and still
   reproduce the same result as applying them all at once via `reserve_spline`.
   The porting doc's own composability findings (§4, `reserve_region_bezier`'s
   docstring) suggest this holds (subtraction is associative), but it should be
   verified directly for the lazy/incremental case specifically, not assumed --
   the sequential per-segment reconstruction in `ecd_bezier_visualization.ipynb`
   Part 7 is the closest existing precedent and the right template for a direct
   check.
3. **Dominance-check identity.** `AStarDominanceCheck` currently dedupes by
   *named vertex* (`stgcs/bfs/dominance_check.py:31-67`). Under a coarse-graph
   search, is "vertex identity" the coarse region name, or does it need to be
   (coarse region, resolved piece) once pieces diverge? Two different agents'
   candidate paths might resolve the same coarse region to different pieces
   depending on what's been reserved by the time each is evaluated -- needs a
   concrete answer before this is anything more than a sketch.
4. **Does resolving a coarse candidate to pieces ever need to consider *multiple*
   candidate piece-sequences per coarse edge** (i.e. genuine backtracking within
   one coarse-level expansion, not just moving on to the next coarse candidate)?
   The "escalate fidelity" bullet above assumes trying alternate surviving pieces
   is enough, but this hasn't been checked against a case where a coarse region
   has fragmented into many small pieces with genuinely different connectivity.
5. **Interaction with the heuristic's own edge-local caching.** `h_mot`'s cache
   key is `(predecessor_name, v_name)` (`heuristics.py:133`) at the *fragment*
   level today; under a coarse search this would presumably move to
   `(predecessor_coarse_name, coarse_name)`, which is an even coarser cache
   (fewer distinct keys, more reuse) -- worth confirming this doesn't lose
   admissibility the same way §3 argues, but stated here explicitly as something
   to re-derive, not assume.

## 7. Relationship to prior art and to this repo's own deferred ideas

This is the same family of idea as **lazy collision checking** (Lazy PRM: defer
expensive edge-feasibility computation until an edge is actually on a candidate
best path) and **Conflict-Based Search** in MAPF (plan optimistically/ignoring
some constraints, detect conflicts, resolve locally, replan -- rather than
encoding every constraint up front).

**It's also, structurally, an adaptive spatial index -- an octree/quadtree, but
with exact rather than blind cuts.** The "refine only where and when a query
actually needs it, leave everything else as one coarse cell" shape of this
proposal is exactly the sparse/adaptive-octree pattern used for occupancy mapping
(e.g. OctoMap): don't eagerly subdivide the whole volume down to leaf resolution,
subdivide a cell only once something inside it demands finer resolution, and only
that cell. The difference from a textbook octree is *where the cuts go*:

- A standard octree/quadtree splits **blindly** -- every subdivision is a fixed
  rule (bisect each axis at the geometric midpoint, always 2^d children),
  independent of what's actually inside the cell. Matching an arbitrarily
  oriented or curved obstacle boundary then requires many levels of recursive
  halving (the classic "staircase" approximation of a slanted edge), and the
  approximation is inherently lossy: a leaf cell is marked wholesale
  occupied/free even though the true boundary cuts through its interior.
- ECD's cuts are **exact and obstacle-derived**: `bezier_decision_boundaries`
  places each cutting hyperplane exactly on one facet of the actual (Minkowski-
  inflated) reservation hull (`stgcs/ecd.py:425-435`), not at a fixed geometric
  midpoint chosen without looking at the data. A slanted or curved swept volume
  (generic under `order=4` -- Bezier reservation hulls are not axis-aligned the
  way `order=2`'s parallelotope tubes are) gets exactly the pieces its own facet
  count requires -- bounded independent of orientation (the porting doc's own
  facet-count bound, `porting-bezier-to-search-based-stgcs.md` §2's `hulls.py`
  entry, cites <=14 true facets for a non-degenerate cubic segment in 2D+time) --
  not an orientation-dependent number of recursive halvings to approximate the
  same boundary to similar fidelity.
- This exactness isn't optional polish here the way it might be for an occupancy
  grid: each "cell" in this scheme is fed directly to a continuous SOCP solver as
  a literal optimization domain, and the optimizer will happily place a
  time-optimal trajectory arbitrarily close to the true free-space boundary. A
  blind octree's approximation error would have to resolve one of two ways --
  mark a boundary-straddling leaf wholesale unsafe (conservative, blocks real
  free-space slivers, costs solution quality) or wholesale safe (unsound, lets the
  solver place a trajectory somewhere actually occupied) -- neither acceptable
  here, which is exactly why ECD's exactness, not just its adaptivity, matters.
- One more precision point worth keeping in mind while developing this further:
  because ECD's cuts can be arbitrarily oriented and a region's facet count
  varies with what's actually reserved against it (not a fixed branching factor),
  "lazy BSP tree with data-derived, exact cutting planes" is a more precise
  description of what this scheme actually builds than "octree" -- the octree
  framing is the right *intuition* for the adaptive-refinement/laziness half of
  the idea, but the actual geometry is closer to an obstacle-adaptive binary
  space partition than to a fixed-branching-factor, axis-aligned spatial index.

Framed against this codebase's own language:
`AGENT.md` §1 explicitly defers "rolling horizon... no `DecompositionStore`
versioning/expiry... reservations, once subtracted, are permanent for the rest of
this phase's runs," citing the goal of keeping "subsequent robots' problems...
clean convex programs" as the reason for choosing eager/permanent instead. This
proposal is a narrower, more specific version of that deferred idea: it does
**not** revisit priority ordering (still fixed-priority, per `AGENT.md` §1), does
**not** reintroduce windowed/partial re-solving, and does **not** ever un-reserve
or expire a reservation once made permanent -- it only changes *when* the
resulting ECD split gets computed (lazily, per-region, on first use) instead of
*eagerly, for the whole graph, on every reservation*. Whether that narrower scope
is enough to get the benefit in §5 without reopening the larger rolling-horizon
question is itself one of the things to establish while developing this idea
further.

## 8. Suggested first validation step

Before any of the harder open questions in §6, the cheapest way to sanity-check
the core premise: rebuild Part 8's 10-agent stress test with a hand-rolled lazy
wrapper (search over the original single "free" region's *name* only, resolve to
current pieces via `reserve_region_bezier`+`root_name` right before each
`solve_convex_restriction` call, no real coarse-vs-fine adjacency logic needed yet
since Part 8's stress test starts from exactly one coarse region) and compare the
search-time-vs-agent-index curve against the eager baseline already measured
(0.001s -> 45s). If lazy resolution alone (even without a multi-region coarse
adjacency graph) flattens that curve, it's strong evidence the graph-size
explosion, not some other factor, was really the bottleneck -- and a much cheaper
experiment than building out the full multi-region piece-level adjacency
machinery from §4/§6 first.

> **Superseded -- see §9.1 and §9.2.4.** This step cannot be run as written: the
> Part 8 stress test starts from exactly *one* coarse region, so "search over the
> original region's name only" is a search over a 1-vertex, 0-edge graph. There is
> no coarse search left to measure. §9.6 Step 1 proposes a replacement experiment
> that answers the same question (is graph size really the bottleneck?) and is
> strictly cheaper.

---

## 9. Discussion: critical review against the actual codebase (2026-07-28)

Written after reading `stgcs/ecd.py`, `stgcs/graph.py`, `stgcs/bfs/*.py`,
`stgcs/st_planner.py` and re-running Part 8 with the profiling data
`SearchPlanner.last_profile` already collects but the notebook never prints. No
code was changed to produce this section. Reproduction: Part 8's exact config
(`STRESS_SEED=7`, `STRESS_W=10`, `order=4`, `safe_radius=0.2`, `tmax=60`), driving
`SearchPlanner` directly and reporting `last_profile` plus
`last_search_algorithm.n_expanded`/`.n_generated` per agent.

### 9.1 The measurement §1 was missing

```
ag   |V|    |E|    wall gcsbuild  search cvxrest   #cvx    exp    gen segs   resv
 3   374   5258    0.70     0.29    0.41    0.38    118     10     40    9   0.96
 4   619   8431    2.24     0.48    1.76    1.71    385     29     65   14   1.71
 5  1045  13799   15.04     0.83   14.21   13.99   1404     91    181   21   1.90
 6  1484  19452    5.75     1.24    4.51    4.32    824     54    181   15   1.33
 7  1769  23143   15.60     1.54   14.06   13.70   2117    114    207   15   3.74
```

(`gcsbuild` = `profile["gcs_construction_time"]`, i.e. `STGCS.get_gcs_instance`;
`search` = `profile["main_search_time"]`; `cvxrest`/`#cvx` =
`profile["convex_restriction_time"]`/`["convex_restriction_calls"]`;
`exp`/`gen` = `SearchAlgorithm.n_expanded`/`.n_generated`; `segs` = solution
segment count; `resv` = `ecd.reserve_spline` wall time.)

Four facts here bound or contradict §1's root-cause story:

1. **Search time is ~97% `solve_convex_restriction` wall time** (13.70 of 14.06s
   at agent 7). Not graph traversal, not `h_mot`, not dominance checking. Drake
   GCS construction (`_ensure_reusable_base_gcs`, rebuilt every agent because
   `_base_gcs_signature` changes) is a separate, smaller ~10%.
2. **The search touches ~6% of the graph.** 114 expansions on 1,769 vertices; 207
   nodes ever *generated*. §1's "the discrete search has to explore an
   ever-larger candidate space every subsequent query" is not what the profile
   shows -- A\* + `h_mot` + `AStarDominanceCheck` already discard ~94% of the
   graph without ever instantiating a node for it.
3. **Branching factor is flat under fragmentation.** `|E|/|V|` = 14.1, 13.6, 13.2,
   13.1, 13.1 across agents 3-7. ECD splitting adds vertices and edges roughly in
   proportion; per-vertex out-degree does not grow. So "bigger graph ⇒ wider
   search" does not hold on this data.
4. The 34× growth in search time decomposes as **expansions (10 → 114, ~11×) ×
   per-solve cost (3.2ms → 6.5ms, ~2×)**, and per-solve cost tracks the solution's
   *segment count* (9 → 15), i.e. path length inside the SOCP -- which no lazy
   scheme removes.

**Honest caveat.** Expansions grew faster than `|V|` (11× vs 4.7×), and a larger
`|V|` does mean more distinct keys for `AStarDominanceCheck._g` to retain, so
more nodes survive dominance. Graph size and intrinsic problem difficulty are
therefore confounded in this data, which is one seed and five usable points. This
run does *not* settle the causal question -- but §9.6 Step 1 does, cheaply.

**The ceiling this puts on the proposal.** Reserve time is 3.74s against 15.60s
wall at agent 7. An oracle that eliminated *all* wasted fragmentation caps out at
~20-24% of wall time, because it does not touch the 13.70s of convex restriction.
§5's "this directly targets the dominant, fastest-growing cost measured in Part 8
(search: 0.001s → 45s)" only follows if lazy refinement also collapses the
convex-restriction *call count* -- and the mechanism that would do that is exactly
the part that breaks in §9.2.

### 9.2 Where the idea breaks

#### 9.2.1 Fatal: the coarse graph cannot represent the paths that exist (§2, §4, §8)

§2 asserts "region *adjacency* doesn't change when a region's *interior* gets an
obstacle carved out of it." In a *space-time* GCS this is false, and the code says
so directly. `update_edge` (`stgcs/ecd.py:106-127`) has two loops: the first
reconnects pieces to their former neighbors, the second connects the pieces **to
each other** via `combinations(new_verts, 2)`. That second loop creates
*intra*-region adjacency -- piece→piece edges inside one coarse region, time-ordered
by `add_edges_bidir` (`stgcs/graph.py:182-196`). The coarse graph has no
representation for these: `add_edges_bidir` early-returns on `u_name == v_name`,
so a coarse region has no self-loop.

Concretely in Part 8: the graph starts as **one** vertex named `free`, so all 4,248
eventual pieces have `root_name == ["free"]`. The coarse graph is 1 vertex, 0
edges. Agent 7's solution is 15 segments -- 15 pieces of `free` in sequence. Under
`expand_node`'s cycle check (`n.has_visited(successor)`,
`stgcs/bfs/best_first_search.py:582-584`) a coarse path cannot revisit `free`, so
the only coarse candidate is `[source, free, target]`. The abstraction map
φ(fine path) → coarse names is not a path in the coarse graph at all; it is a
15-fold repetition of one node.

This is not an artifact of the empty-square test. Threading around a reservation
*in time* means traversing many pieces of the same room, in any map -- that is what
the time dimension in ST-GCS is for. Making a coarse search sound would require a
multigraph with unbounded self-loop multiplicity, at which point it is the fine
graph with a weaker heuristic.

#### 9.2.2 Fatal for the same reason: dominance identity (closes §6.3, negatively)

`AStarDominanceCheck` (`stgcs/bfs/dominance_check.py:47-56`) keys `self._g` on
`n.vertex_name` and prunes any node reaching a vertex at cost ≥ the best seen.
Under coarse identity in Part 8 that dictionary has exactly one entry,
`_g["free"]`, and every candidate path after the first is dominated. Keying on
`(coarse, piece)` restores correctness -- and restores the fine graph. §6.3 is not
open; it resolves against the scheme.

#### 9.2.3 §4's "escalate fidelity" bullet is where the design actually lives, and it is empty

"Try the other surviving piece(s) of the same coarse regions" is, in general, a
shortest-path search in the fine graph restricted to a coarse corridor -- the
standard HPA\*-style refinement step. That step is sound when abstract paths do not
revisit abstract nodes. Here they always do. §6.4 asks whether multiple
piece-sequences per coarse edge are ever needed; Part 8's answer is yes, 15-24 of
them, all inside a single coarse node.

#### 9.2.4 Sequential refinement fragments *worse* than eager batching

`_apply_ecd_pairs` (`stgcs/ecd.py:48-81`) accumulates **all** of one agent's
`ECDPair`s per vertex and passes them to `slice()` in one call, which uses
`pairs[0].bottom_halfspace` and `pairs[-1].top_halfspace` for the before/after
time pieces. Applying pairs one at a time -- what §4's `get_current_pieces` loop
does -- creates a full-extent "top" piece for pair A that pair B then re-slices,
yielding strictly more pieces for the same free volume. This is `AGENT.md`
pitfall 8 verbatim: "Sequential subtraction of overlapping obstacles is *sound*
(subtraction is associative) but fragments badly; batch or union locally where
pieces overlap." As written, the lazy scheme would grow `|V|` *faster* than the
eager one. See §9.3 D for the fix, which is also the best argument for the idea.

#### 9.2.5 Mid-search graph mutation is unhandled (missing from §6)

§4 calls `get_current_pieces` during search. But `SearchPlanner.plan`
(`stgcs/st_planner.py:122`) builds the entire Drake `GCS` once via
`get_gcs_instance(..., reuse_base=True)` before `alg.run`. Refining a region
mid-search invalidates: the cached `_reusable_gcs` and the class-level
`_reusable_gcs_cache` signature, `_convex_restriction_cache` (keyed by
vertex-name tuples), `_edge_cache`, `h_mot`'s `(pred, v)` cache,
`AStarDominanceCheck._g`, and any `SearchNode` sitting in OPEN that names a
now-removed vertex. This is the largest engineering risk in the proposal and §6
does not list it.

#### 9.2.6 Factual correction to §3

§3 cites `tests/unit/test_reserve_region_bezier_unittest.py` as evidence the
primitive is "already unit-tested." **There is no `tests/` directory in this
repo** -- `search-based-bezier-implementation-plan.md` §2 Step 8 says so
explicitly ("No `tests/` directory exists in this repo yet"). That citation is
inherited from the *source* repo via `porting-bezier-to-search-based-stgcs.md` §4's
table. `reserve_region_bezier` is exercised only by
`notebooks/ecd_bezier_visualization.ipynb`. Treat it as demonstrated, not
verified. Every other line reference in §§1-8 checks out against the current code.

### 9.3 What survives, and established alternatives that fit better

What §§1-8 gets right and should be kept: eager fragmentation genuinely does
wasted work (most of the ~1,769 pieces never appear in any search node of any
query); `reserve_region_bezier` really is the right atomic primitive and
`STVertex.root_name` (`stgcs/graph.py:51-69`) really is the bookkeeping a lazy
scheme needs; and §3's admissibility argument is correct -- `h_mot`'s domain is the
predecessor∩current interface (`stgcs/bfs/heuristics.py:100-102`), a coarse
interface is a superset of any piece-level interface, so the bound stays valid and
(since `h_mot` ignores obstacles entirely already) barely looser. §6.5 resolves the
same way.

**A. Lazy edge evaluation (LazySP / LWA\*) -- the trivial solution §7 walked past.**
§7 correctly names lazy collision checking as the family this belongs to, then
applies the laziness to the wrong operation. The profile says 2,117 SOCP solves
for 114 expansions: ~95% of solves are for nodes pushed and never expanded.
`expand_node` (`stgcs/bfs/best_first_search.py:586-601`) solves the full-prefix
convex restriction for *every* successor before pushing it. LWA\* pushes with an
optimistic `g` surrogate and solves only on pop. The machinery for an admissible
surrogate already exists -- `find_min_travel_time_state_to_spatial_point` over the
edge interface, exactly what `h_mot` computes. This targets 97% of search time,
needs no geometry change, and touches nothing in F1-F8.
*Why it might not fit:* `g` currently comes *from* the SOCP, so a genuine lower
bound is needed to order OPEN; and the SOCP is over the whole path prefix, not one
edge, so a lazily-evaluated node may need re-solving on pop. The payoff is exactly
"fraction of OPEN never popped", which `SearchTrace` can count directly
(`status == "open"` and never `"expanded"`).

**B. The coarse graph as an admissible *heuristic*, not a search space.** This is
what the abstraction-heuristic literature (hierarchical A\*, pattern databases)
does, and it sidesteps §9.2.1 entirely, because a heuristic never has to be
realizable as an abstract path. The repo is already half-way there:
`TripletRelaxationHeuristic` maintains `_root_set2set_dist` and
`_root_triplet_lb_cost` keyed on `root_name` -- i.e. coarse region names
(`stgcs/bfs/heuristics.py:235-260`, `:345-358`). That *is* a coarse-graph
abstraction heuristic over a fine search, already implemented. It is descoped for
`order=4` right now, but re-enabling it is a change of which `Heuristic` is passed,
not new machinery.

**C. Prefix-incremental / warm-started convex restriction.** Each expansion
re-solves the whole path from `source` (`gcs_solver.py:302-332`), so extending a
length-*n* path costs O(*n*) and A\* redoes prefix work O(*n*²) per branch. The
observed per-solve doubling (3.2 → 6.5 ms) as segment count goes 9 → 15 is
consistent with this. Warm-starting from the parent's solution is standard and
orthogonal to everything else here.

**D. Lazy + *batched* ECD -- the strongest salvageable version of this proposal,
and one §5 does not claim.** Defer reservations per region, then apply *all*
pending ones in a single `slice()` call. By pitfall 8's own logic that produces
**fewer** pieces than N sequential eager applications, so the graph gets genuinely
*smaller*, not merely refined later. This is a better argument for laziness than
anything in §5, because it attacks `|V|`/`|E|` directly while preserving
exactness. Open risk, to verify rather than assume: `slice`'s own docstring
(`stgcs/ecd.py:186-187`) warns that its `ecd_pairs` "must be collected from a
continuous piece-wise linear trajectory otherwise the slicing would be incorrect"
-- batching pairs from *different agents* in one call may violate that
precondition.

### 9.4 Revised assessment of §5's claims

| §5 claim | Status |
|---|---|
| "Discrete search graph size stays ~constant" | Unsound as stated -- §9.2.1: the coarse graph cannot express the solution paths. |
| "directly targets the dominant, fastest-growing cost" | Not supported -- §9.1: the dominant cost is convex-restriction solves, and the search already ignores ~94% of the graph. |
| "Reservation/fragmentation cost paid only for regions a query needs" | Directionally right, but capped at ~20-24% of wall time, and §9.2.4 says the naive incremental form makes fragmentation worse. |
| "No change to F1-F8 / to what counts as a correct solution" | Holds. Nothing here touches the trajectory representation or the reservation geometry. |

### 9.5 Where the idea would have the most headroom

Part 8 is the *worst* case for this proposal: a 10x10 square with 10 agents all
crossing it starting from a single coarse region -- there is no region that "no
query will ever touch", and there is no coarse adjacency structure to search over.
The `benchmark/environment/` maps (`uav_village.py`, `maze/maze_env.py`,
`grid.py`, `iris.py`) are spatially large with many genuinely-untouched regions and
a real multi-region coarse graph. If the deferred-fragmentation half of this idea
(§9.3 D, without the coarse search) is worth anything, that is where it will show,
and the §9.1 ceiling should be re-measured there before drawing conclusions from
Part 8's numbers alone.

### 9.6 Concrete next steps

**Step 0.** Retire §8's validation step (already marked superseded above).

**Step 1 -- settle the premise with an oracle-pruning experiment.** ~30 lines,
read-only, no new machinery. Take the agent-7 graph; record the vertices the eager
search actually generated (`SearchPlanner(record_trace=True)` → `alg.trace`);
rebuild an `STGCS` with only those vertices and induced edges; re-run the identical
query and compare `main_search_time` and `convex_restriction_calls`. This is the
**absolute upper bound on what any laziness scheme can save**, measured directly.
Also run the realizable variant (prune to a spatial ball / reachable time-cone
around the query, which a lazy scheme could actually compute). Decision rule: if
oracle-pruned search is still ≥60% of the original, this proposal is capped below
its own §5 claim and should be scoped down to a reservation-time optimization
(§9.3 D); if ≤20%, graph size is causal and the idea deserves the investment.

**Step 2 -- measure the LWA\* headroom (§9.3 A), in parallel and independent.**
From the same trace, count nodes with `status == "open"` never marked
`"expanded"`. That fraction × 2,117 is the SOCP calls an LWA\*-style reordering
could defer. Rough arithmetic on the aggregates suggests ~90%, but that is not a
direct count.

**Step 3 -- verify batching soundness (§9.3 D) before building anything lazy.**
Two decidable sub-questions: (a) does `slice()` accept `ECDPair`s drawn from
different trajectories, or does its continuity precondition bite? (b) does
batched-across-agents subtraction produce the same free volume as sequential, with
fewer pieces? `ecd_bezier_visualization.ipynb` Part 7's sequential reconstruction
is the right template, as §6.2 already says. Measure piece count both ways. This
is the one claim in the family that could shrink `|V|` directly.

**Step 4 -- only if Step 1 says graph size is causal.** Build lazy refinement as a
*reservation-scheduling* change with the fine graph intact: defer + batch (Step 3),
keyed on `root_name`, resolved before `get_gcs_instance` rather than mid-search --
which avoids §9.2.5 entirely. Coarse-region *search* (§2, §4) stays out of scope.
Expected payoff bounded by Step 1's oracle number and by §9.1's ~20-24% reserve
share.

**Step 5 -- if a coarse abstraction is still wanted, spend it on the heuristic
(§9.3 B), not the search space.** Concretely: re-enable
`TripletRelaxationHeuristic`'s root-level tables for `order=4` and measure the
expansion-count reduction against `h_mot`. That is where a coarse abstraction is
provably safe.

Steps 1-3 are read-only measurements against code that already exists. The §4
architecture should be held until Step 1 returns a number.
