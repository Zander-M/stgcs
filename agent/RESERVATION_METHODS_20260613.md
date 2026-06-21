# Reservation Methods: BVC and CVT — Implementation & Preliminary Results

**Date:** 2026-06-13  
**Branch:** RH (rolling-horizon)  
**Status:** Initial implementation complete; preliminary benchmark done

---

## What Was Decided and Why

Implemented two new reservation methods as alternatives to ECD, following the design
proposals in `BVC_RESERVATION.md` and `CVT_RESERVATION.md`. Both are exposed as a
`reservation_method` argument (`'ecd'`, `'bvc'`, `'cvt'`) to `prioritized_planning`
in `baselines/rp_stgcs.py`.

---

## Files Changed

| File | Change |
|---|---|
| `mrmp/region_reservation/bvc.py` | New — `bvc_reserve(stgcs, trajectory, safe_radius)` |
| `mrmp/region_reservation/cvt.py` | New — `cvt_reserve(stgcs, trajectories, safe_radius, current_start, current_goal, tmax)` |
| `mrmp/region_reservation/__init__.py` | Exports both functions |
| `baselines/rp_stgcs.py` | Added `reservation_method` kwarg to `prioritized_planning`, `sequential_planning`, `randomized_prioritized_planning` |
| `tests/test_reservation.py` | Benchmark comparing all three methods |
| `tests/debug_bvc.py` | Debug script for 2-agent comparisons |

---

## Key Implementation Decision: Temporal Splitting

**Problem discovered:** The naive approach (adding the BVC halfspace to the full vertex
convex set for all time) caused infeasibility. For example, vertex v might span t=[0, 50],
but agent i only passes through it during t=[2, 4]. Without temporal splitting, the BVC
halfspace cuts j's access to v for ALL 50 time units — blocking j's goal at t=50.

**Solution:** Temporal split (like ECD). For each contested vertex v:
1. **Before** (t < t_overlap_start): full v, no constraint
2. **During** (t_overlap_start to t_overlap_end): v intersected with BVC/CVT halfspace
3. **After** (t_overlap_end to t_v_end): full v, no constraint

This uses the union of all overlapping segment time windows as the "during" interval.
Without this, Trial 3 and Trial 4 of the debug suite showed "No target vertices found"
or "No feasible path" for BVC/CVT; with it, all 2-agent debug trials succeed.

---

## Interface Difference: CVT vs BVC

Unlike ECD/BVC (which accumulate reservations sequentially into a shared stgcs), CVT
**computes a fresh per-agent stgcs** using ALL known trajectories:

```python
# ECD/BVC: cumulative
stgcs = ecd_reserve(stgcs, sol.trajectory, 2*r)   # or bvc_reserve(...)

# CVT: per-agent fresh copy
stgcs_i = cvt_reserve(stgcs, planned_trajs, r, starts[i], goals[i], tmax)
```

CVT requires passing `current_start` and `current_goal` to estimate j's generator via
linear interpolation from start to goal, evaluated at the vertex's time midpoint.

---

## Graph Size Impact (from 2-agent debug suite)

Base STGCS for simple2d: ~47–52 vertices, ~195–241 edges.

After agent 0 plans and reserves:

| Method | Avg vertices | Avg edges |
|--------|-------------|-----------|
| ECD    | ~90         | ~500      |
| BVC    | ~55         | ~245      |
| CVT    | ~50         | ~205      |

BVC and CVT produce **2–3× fewer vertices and edges** than ECD. This directly
improves rounding success in the GCS relaxation step.

---

## Benchmark Results (simple2d, n=4 agents, 10 trials)

Results vary between runs due to stochastic path rounding. Representative observations
across two runs:

| Method | Success rate | Avg time | Peak edges | Collision-safe rate |
|--------|-------------|----------|------------|---------------------|
| ECD    | 0–1/10      | ~5.5s    | n/a (fails)| 1/1 (when succeeds) |
| BVC    | 4–7/10      | ~1.0s    | ~300–350   | ~75–100%            |
| CVT    | 7–9/10      | ~0.75s   | ~220–270   | ~22–60%             |

---

## Safety Analysis

### BVC safety guarantee (within-vertex)
Within any vertex v where the BVC constraint applies, j's trajectory satisfies
`n^T x_j >= max(n^T p_i(t)) + 2r`, which guarantees Euclidean separation >= 2r
from i's trajectory WITHIN v.

**Gap:** Cross-vertex collisions are NOT prevented. Adjacent vertices that i does
NOT pass through have no constraint — j can be at the boundary of v where i is
also at the boundary of an adjacent vertex. This explains the few unsafe BVC solutions
in multi-agent runs.

### CVT safety guarantee (weak)
CVT provides separation from j's estimated Voronoi cell boundary. Since j's generator
is only a linear interpolation estimate, j may end up in the "wrong" Voronoi cell at
runtime, causing collisions. ~30–80% of CVT solutions are unsafe on 4-agent problems.

---

## Tradeoffs

| Dimension | ECD | BVC | CVT |
|---|---|---|---|
| Success rate | Very low (graph blowup) | High | Highest |
| Speed | Slow (~5s) | Fast (~1s) | Fastest (~0.75s) |
| Safety | Guaranteed | Mostly safe (within-vertex) | Not guaranteed |
| Graph complexity | Exponential growth | Linear growth (temporal split) | Similar to BVC |

---

## Open Issues

1. **BVC cross-vertex safety**: The halfspace only constrains j in vertices i passes
   through. Adjacent vertices are unconstrained. Fix: add buffer-zone halfspaces to
   adjacent vertices too, or use exact `IntersectsWith` spatial overlap check.

2. **CVT generator accuracy**: Linear interpolation for j's generator is a rough
   estimate. Better options: (a) use j's GCS relaxation as a generator warmstart,
   (b) iterate Lloyd's algorithm 2–3 times.

3. **CVT temporal safety**: The "before/after" slices of a contested vertex have no
   constraint. If another agent's trajectory occupies these slices (in an adjacent
   vertex), collisions can occur.

4. **Multiple overlapping agents in BVC**: The current BVC implementation computes
   separate halfspaces for each overlapping trajectory and stacks them (intersection).
   The direction `n` is computed independently per segment, which may give inconsistent
   constraints if multiple agents overlap the same vertex.
