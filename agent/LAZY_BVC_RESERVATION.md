# Lazy BVC Reservation: Design Discussion

**Date:** 2026-06-15
**Branch:** RH (rolling-horizon)
**Status:** Design finalized, implementation next
**See also:** `agent/BVC_RESERVATION.md`, `agent/CVT_RESERVATION.md`

---

## Motivation: Problems with the Current Approaches

### ECD: eager and exponential

ECD carves a bounding parallelepiped around each trajectory segment, producing 4 side halfplanes. All 4 complementary sub-regions are materialized immediately as GCS vertices. For $k$ segments overlapping a vertex, this creates $2 + 4k$ sub-vertices. Across agents in PBS, the blowup is multiplicative.

### Current BVC: safe but over-conservative

The current `bvc_reserve` picks **one** halfplane (the centroid-direction BVC halfspace) and restricts the "during" vertex to `v ∩ H`. The other half is silently discarded. This is:

- **Safe**: the halfplane safety certificate (`n · xⱼ ≥ max(n·p_lo, n·p_hi) + 2r`) guarantees `‖xⱼ − xᵢ‖ ≥ 2r`
- **Conservative**: j loses half the space-time region even when its optimal path requires the other side
- **Less generous than ECD**: ECD preserves all 4 flanking regions; current BVC preserves only 1

The failure mode is **false infeasibility** — PBS backtracks on a priority ordering when a valid solution exists on the discarded side.

---

## Core Insight

The collision avoidance constraint is **disjunctive**:

$$j \text{ is safe from } i \;\iff\; (j \text{ satisfies } H_1) \;\lor\; (j \text{ satisfies } H_2) \;\lor\; (j \text{ satisfies } H_3) \;\lor\; (j \text{ satisfies } H_4)$$

where $H_1, H_2, H_3, H_4$ are the four side halfplanes of the ECD parallelepiped around $i$'s trajectory segment.

ECD materializes all four disjuncts as sub-vertices at once. But **for GCS to find a feasible path through vertex $v$, only one disjunct needs to be satisfied.** This means we can:

1. Try one halfplane first — add `v ∩ H_k` as the only sub-vertex
2. If GCS finds a solution, done — cost is 1 sub-vertex
3. If not, **lazily expand**: add the next halfplane's sub-vertex and re-solve
4. Continue until feasible or all four are exhausted (equivalent to ECD)

---

## Algorithm

For each vertex $v$ whose space-time window overlaps agent $i$'s trajectory:

1. **Compute the four halfplanes** $H_1, H_2, H_3, H_4$ from the ECD parallelepiped (same construction as `ecd.py`)
2. **Order them** — try the BVC centroid-direction halfplane first (most likely to work), then the remaining three
3. **Remove** $v$ from the stgcs; add `v ∩ H_1` as the sole "during" sub-vertex; reconnect edges
4. **Solve** agent $j$'s GCS problem
5. **If infeasible**: add `v ∩ H_2` to the stgcs as a second sub-vertex; re-solve
6. **Repeat** for $H_3$, $H_4$ as needed
7. **If all four exhausted and still infeasible**: genuinely infeasible for this priority ordering — PBS backtracks

The before/after temporal slices (outside the overlap window) always keep the full original polytope, same as the current BVC.

---

## Safety Guarantee

Each halfplane $H_k$ is a face of the bounding parallelepiped around $i$'s trajectory. The supporting halfspace argument applies independently to each:

$$n_k \cdot x_j \geq \max_{z \in \text{hull}(i)} n_k^T z + 2r \implies \|x_j - x_i\| \geq 2r$$

Any path found through any `v ∩ Hₖ` is guaranteed collision-free. Safety does not depend on how many halfplanes are tried — each sub-vertex is individually safe.

---

## Completeness

| Halfplanes added | Sub-vertices | Outcome |
|---|---|---|
| 1 | 1 | Feasible if j's path is on that side |
| 2 | 2 | Feasible if j's path is on either of two sides |
| 3 | 3 | Feasible if j's path is on any of three sides |
| 4 | 4 | Equivalent to ECD — complete for this priority ordering |

Worst-case completeness matches ECD. The gain is in the common case where 1 or 2 halfplanes are sufficient.

---

## Generosity Comparison

| Method | Sub-vertices per conflict | Space preserved for $j$ |
|---|---|---|
| Current BVC | 1 (one half-space) | Half of $v$ |
| Lazy BVC (1 halfplane) | 1 | One side of parallelepiped |
| Lazy BVC (fully expanded) | 4 | Same as ECD |
| ECD | $2 + 4k$ (eager) | All 4 sides of parallelepiped |
| CVT | Exactly $n$ (one per agent) | One Voronoi cell per agent |

Lazy BVC starts cheaper than ECD and converges to ECD-equivalent completeness on demand.

---

## Halfplane Ordering Heuristic

The order in which halfplanes are tried determines how often expansion is needed. Recommended ordering:

1. **BVC centroid direction** — the halfplane whose normal points from $i$'s trajectory midpoint toward $v$'s spatial centroid. This is the direction $j$ most naturally avoids $i$ from, and the current BVC already computes it.
2. **Remaining three ECD faces** — ordered by the projection of $v$'s centroid onto each face normal (largest projection first = $j$ is most likely already on that side).

Better heuristics (e.g., using $j$'s unconstrained GCS solution to pick the most compatible halfplane) can be added later without changing the overall algorithm.

---

## Implementation Plan

### New file: `mrmp/region_reservation/lazy_bvc.py`

```python
def lazy_bvc_reserve(stgcs: STGCS, trajectory: List[np.ndarray], safe_radius: float) -> STGCS:
    """
    Reserve space for a planned trajectory using lazy halfplane expansion.

    For each GCS vertex whose space-time window overlaps the trajectory:
    - Temporally split into before / during / after slices
    - For the "during" slice, try ECD halfplanes one at a time (BVC centroid direction first)
    - Add sub-vertices lazily until GCS finds a feasible path or all four are exhausted

    Returns the stgcs with the minimal set of sub-vertices needed for feasibility.
    """
```

Key differences from `bvc_reserve`:
- Compute all 4 ECD halfplanes per segment (not just the BVC centroid direction)
- Start with 1 sub-vertex per conflicted vertex; expand only on infeasibility
- Outer loop re-solves the GCS after each expansion step

### Integration

Same substitution points as BVC:

| File | Change |
|---|---|
| `mrmp/region_reservation/lazy_bvc.py` | New — implement `lazy_bvc_reserve` |
| `mrmp/pbs.py` | Swap `ecd_reserve` / `bvc_reserve` → `lazy_bvc_reserve` |
| `baselines/rp_stgcs.py` | Swap `ecd_reserve` / `bvc_reserve` → `lazy_bvc_reserve` |
| `mrmp/region_reservation/__init__.py` | Export `lazy_bvc_reserve` |

---

## Tradeoffs and Open Questions

### Multiple solve calls per PBS node

The lazy expansion requires re-solving the GCS after each halfplane is added. In the best case (1 halfplane sufficient) this is 1 solve — same as current BVC. In the worst case (all 4 needed) it is 4 solves — worse than ECD which needs only 1 solve on a larger graph. Whether the smaller graph size per solve outweighs the extra solve count depends on the problem.

### Interaction with PBS

PBS already backtracks over priority orderings when a GCS solve is infeasible. Lazy expansion adds a finer-grained inner loop: try more halfplanes before handing infeasibility back to PBS. This reduces unnecessary PBS backtracking at the cost of extra GCS solves within a single PBS node.

### Multiple conflicting segments per vertex

A vertex may overlap multiple trajectory segments from the same agent. The current design applies lazy expansion per segment independently. An alternative is to treat all overlapping segments jointly (compute 4 halfplanes per segment, expand over their Cartesian product) — more complete but potentially expensive.

### Cross-vertex disconnection (open problem)

**The core issue:** halfplane choices for adjacent vertices `v_A` and `v_B` may be mutually incompatible — producing safe sets that are disconnected at their shared boundary:

```
v_A:  [--- H_1 safe ---|--- unsafe ---]
v_B:  [--- unsafe ---|--- H_2 safe ---]
                       ↑
                  boundary — no spatial overlap
```

If j's path requires entering `v_A` on the left (H_1) and exiting into `v_B` on the right (H_2), the transition is infeasible even though a valid path through both vertices exists (e.g., going right in both). Per-vertex independent expansion cannot detect this: adding more halfplanes to `v_A` alone does not fix the incompatibility with `v_B`.

ECD avoids this because `update_edge` materializes all sub-regions for every vertex and only connects pairs whose shared boundary is non-empty — the GCS optimizer finds the globally consistent assignment. Lazy BVC breaks this guarantee by selecting halfplanes independently per vertex.

**Modeling halfplane selection as a search problem:**

A principled fix is to treat the halfplane assignment across all conflicted vertices as a **search problem**:

- **State**: an assignment $\sigma: V_{\text{conflict}} \to \{H_1, H_2, H_3, H_4\}$ — one halfplane chosen per conflicted vertex
- **Initial state**: assign the BVC centroid direction to every conflicted vertex (cheapest guess)
- **Evaluation**: build the stgcs with the current assignment and solve j's GCS problem
- **Conflict detection**: if infeasible, identify which adjacent vertex pairs have empty shared boundaries under the current assignment — these are the cross-vertex conflicts
- **Branching**: for each detected conflict between `(v_A, v_B)`, branch on changing $\sigma(v_A)$ or $\sigma(v_B)$ to a compatible halfplane
- **Termination**: find a feasible assignment or exhaust all combinations (equivalent to ECD)

This is structurally analogous to CBS: plan under the current assignment, detect incompatibilities (conflicts), branch to resolve them one at a time. The search tree has branching factor at most 4 per conflicted vertex pair and depth bounded by the number of conflict edges.

**Key properties of the search formulation:**
- **Sound**: every assignment uses safe halfplanes, so any solution found is collision-free
- **Complete**: if ECD is feasible, the search will find a compatible assignment (since ECD is the fully-expanded case)
- **Lazy**: only branches when a conflict is actually detected — vertices whose halfplane choice is conflict-free are never expanded
- **Guided**: the GCS relaxation value at each search node provides a natural lower bound for best-first search

**Conflict detection:** a cross-vertex conflict between `v_A` and `v_B` is detected when the edge between their chosen sub-vertices has empty intersection. This check is cheap — it is already performed by `update_edge` — so the detection step requires no additional GCS solve.

**Constraint propagation:**

A halfplane assignment for vertex `v_A` does not just affect `v_A` in isolation — it restricts which halfplanes are valid for every adjacent vertex. When $\sigma(v_A) = H_k$ is fixed, any halfplane $H_m$ for a neighbor `v_B` that produces an empty shared boundary with `v_A \cap H_k` is incompatible and can be removed from `v_B`'s domain immediately — before any GCS solve.

This propagation cascades: eliminating halfplane options from `v_B` may further reduce the domains of `v_B`'s neighbors, and so on. If any vertex's domain becomes empty during propagation, the current assignment for some ancestor vertex is provably wrong and must be changed — no GCS solve needed to detect this.

The compatibility check between two adjacent sub-vertices is a cheap HPolyhedron intersection test (feasibility LP), not a full GCS solve. Propagation therefore prunes large portions of the search tree at low cost:

```
assign σ(v_A) = H_1
  → v_B compatible halfplanes: {H_1, H_2}   (H_3, H_4 eliminated by propagation)
  → v_C compatible halfplanes: {H_2}         (H_1, H_3, H_4 eliminated)
  → v_D compatible halfplanes: {}            → backtrack immediately, no GCS solve
```

This is arc consistency (AC-3) applied to the halfplane selection CSP:
- **Variables**: one per conflicted vertex, domain $\{H_1, H_2, H_3, H_4\}$
- **Constraints**: for each edge $(v_A, v_B)$ in the GCS, the chosen sub-vertices must have non-empty shared boundary
- **Propagation**: after each assignment or domain reduction, re-check all incident edges and remove newly-incompatible options

Arc consistency significantly tightens the search before any GCS solve is attempted, and backtracking is triggered by domain wipeout rather than GCS infeasibility — making the inner loop much cheaper on average.

### Relationship to CBS

Lazy BVC with the search formulation operates at two complementary levels:
- **Inner loop (halfplane search)**: branches on which halfplane to assign per conflicted vertex — resolves geometric incompatibilities within a single agent's planning problem
- **Outer loop (PBS)**: branches on priority orderings — resolves multi-agent coordination conflicts

The two loops are independent and can coexist. The inner loop reduces unnecessary PBS backtracking by finding feasible halfplane assignments before declaring a priority ordering infeasible.
