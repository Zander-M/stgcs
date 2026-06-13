# CVT-Based Reservation: Design Discussion

**Date:** 2026-06-13  
**Branch:** RH (rolling-horizon)  
**Status:** Design finalized, implementation pending  
**See also:** `agent/BVC_RESERVATION.md` (alternative method)

---

## Core Idea

Instead of per-pair halfspace constraints (BVC), use a **Centroidal Voronoi Tessellation** to partition each contested space-time vertex into exactly $n_v$ non-overlapping convex cells — one per contesting agent. Each agent plans within its assigned cell.

This is a true space-time tessellation: cells tile the full vertex domain with no overlaps and no gaps.

---

## Construction

### Step 1: Identify contested vertices

A vertex $v$ is contested if $n_v \geq 2$ agents' planned trajectories pass through $v$'s space-time extent. Use AABB pre-filter + `IntersectsWith` for precision (same as existing `ecd.py`).

### Step 2: Compute generators

For each contesting agent $k$, compute a **generator** $g_k \in \mathbb{R}^{d+1}$ — the centroid of agent $k$'s trajectory within $v$'s space-time window $[t_v^{\text{start}}, t_v^{\text{end}}]$:

$$g_k = \frac{1}{2}\left(\text{lerp}_k(t_v^{\text{start}}) + \text{lerp}_k(t_v^{\text{end}})\right)$$

where `lerp_k(t)` interpolates agent $k$'s planned trajectory at time $t$ (using `ShortestPathSolution.lerp`). For the first iteration (before any agents are planned), use linear interpolation from start to goal.

### Step 3: Compute Voronoi cells

The Voronoi cell for agent $k$ within vertex $v$ is:

$$C_k = v \;\cap\; \bigcap_{j \neq k} H_{kj}$$

where $H_{kj}$ is the halfspace of points closer to $g_k$ than to $g_j$:

$$H_{kj} = \left\{ x \in \mathbb{R}^{d+1} : (g_j - g_k)^T x \leq \frac{\|g_j\|^2 - \|g_k\|^2}{2} \right\}$$

Each $C_k$ is a convex polytope with at most $n_v - 1$ additional halfspace constraints beyond $v$'s original faces. All cells are disjoint and $\bigcup_k C_k = v$.

### Step 4: Assign cells to agents

Agent $k$ is constrained to $C_k$ when traversing vertex $v$. In the GCS: replace vertex $v$ with $n_v$ sub-vertices $\{C_1, ..., C_{n_v}\}$ and wire each agent to its assigned sub-vertex only.

### Step 5 (CVT iteration): Update generators

After each agent plans within its cell, update its generator to the centroid of its actual planned trajectory within $v$:

$$g_k^{\text{new}} = \frac{1}{2}\left(\text{lerp}_k^{\text{new}}(t_v^{\text{start}}) + \text{lerp}_k^{\text{new}}(t_v^{\text{end}})\right)$$

Re-partition and re-plan. Repeat until generator positions converge (typically 2–3 iterations). This is Lloyd's algorithm applied to the multi-robot space-time planning problem.

---

## Safety Guarantee

Since the cells are non-overlapping by construction and each agent is constrained to its own cell, two agents in different cells cannot occupy the same space-time point. Combined with a clearance margin:

Offset each bisector halfspace by $r$ in the direction away from $g_k$:

$$H_{kj}^{(r)} = \left\{ x : (g_j - g_k)^T x \leq \frac{\|g_j\|^2 - \|g_k\|^2}{2} - r\right\}$$

This shrinks each cell inward by $r$, leaving a gap of $2r$ between adjacent cells. Any agent staying within its buffered cell is guaranteed $\geq 2r$ Euclidean distance from all other agents.

---

## Graph Size Impact

For a vertex $v$ contested by $k$ agents:

| | ECD | BVC | CVT |
|---|---|---|---|
| Sub-vertices created | 2 + 4k (compounds across agents) | None (halfspace rows added) | Exactly $k$ |
| Halfspaces per sub-vertex | Complex (parallelepiped faces) | 1 per higher-priority agent | $k-1$ (one per other agent) |
| Cells tile the space | No (only feasible regions, interior discarded) | No (per-pair constraints) | Yes (partition) |
| Globally consistent | No | No | Yes |
| Which-side ambiguity | None (graph encodes it) | Exists (forced one side per pair) | None (partition by construction) |

For the complex2d_n4 scenario with 4 agents, a vertex contested by all 4 agents produces:
- ECD: potentially 100+ sub-vertices (compounding)
- BVC: 1 tighter vertex (3 halfspace constraints added)
- CVT: exactly 4 sub-vertices, each with 3 halfspace constraints

---

## The Centroidal Property

A standard Voronoi partition places boundaries at perpendicular bisectors but does not ensure any agent is centered in its cell. The **centroidal** property (each generator = centroid of its cell) ensures:

- Each agent's trajectory is maximally far from cell walls → maximum planning room
- The partition is **energy-minimizing** for the distortion functional $\int_v \|x - g(x)\|^2 dx$ where $g(x)$ is the nearest generator
- Symmetric treatment of all agents — no agent is squeezed against a wall while others have large cells

---

## Iterative Algorithm (Lloyd's)

```
Initialize: generators from linear interpolations (start → goal for each agent)
Repeat until convergence:
    1. Compute Voronoi partition of each contested vertex using current generators
    2. Plan each agent (in priority order) within its assigned Voronoi cells
    3. Update generators to centroids of planned trajectories within each vertex
```

Convergence criterion: $\|g_k^{\text{new}} - g_k^{\text{old}}\| < \epsilon$ for all $k$ and all contested vertices.

In practice: 2–3 iterations is typically sufficient because agents' trajectories don't change drastically between iterations (the GCS relaxation provides a good warm start).

---

## Integration with PBS

### Priority-based integration

In PBS, agents are planned in priority order. When planning agent $j$ with higher-priority agents $\{i_1, ..., i_m\}$ already planned:

1. For each vertex $v$ that $j$ might traverse:
   - Identify which $i_k$ agents also pass through $v$
   - Compute generators: $g_{i_k}$ from their fixed trajectories, $g_j$ from $j$'s linear interpolation or previous iteration
   - Compute Voronoi partition → $C_j$ is $j$'s cell in $v$
2. Replace vertex $v$ with $C_j$ in $j$'s planning graph (the other cells are not added — $j$ only sees its own cell)
3. Plan $j$ within the tightened graph

This is equivalent to BVC with the key difference: the halfspace boundaries are determined by ALL contesting agents simultaneously (not pairwise), giving a globally consistent partition.

### Files to modify

| File | Change |
|---|---|
| `mrmp/region_reservation/cvt.py` | New — implement `cvt_reserve` |
| `mrmp/pbs.py` | Swap `ecd_reserve` → `cvt_reserve` (same interface) |
| `baselines/rp_stgcs.py` | Swap `ecd_reserve` → `cvt_reserve` |
| `mrmp/region_reservation/__init__.py` | Export `cvt_reserve` |

---

## Comparison with BVC

| Dimension | BVC | CVT |
|---|---|---|
| Constraint type | Per-pair halfspace | Global Voronoi cell (per-agent) |
| Number of halfspaces per contested vertex | 1 per higher-priority agent | $n_v - 1$ per agent (same asymptotically) |
| True partition | No | Yes |
| Which-side problem | Resolved by fixing direction (can be wrong) | Resolved by construction (no ambiguity) |
| Agent fairness | Not guaranteed (direction bias) | Guaranteed (centroidal = max clearance) |
| Iteration required | No (one-shot) | Optional but recommended (Lloyd's, 2–3 steps) |
| Implementation complexity | Simple | Moderate (Voronoi computation per vertex) |
| Sensitivity to generator estimate | Low (one halfspace, robust) | Moderate (partition depends on all generators) |

**When BVC is better:** few contested vertices, agents are well-separated, single-iteration planning is required.

**When CVT is better:** densely contested environments, many agents per vertex, solution quality matters more than speed, iterative re-planning is acceptable.

---

## Open Questions

1. **Generator initialization**: using linear interpolation (start → goal) for unplanned agents is a heuristic. How sensitive is the final partition to this initial estimate?

2. **Empty cells**: if a generator is poorly placed, its Voronoi cell within $v$ might be empty (after intersection with $v$'s bounds). Need a fallback (e.g., merge with neighboring cell or skip the vertex).

3. **CBS/PBS interaction**: the CVT partition changes when any agent is replanned. Is it better to recompute the full partition at each PBS node, or cache partitions per vertex and invalidate selectively?

4. **Weighted Voronoi**: if agents have different radii, use power diagrams (weighted Voronoi) where agent $k$'s weight is its robot radius. This gives proportionally larger cells to larger agents.
