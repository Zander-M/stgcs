# BVC-Based Reservation: Design Discussion

**Date:** 2026-06-13  
**Branch:** RH (rolling-horizon)  
**Status:** Design finalized, implementation pending

---

## Background: What ECD Does and Why It Breaks Down

The current multi-robot coordination method is Exact Convex Decomposition (`mrmp/ecd.py`). When agent $i$ is planned, `ecd_reserve(stgcs, trajectory, 2r)` permanently modifies the GCS by carving the robot's spatio-temporal swept volume out of every GCS vertex the trajectory intersects. This is done by slicing each affected vertex into sub-regions using the halfspace faces of a bounding parallelepiped around each trajectory segment.

### Segmentation growth is exponential

For a 2D spatial environment (3D space-time):
- Each trajectory segment generates one `ECDPair` with **4 side halfspaces**
- The `slice()` function creates **4 feasible sub-regions per ECDPair per vertex** plus a `bot` and `top` temporal boundary region
- For $k$ ECDPairs overlapping a single vertex: **2 + 4k** sub-vertices from one original vertex

In PBS (`mrmp/pbs.py:46-47`), `ecd_reserve` is called once per higher-priority agent **on the already-sliced graph**. The expansions compound multiplicatively:
- After agent $i_1$ (k=2 segments): 1 vertex → ~10 sub-vertices
- After agent $i_2$ applied to those 10: potentially ~50-100 sub-vertices
- After agent $i_3$: potentially 200+ sub-vertices

This caused the observed PBS blowup (28-edge graph → 14,912 unique paths, ~9,990 LP solves per replan).

### ECD's structural problem

ECD conflates static graph topology (the environment decomposition) with dynamic coordination constraints (agent reservations). The graph is permanently modified for each PBS node, making it:
- **Non-reversible**: backtracking in CBS/PBS requires rebuilding from scratch
- **Non-reusable**: no structure is shared between PBS sibling nodes even when agent $i$'s trajectory hasn't changed
- **Ill-posed for tree search**: the MICP problem changes shape at every node

---

## Proposed Replacement: BVC-Based Reservation

### Core idea

Instead of slicing vertices, add **linear halfspace constraints** directly to GCS vertex convex sets, derived from a Voronoi-style spatial partition between agents.

For agent $i$'s planned trajectory and agent $j$'s GCS vertex $v$ (their planning domain):

1. Find agent $i$'s trajectory segment within $v$'s time window $[t_v^{\text{start}}, t_v^{\text{end}}]$
2. Compute the **supporting halfspace** of $i$'s space-time segment hull in the direction toward $v$'s centroid:
   - $n = \frac{\hat{p}_v - \hat{p}_i}{\|\hat{p}_v - \hat{p}_i\|}$ (unit vector from $i$'s midpoint toward $v$'s centroid)
   - $b = \max_{x \in \text{seg}_i(v)} n^T x_{\text{spatial}} + 2r$ (supporting value of $i$'s trajectory hull plus clearance)
3. Add the halfspace constraint $n^T x \geq b$ to $v$'s `HPolyhedron` (intersect the two)

### Safety guarantee

Because $b$ uses the supporting value of $i$'s actual trajectory hull (not a bounding box):

$$n^T x_j \geq b = \max_{z \in \text{hull}(i)} n^T z + 2r \implies n^T(x_j - x_i) \geq 2r \implies \|x_j - x_i\| \geq 2r$$

This is sound: any $x_j$ satisfying the halfspace constraint is guaranteed Euclidean distance $\geq 2r$ from $i$'s trajectory within $v$.

### Multiple agents: conjunction stays convex

For $m$ higher-priority agents, agent $j$'s feasible region in vertex $v$ is:

$$V_j = v \cap \bigcap_{i=1}^{m} \{n_i^T x \geq b_i\}$$

This is a **single convex set** — just a tighter HPolyhedron. No sub-vertices, no graph topology change.

### Graph size comparison

| | ECD | BVC |
|---|---|---|
| Sub-regions per original vertex per agent | 2 + 4k (k = #overlapping segments) | 0 (just tighter convex set) |
| Compounding across agents | Multiplicative | Additive (halfspace rows stack) |
| New vertices created | O(k × agents) per vertex | None |
| New edges created | O(k²) (update_edge re-checks all pairs) | None |
| `stgcs.copy()` cost per PBS node | Proportional to expanded graph | Same as original graph |
| Rounding LP count | O(graph_edges³) in practice | ~unchanged from original graph |

### The spatio-temporal key insight

The BVC halfspace is computed in the **full space-time $\mathbb{R}^{d+1}$**, not just spatial $\mathbb{R}^d$. This matters because in the space-time GCS, an agent has an additional degree of freedom: **waiting** (increasing $t$ without moving spatially). A trajectory segment $(x_0, t_0) \to (x_0, t_1)$ with $t_1 > t_0$ is always valid under the velocity constraints in `mrmp/stgcs.py:47-55`.

This resolves the "tight corridor" limitation of pure spatial BVC:
- If $j$ cannot go around $i$ spatially, it waits in time until $i$ has exited the vertex's time window
- In space-time this is always possible unless $i$'s reservation fills the **entire** remaining planning window (a genuine infeasibility regardless of approach)
- In the rolling-horizon setting, the planning window is bounded and reservations are bounded within it

---

## Implementation Plan

### New file: `mrmp/region_reservation/bvc.py`

```python
def bvc_reserve(stgcs: STGCS, trajectory: List[np.ndarray], safe_radius: float) -> STGCS:
    """
    Returns a copy of stgcs with BVC halfspace constraints added to each vertex
    whose space-time window overlaps the given trajectory. Does not modify
    graph topology — only tightens vertex convex sets.
    """
```

For each vertex $v$ in `stgcs.G.vertices`:
1. AABB pre-filter (reuse existing `AABB` from `mrmp/interval.py`) — skip if $v$'s bounds don't overlap trajectory's bounds
2. For each trajectory segment whose time window overlaps `v.itvl`:
   a. Interpolate $i$'s spatial position at `v.itvl`'s midpoint → $\hat{p}_i$
   b. Use $v$'s spatial centroid as $\hat{p}_j$
   c. Compute $n = (\hat{p}_j - \hat{p}_i) / \|\hat{p}_j - \hat{p}_i\|$
   d. Compute $b = n^T \hat{p}_i + 2 \cdot \texttt{safe\_radius}$ (supporting offset along i's endpoint)
   e. Construct `HPolyhedron` for the halfspace in the FULL space-time dimension
   f. Intersect with $v$'s base HPolyhedron: `v_hpoly = v_hpoly.Intersection(halfspace)`
3. If the intersected set is non-empty, add a new vertex with the tighter set; otherwise skip

### Integration into PBS (`mrmp/pbs.py`)

Replace `ecd_reserve` calls with `bvc_reserve`:
```python
# Before:
stgcs_reserved = ecd_reserve(stgcs_reserved, self.sols[k].trajectory, 2 * robot_radius)

# After:
stgcs_reserved = bvc_reserve(stgcs_reserved, self.sols[k].trajectory, robot_radius)
```

Note: `bvc_reserve` takes `robot_radius` (not `2 * robot_radius`) because the `2r` factor is baked into the halfspace offset formula.

### Integration into `baselines/rp_stgcs.py`

Same substitution in `prioritized_planning`:
```python
# Before:
stgcs = ecd_reserve(stgcs, sol.trajectory, 2*robot_radius)

# After:
stgcs = bvc_reserve(stgcs, sol.trajectory, robot_radius)
```

---

## Tradeoffs and Open Questions

### BVC is conservative in one direction

The halfspace direction $n$ is computed from the midpoint of $i$'s segment to $v$'s centroid. This is a fixed direction: it blocks $j$ from being on $i$'s side. If $j$'s optimal path requires approaching from that direction (e.g., $j$ starts on $i$'s side and needs to pass through), the BVC constraint might be overly conservative.

**Mitigation**: in the rolling-horizon / PBS setting, $j$ can wait in time to let $i$ pass (see space-time insight above). The PBS outer loop handles remaining infeasibilities by backtracking to a different priority ordering.

### Direction choice sensitivity

The halfspace direction is a design choice. Options:
1. **Centroid-based** (recommended for first implementation): $n$ = from $i$'s midpoint to $v$'s spatial centroid. Requires no knowledge of $j$'s trajectory.
2. **Goal-directed**: $n$ = from $i$'s midpoint toward $j$'s goal. Slightly more informed.
3. **Relaxation-based**: solve $j$'s GCS relaxation without collision constraints first, extract approximate positions, recompute $n$. Most accurate, requires one extra solve.

### Relationship to existing work

This approach is structurally equivalent to **Buffered Voronoi Cells (BVC)** used in reactive swarm planners (Zhou et al., EGO-Swarm; Chen et al. 2020), adapted to the offline GCS planning setting. The key adaptation is:
- Using the **supporting halfspace of the control-point convex hull** (not just the Euclidean bisector of two points) for a sound safety certificate
- Applying the constraint to GCS vertex sets (not velocity space)
- Leveraging the **space-time dimension** to handle temporal bypassing

---

## Files to Modify

| File | Change |
|---|---|
| `mrmp/region_reservation/bvc.py` | New — implement `bvc_reserve` |
| `mrmp/pbs.py` | Replace `ecd_reserve` with `bvc_reserve` |
| `baselines/rp_stgcs.py` | Replace `ecd_reserve` with `bvc_reserve` |
| `mrmp/region_reservation/__init__.py` | Export `bvc_reserve` |

ECD (`mrmp/ecd.py`) is kept as-is for comparison / fallback.
