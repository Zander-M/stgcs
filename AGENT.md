# ST-GCS — Agent Initialization Guide

## Project Overview

This repository implements **Space-Time Graphs of Convex Sets (ST-GCS)** for Multi-Robot Motion Planning (MRMP). Robots move through a 2D configuration space decomposed into convex polytopes; ST-GCS lifts the space-time trajectory planning problem into a Graph of Convex Sets (GCS) and solves it optimally via convex relaxation + rounding.

Paper: https://arxiv.org/abs/2503.00583

---

## Environment Setup

### Conda environment (Python 3.12)

```bash
conda env create -f environment.yml   # first time
conda activate stgcs
```

If the `stgcs` env already exists:

```bash
conda env update -f environment.yml --prune
conda activate stgcs
```

### Mosek license (required)

Drake uses the Mosek solver for GCS programs. You need a valid license file at:

```
~/mosek/mosek.lic
```

Academic (free) licenses: https://www.mosek.com/products/academic-licenses/

Verify the setup:

```bash
python -c "from pydrake.all import MosekSolver; print(MosekSolver().enabled())"
```

### Running scripts

All scripts must be run from the **project root** (not from inside a subdirectory), because imports use top-level package paths (`from mrmp.stgcs import ...`, `from environment.env import ...`).

```bash
cd /path/to/stgcs
conda activate stgcs
python demos/simple2d_n1.py
```

---

## File Structure

```
stgcs/
├── mrmp/                      # Core algorithm
│   ├── stgcs.py               # STGCS class — builds and solves the GCS problem
│   ├── ecd.py                 # Exact Convex Decomposition for reserving trajectories
│   ├── pbs.py                 # Priority-Based Search (PBS) multi-robot planner
│   ├── graph.py               # Graph/Vertex/Edge wrappers around Drake GCS
│   ├── interval.py            # Interval and AABB helpers
│   ├── utils.py               # HPolyhedron utilities, visualization, Mosek solver setup
│   └── geometry/              # Convex set abstractions (Polyhedron, Point, Ellipsoid)
│
├── environment/               # Environment and obstacle definitions
│   ├── env.py                 # Env class — C-space, static/dynamic obstacles, animation
│   ├── obstacle.py            # StaticObstacle, DynamicSphere, ConcatDynamicSphere
│   ├── examples.py            # Pre-built environments (SIMPLE2D_*, EMPTY2D, COMPLEX2D)
│   └── problems.py            # MRMP problem generation (random start/goal sampling)
│
├── baselines/                 # Comparison planners
│   ├── rp_stgcs.py            # Sequential (SP) and Randomized-Prioritized (RP) + ST-GCS
│   ├── sp_strrtstar.py        # Sequential planner + ST-RRT* single-robot planner
│   ├── sp_tprm.py             # Sequential planner + T-PRM single-robot planner
│   ├── st_rrt_star/           # ST-RRT* implementation (planner.py, tree.py, state.py)
│   └── tprm/                  # T-PRM implementation (planner.py, temporal_graph.py)
│
├── demos/                     # Runnable demonstrations
│   ├── simple2d_n1.py         # Single robot, simple map — compares ST-GCS vs ST-RRT*
│   ├── simple2d_n{4,8,12,16}.py  # Multi-robot on simple map
│   ├── complex2d_n{4,12,16,20}.py # Multi-robot on complex map
│   ├── empty_n{2,4}.py        # Multi-robot on empty map
│   ├── iros_formation.py      # Letter-formation demo (PBS + ST-GCS, 8 robots, 3 phases)
│   └── iros_rearrange.py      # Rearrangement demo (PBS + ST-GCS, 8 robots, 3 phases)
│
├── data/                      # Problem sets (.ps) and experiment results (.pkl, .gs)
├── exp.py                     # Batch experiment runner (CLI: -problem_set, -method)
├── plot.py                    # Plot experiment results from .pkl files
├── requirements.txt           # Pip dependencies (pinned)
└── environment.yml            # Conda environment spec
```

---

## Key Concepts

### STGCS class (`mrmp/stgcs.py`)

- `STGCS.from_env(env, t0, tmax, vlimit)` — builds the space-time GCS from a convex C-space decomposition
- `stgcs.solve(start, goal, t_start)` — solves for a single robot; returns `ShortestPathSolution`
- `ShortestPathSolution.lerp(t)` — interpolates position at time `t`
- `ShortestPathSolution.cost` — travel time (arrival time minus start time)

### Multi-robot planners

| Function | File | Strategy |
|---|---|---|
| `PBS` | `mrmp/pbs.py` | Priority-Based Search; re-plans with conflict constraints |
| `sequential_planning` | `baselines/rp_stgcs.py` | Fixed ordering, ST-GCS single-robot |
| `randomized_prioritized_planning` | `baselines/rp_stgcs.py` | Random orderings, ST-GCS single-robot |
| `sequential_planning` | `baselines/sp_strrtstar.py` | Fixed ordering, ST-RRT* single-robot |
| `sequential_planning` | `baselines/sp_tprm.py` | Fixed ordering, T-PRM single-robot |

### Dynamic obstacle reservation (`mrmp/ecd.py`)

After each robot's trajectory is solved, `ecd_reserve(stgcs, trajectory, clearance)` carves the robot's swept volume out of the GCS so subsequent robots avoid it.

### Environments

Pre-built environments live in `environment/examples.py`:
- `SIMPLE2D_4DynamicSPHERE` — simple 2D map with 4 dynamic sphere obstacles
- `EMPTY2D` — open rectangular workspace
- `COMPLEX2D` — complex 2D map with many static obstacles

---

## Running Experiments

Problem sets (`.ps` files) are in `data/`. Run with:

```bash
python exp.py -problem_set simple2d -method PBS+STGCS
```

Available methods: `PBS+STGCS`, `PP+STGCS`, `SP+STGCS`, `SP+STRRTStar`, `SP+STRRTStar-C`, `SP+TPRM`, `SP+TPRM-C`, `STGCS_graph_size`

Results are saved as `.pkl` files in the working directory. Plot with:

```bash
python plot.py
```

---

## Common Pitfalls

- **`pydrake` import errors**: The `drake` pip package provides the `pydrake` module. Do not separately install the PyPI `pydrake` package (version 0.2.1) — it overwrites Drake's `pydrake/__init__.py` and breaks imports. If this happens, run `pip uninstall pydrake && pip install --force-reinstall drake==1.36.0`.
- **No Mosek license**: `MosekSolver.Solve()` will fail silently or raise. Place the license at `~/mosek/mosek.lic`.
- **Wrong working directory**: Imports like `from mrmp.stgcs import STGCS` only work from the project root.
- **`all.py` import error with `getDrakePath`**: Caused by the pydrake/drake conflict above. Force-reinstall drake to fix.
