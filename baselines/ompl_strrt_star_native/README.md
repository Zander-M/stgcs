# Official OMPL ST-RRT* Native Backend

This optional extension builds `baselines._ompl_strrt_star_native`, which is used
by `baselines.ompl_strrt_star.OfficialOMPLSTRRTStar`.

It requires official OMPL headers/libraries and pybind11. This repository does
not vendor either dependency.

Example build, from the repository root:

```bash
/Users/jingtao/miniconda3/envs/gcs/bin/cmake \
  -S baselines/ompl_strrt_star_native \
  -B /private/tmp/stgcs_ompl_strrt_build \
  -G Ninja \
  -DPython_EXECUTABLE=/Users/jingtao/miniconda3/envs/gcs/bin/python \
  -DCMAKE_PREFIX_PATH=/Users/jingtao/miniconda3/envs/gcs
/Users/jingtao/miniconda3/envs/gcs/bin/cmake --build /private/tmp/stgcs_ompl_strrt_build
```

Once `baselines._ompl_strrt_star_native` is importable, run the ST performance
comparison rows explicitly with:

```bash
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
  /Users/jingtao/miniconda3/envs/gcs/bin/python \
  -m exp.base.performance_comparison_run_search \
  data/st_planning/manifest.json \
  --planner ompl_strrt_star \
  --budget 600 \
  --output-root data/st_planning/performance_comparison/results_ompl_strrt_star
```

`ompl_strrt_star` expands to `OMPL ST-RRT*-C(first)` and
`OMPL ST-RRT*-C(final)`. The legacy `strrt` planner key is an alias for this
OMPL backend, so `--planner all` also uses the OMPL implementation.

The adapter only records exact OMPL solutions as successful rows. OMPL
approximate solutions are written as failures because they may not reach the
spatial goal. If an older run already wrote `OMPL ST-RRT*-C(first).csv` or
`OMPL ST-RRT*-C(final).csv`, remove those stale files or use a fresh
`--output-root`; the runner resumes completed rows and will not overwrite them.
By default, Python callers leave OMPL's time dimension unbounded so ST-RRT*
uses its own time-bound expansion strategy. A finite `time_upper_bound` remains
available for genuinely windowed queries. The native binding captures the first
exact OMPL solution through OMPL's intermediate-solution callback, then
continues until the runtime budget to report the final exact solution. The
local binding keeps OMPL's narrowed time bound after each solution but does not
run OMPL 1.7's destructive prune step, which can leave stale connection
pointers when planning continues after the first solution.

`FixedPrioritySTRRTStarPlanner` runs each low-level OMPL call in a forked child
process. OMPL's conditional sampler keeps process-global RNG state internally;
isolating each low-level call prevents earlier agents or earlier benchmark rows
from changing whether a later agent receives an exact first solution.

The extension implements a custom OMPL spatial state space with distance
`max(abs(dx_i) / vlimit_i)`, then wraps it in OMPL `SpaceTimeStateSpace` with
`vMax=1`. This preserves the velocity feasibility semantics used by this
codebase while running official `ompl::geometric::STRRTstar`.

The native motion validator owns static 2D Env geometry checks. Polygonal
obstacles are checked against the robot's swept L-infinity footprint, and
grid-style C-space cells use the same segment-cover test as `Env.is_segment_in_CSpace`.
Python callbacks are reserved for extra time-dependent constraints, such as
dynamic obstacles and fixed-priority reservations.
