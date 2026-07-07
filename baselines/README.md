# Baselines

This folder contains baseline planner adapters used by the ST-GCS benchmark and
experiment runners. The core ST-GCS planners live under `stgcs/`; this directory
keeps comparison methods and their native bindings in one place.

## Contents

- `common.py`: shared result containers and geometry helpers for baseline
  planners.
- `zeta_sipp/`: single-agent Zeta*-SIPP implementation for 2D ST-planning
  comparisons.
- `pbs_zeta_sipp.py`: priority-based multi-robot search using Zeta*-SIPP as the
  low-level planner.
- `ompl_strrt_star.py`: Python adapter for the optional official OMPL ST-RRT*
  native backend.
- `sp_strrtstar.py`: fixed-priority MRMP planner built from repeated ST-RRT*
  calls.
- `ompl_kcbs.py`: adapter for the optional official OMPL K-CBS native backend.
- `cb_gcs.py`: conflict-based GCS/T-GCS baseline for 2D MRMP comparisons.
- `ompl_strrt_star_native/` and `ompl_kcbs_native/`: CMake/pybind11 native
  backend sources and build notes.

## Usage

Run these baselines through the experiment CLIs so manifests, budgets, output
paths, resume behavior, and result schemas stay consistent:

```bash
python -m experiments.st_runners.performance_comparison_run_search \
  data/instances/st_planning/manifest.json \
  --planner strrt zeta \
  --budget 600

python -m experiments.mrmp_runners.run_performance_comparison \
  data/instances/mrmp/performance_comparison/manifest_n10.json \
  --planner fixed-pp-st-rrt-star pbs-zeta-sipp cb-gcs kcbs \
  --budget 180
```

You can also check `demos/demo_benchmark_comparison.py` for a qualitative comparison demo.

The OMPL-backed planners require their native modules to be built first. See the
README files in `ompl_strrt_star_native/` and `ompl_kcbs_native/` for the
backend-specific build steps.
