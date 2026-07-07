# Experiments

This folder contains repository-local scripts for building benchmark artifacts,
running ST and MRMP experiments, and regenerating figures. Run them from the
repository root with `python -m ...`; the `experiments` package is a script
surface, not an installed library component.

For local runs, keep Matplotlib and font caches in writable temp folders:

```bash
export MPLCONFIGDIR=/private/tmp/stgcs-mpl-cache
export XDG_CACHE_HOME=/private/tmp/stgcs-xdg-cache
```

Use `--limit 1` or a small manifest when checking a runner before launching a
full benchmark.

## Layout

- `build_base_manifest.py`: build shared base ST-GCS manifests under
  `data/stgcs_base`.
- `compute_heuristics.py`: precompute shared base heuristic caches.
- `st_runners/`: ST-planning manifest builders and experiment runners.
- `mrmp_runners/`: MRMP manifest builders and experiment runners.
- `plot/`: figure and table scripts that consume CSV results.
- `common.py` and `mpl_paper.py`: shared script utilities.

Reusable benchmark definitions live in `benchmark/`, not here. In particular,
`benchmark.offline_heuristics.BaseOfflineHeuristicStore` defines heuristic
cache paths and loading behavior; `experiments.compute_heuristics` is only the
CLI that generates those caches.

## Shared Base Artifacts

Build base manifests:

```bash
python -m experiments.build_base_manifest grid2d \
  --output-root data/stgcs_base \
  --count-per-size 100

python -m experiments.build_base_manifest maze \
  --output-root data/stgcs_base \
  --count-total 500

python -m experiments.build_base_manifest iris-2d \
  --output-root data/stgcs_base \
  --count-total 500
```

Precompute LBG caches for a base manifest:

```bash
python -m experiments.compute_heuristics \
  data/stgcs_base/grid2d/manifest.json \
  --workers 4
```

Precompute LBG and TD caches:

```bash
python -m experiments.compute_heuristics \
  data/stgcs_base/grid2d/manifest.json \
  --workers 4 \
  --td-timeout-secs 60
```

The same command can consume an ST manifest; it resolves the referenced base
records through `--base-root`:

```bash
python -m experiments.compute_heuristics \
  data/instances/st_planning/manifest.json \
  --base-root data/stgcs_base \
  --workers 4 \
  --td-timeout-secs 60
```

Use `--force` only when the existing cache files should be recomputed.

## ST Runners

Build the dynamic-obstacle ST manifest used by the heuristic ablation:

```bash
python -m experiments.st_runners.heuristic_ablation_st_manifest \
  --base-root data/stgcs_base \
  --output data/instances/st_planning/manifest.json
```

Run the heuristic ablation:

```bash
python -m experiments.st_runners.heuristic_ablation_run_search \
  data/instances/st_planning/manifest.json \
  --base-root data/stgcs_base \
  --output-root data/results/st_planning/heuristic_ablation \
  --budget 600 \
  --limit 1
```

Run the heuristic-inflation ablation:

```bash
python -m experiments.st_runners.heuristic_inflation_run_search \
  data/instances/st_planning/manifest.json \
  --base-root data/stgcs_base \
  --output-root data/results/st_planning/heuristic_inflation \
  --budget 600 \
  --limit 1
```

Run the ST performance comparison:

```bash
python -m experiments.st_runners.performance_comparison_run_search \
  data/instances/st_planning/manifest.json \
  --base-root data/stgcs_base \
  --output-root data/results/st_planning/performance_comparison \
  --planner ipc \
  --budget 600 \
  --limit 1
```

Run the MICP timeout wrapper when isolating MICP rows:

```bash
PYTHON_BIN=/Users/jingtao/miniconda3/envs/gcs/bin/python \
MANIFEST=data/instances/st_planning/manifest.json \
PLANNERS="micp micpg" \
LIMIT=1 \
bash experiments/st_runners/run_st_performance_micp_timeout.sh
```

Build the heuristic-computation scaling manifest and caches:

```bash
python -m experiments.st_runners.heur_computation_time_scaling \
  --output-root data/stgcs_base/heur_computation_time_scaling \
  --td-timeout-secs 10000
```

## MRMP Runners

Build MRMP manifests:

```bash
python -m experiments.mrmp_runners.build_manifest grid2d \
  --benchmark pbs-node-expansion \
  --base-root data/stgcs_base

python -m experiments.mrmp_runners.build_manifest all \
  --benchmark performance-comparison \
  --replicates-per-robot-count 1 \
  --base-root data/stgcs_base
```

Run full-horizon PBS variants:

```bash
python -m experiments.mrmp_runners.run_search \
  data/instances/mrmp/pbs_node_expansion/grid2d/manifest.json \
  --base-root data/stgcs_base \
  --budget 300 \
  --limit 1
```

Run the windowed-coordination ablation:

```bash
python -m experiments.mrmp_runners.run_windowed_coordination \
  --manifest-root data/instances/mrmp/windowed_coordination \
  --base-root data/stgcs_base \
  --output-root data/results/mrmp/windowed_coordination \
  --budget 600 \
  --limit 1
```

Run the MRMP performance comparison:

```bash
python -m experiments.mrmp_runners.run_performance_comparison \
  data/instances/mrmp/performance_comparison/manifest_n10.json \
  --base-root data/stgcs_base \
  --output-root data/results/mrmp/performance_comparison \
  --planner wpbs \
  --budget 180 \
  --limit 1
```

`experiments/mrmp_runners/run_fixed_pp_strrt_star_performance_comparison.sh`
is a small shell convenience wrapper for repeated MRMP performance-comparison
runs over the numbered manifests.

## Plot Scripts

Plot scripts read CSVs from `data/results/...` and write figures or tables to
`latex/figs/...` and `latex/tables/...` by default. Run them as modules:

```bash
python -m experiments.plot.plot_st_performance_comparison
python -m experiments.plot.plot_st_heuristic_ablation_groups
python -m experiments.plot.plot_st_domination_ablation_groups
python -m experiments.plot.plot_heuristic_inflation_and_scaling

python -m experiments.plot.plot_mrmp_pbs_node_expansion
python -m experiments.plot.plot_mrmp_windowed_coordination
python -m experiments.plot.plot_mrmp_performance_comparison
python -m experiments.plot.plot_mrmp_wpbs_runtime_breakdown
```

Use each script's `--help` output for custom `--results-root`,
`--output-prefix`, `--budget`, and format options.

## Quick Checks

Check all main entrypoints without running experiments:

```bash
python -m experiments.build_base_manifest --help
python -m experiments.compute_heuristics --help
python -m experiments.st_runners.performance_comparison_run_search --help
python -m experiments.mrmp_runners.run_performance_comparison --help
python -m experiments.plot.plot_st_performance_comparison --help
```

Compile the experiment scripts:

```bash
find experiments -name '*.py' -print0 | xargs -0 python -m py_compile
```
