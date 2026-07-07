# ST-GCS

This repository implements Graphs of Space-Time Convex Sets (ST-GCS) for
spatiotemporal and multi-robot motion planning.

- Branch *master*: Jingtao Tang, Zining Mao, Lufan Yang, and Hang Ma. "Search-Based Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets." [[paper]](https://arxiv.org/abs/2607.00444), [[project]](https://sites.google.com/view/stgcs)
- Branch *iros*: Jingtao Tang, Zining Mao, Lufan Yang, and Hang Ma. "Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning." [[paper]](https://arxiv.org/abs/2503.00583), [[project]](https://sites.google.com/view/stgcs)

## Installation

Create a conda environment with the native `cddlib` and GMP headers required by
`pycddlib`, then install this repository in editable mode:

```bash
conda create -n stgcs -c conda-forge python=3.11 pip cddlib gmp
conda activate stgcs
CFLAGS="-I$CONDA_PREFIX/include" LDFLAGS="-L$CONDA_PREFIX/lib" pip install -e .
```

[Mosek](https://www.mosek.com/) should also be installed and licensed. Drake
uses it to solve the GCS programs. Gurobi can also be used through Drake, but
that requires a source build of Drake; see the
[Drake installation guide](https://drake.mit.edu/installation.html).

For local plotting or benchmark runs, it is often useful to keep Matplotlib and
font caches in writable temp folders:

```bash
export MPLCONFIGDIR=/private/tmp/stgcs-mpl-cache
export XDG_CACHE_HOME=/private/tmp/stgcs-xdg-cache
```

## Quick Start

Check that the editable install imports:

```bash
python -c "import stgcs; print('ST-GCS import OK')"
```

Run the cached MRMP instance demo and open the local viewer:

```bash
python -m demos.demo_benchmark_instances
```

Run one ST-planning benchmark row:

```bash
python -m experiments.st_runners.performance_comparison_run_search \
  data/instances/st_planning/manifest.json \
  --planner ipc \
  --budget 60 \
  --limit 1
```

Run one MRMP performance-comparison row:

```bash
python -m experiments.mrmp_runners.run_performance_comparison \
  data/instances/mrmp/performance_comparison/manifest_n10.json \
  --planner wpbs \
  --budget 180 \
  --limit 1
```

The demo viewer serves at `http://127.0.0.1:8765/` and stays active until the
process is interrupted.

## Repository Layout

- `stgcs/`: core ST-GCS graph, low-level search, PBS, windowed coordination,
  trajectory, collision, and postprocessing code.
- `benchmark/`: benchmark records, manifest definitions, environment builders,
  and planner labels used by the experiment runners.
- `experiments/`: repository-local experiment scripts. Shared helpers live at
  the package root, ST runners live in `experiments/st_runners/`, MRMP runners
  live in `experiments/mrmp_runners/`, and figure scripts live in
  `experiments/plot/`. See `experiments/README.md` for script usage.
- `baselines/`: baseline planner adapters and optional native backend sources.
  See `baselines/README.md` for the baseline-specific overview.
- `demos/`: small runnable demos that load benchmark instances and saved
  solutions, optionally rerun planning/trajopt, and start the WebGL viewer. See
  `demos/README.md` for demo commands and rerun options.
- `visualization/`: viewer server, viewer manifest generation, and plotting or
  visualization helpers.
- `data/`: benchmark manifests, cached base instances and heuristics, saved
  viewer solutions, and result CSVs.

## Benchmarks

Build or refresh MRMP manifests with:

```bash
python -m experiments.mrmp_runners.build_manifest grid2d \
  --benchmark performance-comparison \
  --replicates-per-robot-count 1
```

Run benchmark modules with `--help` to see planner keys, manifest defaults, and
output locations:

```bash
python -m experiments.st_runners.performance_comparison_run_search --help
python -m experiments.mrmp_runners.run_performance_comparison --help
```

Large benchmark runs can be expensive. Use `--limit` for smoke tests and run
from the repository root so the repository-local `experiments` modules and data
paths resolve correctly.

## Optional Baselines

Some baselines are pure Python adapters; others require native OMPL backends.
The native modules are not vendored by this repository. Build notes for the
optional backends are in:

- `baselines/ompl_strrt_star_native/README.md`
- `baselines/ompl_kcbs_native/README.md`

## BibTeX

```bibtex
@misc{tang2026searchbasedspatiotemporalmultirobotmotion,
  title={Search-Based Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets},
  author={Jingtao Tang and Zining Mao and Lufan Yang and Hang Ma},
  year={2026},
  eprint={2607.00444},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2607.00444}
}
```

IROS version:

```bibtex
@misc{tang2025spacetimegraphsconvexsets,
  title={Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning},
  author={Jingtao Tang and Zining Mao and Lufan Yang and Hang Ma},
  year={2025},
  eprint={2503.00583},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2503.00583},
}
```

## License

ST-GCS is released under GPL version 3. See `LICENSE.txt` for details.
