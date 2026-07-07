# Demos

This folder contains runnable demos that load benchmark instances, reuse saved
solution JSONs when available, optionally rerun the planning pipeline, and open
the WebGL viewer.

Run demos from the repository root with `python -m ...`:

```bash
export MPLCONFIGDIR=/private/tmp/stgcs-mpl-cache
export XDG_CACHE_HOME=/private/tmp/stgcs-xdg-cache
```

Most demos write or read solutions under `data/solutions/...`. If the expected
MRMP and trajopt solution JSONs already exist and match the current demo
configuration, the demo loads them instead of solving again.

## Large MRMP Demos

These demos load saved MRMP and trajopt JSONs by default. Use `--force-solve`
to rerun MRMP, `--force-trajopt` to rerun trajopt, `--no-trajopt` to skip
trajopt, and `--no-serve` to run or load solutions without starting the viewer.

View the checked-in 50-robot four-room solution:

```bash
python -m demos.demo_r50_four_room
```

Run the full 50-robot pipeline without starting the viewer:

```bash
python -m demos.demo_r50_four_room --no-serve
```

Rerun both MRMP and trajopt:

```bash
python -m demos.demo_r50_four_room --force-solve --force-trajopt
```

View the checked-in 100-robot empty-workspace solution:

```bash
python -m demos.demo_r100_empty
```

Rerun only trajopt from the saved MRMP solution:

```bash
python -m demos.demo_r100_empty --force-trajopt
```

View the checked-in 32-robot sphere exchange solution:

```bash
python -m demos.demo_r32_sphere_exchange
```

Rerun the sphere MRMP solve with a different child-expansion rule:

```bash
python -m demos.demo_r32_sphere_exchange \
  --force-solve \
  --force-trajopt \
  --child-expansion-rule num_conflicts
```

View the checked-in 48-UAV village solution:

```bash
python -m demos.demo_r48_village
```

The village demo uses port `8780` by default. Run it on another port with:

```bash
python -m demos.demo_r48_village --port 8870
```

It also maintains intermediate cache files:

- `data/solutions/demo_r48_village/mrmp_cache.json`
- `data/solutions/demo_r48_village/trajopt_cache.json`

Pass `--force-solve` or `--force-trajopt` when those cached stages should be
recomputed.

## Sequential MRMP Demos

The sequential demos are script-defined tasks. Their default `serve` phase runs
MRMP stage by stage, runs trajopt, saves JSONs, and starts a viewer. They reuse
valid saved JSONs under `data/solutions/demo_sequential_tasks/...`.

Run the formation-control demo:

```bash
python -m demos.demo_stgcs_formation_control
```

Run it without opening the viewer:

```bash
python -m demos.demo_stgcs_formation_control --no-serve
```

Plot only the stage target configurations:

```bash
python -m demos.demo_stgcs_formation_control targets \
  --targets-output /private/tmp/formation_targets.png \
  --no-show-targets
```

Run the rearrangement demo:

```bash
python -m demos.demo_stgcs_rearrangement
```

Sequential demos do not expose `--force-solve` or `--force-trajopt`. To rerun
from scratch, delete the output directory or pass a fresh `--output-dir`:

```bash
python -m demos.demo_stgcs_rearrangement \
  --output-dir /private/tmp/stgcs-rearrangement-rerun \
  --no-serve
```

Sequential viewers use port `8766` by default.

## Benchmark Viewer Demos

These demos select fixed benchmark instances and save viewer solution payloads.
They do not expose CLI flags; edit the constants near the top of the script to
change the selected instance or planner.

View one 20-robot benchmark instance with Windowed-PBS+BFS:

```bash
python -m demos.demo_benchmark_instances
```

The script reuses files in `data/solutions/benchmark_instances/`. Delete the
corresponding JSON if the selected benchmark instance should be solved again.

View the 10-robot benchmark-comparison demo:

```bash
python -m demos.demo_benchmark_comparison
```

The script tries to load or compute the configured planners and writes the
combined viewer payload under `data/solutions/benchmark_comparison/...`. Native
baseline planners are skipped when their optional native modules are not built
and no saved solution is available.

## Viewer

When a demo starts the viewer, it prints a local URL such as:

```text
http://127.0.0.1:8765/
```

The process stays active until interrupted. Use `--no-serve` when only the
solution JSONs or printed summary are needed.

## Quick Checks

Check CLI availability without solving:

```bash
python -m demos.demo_r50_four_room --help
python -m demos.demo_r100_empty --help
python -m demos.demo_r32_sphere_exchange --help
python -m demos.demo_r48_village --help
python -m demos.demo_stgcs_formation_control --help
python -m demos.demo_stgcs_rearrangement --help
```

Compile the demo scripts:

```bash
find demos -name '*.py' -print0 | xargs -0 python -m py_compile
```
