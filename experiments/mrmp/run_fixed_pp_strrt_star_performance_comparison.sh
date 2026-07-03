#!/usr/bin/env bash
set -euo pipefail

for n in 2 4 6 8 10 12 14 16 18 20; do
    python -O -m exp.mrmp.run_performance_comparison \
        "data/mrmp/performance_comparison/manifest_n${n}.json" \
        --output-root data/mrmp/performance_comparison_wpbs_profiled \
        --budget 180 \
        --planner wpbs
done