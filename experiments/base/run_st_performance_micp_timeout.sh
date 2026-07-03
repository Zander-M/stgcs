#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/Users/jingtao/miniconda3/envs/gcs/bin/python}"
MANIFEST="${MANIFEST:-data/st_planning/stress_test/manifest.json}"
BASE_ROOT="${BASE_ROOT:-data/stgcs_base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/st_planning/performance_comparison/results}"
BUDGET="${BUDGET:-600}"
TIMEOUT_SECS="${TIMEOUT_SECS:-$BUDGET}"
PLANNERS="${PLANNERS:-micp micpg}"
LIMIT="${LIMIT:-}"
INSTANCE_IDS="${INSTANCE_IDS:-}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

cd "$REPO_ROOT"

instance_ids=()
if [[ -n "$INSTANCE_IDS" ]]; then
    read -r -a instance_ids <<< "$INSTANCE_IDS"
else
    while IFS= read -r instance_id; do
        instance_ids+=("$instance_id")
    done < <("$PYTHON_BIN" - "$MANIFEST" "$LIMIT" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
limit_text = sys.argv[2]
limit = None if not limit_text else int(limit_text)
with open(manifest_path) as fh:
    records = json.load(fh)
for record in records[:limit]:
    print(record["instance_id"])
PY
)
fi

read -r -a planner_keys <<< "$PLANNERS"

run_with_timeout() {
    local timeout_secs="$1"
    shift
    local marker
    marker="$(mktemp "${TMPDIR:-/tmp}/st-perf-micp-timeout.XXXXXX")"
    rm -f "$marker"

    "$@" &
    local child_pid=$!
    (
        sleep "$timeout_secs"
        if kill -0 "$child_pid" 2>/dev/null; then
            touch "$marker"
            kill "$child_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$child_pid" 2>/dev/null || true
        fi
    ) &
    local watchdog_pid=$!

    local status=0
    if wait "$child_pid" 2>/dev/null; then
        status=0
    else
        status=$?
    fi

    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

    if [[ -f "$marker" ]]; then
        rm -f "$marker"
        return 124
    fi
    rm -f "$marker"
    return "$status"
}

for instance_id in ${instance_ids[@]+"${instance_ids[@]}"}; do
    for planner in ${planner_keys[@]+"${planner_keys[@]}"}; do
        echo "${instance_id} | ${planner} | budget=${BUDGET} | timeout=${TIMEOUT_SECS}"
        if run_with_timeout "$TIMEOUT_SECS" \
            "$PYTHON_BIN" -m exp.base.performance_comparison_micp_wrapper \
            "$MANIFEST" \
            --base-root "$BASE_ROOT" \
            --output-root "$OUTPUT_ROOT" \
            --budget "$BUDGET" \
            --planner "$planner" \
            --instance-id "$instance_id"; then
            continue
        else
            status=$?
        fi

        if [[ "$status" -eq 124 ]]; then
            echo "${instance_id} | ${planner} | budget=${BUDGET} | timed out; recording failure row"
            "$PYTHON_BIN" -m exp.base.performance_comparison_micp_wrapper \
                "$MANIFEST" \
                --base-root "$BASE_ROOT" \
                --output-root "$OUTPUT_ROOT" \
                --budget "$BUDGET" \
                --planner "$planner" \
                --instance-id "$instance_id" \
                --mark-timeout-failure
        else
            exit "$status"
        fi
    done
done
