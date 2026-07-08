import gc
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stgcs-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stgcs-xdg-cache")

from benchmark.base import BaseManifestStore
from benchmark.manifests.mrmp import load_manifest
from benchmark.planners.mrmp import MRMPPerformanceComparison
from visualization.viewer.solution_visualization import SolutionVisualizationService, ViewerMRMPSolutionRun
from visualization.viewer_utils import print_json, start_instance_viewer


MANIFEST_PATH = ROOT / "data/instances/mrmp/performance_comparison/manifest_n20.json"
BASE_ROOT = ROOT / "data/stgcs_base"
OUTPUT_ROOT = ROOT / "data/solutions/benchmark_instances"
BUDGET = 180.0
PLANNER_KEY = MRMPPerformanceComparison.WINDOWED_PBS_KEY
solution_path = lambda record: OUTPUT_ROOT / f"{record.instance_id}-{PLANNER_KEY}.json"


# Uncomment one INSTANCE line to switch domains.
# the instance name can be found in the manifest file in MANIFEST_PATH.
INSTANCE = ("grid2d", "mrmp-base-grid2d-s4-seed00002-20-robot-intersection")
# INSTANCE = ("maze", "mrmp-base-maze-10x10-seed00000-20-robot-intersection")
# INSTANCE = ("iris-2d", "mrmp-base-iris-2d-m10-seed00008-20-robot-intersection")


def run_wpbs(record):
    from benchmark.offline_heuristics import BaseOfflineHeuristicStore
    from experiments.mrmp_runners.common import MRMPExperiment

    path = solution_path(record)
    if path.exists():
        print(f"Reusing {record.instance_id}")
        return json.loads(path.read_text())

    print(f"Running {record.instance_id}")
    base_manifest_path = BaseManifestStore.manifest_path(BASE_ROOT, domain_key=record.domain_key)
    base_record = BaseManifestStore.records_by_id(base_manifest_path)[record.base_instance_id]
    instance = queries = None
    try:
        instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
        BaseOfflineHeuristicStore.prepare_instance_for_search(
            instance,
            base_manifest_path,
            base_record,
            MRMPPerformanceComparison.required_heuristics(),
            online_h_tab_timeout_secs=float(BUDGET),
        )
        solutions, entry = MRMPExperiment.run_windowed_pbs_spec_with_solutions(
            instance,
            queries,
            MRMPPerformanceComparison.LOW_LEVEL_SPEC,
            budget=float(BUDGET),
            window_span_factor=MRMPPerformanceComparison.WINDOW_SPAN_FACTOR,
            dynamic_window_adjustment=MRMPPerformanceComparison.DYNAMIC_WINDOW_ADJUSTMENT,
            child_expansion_mode=MRMPPerformanceComparison.CHILD_EXPANSION_MODE,
            execution_horizon_factor=MRMPPerformanceComparison.EXECUTION_HORIZON_FACTOR,
        )
        path = solution_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = SolutionVisualizationService.mrmp_run_to_viewer_json(
            record,
            ViewerMRMPSolutionRun(
                planner_key=PLANNER_KEY,
                planner_name=MRMPPerformanceComparison.planner_name(PLANNER_KEY),
                budget=float(BUDGET),
                status="SUCCESS" if entry.is_success else "FAIL",
                entry=entry,
                solutions=solutions,
            ),
        )
        path.write_text(json.dumps(payload, indent=2))
    finally:
        del queries
        del instance
        gc.collect()

    return payload


if __name__ == "__main__":
    domain_key, instance_id = INSTANCE
    records = {record.instance_id: record for record in load_manifest(MANIFEST_PATH)}
    record = records.get(instance_id)
    if record is None:
        raise ValueError(f"{instance_id!r} is not in {MANIFEST_PATH}.")
    if record.domain_key != domain_key:
        raise ValueError(f"{instance_id!r} belongs to {record.domain_key!r}, not {domain_key!r}.")

    payload = run_wpbs(record)

    print_json({
        "domain": record.domain_key,
        "instance_id": record.instance_id,
        "success": bool(payload["is_success"]),
        "runtime": float(payload["runtime"]),
        "sum_of_costs": float(payload["cost"]),
        "makespan": float(payload["makespan"]),
        "completed_agents": int(payload["num_completed_agents"]),
        "solution": str(solution_path(record).relative_to(ROOT)),
    })

    start_instance_viewer(
        MANIFEST_PATH,
        BASE_ROOT,
        record.instance_id,
        solution_path(record),
    )
