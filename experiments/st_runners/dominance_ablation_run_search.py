from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from experiments.st_runners.heuristic_ablation_run_search import STHeuristicAblationRunner
from experiments.st_runners.heuristic_ablation_st_manifest import STHeuristicAblationManifestBuilder
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from benchmark.planners.mrmp import SearchPlannerSpec


class STDominanceAblationRunCLI:
    DEFAULT_MANIFEST = Path("data/instances/st_planning/manifest.json")
    DEFAULT_OUTPUT_ROOT = Path("data/results/st_planning/dominance_ablation")
    DEFAULT_BUDGET = 600.0
    DOMINANCE_CHECK_SPECS: tuple[SearchPlannerSpec, ...] = (
        SearchPlannerSpec("h_max", ("GUB",)),
        SearchPlannerSpec("h_max", ("GUB", "delta_set")),
        SearchPlannerSpec("h_max", ("GUB", "delta_pos")),
        SearchPlannerSpec("h_max", ("GUB", "delta_state")),
    )

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Run the BFS dominance-check ablation on the ST planning manifest."
        )
        parser.add_argument("manifest", type=Path, nargs="?", default=cls.DEFAULT_MANIFEST)
        parser.add_argument("--budget", type=float, nargs="+", default=(cls.DEFAULT_BUDGET,))
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--output-root", type=Path, default=cls.DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args()
        for budget in args.budget:
            if not math.isfinite(float(budget)) or float(budget) <= 0.0:
                parser.error("--budget values must be finite and positive.")
        return args

    @classmethod
    def search_specs(cls) -> tuple[SearchPlannerSpec, ...]:
        return cls.DOMINANCE_CHECK_SPECS

    @staticmethod
    def budgets(args: argparse.Namespace) -> tuple[float, ...]:
        return tuple(float(budget) for budget in args.budget)

    @staticmethod
    def budget_label(budget: float) -> str:
        if math.isinf(budget):
            return "inf"
        if math.isclose(budget, round(budget)):
            return str(int(round(budget)))
        return f"{budget:g}"

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        records = STHeuristicAblationManifestBuilder.load_manifest(Path(args.manifest))
        if args.limit is not None:
            records = records[: int(args.limit)]
        specs: Sequence[SearchPlannerSpec] = cls.search_specs()
        budgets = cls.budgets(args)
        output_paths = STHeuristicAblationRunner.result_paths(args.output_root, specs=specs)
        completed = {
            planner_name: STHeuristicAblationRunner.completed_result_keys(path)
            for planner_name, path in output_paths.items()
        }

        for record in records:
            pending_runs: list[tuple[float, SearchPlannerSpec, str]] = []
            for budget in budgets:
                budget_value = float(budget)
                for spec in specs:
                    planner_name = spec.name
                    key = STHeuristicAblationRunner.result_key(record.instance_id, budget_value)
                    if key in completed[planner_name]:
                        print(
                            f"{record.instance_id} | {planner_name} "
                            f"| budget={cls.budget_label(budget_value)} | skipped (resume)"
                        )
                        continue
                    pending_runs.append((budget_value, spec, planner_name))

            if not pending_runs:
                continue

            instance, query, base_manifest_path, base_record = STHeuristicAblationRunner.reconstruct_instance(
                record,
                args.base_root,
            )
            BaseOfflineHeuristicStore.prepare_instance_for_search(
                instance,
                base_manifest_path,
                base_record,
                required_heuristics={spec.heuristic for _, spec, _ in pending_runs},
                online_h_tab_timeout_secs=max(budget for budget, _, _ in pending_runs),
            )
            for budget, spec, planner_name in pending_runs:
                entry = STHeuristicAblationRunner.run_search_spec(instance, query, spec, budget)
                STHeuristicAblationRunner.append_result_row(
                    output_paths[planner_name],
                    planner_name,
                    record,
                    budget,
                    entry,
                )
                completed[planner_name].add(STHeuristicAblationRunner.result_key(record.instance_id, budget))
                print(
                    f"{record.instance_id} | {planner_name} | budget={cls.budget_label(budget)} "
                    f"| success={entry.is_success} | runtime={entry.runtime:.3f}"
                )


if __name__ == "__main__":
    STDominanceAblationRunCLI.main()
