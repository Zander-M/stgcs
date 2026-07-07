from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from experiments.st_runners.heuristic_ablation_run_search import STHeuristicAblationRunner
from experiments.st_runners.heuristic_ablation_st_manifest import STHeuristicAblationManifestBuilder
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from benchmark.planners.mrmp import SearchPlannerSpec


class STHeuristicInflationRunCLI:
    DEFAULT_MANIFEST = Path("data/instances/st_planning/manifest.json")
    DEFAULT_OUTPUT_ROOT = Path("data/results/st_planning/heuristic_inflation")
    DEFAULT_BUDGET = 600.0
    DEFAULT_HEURISTICS = ("SC", "LBG", "TD", "Max")
    DEFAULT_EPSILONS = (1.25, 2.5, 5.0, 10.0)
    DOMINATION_STACK = ("GUB",)

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Run the BFS heuristic-inflation ablation on the ST heuristic-ablation manifest. "
                "All runs use only the GUB domination check."
            )
        )
        parser.add_argument("manifest", type=Path, nargs="?", default=cls.DEFAULT_MANIFEST)
        parser.add_argument("--budget", type=float, nargs="+", default=(cls.DEFAULT_BUDGET,))
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--output-root", type=Path, default=cls.DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--heuristics", nargs="+", default=cls.DEFAULT_HEURISTICS)
        parser.add_argument("--epsilons", type=float, nargs="+", default=cls.DEFAULT_EPSILONS)
        args = parser.parse_args()
        for budget in args.budget:
            if not math.isfinite(float(budget)) or float(budget) <= 0.0:
                parser.error("--budget values must be finite and positive.")
        for epsilon in args.epsilons:
            if not math.isfinite(float(epsilon)) or float(epsilon) < 1.0:
                parser.error("--epsilons values must be finite and at least 1.0.")
        return args

    @classmethod
    def search_specs(
        cls,
        heuristics: Sequence[str] | None = None,
        epsilons: Sequence[float] | None = None,
    ) -> tuple[SearchPlannerSpec, ...]:
        selected_heuristics = cls.DEFAULT_HEURISTICS if heuristics is None else tuple(str(value) for value in heuristics)
        raw_epsilons = cls.DEFAULT_EPSILONS if epsilons is None else tuple(float(value) for value in epsilons)
        selected_epsilons = tuple(epsilon for epsilon in raw_epsilons if not math.isclose(epsilon, 1.0))
        return tuple(
            SearchPlannerSpec(heuristic, cls.DOMINATION_STACK, epsilon=epsilon)
            for heuristic in selected_heuristics
            for epsilon in selected_epsilons
        )

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
        specs: Sequence[SearchPlannerSpec] = cls.search_specs(args.heuristics, args.epsilons)
        if not specs:
            print("No inflation runs requested after skipping epsilon=1.0.")
            return
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
            required_heuristics = {spec.heuristic for _, spec, _ in pending_runs}
            BaseOfflineHeuristicStore.prepare_instance_for_search(
                instance,
                base_manifest_path,
                base_record,
                required_heuristics=required_heuristics,
                online_td_timeout_secs=max(budget for budget, _, _ in pending_runs),
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
    STHeuristicInflationRunCLI.main()
