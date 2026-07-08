from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Sequence

from benchmark.base import BaseInstanceFactory, BaseManifestStore
from experiments.st_runners.heuristic_ablation_st_manifest import (
    STHeuristicAblationResultEntry,
    STHeuristicAblationResultStore,
    STHeuristicAblationManifestBuilder,
    STHeuristicAblationRecord,
)
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from benchmark.instance import Instance
from experiments.mrmp_runners.common import MRMPExperiment
from benchmark.planners.mrmp import SEARCH_GROUPS, SearchPlannerSpec
from stgcs.st_planner import STPlanStatus


class STHeuristicAblationRunner:
    RESULT_FIELDS = STHeuristicAblationResultStore.RESULT_FIELDS

    @classmethod
    def result_key(cls, instance_id: str, budget: float) -> tuple[str, float]:
        return STHeuristicAblationResultStore.result_key(instance_id, budget)

    @staticmethod
    def planner_output_path(output_root: str | Path, planner_name: str) -> Path:
        return STHeuristicAblationResultStore.planner_output_path(output_root, planner_name)

    @classmethod
    def load_result_rows(cls, path: str | Path) -> list[Dict[str, object]]:
        return STHeuristicAblationResultStore.load_result_rows(path)

    @classmethod
    def completed_result_keys(cls, output_path: Path) -> set[tuple[str, float]]:
        return STHeuristicAblationResultStore.completed_result_keys(output_path)

    @classmethod
    def result_paths(
        cls,
        output_root: str | Path,
        specs: Sequence[SearchPlannerSpec] | None = None,
    ) -> Dict[str, Path]:
        selected_specs = SEARCH_GROUPS["heuristic"] if specs is None else specs
        return {
            spec.name: cls.planner_output_path(output_root, spec.name)
            for spec in selected_specs
        }

    @classmethod
    def base_manifest_path(cls, base_root: str | Path, record: STHeuristicAblationRecord) -> Path:
        return BaseManifestStore.manifest_path(base_root, domain_key=record.source_domain)

    @classmethod
    def reconstruct_instance(
        cls,
        record: STHeuristicAblationRecord,
        base_root: str | Path,
    ):
        base_manifest_path = cls.base_manifest_path(base_root, record)
        base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
        instance = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
        env = instance.env.copy()
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(record.dynamic_obstacles)
        instance.env = env
        instance.stgcs = Instance.build_stgcs_from_env(
            env,
            tmax=STHeuristicAblationManifestBuilder.TMAX,
            vlimit=STHeuristicAblationManifestBuilder.VLIMIT,
        )
        return instance, record.query.to_query(), base_manifest_path, base_record

    @staticmethod
    def run_search_spec(
        instance,
        query,
        spec: SearchPlannerSpec,
        runtime_limit_secs: float,
    ) -> STHeuristicAblationResultEntry:
        planner = MRMPExperiment._build_low_level_planner(
            instance,
            spec,
            runtime_limit_secs=runtime_limit_secs,
        )
        sol, runtime, status = planner.plan(instance.stgcs, query)
        algorithm = planner.last_search_algorithm
        return STHeuristicAblationResultEntry(
            is_success=status != STPlanStatus.FAIL,
            runtime=float(runtime),
            cost=float("inf") if sol is None else float(sol.duration),
            num_expanded_nodes=0 if algorithm is None else int(algorithm.n_expanded),
            num_generated_nodes=0 if algorithm is None else int(algorithm.n_generated),
        )

    @classmethod
    def append_result_row(
        cls,
        output_path: str | Path,
        planner_name: str,
        record: STHeuristicAblationRecord,
        budget: float,
        entry: STHeuristicAblationResultEntry,
    ) -> None:
        STHeuristicAblationResultStore.append_result_row(output_path, planner_name, record, budget, entry)


class STHeuristicAblationRunCLI:
    H_TAB_GUB_PLANNER = "Search(h_tab+GUB)"
    HEURISTIC_RUN_ORDER = ("h_mot", "h_tab", "h_tri", "h_max", "h_zero")
    DEFAULT_BUDGET = 600.0

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Run the BFS heuristic ablation on the dynamic-obstacle ST manifest."
        )
        parser.add_argument("manifest", type=Path)
        parser.add_argument("--budget", type=float, nargs="+", default=(cls.DEFAULT_BUDGET,))
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--output-root", type=Path, default=Path("data/results/st_planning/heuristic_ablation"))
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--h-tab-only",
            action="store_true",
            help="Run only Search(h_tab+GUB) for the requested budget values.",
        )
        args = parser.parse_args()
        for budget in args.budget:
            if not math.isfinite(float(budget)) or float(budget) <= 0.0:
                parser.error("--budget values must be finite and positive.")
        return args

    @classmethod
    def search_specs(cls, h_tab_only: bool) -> tuple[SearchPlannerSpec, ...]:
        source_specs = tuple(SEARCH_GROUPS["heuristic"])
        ordered_specs: list[SearchPlannerSpec] = []
        for heuristic in cls.HEURISTIC_RUN_ORDER:
            matches = tuple(spec for spec in source_specs if spec.heuristic == heuristic)
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one heuristic spec for {heuristic!r}, got {len(matches)}.")
            ordered_specs.append(matches[0])
        specs = tuple(ordered_specs)
        if not h_tab_only:
            return specs
        selected = tuple(spec for spec in specs if spec.name == cls.H_TAB_GUB_PLANNER)
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one {cls.H_TAB_GUB_PLANNER} spec, got {len(selected)}.")
        return selected

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
        specs: Sequence[SearchPlannerSpec] = cls.search_specs(args.h_tab_only)
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
            required_heuristics = {spec.heuristic for _, spec, _ in pending_runs if not spec.exact_astar}
            BaseOfflineHeuristicStore.prepare_instance_for_search(
                instance,
                base_manifest_path,
                base_record,
                required_heuristics=required_heuristics,
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
    STHeuristicAblationRunCLI.main()
