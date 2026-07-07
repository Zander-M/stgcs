from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from experiments.st_runners.heuristic_ablation_run_search import STHeuristicAblationRunner
from experiments.st_runners.heuristic_ablation_st_manifest import (
    STHeuristicAblationManifestBuilder,
    STHeuristicAblationRecord,
    STHeuristicAblationResultEntry,
    STHeuristicAblationResultStore,
)
from experiments.st_runners.performance_comparison_run_search import (
    STPerformanceComparisonRunCLI,
    STPerformanceComparisonRunner,
)


class STPerformanceComparisonMICPWrapperCLI:
    MICP_PLANNERS = (
        STPerformanceComparisonRunner.MICP_PLANNER,
        STPerformanceComparisonRunner.MICP_ROUNDING_PLANNER,
    )

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Run or mark one MICP ST performance-comparison row."
        )
        parser.add_argument("manifest", type=Path)
        parser.add_argument("--instance-id", required=True)
        parser.add_argument("--planner", required=True, help="MICP planner key/name: micp or micpg.")
        parser.add_argument("--budget", type=float, required=True)
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument(
            "--output-root",
            type=Path,
            default=STPerformanceComparisonRunCLI.DEFAULT_OUTPUT_ROOT,
        )
        parser.add_argument(
            "--mark-timeout-failure",
            action="store_true",
            help="Append a failed timeout row if the target row is still missing.",
        )
        args = parser.parse_args()
        if not math.isfinite(float(args.budget)) or float(args.budget) <= 0.0:
            parser.error("--budget must be finite and positive.")
        try:
            args.planner_name = cls.micp_planner_name(args.planner)
        except ValueError as exc:
            parser.error(str(exc))
        return args

    @classmethod
    def micp_planner_name(cls, value: str) -> str:
        planner_name = STPerformanceComparisonRunner.planner_name_from_cli_value(value)
        if planner_name not in cls.MICP_PLANNERS:
            choices = ", ".join(("micp", "micpg", *cls.MICP_PLANNERS))
            raise ValueError(f"Planner {value!r} is not a MICP planner. Valid values: {choices}.")
        return planner_name

    @staticmethod
    def record_by_instance_id(
        records: Sequence[STHeuristicAblationRecord],
        instance_id: str,
    ) -> STHeuristicAblationRecord:
        matches = [record for record in records if record.instance_id == instance_id]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one ST record for instance_id {instance_id!r}, found {len(matches)}.")
        return matches[0]

    @staticmethod
    def timeout_entry(budget: float) -> STHeuristicAblationResultEntry:
        return STHeuristicAblationResultEntry(
            is_success=False,
            runtime=float(budget),
            cost=math.inf,
            num_expanded_nodes=0,
            num_generated_nodes=0,
        )

    @classmethod
    def append_entry_if_missing(
        cls,
        output_root: str | Path,
        planner_name: str,
        record: STHeuristicAblationRecord,
        budget: float,
        entry: STHeuristicAblationResultEntry,
    ) -> bool:
        output_path = STPerformanceComparisonRunner.result_paths(output_root, (planner_name,))[planner_name]
        key = STHeuristicAblationResultStore.result_key(record.instance_id, float(budget))
        if key in STHeuristicAblationResultStore.completed_result_keys(output_path):
            print(
                f"{record.instance_id} | {planner_name} "
                f"| budget={STPerformanceComparisonRunCLI.budget_label(float(budget))} | skipped (resume)"
            )
            return False
        STHeuristicAblationResultStore.append_result_row(
            output_path,
            planner_name,
            record,
            float(budget),
            entry,
        )
        return True

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        records = STHeuristicAblationManifestBuilder.load_manifest(Path(args.manifest))
        record = cls.record_by_instance_id(records, str(args.instance_id))
        budget = float(args.budget)
        planner_name = str(args.planner_name)

        if args.mark_timeout_failure:
            appended = cls.append_entry_if_missing(
                args.output_root,
                planner_name,
                record,
                budget,
                cls.timeout_entry(budget),
            )
            action = "timeout recorded" if appended else "timeout skipped"
            print(
                f"{record.instance_id} | {planner_name} "
                f"| budget={STPerformanceComparisonRunCLI.budget_label(budget)} | {action}"
            )
            return

        output_path = STPerformanceComparisonRunner.result_paths(args.output_root, (planner_name,))[planner_name]
        key = STHeuristicAblationResultStore.result_key(record.instance_id, budget)
        if key in STHeuristicAblationResultStore.completed_result_keys(output_path):
            print(
                f"{record.instance_id} | {planner_name} "
                f"| budget={STPerformanceComparisonRunCLI.budget_label(budget)} | skipped (resume)"
            )
            return

        instance, query, _, _ = STHeuristicAblationRunner.reconstruct_instance(record, args.base_root)
        entry = STPerformanceComparisonRunner.run_planner(
            planner_name,
            instance,
            query,
            record,
            budget,
            seed_offset=0,
        )
        cls.append_entry_if_missing(args.output_root, planner_name, record, budget, entry)
        print(
            f"{record.instance_id} | {planner_name} | budget={STPerformanceComparisonRunCLI.budget_label(budget)} "
            f"| success={entry.is_success} | runtime={entry.runtime:.3f} | cost={entry.cost:.3f}"
        )


if __name__ == "__main__":
    STPerformanceComparisonMICPWrapperCLI.main()
