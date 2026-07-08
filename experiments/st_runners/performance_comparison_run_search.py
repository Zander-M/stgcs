from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Dict, Sequence
import zlib

from baselines.common import ShortestPathSolution
from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar, OfficialOMPLSTRRTStarOptions
from baselines.zeta_sipp import ZetaStarSIPPPlanner
from experiments.st_runners.heuristic_ablation_run_search import STHeuristicAblationRunner
from experiments.st_runners.heuristic_ablation_st_manifest import (
    STHeuristicAblationManifestBuilder,
    STHeuristicAblationRecord,
    STHeuristicAblationResultEntry,
    STHeuristicAblationResultStore,
)
from experiments.st_runners.dominance_stress_manifest import STDominanceStressManifestBuilder
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from benchmark.planners.mrmp import SearchPlannerSpec
from stgcs.st_planner import MICPPlanner, STPlanStatus


class STPerformanceComparisonRunner:
    DELTA_POS_SPEC = SearchPlannerSpec("h_max", ("GUB", "delta_pos"), epsilon=10.0)
    DELTA_SET_SPEC = SearchPlannerSpec("h_max", ("GUB", "delta_set"), epsilon=10.0)
    DELTA_SET_EPS1_SPEC = SearchPlannerSpec("h_max", ("GUB", "delta_set"), epsilon=1.0)
    SEARCH_SPECS = (DELTA_POS_SPEC, DELTA_SET_SPEC, DELTA_SET_EPS1_SPEC)
    SEARCH_SPEC_BY_PLANNER = {spec.name: spec for spec in SEARCH_SPECS}
    DELTA_POS_PLANNER = DELTA_POS_SPEC.name
    DELTA_SET_PLANNER = DELTA_SET_SPEC.name
    DELTA_SET_EPS1_PLANNER = DELTA_SET_EPS1_SPEC.name
    MICP_PLANNER = "MICP"
    MICP_ROUNDING_PLANNER = "MICP(g)"
    ST_RRT_PLANNER = "OMPL ST-RRT*-C"
    ST_RRT_FIRST_PLANNER = "OMPL ST-RRT*-C(first)"
    ST_RRT_FINAL_PLANNER = "OMPL ST-RRT*-C(final)"
    ST_RRT_OUTPUT_PLANNERS = (ST_RRT_FIRST_PLANNER, ST_RRT_FINAL_PLANNER)
    OMPL_ST_RRT_PLANNER = ST_RRT_PLANNER
    OMPL_ST_RRT_FIRST_PLANNER = ST_RRT_FIRST_PLANNER
    OMPL_ST_RRT_FINAL_PLANNER = ST_RRT_FINAL_PLANNER
    OMPL_ST_RRT_OUTPUT_PLANNERS = (OMPL_ST_RRT_FIRST_PLANNER, OMPL_ST_RRT_FINAL_PLANNER)
    ZETA_SIPP_PLANNER = "Zeta*-SIPP-C"
    ZETA_SIPP_2R_PLANNER = "Zeta*-SIPP-C(2r)"
    ZETA_SIPP_PLANNERS = (ZETA_SIPP_PLANNER, ZETA_SIPP_2R_PLANNER)
    BASELINE_PLANNERS = (
        MICP_PLANNER,
        MICP_ROUNDING_PLANNER,
        *ST_RRT_OUTPUT_PLANNERS,
        *ZETA_SIPP_PLANNERS,
    )
    PLANNERS = (DELTA_POS_PLANNER, DELTA_SET_PLANNER, DELTA_SET_EPS1_PLANNER, *BASELINE_PLANNERS)
    PLANNER_KEY_TO_NAME = {
        "all": "all",
        "delta-pos": DELTA_POS_PLANNER,
        "delta-set": DELTA_SET_PLANNER,
        "delta-set-eps1": DELTA_SET_EPS1_PLANNER,
        "delta-set-epsilon1": DELTA_SET_EPS1_PLANNER,
        "micp": MICP_PLANNER,
        "micpg": MICP_ROUNDING_PLANNER,
        "micp-g": MICP_ROUNDING_PLANNER,
        "strrt": ST_RRT_PLANNER,
        "st-rrt": ST_RRT_PLANNER,
        "strrt-first": ST_RRT_FIRST_PLANNER,
        "st-rrt-first": ST_RRT_FIRST_PLANNER,
        "strrt-final": ST_RRT_FINAL_PLANNER,
        "st-rrt-final": ST_RRT_FINAL_PLANNER,
        "ompl-strrt": OMPL_ST_RRT_PLANNER,
        "ompl_strrt": OMPL_ST_RRT_PLANNER,
        "ompl-strrt-star": OMPL_ST_RRT_PLANNER,
        "ompl_strrt_star": OMPL_ST_RRT_PLANNER,
        "ompl-st-rrt": OMPL_ST_RRT_PLANNER,
        "ompl-strrt-first": OMPL_ST_RRT_FIRST_PLANNER,
        "ompl_strrt_first": OMPL_ST_RRT_FIRST_PLANNER,
        "ompl-strrt-star-first": OMPL_ST_RRT_FIRST_PLANNER,
        "ompl_strrt_star_first": OMPL_ST_RRT_FIRST_PLANNER,
        "ompl-st-rrt-first": OMPL_ST_RRT_FIRST_PLANNER,
        "ompl-strrt-final": OMPL_ST_RRT_FINAL_PLANNER,
        "ompl_strrt_final": OMPL_ST_RRT_FINAL_PLANNER,
        "ompl-strrt-star-final": OMPL_ST_RRT_FINAL_PLANNER,
        "ompl_strrt_star_final": OMPL_ST_RRT_FINAL_PLANNER,
        "ompl-st-rrt-final": OMPL_ST_RRT_FINAL_PLANNER,
        "zeta": ZETA_SIPP_PLANNER,
        "zeta-r": ZETA_SIPP_PLANNER,
        "zeta-sipp": ZETA_SIPP_PLANNER,
        "zeta2r": ZETA_SIPP_2R_PLANNER,
        "zeta-2r": ZETA_SIPP_2R_PLANNER,
        "zeta-sipp-2r": ZETA_SIPP_2R_PLANNER,
    }
    MICP_ROUNDED_PATH_SCALE = 1000.0
    @classmethod
    def planner_names(cls, requested: Sequence[str] | None = None) -> tuple[str, ...]:
        if requested is None:
            return cls.PLANNERS
        selected: list[str] = []
        for value in requested:
            name = cls.planner_name_from_cli_value(value)
            if name == "all":
                return cls.PLANNERS
            names = cls.expand_planner_name(name)
            for selected_name in names:
                if selected_name not in selected:
                    selected.append(selected_name)
        if not selected:
            raise ValueError("At least one ST performance-comparison planner must be selected.")
        return tuple(selected)

    @classmethod
    def planner_name_from_cli_value(cls, value: str) -> str:
        token = str(value)
        if token in cls.PLANNERS:
            return token
        if token in (cls.ST_RRT_PLANNER, *cls.ST_RRT_OUTPUT_PLANNERS):
            return token
        key = token.lower()
        if key in cls.PLANNER_KEY_TO_NAME:
            return cls.PLANNER_KEY_TO_NAME[key]
        choices = ", ".join(cls.planner_cli_choices())
        raise ValueError(f"Unknown planner {value!r}. Valid planner keys/names: {choices}.")

    @classmethod
    def expand_planner_name(cls, planner_name: str) -> tuple[str, ...]:
        if planner_name == cls.ST_RRT_PLANNER:
            return cls.ST_RRT_OUTPUT_PLANNERS
        return (planner_name,)

    @classmethod
    def planner_cli_choices(cls) -> tuple[str, ...]:
        return (
            *cls.PLANNER_KEY_TO_NAME.keys(),
            cls.ST_RRT_PLANNER,
            cls.OMPL_ST_RRT_PLANNER,
            *cls.OMPL_ST_RRT_OUTPUT_PLANNERS,
            *cls.PLANNERS,
        )

    @classmethod
    def planner_cli_help(cls) -> str:
        pairs = [
            f"delta-pos={cls.DELTA_POS_PLANNER}",
            f"delta-set={cls.DELTA_SET_PLANNER}",
            f"delta-set-eps1={cls.DELTA_SET_EPS1_PLANNER}",
            f"micp={cls.MICP_PLANNER}",
            f"micpg={cls.MICP_ROUNDING_PLANNER}",
            f"strrt/ompl-strrt={cls.ST_RRT_FIRST_PLANNER}+{cls.ST_RRT_FINAL_PLANNER}",
            f"zeta={cls.ZETA_SIPP_PLANNER}",
            f"zeta2r={cls.ZETA_SIPP_2R_PLANNER}",
        ]
        return "Planner keys: all, " + ", ".join(pairs)

    @classmethod
    def result_paths(cls, output_root: str | Path, planners: Sequence[str] | None = None) -> Dict[str, Path]:
        selected = cls.PLANNERS if planners is None else tuple(
            selected_name
            for planner_name in planners
            for selected_name in cls.expand_planner_name(planner_name)
        )
        return {
            planner_name: STHeuristicAblationResultStore.planner_output_path(output_root, planner_name)
            for planner_name in selected
        }

    @staticmethod
    def seed_for_record(record: STHeuristicAblationRecord, seed_offset: int) -> int:
        base_seed = zlib.crc32(str(record.instance_id).encode("utf-8")) & 0xFFFFFFFF
        return int((base_seed + int(seed_offset)) % (2**31 - 1))

    @classmethod
    def micp_rounding_paths(cls, edge_count: int) -> int:
        num_edges = int(edge_count)
        if num_edges <= 1:
            raise ValueError(f"MICP(g) requires |E| > 1 to set rounded paths, got |E|={num_edges}.")
        return int(math.ceil(cls.MICP_ROUNDED_PATH_SCALE * math.log(num_edges)))

    @staticmethod
    def zeta_sipp_cell_size(instance, multiplier: float) -> float:
        return float(multiplier) * float(instance.env.robot_radius)

    @staticmethod
    def entry_from_shortest_path_solution(
        solution: ShortestPathSolution,
        measured_runtime: float,
        budget: float,
    ) -> STHeuristicAblationResultEntry:
        runtime = float(solution.time)
        if not math.isfinite(runtime) or runtime < 0.0:
            runtime = float(measured_runtime)
        if not bool(solution.is_success):
            runtime = min(max(runtime, 0.0), float(budget))
        return STHeuristicAblationResultEntry(
            is_success=bool(solution.is_success),
            runtime=float(runtime),
            cost=float(solution.cost) if bool(solution.is_success) else math.inf,
            num_expanded_nodes=0,
            num_generated_nodes=0,
        )

    @staticmethod
    def cost_label(cost: float | None) -> str:
        if cost is None:
            return "none"
        if not math.isfinite(float(cost)):
            return "inf"
        return f"{float(cost):.3f}"

    @classmethod
    def search_spec_for_planner(cls, planner_name: str) -> SearchPlannerSpec:
        try:
            return cls.SEARCH_SPEC_BY_PLANNER[str(planner_name)]
        except KeyError as exc:
            raise ValueError(f"Planner {planner_name!r} is not an ST search comparison planner.") from exc

    @classmethod
    def run_search_planner(
        cls,
        planner_name: str,
        instance,
        query,
        budget: float,
    ) -> STHeuristicAblationResultEntry:
        return STHeuristicAblationRunner.run_search_spec(
            instance,
            query,
            cls.search_spec_for_planner(planner_name),
            float(budget),
        )

    @staticmethod
    def run_micp(
        instance,
        query,
        budget: float,
        max_rounded_paths: int,
    ) -> STHeuristicAblationResultEntry:
        planner = MICPPlanner(
            max_rounded_paths=int(max_rounded_paths),
            runtime_limit_secs=float(budget),
        )
        sol, runtime, status = planner.plan(instance.stgcs, query)
        return STHeuristicAblationResultEntry(
            is_success=status != STPlanStatus.FAIL,
            runtime=float(runtime),
            cost=math.inf if sol is None else float(sol.duration),
            num_expanded_nodes=0,
            num_generated_nodes=0,
        )

    @classmethod
    def run_strrt(
        cls,
        instance,
        query,
        budget: float,
        seed: int,
    ) -> Dict[str, STHeuristicAblationResultEntry]:
        return cls.run_ompl_strrt(instance, query, budget, seed=seed)

    @classmethod
    def run_ompl_strrt(
        cls,
        instance,
        query,
        budget: float,
        seed: int,
    ) -> Dict[str, STHeuristicAblationResultEntry]:
        options = OfficialOMPLSTRRTStarOptions(
            max_runtime_in_secs=float(budget),
            return_first_valid=False,
        )
        planner = OfficialOMPLSTRRTStar(instance.env.copy(), int(seed), float(query.vlimit))
        ts = time.perf_counter()
        snapshots = planner.solve_with_snapshots(
            query.start,
            query.goal,
            t0=float(query.t_start),
            options=options,
        )
        measured_runtime = time.perf_counter() - ts
        return {
            cls.OMPL_ST_RRT_FIRST_PLANNER: cls.entry_from_shortest_path_solution(
                snapshots.first_solution,
                measured_runtime,
                budget,
            ),
            cls.OMPL_ST_RRT_FINAL_PLANNER: cls.entry_from_shortest_path_solution(
                snapshots.final_solution,
                measured_runtime,
                budget,
            ),
        }

    @classmethod
    def run_zeta_sipp(
        cls,
        instance,
        query,
        budget: float,
        seed: int,
        cell_size_multiplier: float = 1.0,
    ) -> STHeuristicAblationResultEntry:
        if int(instance.env.dim) != 2:
            raise ValueError("Zeta*-SIPP performance comparison is defined only for 2D ST-planning domains.")
        planner = ZetaStarSIPPPlanner(
            instance.env.copy(),
            int(seed),
            float(query.vlimit),
            cell_size=cls.zeta_sipp_cell_size(instance, cell_size_multiplier),
            runtime_limit_secs=float(budget),
        )
        ts = time.perf_counter()
        solution = planner.solve(
            query.start,
            query.goal,
            t_start=float(query.t_start),
            t_max=STHeuristicAblationManifestBuilder.TMAX,
            is_stay=bool(query.is_stay),
        )
        return cls.entry_from_shortest_path_solution(solution, time.perf_counter() - ts, budget)

    @classmethod
    def run_planner(
        cls,
        planner_name: str,
        instance,
        query,
        record: STHeuristicAblationRecord,
        budget: float,
        seed_offset: int,
    ) -> STHeuristicAblationResultEntry:
        if planner_name in cls.SEARCH_SPEC_BY_PLANNER:
            return cls.run_search_planner(planner_name, instance, query, budget)
        if planner_name == cls.MICP_PLANNER:
            return cls.run_micp(instance, query, budget, max_rounded_paths=0)
        if planner_name == cls.MICP_ROUNDING_PLANNER:
            max_rounded_paths = cls.micp_rounding_paths(record.stgcs_num_edges)
            return cls.run_micp(instance, query, budget, max_rounded_paths=max_rounded_paths)
        seed = cls.seed_for_record(record, seed_offset)
        if planner_name in cls.ST_RRT_OUTPUT_PLANNERS:
            return cls.run_strrt(instance, query, budget, seed=seed)[planner_name]
        if planner_name == cls.ZETA_SIPP_PLANNER:
            return cls.run_zeta_sipp(instance, query, budget, seed=seed, cell_size_multiplier=1.0)
        if planner_name == cls.ZETA_SIPP_2R_PLANNER:
            return cls.run_zeta_sipp(instance, query, budget, seed=seed, cell_size_multiplier=2.0)
        raise ValueError(f"Unknown ST performance-comparison planner {planner_name!r}.")


class STPerformanceComparisonRunCLI:
    DEFAULT_MANIFEST = STDominanceStressManifestBuilder.DEFAULT_OUTPUT
    DEFAULT_OUTPUT_ROOT = Path("data/results/st_planning/performance_comparison")
    DEFAULT_BUDGET = 600.0

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Run the ST-planning performance comparison: "
                f"{STPerformanceComparisonRunner.DELTA_POS_PLANNER}, "
                f"{STPerformanceComparisonRunner.DELTA_SET_PLANNER}, "
                f"{STPerformanceComparisonRunner.DELTA_SET_EPS1_PLANNER}, MICP, MICP(g), "
                "OMPL ST-RRT*-C, and Zeta*-SIPP variants on a shared ST manifest."
            )
        )
        parser.add_argument("manifest", type=Path, nargs="?", default=cls.DEFAULT_MANIFEST)
        parser.add_argument("--budget", type=float, nargs="+", default=(cls.DEFAULT_BUDGET,))
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--output-root", type=Path, default=cls.DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--seed-offset", type=int, default=0)
        parser.add_argument(
            "--planner",
            nargs="+",
            default=("all",),
            metavar="PLANNER",
            help=STPerformanceComparisonRunner.planner_cli_help(),
        )
        args = parser.parse_args()
        for budget in args.budget:
            if not math.isfinite(float(budget)) or float(budget) <= 0.0:
                parser.error("--budget values must be finite and positive.")
        if args.limit is not None and int(args.limit) <= 0:
            parser.error("--limit must be positive when provided.")
        try:
            args.planner_names = STPerformanceComparisonRunner.planner_names(args.planner)
        except ValueError as exc:
            parser.error(str(exc))
        return args

    @staticmethod
    def budget_label(budget: float) -> str:
        if math.isinf(budget):
            return "inf"
        if math.isclose(budget, round(budget)):
            return str(int(round(budget)))
        return f"{budget:g}"

    @staticmethod
    def budgets(args: argparse.Namespace) -> tuple[float, ...]:
        return tuple(float(budget) for budget in args.budget)

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        records = STHeuristicAblationManifestBuilder.load_manifest(Path(args.manifest))
        if args.limit is not None:
            records = records[: int(args.limit)]
        planner_names = tuple(args.planner_names)
        print(f"Selected planners: {', '.join(planner_names)}")
        budgets = cls.budgets(args)
        output_paths = STPerformanceComparisonRunner.result_paths(args.output_root, planner_names)
        completed = {
            planner_name: STHeuristicAblationResultStore.completed_result_keys(path)
            for planner_name, path in output_paths.items()
        }

        for record in records:
            pending_runs: list[tuple[float, str]] = []
            for budget in budgets:
                budget_value = float(budget)
                key = STHeuristicAblationResultStore.result_key(record.instance_id, budget_value)
                for planner_name in planner_names:
                    if key in completed[planner_name]:
                        print(
                            f"{record.instance_id} | {planner_name} "
                            f"| budget={cls.budget_label(budget_value)} | skipped (resume)"
                        )
                        continue
                    pending_runs.append((budget_value, planner_name))
            if not pending_runs:
                continue

            instance, query, base_manifest_path, base_record = STHeuristicAblationRunner.reconstruct_instance(
                record,
                args.base_root,
            )
            required_heuristics = {
                STPerformanceComparisonRunner.search_spec_for_planner(planner_name).heuristic
                for _, planner_name in pending_runs
                if planner_name in STPerformanceComparisonRunner.SEARCH_SPEC_BY_PLANNER
            }
            if required_heuristics:
                BaseOfflineHeuristicStore.prepare_instance_for_search(
                    instance,
                    base_manifest_path,
                    base_record,
                    required_heuristics=required_heuristics,
                    online_h_tab_timeout_secs=max(budget for budget, _ in pending_runs),
                )

            handled_runs: set[tuple[float, str]] = set()
            for budget in budgets:
                budget_value = float(budget)
                strrt_planners = [
                    planner_name for pending_budget, planner_name in pending_runs
                    if math.isclose(float(pending_budget), budget_value)
                    and planner_name in STPerformanceComparisonRunner.ST_RRT_OUTPUT_PLANNERS
                ]
                if not strrt_planners:
                    continue
                seed = STPerformanceComparisonRunner.seed_for_record(record, int(args.seed_offset))
                progress_label = (
                    f"{record.instance_id} | {STPerformanceComparisonRunner.ST_RRT_PLANNER} "
                    f"| budget={cls.budget_label(budget_value)}"
                )
                print(f"{progress_label} | started | seed={seed}", flush=True)
                strrt_entries = STPerformanceComparisonRunner.run_strrt(
                    instance,
                    query,
                    budget_value,
                    seed=seed,
                )
                for planner_name in strrt_planners:
                    entry = strrt_entries[planner_name]
                    STHeuristicAblationResultStore.append_result_row(
                        output_paths[planner_name],
                        planner_name,
                        record,
                        budget_value,
                        entry,
                    )
                    completed[planner_name].add(
                        STHeuristicAblationResultStore.result_key(record.instance_id, budget_value)
                    )
                    handled_runs.add((budget_value, planner_name))
                    print(
                        f"{record.instance_id} | {planner_name} | budget={cls.budget_label(budget_value)} "
                        f"| success={entry.is_success} | runtime={entry.runtime:.3f} | cost={entry.cost:.3f}"
                    )

            for budget, planner_name in pending_runs:
                budget_value = float(budget)
                if (budget_value, planner_name) in handled_runs:
                    continue
                entry = STPerformanceComparisonRunner.run_planner(
                    planner_name,
                    instance,
                    query,
                    record,
                    budget_value,
                    int(args.seed_offset),
                )
                STHeuristicAblationResultStore.append_result_row(
                    output_paths[planner_name],
                    planner_name,
                    record,
                    budget_value,
                    entry,
                )
                completed[planner_name].add(
                    STHeuristicAblationResultStore.result_key(record.instance_id, budget_value)
                )
                print(
                    f"{record.instance_id} | {planner_name} | budget={cls.budget_label(budget_value)} "
                    f"| success={entry.is_success} | runtime={entry.runtime:.3f} | cost={entry.cost:.3f}"
                )


if __name__ == "__main__":
    STPerformanceComparisonRunCLI.main()
