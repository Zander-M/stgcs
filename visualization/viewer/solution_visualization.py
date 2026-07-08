from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from baselines.common import ShortestPathSolution
from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar, OfficialOMPLSTRRTStarOptions
from baselines.zeta_sipp import ZetaStarSIPPPlanner
from benchmark.base import BaseManifestStore
from benchmark.manifests.mrmp import MRMPBenchmarkRecord
from benchmark.planners.mrmp import MRMPPerformanceComparison, SearchPlannerSpec
from benchmark.manifests.st_planning import (
    STHeuristicAblationRecord,
    STHeuristicAblationResultEntry,
    STPlanningManifestStore,
)
from benchmark.planners.st_planning import STPerformanceComparisonConfig as STPerformanceComparisonRunner
from stgcs.st_planner import MICPPlanner, STPlanStatus
from stgcs.trajectory import STTrajectory


@dataclass(frozen=True)
class ViewerSolutionRun:
    planner_key: str
    planner_name: str
    budget: float
    status: str
    entry: STHeuristicAblationResultEntry
    solution: STTrajectory | ShortestPathSolution | None


@dataclass(frozen=True)
class ViewerMRMPSolutionRun:
    planner_key: str
    planner_name: str
    budget: float
    status: str
    entry: Any
    solutions: Sequence[STTrajectory] | None


class SolutionVisualizationService:
    DEFAULT_BUDGET = 10.0
    ST_PLANNER_OPTIONS = (
        ("delta-pos", STPerformanceComparisonRunner.DELTA_POS_PLANNER, (2, 3)),
        ("delta-set", STPerformanceComparisonRunner.DELTA_SET_PLANNER, (2, 3)),
        ("delta-set-eps1", STPerformanceComparisonRunner.DELTA_SET_EPS1_PLANNER, (2, 3)),
        ("micp", STPerformanceComparisonRunner.MICP_PLANNER, (2, 3)),
        ("micpg", STPerformanceComparisonRunner.MICP_ROUNDING_PLANNER, (2, 3)),
        ("strrt-first", STPerformanceComparisonRunner.ST_RRT_FIRST_PLANNER, (2, 3)),
        ("strrt-final", STPerformanceComparisonRunner.ST_RRT_FINAL_PLANNER, (2, 3)),
        ("zeta", STPerformanceComparisonRunner.ZETA_SIPP_PLANNER, (2,)),
        ("zeta2r", STPerformanceComparisonRunner.ZETA_SIPP_2R_PLANNER, (2,)),
    )
    MRMP_PLANNER_BUTTONS = {
        MRMPPerformanceComparison.KCBS_KEY: "K-CBS",
        MRMPPerformanceComparison.CB_GCS_KEY: "CB-GCS",
        MRMPPerformanceComparison.WINDOWED_PBS_KEY: "wPBS",
        MRMPPerformanceComparison.WINDOWED_PP_KEY: "wPP",
        MRMPPerformanceComparison.PBS_KEY: "PBS",
    }
    MRMP_VIEWER_PLANNER_KEYS = (
        MRMPPerformanceComparison.KCBS_KEY,
        MRMPPerformanceComparison.CB_GCS_KEY,
        MRMPPerformanceComparison.WINDOWED_PBS_KEY,
        MRMPPerformanceComparison.WINDOWED_PP_KEY,
        MRMPPerformanceComparison.PBS_KEY,
    )

    def __init__(self, base_root: str | Path) -> None:
        self.base_root = Path(base_root).resolve()

    @classmethod
    def planner_options_for_kind(cls, problem_kind: str) -> list[Dict[str, Any]]:
        if problem_kind == "st_heuristic_ablation":
            return [
                {"key": key, "name": name, "space_dims": list(space_dims)}
                for key, name, space_dims in cls.ST_PLANNER_OPTIONS
            ]
        if problem_kind == "mrmp":
            options: list[Dict[str, Any]] = []
            for key in cls.MRMP_VIEWER_PLANNER_KEYS:
                option = {
                    "key": key,
                    "name": MRMPPerformanceComparison.planner_name(key),
                    "button_label": cls.MRMP_PLANNER_BUTTONS[key],
                }
                if key in {
                    MRMPPerformanceComparison.CB_GCS_KEY,
                    MRMPPerformanceComparison.KCBS_KEY,
                }:
                    option["space_dims"] = [2]
                options.append(option)
            return options
        return []

    @classmethod
    def planner_payload(cls) -> Dict[str, Any]:
        return {
            "ok": True,
            "default_budget": cls.DEFAULT_BUDGET,
            "default_window_alpha": MRMPPerformanceComparison.WINDOW_SPAN_FACTOR,
            "default_window_beta": MRMPPerformanceComparison.EXECUTION_HORIZON_FACTOR,
            "default_epsilon": MRMPPerformanceComparison.LOW_LEVEL_SPEC.epsilon,
            "planners": {
                "st_heuristic_ablation": cls.planner_options_for_kind("st_heuristic_ablation"),
                "mrmp": cls.planner_options_for_kind("mrmp"),
                "base": [],
            },
        }

    @staticmethod
    def finite_float_or_none(value: float | int | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric

    @staticmethod
    def positive_finite_float(value: float | int, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return numeric

    @staticmethod
    def unit_interval_float(value: float | int, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return numeric

    @staticmethod
    def _trajectory_points(solution: STTrajectory | ShortestPathSolution | None) -> tuple[int, Sequence[Any]]:
        if solution is None:
            return 0, []
        if isinstance(solution, ShortestPathSolution):
            return int(solution.dim), solution.trajectory
        return int(solution.dim), solution.points

    @classmethod
    def trajectory_to_viewer_json(
        cls,
        solution: STTrajectory | ShortestPathSolution | None,
    ) -> Dict[str, Any]:
        dim, raw_points = cls._trajectory_points(solution)
        if dim <= 1 or not raw_points:
            return {
                "dimension": max(dim - 1, 0),
                "segments": [],
                "path": [],
                "time_range": [0.0, 0.0],
            }

        segments: list[Dict[str, Any]] = []
        path: list[list[float]] = []
        min_time = math.inf
        max_time = -math.inf
        for point in raw_points:
            values = np.asarray(point, dtype=float).reshape(-1)
            if values.size != 2 * dim:
                raise ValueError(
                    f"Solution segment has {values.size} values, expected {2 * dim} for dim={dim}."
                )
            start = values[: dim - 1]
            goal = values[dim: 2 * dim - 1]
            t_start = float(values[dim - 1])
            t_end = float(values[-1])
            if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
                raise ValueError("Solution contains a non-finite spatial coordinate.")
            if not math.isfinite(t_start) or not math.isfinite(t_end):
                raise ValueError("Solution contains a non-finite time coordinate.")
            segment = {
                "start": start.tolist(),
                "goal": goal.tolist(),
                "t_start": t_start,
                "t_end": t_end,
            }
            segments.append(segment)
            if not path:
                path.append(segment["start"])
            path.append(segment["goal"])
            min_time = min(min_time, t_start, t_end)
            max_time = max(max_time, t_start, t_end)

        if not math.isfinite(min_time) or not math.isfinite(max_time):
            min_time = max_time = 0.0
        return {
            "dimension": dim - 1,
            "segments": segments,
            "path": path,
            "time_range": [float(min_time), float(max_time)],
        }

    @classmethod
    def run_to_viewer_json(
        cls,
        record: STHeuristicAblationRecord,
        run: ViewerSolutionRun,
    ) -> Dict[str, Any]:
        return {
            "ok": True,
            "instance_id": record.instance_id,
            "planner_key": run.planner_key,
            "planner_name": run.planner_name,
            "budget": cls.finite_float_or_none(run.budget),
            "status": run.status,
            "is_success": bool(run.entry.is_success),
            "runtime": cls.finite_float_or_none(run.entry.runtime),
            "cost": cls.finite_float_or_none(run.entry.cost),
            "num_expanded_nodes": int(run.entry.num_expanded_nodes),
            "num_generated_nodes": int(run.entry.num_generated_nodes),
            "trajectory": cls.trajectory_to_viewer_json(run.solution if run.entry.is_success else None),
        }

    @classmethod
    def mrmp_run_to_viewer_json(
        cls,
        record: MRMPBenchmarkRecord,
        run: ViewerMRMPSolutionRun,
    ) -> Dict[str, Any]:
        trajectories = [
            {
                **cls.trajectory_to_viewer_json(solution),
                "agent_index": agent_idx,
            }
            for agent_idx, solution in enumerate([] if run.solutions is None else run.solutions)
        ]
        return {
            "ok": True,
            "instance_id": record.instance_id,
            "planner_key": run.planner_key,
            "planner_name": run.planner_name,
            "budget": cls.finite_float_or_none(run.budget),
            "status": run.status,
            "is_success": bool(run.entry.is_success),
            "runtime": cls.finite_float_or_none(run.entry.runtime),
            "cost": cls.finite_float_or_none(run.entry.sum_of_costs),
            "makespan": cls.finite_float_or_none(run.entry.makespan),
            "num_completed_agents": int(run.entry.num_completed_agents),
            "pbs_calls": int(run.entry.pbs_calls),
            "pbs_runtime": cls.finite_float_or_none(run.entry.pbs_runtime),
            "pbs_popped_nodes": int(run.entry.pbs_popped_nodes),
            "pbs_generated_children": int(run.entry.pbs_generated_children),
            "pbs_update_calls": int(run.entry.pbs_update_calls),
            "num_expanded_nodes": int(run.entry.pbs_popped_nodes),
            "num_generated_nodes": int(run.entry.pbs_generated_children),
            "trajectories": trajectories,
        }

    @staticmethod
    def _entry_from_st_plan(
        solution: STTrajectory | None,
        runtime: float,
        status: STPlanStatus,
        algorithm: Any = None,
    ) -> STHeuristicAblationResultEntry:
        return STHeuristicAblationResultEntry(
            is_success=status != STPlanStatus.FAIL,
            runtime=float(runtime),
            cost=math.inf if solution is None else float(solution.duration),
            num_expanded_nodes=0 if algorithm is None else int(algorithm.n_expanded),
            num_generated_nodes=0 if algorithm is None else int(algorithm.n_generated),
        )

    @staticmethod
    def _require_single_planner(planner_key: str) -> str:
        planner_name = STPerformanceComparisonRunner.planner_name_from_cli_value(planner_key)
        expanded = STPerformanceComparisonRunner.expand_planner_name(planner_name)
        if len(expanded) != 1:
            raise ValueError(
                f"Planner key {planner_key!r} expands to multiple result streams; choose one explicit stream."
            )
        return expanded[0]

    @staticmethod
    def _run_search_solution(
        instance,
        query,
        planner_name: str,
        budget: float,
    ) -> tuple[STTrajectory | None, STHeuristicAblationResultEntry, str]:
        from experiments.mrmp_runners.common import MRMPExperiment

        planner = MRMPExperiment._build_low_level_planner(
            instance,
            STPerformanceComparisonRunner.search_spec_for_planner(planner_name),
            runtime_limit_secs=float(budget),
        )
        solution, runtime, status = planner.plan(instance.stgcs, query)
        entry = SolutionVisualizationService._entry_from_st_plan(
            solution,
            runtime,
            status,
            planner.last_search_algorithm,
        )
        return solution, entry, status.name

    @staticmethod
    def _run_micp_solution(
        instance,
        query,
        budget: float,
        max_rounded_paths: int,
    ) -> tuple[STTrajectory | None, STHeuristicAblationResultEntry, str]:
        planner = MICPPlanner(
            max_rounded_paths=int(max_rounded_paths),
            runtime_limit_secs=float(budget),
        )
        solution, runtime, status = planner.plan(instance.stgcs, query)
        entry = SolutionVisualizationService._entry_from_st_plan(solution, runtime, status)
        return solution, entry, status.name

    @staticmethod
    def _run_strrt_solution(
        instance,
        query,
        budget: float,
        seed: int,
        planner_name: str,
    ) -> tuple[ShortestPathSolution, STHeuristicAblationResultEntry, str]:
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
        solution = (
            snapshots.first_solution
            if planner_name == STPerformanceComparisonRunner.ST_RRT_FIRST_PLANNER
            else snapshots.final_solution
        )
        entry = STPerformanceComparisonRunner.entry_from_shortest_path_solution(
            solution,
            measured_runtime,
            budget,
        )
        return solution, entry, "SUCCESS" if solution.is_success else "FAIL"

    @staticmethod
    def _run_zeta_sipp_solution(
        instance,
        query,
        budget: float,
        seed: int,
        cell_size_multiplier: float,
    ) -> tuple[ShortestPathSolution, STHeuristicAblationResultEntry, str]:
        if int(instance.env.dim) != 2:
            raise ValueError("Zeta*-SIPP performance comparison is defined only for 2D ST-planning domains.")
        planner = ZetaStarSIPPPlanner(
            instance.env.copy(),
            int(seed),
            float(query.vlimit),
            cell_size=STPerformanceComparisonRunner.zeta_sipp_cell_size(instance, cell_size_multiplier),
            runtime_limit_secs=float(budget),
        )
        ts = time.perf_counter()
        solution = planner.solve(
            query.start,
            query.goal,
            t_start=float(query.t_start),
            t_max=STPlanningManifestStore.TMAX,
            is_stay=bool(query.is_stay),
        )
        entry = STPerformanceComparisonRunner.entry_from_shortest_path_solution(
            solution,
            time.perf_counter() - ts,
            budget,
        )
        return solution, entry, "SUCCESS" if solution.is_success else "FAIL"

    def run_st_solution(
        self,
        record: STHeuristicAblationRecord,
        planner_key: str,
        budget: float,
        seed_offset: int = 0,
    ) -> Dict[str, Any]:
        budget_value = self.positive_finite_float(budget, "Solution budget")
        planner_name = self._require_single_planner(planner_key)

        instance, query, base_manifest_path, base_record = STPlanningManifestStore.reconstruct_instance(
            record,
            self.base_root,
        )
        if planner_name in STPerformanceComparisonRunner.SEARCH_SPEC_BY_PLANNER:
            from benchmark.offline_heuristics import BaseOfflineHeuristicStore

            search_spec = STPerformanceComparisonRunner.search_spec_for_planner(planner_name)
            BaseOfflineHeuristicStore.prepare_instance_for_search(
                instance,
                base_manifest_path,
                base_record,
                required_heuristics={search_spec.heuristic},
                online_h_tab_timeout_secs=budget_value,
            )
            solution, entry, status = self._run_search_solution(instance, query, planner_name, budget_value)
        elif planner_name == STPerformanceComparisonRunner.MICP_PLANNER:
            solution, entry, status = self._run_micp_solution(
                instance,
                query,
                budget_value,
                max_rounded_paths=0,
            )
        elif planner_name == STPerformanceComparisonRunner.MICP_ROUNDING_PLANNER:
            solution, entry, status = self._run_micp_solution(
                instance,
                query,
                budget_value,
                max_rounded_paths=STPerformanceComparisonRunner.micp_rounding_paths(record.stgcs_num_edges),
            )
        elif planner_name in STPerformanceComparisonRunner.ST_RRT_OUTPUT_PLANNERS:
            seed = STPerformanceComparisonRunner.seed_for_record(record, seed_offset)
            solution, entry, status = self._run_strrt_solution(
                instance,
                query,
                budget_value,
                seed=seed,
                planner_name=planner_name,
            )
        elif planner_name == STPerformanceComparisonRunner.ZETA_SIPP_PLANNER:
            seed = STPerformanceComparisonRunner.seed_for_record(record, seed_offset)
            solution, entry, status = self._run_zeta_sipp_solution(
                instance,
                query,
                budget_value,
                seed=seed,
                cell_size_multiplier=1.0,
            )
        elif planner_name == STPerformanceComparisonRunner.ZETA_SIPP_2R_PLANNER:
            seed = STPerformanceComparisonRunner.seed_for_record(record, seed_offset)
            solution, entry, status = self._run_zeta_sipp_solution(
                instance,
                query,
                budget_value,
                seed=seed,
                cell_size_multiplier=2.0,
            )
        else:
            raise ValueError(f"Unknown ST solution planner {planner_name!r}.")

        return self.run_to_viewer_json(
            record,
            ViewerSolutionRun(
                planner_key=str(planner_key),
                planner_name=planner_name,
                budget=budget_value,
                status=status,
                entry=entry,
                solution=solution,
            ),
        )

    @staticmethod
    def mrmp_low_level_spec(epsilon: float) -> SearchPlannerSpec:
        base = MRMPPerformanceComparison.LOW_LEVEL_SPEC
        return SearchPlannerSpec(
            heuristic=base.heuristic,
            dominance_checks=base.dominance_checks,
            epsilon=float(epsilon),
            exact_astar=base.exact_astar,
        )

    @classmethod
    def mrmp_planner_name(
        cls,
        planner_key: str,
        low_level_spec: SearchPlannerSpec,
        window_span_factor: float,
        execution_horizon_factor: float,
    ) -> str:
        low_level_name = low_level_spec.name
        alpha = float(window_span_factor)
        beta = float(execution_horizon_factor)
        if planner_key == MRMPPerformanceComparison.PBS_KEY:
            return f"PBS-{MRMPPerformanceComparison.CHILD_EXPANSION_LABEL} + {low_level_name}"
        if planner_key == MRMPPerformanceComparison.WINDOWED_PBS_KEY:
            return f"Windowed PBS(alpha={alpha:g},beta={beta:g},rule=NC) + {low_level_name}"
        if planner_key == MRMPPerformanceComparison.WINDOWED_PP_KEY:
            return f"Windowed PP(alpha={alpha:g},beta={beta:g},order=index) + {low_level_name}"
        if planner_key == MRMPPerformanceComparison.CB_GCS_KEY:
            return MRMPPerformanceComparison.planner_name(planner_key)
        if planner_key == MRMPPerformanceComparison.KCBS_KEY:
            return MRMPPerformanceComparison.planner_name(planner_key)
        raise KeyError(f"Unknown MRMP viewer planner {planner_key!r}.")

    def run_mrmp_solution(
        self,
        record: MRMPBenchmarkRecord,
        planner_key: str,
        budget: float,
        window_span_factor: float,
        execution_horizon_factor: float,
        epsilon: float,
    ) -> Dict[str, Any]:
        budget_value = self.positive_finite_float(budget, "Solution budget")
        alpha = self.positive_finite_float(window_span_factor, "MRMP window alpha")
        beta = self.unit_interval_float(execution_horizon_factor, "MRMP window beta")
        epsilon_value = self.positive_finite_float(epsilon, "MRMP low-level epsilon")
        low_level_spec = self.mrmp_low_level_spec(epsilon_value)
        planner_name = self.mrmp_planner_name(planner_key, low_level_spec, alpha, beta)

        base_manifest_path = BaseManifestStore.manifest_path(self.base_root, domain_key=record.domain_key)
        base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
        from experiments.mrmp_runners.common import MRMPExperiment

        instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)

        if planner_key in MRMPPerformanceComparison.SEARCH_BASED_PLANNER_KEYS:
            from benchmark.offline_heuristics import BaseOfflineHeuristicStore

            BaseOfflineHeuristicStore.prepare_instance_for_search(
                instance,
                base_manifest_path,
                base_record,
                required_heuristics={low_level_spec.heuristic},
                online_h_tab_timeout_secs=budget_value,
            )

        if planner_key == MRMPPerformanceComparison.PBS_KEY:
            solutions, entry = MRMPExperiment.run_full_horizon_pbs_spec_with_solutions(
                instance,
                queries,
                low_level_spec,
                budget=budget_value,
                child_expansion_mode=MRMPPerformanceComparison.CHILD_EXPANSION_MODE,
            )
        elif planner_key == MRMPPerformanceComparison.WINDOWED_PBS_KEY:
            solutions, entry = MRMPExperiment.run_windowed_pbs_spec_with_solutions(
                instance,
                queries,
                low_level_spec,
                budget=budget_value,
                window_span_factor=alpha,
                dynamic_window_adjustment=MRMPPerformanceComparison.DYNAMIC_WINDOW_ADJUSTMENT,
                child_expansion_mode=MRMPPerformanceComparison.CHILD_EXPANSION_MODE,
                execution_horizon_factor=beta,
            )
        elif planner_key == MRMPPerformanceComparison.WINDOWED_PP_KEY:
            solutions, entry = MRMPExperiment.run_windowed_pp_spec_with_solutions(
                instance,
                queries,
                low_level_spec,
                budget=budget_value,
                window_span_factor=alpha,
                dynamic_window_adjustment=MRMPPerformanceComparison.DYNAMIC_WINDOW_ADJUSTMENT,
                execution_horizon_factor=beta,
                priority_order=None,
            )
        elif planner_key == MRMPPerformanceComparison.CB_GCS_KEY:
            solutions, entry = MRMPExperiment.run_cb_gcs_with_solutions(
                instance,
                queries,
                budget=budget_value,
                time_horizon=MRMPPerformanceComparison.CB_GCS_TIME_HORIZON,
                time_step=MRMPPerformanceComparison.CB_GCS_TIME_STEP,
                max_rounded_paths=MRMPPerformanceComparison.micp_rounding_paths(record.stgcs_num_edges),
            )
        elif planner_key == MRMPPerformanceComparison.KCBS_KEY:
            solutions, entry = MRMPExperiment.run_ompl_kcbs_with_solutions(
                instance,
                queries,
                budget=budget_value,
                low_level_solve_time=MRMPPerformanceComparison.KCBS_LOW_LEVEL_SOLVE_TIME,
                propagation_step_size=MRMPPerformanceComparison.KCBS_PROPAGATION_STEP_SIZE,
                min_control_duration=MRMPPerformanceComparison.KCBS_MIN_CONTROL_DURATION,
                max_control_duration=MRMPPerformanceComparison.KCBS_MAX_CONTROL_DURATION,
                goal_tolerance=MRMPPerformanceComparison.KCBS_GOAL_TOLERANCE,
                goal_bias=MRMPPerformanceComparison.KCBS_GOAL_BIAS,
                intermediate_states=MRMPPerformanceComparison.KCBS_INTERMEDIATE_STATES,
                num_threads=MRMPPerformanceComparison.KCBS_NUM_THREADS,
                seed=MRMPPerformanceComparison.KCBS_SEED,
            )
        else:
            raise KeyError(f"Unknown MRMP viewer planner {planner_key!r}.")

        return self.mrmp_run_to_viewer_json(
            record,
            ViewerMRMPSolutionRun(
                planner_key=str(planner_key),
                planner_name=planner_name,
                budget=budget_value,
                status="SUCCESS" if entry.is_success else "FAIL",
                entry=entry,
                solutions=solutions,
            ),
        )

    def run_solution(
        self,
        record: STHeuristicAblationRecord | MRMPBenchmarkRecord,
        planner_key: str,
        budget: float,
        seed_offset: int = 0,
        window_span_factor: float = MRMPPerformanceComparison.WINDOW_SPAN_FACTOR,
        execution_horizon_factor: float = MRMPPerformanceComparison.EXECUTION_HORIZON_FACTOR,
        epsilon: float = MRMPPerformanceComparison.LOW_LEVEL_SPEC.epsilon,
    ) -> Dict[str, Any]:
        if isinstance(record, STHeuristicAblationRecord):
            return self.run_st_solution(record, planner_key, budget, seed_offset=seed_offset)
        if isinstance(record, MRMPBenchmarkRecord):
            return self.run_mrmp_solution(
                record,
                planner_key,
                budget,
                window_span_factor=window_span_factor,
                execution_horizon_factor=execution_horizon_factor,
                epsilon=epsilon,
            )
        raise TypeError(f"Unsupported solution record type {type(record)!r}.")
