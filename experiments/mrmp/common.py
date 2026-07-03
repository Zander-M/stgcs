from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from baselines.cb_gcs import CBGCSOptions, CBGCSPlanner
from baselines.ompl_kcbs import OfficialOMPLKCBSOptions, OfficialOMPLKCBSPlanner
from baselines.pbs_zeta_sipp import ZetaSIPPPriorityBasedSearch
from baselines.sp_strrtstar import FixedPrioritySTRRTStarPlanner, FixedPrioritySTRRTStarSnapshot
from experiments.base.manifest import BaseBenchmarkRecord
from experiments.base.common import BaseInstanceFactory
from experiments.common import MPResultEntry
from experiments.mrmp.manifest import MRMPBenchmarkRecord
from experiments.mrmp.planner_defs import SearchPlannerSpec
from stgcs.bfs.best_first_search import SearchAlgorithm
from stgcs.bfs.domination_check import (
    AStar_DC,
    ExactSetContainment_DC,
    GlobalUpperBound_DC,
    InexactSetContainment_DC,
    Sampling_DC,
)
from stgcs.bfs.heuristics import HeurLowerBoundGraph, HeurShortCut, HeurZero, MaxHeuristic
from stgcs.pbs import ChildExpansionMode, PriorityBasedSearch
from stgcs.mrmp_planner import MRMPQuery, pp, windowed_pbs, windowed_pp
from stgcs.st_planner import SearchPlanner


@dataclass(frozen=True)
class MRMPResultEntry:
    is_success: bool
    runtime: float
    sum_of_costs: float
    makespan: float
    num_completed_agents: int
    pbs_calls: int
    pbs_runtime: float
    pbs_popped_nodes: int
    pbs_generated_children: int
    pbs_update_calls: int
    mp_calls: int
    mp_runtime: float
    ecd_calls: int
    ecd_runtime: float
    cc_calls: int
    cc_runtime: float
    mp_gcs_calls: int = 0
    mp_gcs_runtime: float = 0.0
    mp_gub_calls: int = 0
    mp_gub_runtime: float = 0.0
    mp_search_calls: int = 0
    mp_search_runtime: float = 0.0
    mp_convex_restriction_calls: int = 0
    mp_convex_restriction_runtime: float = 0.0
    mp_domination_check_calls: int = 0
    mp_domination_check_runtime: float = 0.0
    mp_domination_convex_restriction_calls: int = 0
    mp_domination_convex_restriction_runtime: float = 0.0


class MRMPExperiment:
    @staticmethod
    def reconstruct_instance(
        record: MRMPBenchmarkRecord,
        base_record: BaseBenchmarkRecord,
        compute_heuristics: bool = False,
    ) -> Tuple[object, MRMPQuery]:
        instance = BaseInstanceFactory.from_record(base_record, compute_heuristics=compute_heuristics)
        return instance, MRMPQuery([query.to_query() for query in record.queries])

    @staticmethod
    def exact_reference_solve(instance, query, budget: float = float("inf")) -> Tuple[object | None, MPResultEntry]:
        if instance.lbg is None:
            gcs = instance.stgcs.get_gcs_instance().gcs
            if instance.sc_heur is None:
                instance.sc_heur = HeurShortCut(instance.stgcs)
            instance.lbg = HeurLowerBoundGraph(instance.stgcs, gcs, use_update=True)

        gcs_instance = instance.stgcs.get_gcs_instance(query)
        if gcs_instance is None:
            return None, MPResultEntry(False, 0.0, math.inf)

        astar = SearchAlgorithm(
            heuristics=instance.lbg,
            domination_checker=[AStar_DC()],
        )
        sol = astar.run(instance.stgcs, gcs_instance.gcs, timeout_seconds=budget)
        if sol is None:
            return None, MPResultEntry(False, astar.runtime, math.inf, astar.n_expanded, astar.n_generated)
        return sol, MPResultEntry(True, astar.runtime, sol.duration, astar.n_expanded, astar.n_generated)

    @staticmethod
    def _heuristic_by_name(instance, name: str):
        if name == "Zero":
            return HeurZero(instance.stgcs)
        if name == "SC":
            if instance.sc_heur is None:
                instance.sc_heur = HeurShortCut(instance.stgcs)
            return instance.sc_heur
        if name == "LBG":
            if instance.lbg is None:
                raise ValueError("Offline LBG heuristic is not loaded.")
            return instance.lbg
        if name == "TD":
            if instance.td_heur is None:
                raise ValueError("Offline TD heuristic is not loaded.")
            return instance.td_heur
        if name == "Max":
            return MaxHeuristic.from_instance(instance)
        raise ValueError(f"Unknown heuristic {name!r}")

    @staticmethod
    def _domination_by_names(
        instance,
        names: Iterable[str],
        ub_cost: float,
        ub_runtime: float,
        epsilon: float,
    ):
        vmin = -instance.stgcs.vlimit * np.ones(instance.stgcs.dimension)
        vmax = instance.stgcs.vlimit * np.ones(instance.stgcs.dimension)
        checkers = []
        for name in names:
            if name == "GUB":
                checkers.append(GlobalUpperBound_DC(ub_cost, ub_runtime, epsilon))
            elif name == "IPC":
                checkers.append(Sampling_DC())
            elif name == "ISC":
                checkers.append(InexactSetContainment_DC(vmin, vmax))
            elif name == "ESC":
                checkers.append(
                    ExactSetContainment_DC(
                        vmin,
                        vmax,
                        instance.stgcs.tmax,
                        option=ExactSetContainment_DC.Option.VERTEX_ONLY,
                    )
                )
            else:
                raise ValueError(f"Unknown domination checker {name!r}")
        return checkers

    @classmethod
    def _build_low_level_planner(
        cls,
        instance,
        spec: SearchPlannerSpec,
        runtime_limit_secs: float,
    ) -> SearchPlanner:
        if spec.exact_astar:
            raise ValueError("MRMP runner only supports low-level SearchPlanner specs.")
        heuristic = cls._heuristic_by_name(instance, spec.heuristic)
        domination = cls._domination_by_names(instance, spec.domination, math.inf, 0.0, spec.epsilon)
        return SearchPlanner(
            dc_list=domination,
            heur=heuristic,
            eps=spec.epsilon,
            runtime_limit_secs=runtime_limit_secs,
        )

    @staticmethod
    def _num_completed_agents(solutions: Sequence[object], queries: Sequence[object]) -> int:
        completed = 0
        for solution, query in zip(solutions, queries):
            if getattr(solution, "size", 0) == 0:
                continue
            if np.allclose(solution.xT[:-1], query.goal, atol=1e-6):
                completed += 1
        return completed

    @staticmethod
    def _solution_metrics(solutions: Sequence[object]) -> Tuple[float, float]:
        return (
            float(sum(solution.duration for solution in solutions)),
            float(max(solution.duration for solution in solutions)),
        )

    @staticmethod
    def _zero_profile_stats() -> np.ndarray:
        return np.zeros(2, dtype=float)

    @classmethod
    def _profile_stats(cls, profiler: Dict[str, np.ndarray], key: str) -> np.ndarray:
        stats = profiler.get(key)
        if stats is None:
            return cls._zero_profile_stats()
        return np.asarray(stats, dtype=float)

    @classmethod
    def _build_result_entry(
        cls,
        solutions: Sequence[object] | None,
        queries: Sequence[object],
        is_success: bool,
        runtime: float,
        pbs_calls: int,
        pbs_runtime: float,
        pbs_popped_nodes: int,
        pbs_generated_children: int,
        pbs_update_calls: int,
        mp_stats: np.ndarray,
        ecd_stats: np.ndarray,
        cc_stats: np.ndarray,
        mp_gcs_stats: np.ndarray | None = None,
        mp_gub_stats: np.ndarray | None = None,
        mp_search_stats: np.ndarray | None = None,
        mp_convex_restriction_stats: np.ndarray | None = None,
        mp_domination_check_stats: np.ndarray | None = None,
        mp_domination_convex_restriction_stats: np.ndarray | None = None,
    ) -> MRMPResultEntry:
        mp_gcs_stats = cls._zero_profile_stats() if mp_gcs_stats is None else np.asarray(mp_gcs_stats, dtype=float)
        mp_gub_stats = cls._zero_profile_stats() if mp_gub_stats is None else np.asarray(mp_gub_stats, dtype=float)
        mp_search_stats = (
            cls._zero_profile_stats() if mp_search_stats is None else np.asarray(mp_search_stats, dtype=float)
        )
        mp_convex_restriction_stats = (
            cls._zero_profile_stats()
            if mp_convex_restriction_stats is None
            else np.asarray(mp_convex_restriction_stats, dtype=float)
        )
        mp_domination_check_stats = (
            cls._zero_profile_stats()
            if mp_domination_check_stats is None
            else np.asarray(mp_domination_check_stats, dtype=float)
        )
        mp_domination_convex_restriction_stats = (
            cls._zero_profile_stats()
            if mp_domination_convex_restriction_stats is None
            else np.asarray(mp_domination_convex_restriction_stats, dtype=float)
        )
        if is_success:
            if solutions is None:
                raise ValueError("Successful MRMP coordination must return solutions.")
            sum_of_costs, makespan = cls._solution_metrics(solutions)
            completed_agents = len(queries)
        else:
            sum_of_costs = math.inf
            makespan = math.inf
            completed_agents = cls._num_completed_agents([] if solutions is None else solutions, queries)
        return MRMPResultEntry(
            is_success=bool(is_success),
            runtime=float(runtime),
            sum_of_costs=sum_of_costs,
            makespan=makespan,
            num_completed_agents=completed_agents,
            pbs_calls=int(pbs_calls),
            pbs_runtime=float(pbs_runtime),
            pbs_popped_nodes=int(pbs_popped_nodes),
            pbs_generated_children=int(pbs_generated_children),
            pbs_update_calls=int(pbs_update_calls),
            mp_calls=int(mp_stats[0]),
            mp_runtime=float(mp_stats[1]),
            ecd_calls=int(ecd_stats[0]),
            ecd_runtime=float(ecd_stats[1]),
            cc_calls=int(cc_stats[0]),
            cc_runtime=float(cc_stats[1]),
            mp_gcs_calls=int(mp_gcs_stats[0]),
            mp_gcs_runtime=float(mp_gcs_stats[1]),
            mp_gub_calls=int(mp_gub_stats[0]),
            mp_gub_runtime=float(mp_gub_stats[1]),
            mp_search_calls=int(mp_search_stats[0]),
            mp_search_runtime=float(mp_search_stats[1]),
            mp_convex_restriction_calls=int(mp_convex_restriction_stats[0]),
            mp_convex_restriction_runtime=float(mp_convex_restriction_stats[1]),
            mp_domination_check_calls=int(mp_domination_check_stats[0]),
            mp_domination_check_runtime=float(mp_domination_check_stats[1]),
            mp_domination_convex_restriction_calls=int(mp_domination_convex_restriction_stats[0]),
            mp_domination_convex_restriction_runtime=float(mp_domination_convex_restriction_stats[1]),
        )

    @classmethod
    def _run_full_horizon_pbs_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        child_expansion_mode: ChildExpansionMode,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        pbs = PriorityBasedSearch(
            instance.stgcs,
            planner,
            instance.env.robot_radius,
            child_expansion_mode=child_expansion_mode,
        )
        solutions, runtime, success = pbs.run(list(queries), budget, verbose=False)
        entry = cls._build_result_entry(
            solutions=solutions,
            queries=queries,
            is_success=success,
            runtime=runtime,
            pbs_calls=1,
            pbs_runtime=runtime,
            pbs_popped_nodes=pbs.num_popped_nodes,
            pbs_generated_children=pbs.num_generated_children,
            pbs_update_calls=pbs.num_update_calls,
            mp_stats=pbs._profiler["mp"],
            ecd_stats=pbs._profiler["ecd"],
            cc_stats=pbs._profiler["cc"],
            mp_gcs_stats=cls._profile_stats(pbs._profiler, "gcs"),
            mp_gub_stats=cls._profile_stats(pbs._profiler, "gub"),
            mp_search_stats=cls._profile_stats(pbs._profiler, "search"),
            mp_convex_restriction_stats=cls._profile_stats(pbs._profiler, "cr"),
            mp_domination_check_stats=cls._profile_stats(pbs._profiler, "dc"),
            mp_domination_convex_restriction_stats=cls._profile_stats(pbs._profiler, "dc_cr"),
        )
        return solutions, entry

    @classmethod
    def _run_full_horizon_pbs(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        child_expansion_mode: ChildExpansionMode,
    ) -> MRMPResultEntry:
        return cls._run_full_horizon_pbs_with_solutions(
            instance,
            queries,
            planner,
            budget,
            child_expansion_mode,
        )[1]

    @classmethod
    def run_full_horizon_pbs_spec_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        child_expansion_mode: ChildExpansionMode,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = cls._build_low_level_planner(instance, spec, runtime_limit_secs=budget)
        return cls._run_full_horizon_pbs_with_solutions(
            instance,
            queries,
            planner,
            budget,
            child_expansion_mode,
        )

    @classmethod
    def run_full_horizon_pbs_spec(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        child_expansion_mode: ChildExpansionMode,
    ) -> MRMPResultEntry:
        return cls.run_full_horizon_pbs_spec_with_solutions(
            instance,
            queries,
            spec,
            budget,
            child_expansion_mode,
        )[1]

    @classmethod
    def _run_fixed_priority_planning_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        priority_order: Sequence[int] | None = None,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        solutions, result = pp(
            instance.stgcs,
            planner,
            queries,
            instance.env.robot_radius,
            timeout_secs=budget,
            priority_order=priority_order,
        )
        entry = cls._build_result_entry(
            solutions=solutions,
            queries=queries,
            is_success=result.success,
            runtime=float(result.runtime[0]),
            pbs_calls=int(result.wpp[0]),
            pbs_runtime=float(result.wpp[1]),
            pbs_popped_nodes=int(result.pbs_popped_nodes),
            pbs_generated_children=int(result.pbs_generated_children),
            pbs_update_calls=int(result.pbs_update_calls),
            mp_stats=result.mp,
            ecd_stats=result.ecd,
            cc_stats=result.cc,
            mp_gcs_stats=result.gcs,
            mp_gub_stats=result.gub,
            mp_search_stats=result.search,
            mp_convex_restriction_stats=result.cr,
            mp_domination_check_stats=result.dc,
            mp_domination_convex_restriction_stats=result.dc_cr,
        )
        return solutions, entry

    @classmethod
    def _run_fixed_priority_planning(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        priority_order: Sequence[int] | None = None,
    ) -> MRMPResultEntry:
        return cls._run_fixed_priority_planning_with_solutions(
            instance,
            queries,
            planner,
            budget,
            priority_order,
        )[1]

    @classmethod
    def run_fixed_priority_planning_spec_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        priority_order: Sequence[int] | None = None,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = cls._build_low_level_planner(instance, spec, runtime_limit_secs=budget)
        return cls._run_fixed_priority_planning_with_solutions(
            instance,
            queries,
            planner,
            budget,
            priority_order,
        )

    @classmethod
    def run_fixed_priority_planning_spec(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        priority_order: Sequence[int] | None = None,
    ) -> MRMPResultEntry:
        return cls.run_fixed_priority_planning_spec_with_solutions(
            instance,
            queries,
            spec,
            budget,
            priority_order,
        )[1]

    @classmethod
    def _run_windowed_pbs_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        child_expansion_mode: ChildExpansionMode,
        execution_horizon_factor: float = 1.0,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        window_span = float(window_span_factor) * instance.env.robot_radius / instance.stgcs.vlimit
        solutions, result = windowed_pbs(
            instance.stgcs,
            planner,
            queries,
            instance.env.robot_radius,
            window_span=window_span,
            execution_horizon_factor=execution_horizon_factor,
            timeout_secs=budget,
            dynamic_window_adjustment=dynamic_window_adjustment,
            child_expansion_mode=child_expansion_mode,
        )
        entry = cls._build_result_entry(
            solutions=solutions,
            queries=queries,
            is_success=result.success,
            runtime=float(result.runtime[0]),
            pbs_calls=int(result.wpbs[0]),
            pbs_runtime=float(result.wpbs[1]),
            pbs_popped_nodes=result.pbs_popped_nodes,
            pbs_generated_children=result.pbs_generated_children,
            pbs_update_calls=result.pbs_update_calls,
            mp_stats=result.mp,
            ecd_stats=result.ecd,
            cc_stats=result.cc,
            mp_gcs_stats=result.gcs,
            mp_gub_stats=result.gub,
            mp_search_stats=result.search,
            mp_convex_restriction_stats=result.cr,
            mp_domination_check_stats=result.dc,
            mp_domination_convex_restriction_stats=result.dc_cr,
        )
        return solutions, entry

    @classmethod
    def _run_windowed_pbs(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        child_expansion_mode: ChildExpansionMode,
        execution_horizon_factor: float = 1.0,
    ) -> MRMPResultEntry:
        return cls._run_windowed_pbs_with_solutions(
            instance,
            queries,
            planner,
            budget,
            window_span_factor,
            dynamic_window_adjustment,
            child_expansion_mode,
            execution_horizon_factor,
        )[1]

    @classmethod
    def run_windowed_pbs_spec_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        child_expansion_mode: ChildExpansionMode,
        execution_horizon_factor: float = 1.0,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = cls._build_low_level_planner(instance, spec, runtime_limit_secs=budget)
        return cls._run_windowed_pbs_with_solutions(
            instance,
            queries,
            planner,
            budget,
            window_span_factor,
            dynamic_window_adjustment,
            child_expansion_mode,
            execution_horizon_factor,
        )

    @classmethod
    def run_windowed_pbs_spec(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        child_expansion_mode: ChildExpansionMode,
        execution_horizon_factor: float = 1.0,
    ) -> MRMPResultEntry:
        return cls.run_windowed_pbs_spec_with_solutions(
            instance,
            queries,
            spec,
            budget,
            window_span_factor,
            dynamic_window_adjustment,
            child_expansion_mode,
            execution_horizon_factor,
        )[1]

    @classmethod
    def _run_windowed_pp_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        execution_horizon_factor: float = 1.0,
        priority_order: Sequence[int] | None = None,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        window_span = float(window_span_factor) * instance.env.robot_radius / instance.stgcs.vlimit
        solutions, result = windowed_pp(
            instance.stgcs,
            planner,
            queries,
            instance.env.robot_radius,
            window_span=window_span,
            execution_horizon_factor=execution_horizon_factor,
            timeout_secs=budget,
            dynamic_window_adjustment=dynamic_window_adjustment,
            priority_order=priority_order,
        )
        entry = cls._build_result_entry(
            solutions=solutions,
            queries=queries,
            is_success=result.success,
            runtime=float(result.runtime[0]),
            pbs_calls=int(result.wpp[0]),
            pbs_runtime=float(result.wpp[1]),
            pbs_popped_nodes=0,
            pbs_generated_children=0,
            pbs_update_calls=0,
            mp_stats=result.mp,
            ecd_stats=result.ecd,
            cc_stats=result.cc,
            mp_gcs_stats=result.gcs,
            mp_gub_stats=result.gub,
            mp_search_stats=result.search,
            mp_convex_restriction_stats=result.cr,
            mp_domination_check_stats=result.dc,
            mp_domination_convex_restriction_stats=result.dc_cr,
        )
        return solutions, entry

    @classmethod
    def _run_windowed_pp(
        cls,
        instance,
        queries: Sequence[object],
        planner: SearchPlanner,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        execution_horizon_factor: float = 1.0,
        priority_order: Sequence[int] | None = None,
    ) -> MRMPResultEntry:
        return cls._run_windowed_pp_with_solutions(
            instance,
            queries,
            planner,
            budget,
            window_span_factor,
            dynamic_window_adjustment,
            execution_horizon_factor,
            priority_order,
        )[1]

    @classmethod
    def run_windowed_pp_spec_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        execution_horizon_factor: float = 1.0,
        priority_order: Sequence[int] | None = None,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = cls._build_low_level_planner(instance, spec, runtime_limit_secs=budget)
        return cls._run_windowed_pp_with_solutions(
            instance,
            queries,
            planner,
            budget,
            window_span_factor,
            dynamic_window_adjustment,
            execution_horizon_factor,
            priority_order,
        )

    @classmethod
    def run_windowed_pp_spec(
        cls,
        instance,
        queries: Sequence[object],
        spec: SearchPlannerSpec,
        budget: float,
        window_span_factor: float,
        dynamic_window_adjustment: bool,
        execution_horizon_factor: float = 1.0,
        priority_order: Sequence[int] | None = None,
    ) -> MRMPResultEntry:
        return cls.run_windowed_pp_spec_with_solutions(
            instance,
            queries,
            spec,
            budget,
            window_span_factor,
            dynamic_window_adjustment,
            execution_horizon_factor,
            priority_order,
        )[1]

    @classmethod
    def _entry_from_fixed_priority_strrt_star_snapshot(
        cls,
        snapshot: FixedPrioritySTRRTStarSnapshot,
        queries: Sequence[object],
    ) -> MRMPResultEntry:
        return cls._build_result_entry(
            solutions=snapshot.solutions,
            queries=queries,
            is_success=snapshot.success,
            runtime=snapshot.runtime,
            pbs_calls=1,
            pbs_runtime=snapshot.runtime,
            pbs_popped_nodes=0,
            pbs_generated_children=0,
            pbs_update_calls=0,
            mp_stats=np.array([snapshot.low_level_calls, snapshot.low_level_runtime], dtype=float),
            ecd_stats=np.zeros(2, dtype=float),
            cc_stats=np.array([snapshot.collision_checks, snapshot.collision_runtime], dtype=float),
        )

    @classmethod
    def run_fixed_priority_strrt_star_snapshots(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
    ) -> Tuple[MRMPResultEntry, MRMPResultEntry]:
        planner = FixedPrioritySTRRTStarPlanner()
        result = planner.solve(instance, queries, timeout_secs=budget)
        return (
            cls._entry_from_fixed_priority_strrt_star_snapshot(result.first_solution, queries),
            cls._entry_from_fixed_priority_strrt_star_snapshot(result.final_solution, queries),
        )

    @classmethod
    def run_fixed_priority_strrt_star(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
    ) -> MRMPResultEntry:
        _, final_entry = cls.run_fixed_priority_strrt_star_snapshots(instance, queries, budget)
        return final_entry

    @staticmethod
    def _zeta_sipp_cell_size(instance, multiplier: float) -> float:
        return float(multiplier) * float(instance.env.robot_radius)

    @classmethod
    def run_pbs_zeta_sipp_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        child_expansion_mode: ChildExpansionMode,
        cell_size_multiplier: float = 1.0,
        seed: int = 0,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = ZetaSIPPPriorityBasedSearch(
            instance.stgcs,
            instance.env,
            robot_radius=instance.env.robot_radius,
            child_expansion_mode=child_expansion_mode,
            seed=int(seed),
            cell_size=cls._zeta_sipp_cell_size(instance, cell_size_multiplier),
            runtime_limit_secs=float(budget),
        )
        solutions, runtime, success = planner.run(list(queries), budget, verbose=False)
        entry = cls._build_result_entry(
            solutions=solutions,
            queries=queries,
            is_success=success,
            runtime=runtime,
            pbs_calls=1,
            pbs_runtime=runtime,
            pbs_popped_nodes=planner.num_popped_nodes,
            pbs_generated_children=planner.num_generated_children,
            pbs_update_calls=planner.num_update_calls,
            mp_stats=planner._profiler["mp"],
            ecd_stats=planner._profiler["ecd"],
            cc_stats=planner._profiler["cc"],
            mp_gcs_stats=cls._profile_stats(planner._profiler, "gcs"),
            mp_gub_stats=cls._profile_stats(planner._profiler, "gub"),
            mp_search_stats=cls._profile_stats(planner._profiler, "search"),
            mp_convex_restriction_stats=cls._profile_stats(planner._profiler, "cr"),
            mp_domination_check_stats=cls._profile_stats(planner._profiler, "dc"),
            mp_domination_convex_restriction_stats=cls._profile_stats(planner._profiler, "dc_cr"),
        )
        return solutions, entry

    @classmethod
    def run_pbs_zeta_sipp(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        child_expansion_mode: ChildExpansionMode,
        cell_size_multiplier: float = 1.0,
        seed: int = 0,
    ) -> MRMPResultEntry:
        return cls.run_pbs_zeta_sipp_with_solutions(
            instance,
            queries,
            budget,
            child_expansion_mode,
            cell_size_multiplier,
            seed,
        )[1]

    @classmethod
    def _cb_gcs_result_entry(
        cls,
        result,
        queries: Sequence[object],
    ) -> MRMPResultEntry:
        if result.success:
            if result.solutions is None:
                raise ValueError("Successful CB-GCS coordination must return solutions.")
            sum_of_costs, makespan = cls._solution_metrics(result.solutions)
            completed_agents = len(queries)
        else:
            makespan = math.inf
            completed_agents = cls._num_completed_agents([] if result.solutions is None else result.solutions, queries)
            sum_of_costs = math.inf
        return MRMPResultEntry(
            is_success=bool(result.success),
            runtime=float(result.runtime),
            sum_of_costs=sum_of_costs,
            makespan=makespan,
            num_completed_agents=completed_agents,
            pbs_calls=1,
            pbs_runtime=float(result.runtime),
            pbs_popped_nodes=int(result.popped_nodes),
            pbs_generated_children=int(result.generated_children),
            pbs_update_calls=int(result.update_calls),
            mp_calls=int(result.low_level_calls),
            mp_runtime=float(result.low_level_runtime),
            ecd_calls=0,
            ecd_runtime=0.0,
            cc_calls=int(result.conflict_checks),
            cc_runtime=float(result.conflict_runtime),
        )

    @classmethod
    def run_cb_gcs_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        time_horizon: float | None = None,
        max_rounded_paths: int = 100,
        time_step: float | None = None,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = CBGCSPlanner(
            CBGCSOptions(
                time_horizon=time_horizon,
                time_step=time_step,
                max_rounded_paths=int(max_rounded_paths),
            )
        )
        result = planner.solve(instance, queries, timeout_secs=budget)
        return result.solutions, cls._cb_gcs_result_entry(result, queries)

    @classmethod
    def run_cb_gcs(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        time_horizon: float | None = None,
        max_rounded_paths: int = 100,
        time_step: float | None = None,
    ) -> MRMPResultEntry:
        return cls.run_cb_gcs_with_solutions(
            instance,
            queries,
            budget,
            time_horizon=time_horizon,
            max_rounded_paths=max_rounded_paths,
            time_step=time_step,
        )[1]

    @classmethod
    def _entry_from_ompl_kcbs_result(
        cls,
        result,
        queries: Sequence[object],
    ) -> MRMPResultEntry:
        if result.success:
            if result.solutions is None:
                raise ValueError("Successful K-CBS coordination must return solutions.")
            makespan = float(max(solution.duration for solution in result.solutions))
            completed_agents = len(queries)
            sum_of_costs = float(result.cost)
        else:
            makespan = math.inf
            completed_agents = cls._num_completed_agents([] if result.solutions is None else result.solutions, queries)
            sum_of_costs = math.inf
        root_solve_time = float(result.root_solve_time)
        return MRMPResultEntry(
            is_success=bool(result.success),
            runtime=float(result.runtime),
            sum_of_costs=sum_of_costs,
            makespan=makespan,
            num_completed_agents=completed_agents,
            pbs_calls=1,
            pbs_runtime=float(result.runtime),
            pbs_popped_nodes=int(result.num_nodes_expanded),
            pbs_generated_children=0,
            pbs_update_calls=0,
            mp_calls=len(queries) if root_solve_time >= 0.0 else 0,
            mp_runtime=max(root_solve_time, 0.0),
            ecd_calls=0,
            ecd_runtime=0.0,
            cc_calls=0,
            cc_runtime=0.0,
        )

    @classmethod
    def run_ompl_kcbs_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        low_level_solve_time: float = 1.0,
        propagation_step_size: float = 0.1,
        min_control_duration: int = 1,
        max_control_duration: int = 10,
        goal_tolerance: float = 1e-3,
        goal_bias: float = 0.05,
        intermediate_states: bool = False,
        merge_bound: int | None = None,
        num_threads: int = 4,
        seed: int = 0,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = OfficialOMPLKCBSPlanner(
            OfficialOMPLKCBSOptions(
                max_runtime_in_secs=float(budget),
                low_level_solve_time=float(low_level_solve_time),
                propagation_step_size=float(propagation_step_size),
                min_control_duration=int(min_control_duration),
                max_control_duration=int(max_control_duration),
                goal_tolerance=float(goal_tolerance),
                goal_bias=float(goal_bias),
                intermediate_states=bool(intermediate_states),
                merge_bound=merge_bound,
                num_threads=int(num_threads),
            ),
            seed=int(seed),
        )
        result = planner.solve(instance, queries, timeout_secs=budget)
        return result.solutions, cls._entry_from_ompl_kcbs_result(result, queries)

    @classmethod
    def run_ompl_kcbs(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        low_level_solve_time: float = 1.0,
        propagation_step_size: float = 0.1,
        min_control_duration: int = 1,
        max_control_duration: int = 10,
        goal_tolerance: float = 1e-3,
        goal_bias: float = 0.05,
        intermediate_states: bool = False,
        merge_bound: int | None = None,
        num_threads: int = 4,
        seed: int = 0,
    ) -> MRMPResultEntry:
        return cls.run_ompl_kcbs_with_solutions(
            instance,
            queries,
            budget,
            low_level_solve_time=low_level_solve_time,
            propagation_step_size=propagation_step_size,
            min_control_duration=min_control_duration,
            max_control_duration=max_control_duration,
            goal_tolerance=goal_tolerance,
            goal_bias=goal_bias,
            intermediate_states=intermediate_states,
            merge_bound=merge_bound,
            num_threads=num_threads,
            seed=seed,
        )[1]

    @staticmethod
    def planner_output_path(
        output_root: str | Path,
        domain_key: str,
        planner_name: str,
    ) -> Path:
        target_dir = Path(output_root) / domain_key / "results"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = planner_name.replace("/", "_")
        return target_dir / f"{safe_name}.csv"

    @staticmethod
    def append_result_row(
        output_path: str | Path,
        instance_name: str,
        record: MRMPBenchmarkRecord,
        budget: float,
        entry: MRMPResultEntry,
    ) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    instance_name,
                    entry.is_success,
                    entry.runtime,
                    entry.sum_of_costs,
                    entry.makespan,
                    entry.num_completed_agents,
                    entry.pbs_calls,
                    entry.pbs_runtime,
                    entry.pbs_popped_nodes,
                    entry.pbs_generated_children,
                    entry.pbs_update_calls,
                    entry.mp_calls,
                    entry.mp_runtime,
                    entry.ecd_calls,
                    entry.ecd_runtime,
                    entry.cc_calls,
                    entry.cc_runtime,
                    record.instance_id,
                    budget,
                    record.traffic_family,
                    record.traffic_tier,
                    record.num_agents,
                    record.stgcs_num_vertices,
                    record.stgcs_num_edges,
                    entry.mp_gcs_calls,
                    entry.mp_gcs_runtime,
                    entry.mp_gub_calls,
                    entry.mp_gub_runtime,
                    entry.mp_search_calls,
                    entry.mp_search_runtime,
                    entry.mp_convex_restriction_calls,
                    entry.mp_convex_restriction_runtime,
                    entry.mp_domination_check_calls,
                    entry.mp_domination_check_runtime,
                    entry.mp_domination_convex_restriction_calls,
                    entry.mp_domination_convex_restriction_runtime,
                ]
            )

    @staticmethod
    def load_result_rows(path: str | Path) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        with Path(path).open(newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 19:
                    continue
                rows.append(
                    {
                        "name": row[0],
                        "is_success": row[1].lower() == "true",
                        "runtime": float(row[2]) if row[2] != "inf" else math.inf,
                        "sum_of_costs": float(row[3]) if row[3] != "inf" else math.inf,
                        "makespan": float(row[4]) if row[4] != "inf" else math.inf,
                        "num_completed_agents": int(float(row[5])),
                        "pbs_calls": int(float(row[6])),
                        "pbs_runtime": float(row[7]) if row[7] != "inf" else math.inf,
                        "pbs_popped_nodes": int(float(row[8])),
                        "pbs_generated_children": int(float(row[9])),
                        "pbs_update_calls": int(float(row[10])),
                        "mp_calls": int(float(row[11])),
                        "mp_runtime": float(row[12]) if row[12] != "inf" else math.inf,
                        "ecd_calls": int(float(row[13])),
                        "ecd_runtime": float(row[14]) if row[14] != "inf" else math.inf,
                        "cc_calls": int(float(row[15])),
                        "cc_runtime": float(row[16]) if row[16] != "inf" else math.inf,
                        "instance_id": row[17],
                        "budget": float(row[18]),
                        "traffic_family": row[19] if len(row) > 19 else "",
                        "traffic_tier": row[20] if len(row) > 20 else "",
                        "num_agents": int(float(row[21])) if len(row) > 21 else 0,
                        "stgcs_num_vertices": int(float(row[22])) if len(row) > 22 else 0,
                        "stgcs_num_edges": int(float(row[23])) if len(row) > 23 else 0,
                        "mp_gcs_calls": int(float(row[24])) if len(row) > 24 else 0,
                        "mp_gcs_runtime": float(row[25]) if len(row) > 25 and row[25] != "inf" else 0.0,
                        "mp_gub_calls": int(float(row[26])) if len(row) > 26 else 0,
                        "mp_gub_runtime": float(row[27]) if len(row) > 27 and row[27] != "inf" else 0.0,
                        "mp_search_calls": int(float(row[28])) if len(row) > 28 else 0,
                        "mp_search_runtime": float(row[29]) if len(row) > 29 and row[29] != "inf" else 0.0,
                        "mp_convex_restriction_calls": int(float(row[30])) if len(row) > 30 else 0,
                        "mp_convex_restriction_runtime": (
                            float(row[31]) if len(row) > 31 and row[31] != "inf" else 0.0
                        ),
                        "mp_domination_check_calls": int(float(row[32])) if len(row) > 32 else 0,
                        "mp_domination_check_runtime": (
                            float(row[33]) if len(row) > 33 and row[33] != "inf" else 0.0
                        ),
                        "mp_domination_convex_restriction_calls": int(float(row[34])) if len(row) > 34 else 0,
                        "mp_domination_convex_restriction_runtime": (
                            float(row[35]) if len(row) > 35 and row[35] != "inf" else 0.0
                        ),
                        "has_detailed_profile": len(row) > 33,
                        "has_disjoint_breakdown_profile": len(row) > 35,
                    }
                )
        return rows
