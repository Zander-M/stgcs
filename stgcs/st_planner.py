from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import time
import numpy as np

from stgcs.graph import STGCS
from stgcs.gcs_solver import solve
from stgcs.trajectory import STTrajectory
from stgcs.bfs.best_first_search import SearchAlgorithm, DominanceCheck, Heuristic
from stgcs.bfs.dominance_check import GlobalUpperBoundDominanceCheck, AStarDominanceCheck

import logging
logger = logging.getLogger(__name__)


class STPlanStatus(str, Enum):
    SUCCESS = "success"
    GUB_FALLBACK = "gub_fallback"
    FAIL = "fail"

    @classmethod
    def from_solution(cls, sol: Optional[STTrajectory]) -> STPlanStatus:
        return cls.SUCCESS if sol is not None else cls.FAIL


PlanResult = Tuple[Optional[STTrajectory], float, STPlanStatus]


@dataclass
class MPQuery:
    start: np.ndarray
    goal: np.ndarray
    t_start: float
    is_stay: bool
    vlimit: float

    def __repr__(self):
        dim = self.start.shape[0]
        x_start = ", ".join([f"{self.start[i]:.4f}" for i in range(dim)])
        p_goal = ", ".join([f"{self.goal[i]:.4f}" for i in range(dim)])
        return f"MPQuery(xs=[{x_start}, {self.t_start:.4f}], pg=[{p_goal}], stay={self.is_stay}, vlimit={self.vlimit})"


class STPlanner(ABC):

    def __init__(self, runtime_limit_secs:float) -> None:
        self.runtime_limit_secs = runtime_limit_secs
    
    @abstractmethod
    def plan(self, stgcs:STGCS, mp_query:MPQuery) -> PlanResult:
        pass


class MICPPlanner(STPlanner):

    def __init__(self, max_rounded_paths:int=100, runtime_limit_secs:float=float('inf')) -> None:
        super().__init__(runtime_limit_secs)
        self.rounding = True if max_rounded_paths > 0 else False
        self.max_rounded_paths = max_rounded_paths
    
    def plan(self, stgcs:STGCS, mp_query:MPQuery) -> PlanResult:
        ts = time.perf_counter()
        gcs_instance = stgcs.get_gcs_instance(mp_query)
        if gcs_instance is None:
            logger.warning("Unsolveable instance; cannot find source/target vertices")
            return None, time.perf_counter() - ts, STPlanStatus.FAIL
        sol = solve(gcs_instance, 
              rounding = self.rounding, 
              max_rounded_paths = self.max_rounded_paths, 
              max_runtime = self.runtime_limit_secs
        )
        return sol, time.perf_counter() - ts, STPlanStatus.from_solution(sol)


class SearchPlanner(STPlanner):

    def __init__(
        self, dc_list:List[DominanceCheck], heur:Heuristic, eps:float=1.0,
        runtime_limit_secs:float=float('inf'),
        record_trace: bool = False,
        enable_gub_fallback: bool = False,
    ) -> None:
        super().__init__(runtime_limit_secs)
        self.dc_list = dc_list
        self.heur = heur
        self.eps = eps
        self.record_trace = record_trace
        self.enable_gub_fallback = enable_gub_fallback
        self.gub_planner = None
        self.gub_dc = None
        self.last_search_algorithm: Optional[SearchAlgorithm] = None
        self.last_search_trace = None
        self.last_profile: Dict[str, float] = {}
        for dc in self.dc_list:
            # set up UB planner as A* if GlobalUpperBoundDominanceCheck is used
            if isinstance(dc, GlobalUpperBoundDominanceCheck):
                self.gub_planner = SearchAlgorithm(
                    heuristics = self.heur,
                    dominance_checks = [AStarDominanceCheck()],
                )
                self.gub_dc = dc
    
    def plan(self, stgcs:STGCS, mp_query:MPQuery) -> PlanResult:
        profile = {
            "gcs_construction_time": 0.0,
            "gub_time": 0.0,
            "main_search_time": 0.0,
            "convex_restriction_time": 0.0,
            "convex_restriction_calls": 0,
            "dominance_check_time": 0.0,
            "dominance_convex_restriction_time": 0.0,
            "dominance_convex_restriction_calls": 0,
        }
        self.last_profile = profile

        ts = time.perf_counter()
        mp_gcs_instance = stgcs.get_gcs_instance(mp_query, reuse_base=True)
        profile["gcs_construction_time"] = time.perf_counter() - ts
        if mp_gcs_instance is None:
            logger.warning("Unsolveable instance; cannot find source/target vertices")
            stgcs._clean_source_targets()
            return None, 0.0, STPlanStatus.FAIL

        try:
            convex_restriction_cache = {}
            gub_sol: Optional[STTrajectory] = None
            if self.gub_planner:
                self.gub_planner.reset_dominance_checks()
                self.gub_planner.set_convex_restriction_cache(convex_restriction_cache)
                ts = time.perf_counter()
                gub_sol = self.gub_planner.run(stgcs, mp_gcs_instance.gcs)
                profile["gub_time"] = time.perf_counter() - ts
                profile["convex_restriction_time"] += self.gub_planner.convex_restriction_time
                profile["convex_restriction_calls"] += self.gub_planner.convex_restriction_calls
                profile["dominance_check_time"] += self.gub_planner.dc_time
                profile["dominance_convex_restriction_time"] += (
                    float(getattr(self.gub_planner, "dominance_convex_restriction_time", 0.0))
                )
                profile["dominance_convex_restriction_calls"] += (
                    int(getattr(self.gub_planner, "dominance_convex_restriction_calls", 0))
                )
                if gub_sol is not None:
                    self.gub_dc.reset(gub_sol.duration, profile["gub_time"], self.eps)
                else:
                    self.gub_dc.reset(float('inf'), profile["gub_time"], self.eps)

            # reset dominance checkers
            for dc in self.dc_list:
                if not isinstance(dc, GlobalUpperBoundDominanceCheck):
                    dc.reset()

            alg = SearchAlgorithm(
                heuristics = self.heur,
                heuristic_inflation_factor = self.eps,
                dominance_checks = self.dc_list,
                record_trace = self.record_trace,
                convex_restriction_cache = convex_restriction_cache,
            )
            ts = time.perf_counter()
            sol = alg.run(stgcs, mp_gcs_instance.gcs, timeout_seconds=self.runtime_limit_secs)
            profile["main_search_time"] = time.perf_counter() - ts
            dominance_cr_time = float(getattr(alg, "dominance_convex_restriction_time", 0.0))
            dominance_cr_calls = int(getattr(alg, "dominance_convex_restriction_calls", 0))
            profile["convex_restriction_time"] += (
                alg.convex_restriction_time
                + dominance_cr_time
            )
            profile["convex_restriction_calls"] += (
                alg.convex_restriction_calls
                + dominance_cr_calls
            )
            profile["dominance_check_time"] += alg.dc_time
            profile["dominance_convex_restriction_time"] += dominance_cr_time
            profile["dominance_convex_restriction_calls"] += dominance_cr_calls
            self.last_search_algorithm = alg
            self.last_search_trace = alg.trace
            status = STPlanStatus.from_solution(sol)
            if status == STPlanStatus.FAIL and self.enable_gub_fallback and gub_sol is not None:
                sol = gub_sol
                status = STPlanStatus.GUB_FALLBACK
            return sol, alg.runtime, status
        finally:
            mp_gcs_instance.cleanup()
            stgcs._clean_source_targets()
