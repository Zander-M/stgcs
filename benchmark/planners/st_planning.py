from __future__ import annotations

import math
from typing import Sequence
import zlib

from baselines.common import ShortestPathSolution
from benchmark.planners.mrmp import SearchPlannerSpec
from benchmark.manifests.st_planning import (
    STHeuristicAblationRecord,
    STHeuristicAblationResultEntry,
)


class STPerformanceComparisonConfig:
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

    @classmethod
    def search_spec_for_planner(cls, planner_name: str) -> SearchPlannerSpec:
        try:
            return cls.SEARCH_SPEC_BY_PLANNER[str(planner_name)]
        except KeyError as exc:
            raise ValueError(f"Planner {planner_name!r} is not an ST search comparison planner.") from exc
