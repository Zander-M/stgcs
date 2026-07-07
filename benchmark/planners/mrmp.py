from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Tuple

from stgcs.pbs import DEFAULT_CHILD_EXPANSION_MODE, ChildExpansionMode


@dataclass(frozen=True)
class SearchPlannerSpec:
    heuristic: str
    domination: Tuple[str, ...]
    epsilon: float = 1.0
    exact_astar: bool = False

    @staticmethod
    def epsilon_label(epsilon: float) -> str:
        value = float(epsilon)
        if math.isclose(value, 1.0):
            return ""
        return f",eps={value:g}"

    @property
    def name(self) -> str:
        if self.exact_astar:
            return "Search(LBG+Astar)"
        epsilon_label = self.epsilon_label(self.epsilon)
        if self.heuristic == "Zero":
            return f"Search({'+'.join(self.domination)}{epsilon_label})"
        return f"Search({self.heuristic}+{'+'.join(self.domination)}{epsilon_label})"


@dataclass(frozen=True)
class PBSExpansionRuleSpec:
    key: str
    label: str
    child_expansion_mode: ChildExpansionMode

    def planner_name(self, low_level_spec: SearchPlannerSpec) -> str:
        return f"PBS-{self.label} + {low_level_spec.name}"


class PBSExpansionAblation:
    LOW_LEVEL_SPEC = SearchPlannerSpec("Max", ("GUB", "IPC"), epsilon=1.0)
    DEFAULT_RULE_KEY = "num_conflicts"
    RULES: Tuple[PBSExpansionRuleSpec, ...] = (
        PBSExpansionRuleSpec("lazy", "Lazy", ChildExpansionMode.LAZY),
        PBSExpansionRuleSpec("soc", "SOC", ChildExpansionMode.SOC),
        PBSExpansionRuleSpec("makespan", "Makespan", ChildExpansionMode.MAKESPAN),
        PBSExpansionRuleSpec("num_conflicts", "NumConflicts", ChildExpansionMode.NUM_CONFLICTS),
    )
    DEFAULT_OUTPUT_ROOT = "data/results/mrmp/pbs_node_expansion"
    DEFAULT_MANIFEST_ROOT = "data/instances/mrmp/pbs_node_expansion"

    @classmethod
    def rule_by_key(cls, key: str) -> PBSExpansionRuleSpec:
        for rule in cls.RULES:
            if rule.key == key:
                return rule
        raise KeyError(f"Unknown PBS child-expansion rule {key!r}")

    @classmethod
    def default_rule(cls) -> PBSExpansionRuleSpec:
        return cls.rule_by_key(cls.DEFAULT_RULE_KEY)

    @classmethod
    def planner_name(cls, rule: PBSExpansionRuleSpec) -> str:
        return rule.planner_name(cls.LOW_LEVEL_SPEC)

    @classmethod
    def required_heuristics(cls) -> set[str]:
        return {cls.LOW_LEVEL_SPEC.heuristic}


@dataclass(frozen=True)
class WindowedCoordinationSpec:
    window_span_factor: float
    dynamic_window_adjustment: bool
    child_expansion_mode: ChildExpansionMode = DEFAULT_CHILD_EXPANSION_MODE
    execution_horizon_factor: float = 1.0

    @staticmethod
    def factor_label(window_span_factor: float) -> str:
        return f"{float(window_span_factor):g}"

    @staticmethod
    def factor_key(window_span_factor: float) -> str:
        return WindowedCoordinationSpec.factor_label(window_span_factor).replace(".", "p")

    @property
    def mode_key(self) -> str:
        return "dynamic" if self.dynamic_window_adjustment else "fixed"

    @property
    def key(self) -> str:
        base_key = f"{self.mode_key}_{self.factor_key(self.window_span_factor)}"
        if math.isclose(float(self.execution_horizon_factor), 1.0):
            return base_key
        return f"{base_key}_beta{self.factor_key(self.execution_horizon_factor)}"

    @property
    def label(self) -> str:
        mode_label = "Dynamic" if self.dynamic_window_adjustment else "Fixed"
        label = f"{mode_label} {self.factor_label(self.window_span_factor)}"
        if math.isclose(float(self.execution_horizon_factor), 1.0):
            return label
        return f"{label} beta={self.factor_label(self.execution_horizon_factor)}"

    def planner_name(self, low_level_spec: SearchPlannerSpec) -> str:
        rule = PBSExpansionAblation.default_rule()
        return f"WC-{self.label} + wPBS-{rule.label} + {low_level_spec.name}"


class WindowedCoordinationAblation:
    LOW_LEVEL_SPEC = SearchPlannerSpec("Max", ("GUB", "IPC"), epsilon=10.0)
    DEFAULT_OUTPUT_ROOT = "data/results/mrmp/windowed_coordination"
    DEFAULT_MANIFEST_ROOT = "data/instances/mrmp/windowed_coordination"
    DEFAULT_REFERENCE_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    DOMAINS: Tuple[str, ...] = ("grid2d", "maze", "iris-2d")
    DEFAULT_SPAN_FACTORS: Tuple[float, ...] = (1.0, 2.5, 5.0, 10.0)
    DEFAULT_MODES: Tuple[str, ...] = ("fixed", "dynamic")
    DEFAULT_DYNAMIC_EXECUTION_HORIZON_FACTORS: Tuple[float, ...] = (0.5,)

    @classmethod
    def spec(
        cls,
        mode: str,
        window_span_factor: float,
        execution_horizon_factor: float = 1.0,
    ) -> WindowedCoordinationSpec:
        if mode not in cls.DEFAULT_MODES:
            raise KeyError(f"Unknown windowed coordination mode {mode!r}")
        return WindowedCoordinationSpec(
            window_span_factor=float(window_span_factor),
            dynamic_window_adjustment=mode == "dynamic",
            child_expansion_mode=PBSExpansionAblation.default_rule().child_expansion_mode,
            execution_horizon_factor=float(execution_horizon_factor),
        )

    @classmethod
    def specs(
        cls,
        window_span_factors: Tuple[float, ...] | list[float] | None = None,
        modes: Tuple[str, ...] | list[str] | None = None,
    ) -> Tuple[WindowedCoordinationSpec, ...]:
        factors = cls.DEFAULT_SPAN_FACTORS if window_span_factors is None else tuple(float(v) for v in window_span_factors)
        mode_values = cls.DEFAULT_MODES if modes is None else tuple(modes)
        return tuple(cls.spec(mode, factor) for factor in factors for mode in mode_values)

    @classmethod
    def run_specs(
        cls,
        window_span_factors: Tuple[float, ...] | list[float] | None = None,
        modes: Tuple[str, ...] | list[str] | None = None,
    ) -> Tuple[WindowedCoordinationSpec, ...]:
        factors = cls.DEFAULT_SPAN_FACTORS if window_span_factors is None else tuple(float(v) for v in window_span_factors)
        mode_values = cls.DEFAULT_MODES if modes is None else tuple(modes)
        specs: list[WindowedCoordinationSpec] = []
        for factor in factors:
            for mode in mode_values:
                specs.append(cls.spec(mode, factor))
                if mode == "dynamic":
                    specs.extend(
                        cls.spec(mode, factor, execution_horizon_factor=beta)
                        for beta in cls.DEFAULT_DYNAMIC_EXECUTION_HORIZON_FACTORS
                    )
        return tuple(specs)

    @classmethod
    def planner_name(cls, spec: WindowedCoordinationSpec) -> str:
        return spec.planner_name(cls.LOW_LEVEL_SPEC)

    @classmethod
    def required_heuristics(cls) -> set[str]:
        return {cls.LOW_LEVEL_SPEC.heuristic}


class MRMPPerformanceComparison:
    LOW_LEVEL_SPEC = SearchPlannerSpec("Max", ("GUB", "IPC"), epsilon=10.0)
    DEFAULT_OUTPUT_ROOT = "data/results/mrmp/performance_comparison"
    DEFAULT_MANIFEST_ROOT = "data/instances/mrmp/performance_comparison"
    DOMAINS: Tuple[str, ...] = ("grid2d", "maze", "iris-2d")
    FAMILY = "intersection"
    MIN_NUM_AGENTS = 2
    MAX_NUM_AGENTS = 20
    NUM_AGENTS_STEP = 2
    PBS_KEY = "pbs"
    PP_BFS_KEY = "pp-bfs"
    WINDOWED_PBS_KEY = "wpbs"
    WINDOWED_PP_KEY = "wpp"
    PBS_ZETA_SIPP_KEY = "pbs-zeta-sipp"
    FIXED_PP_ST_RRT_STAR_KEY = "fixed-pp-st-rrt-star"
    FIXED_PP_ST_RRT_STAR_FIRST_KEY = "fixed-pp-st-rrt-star-first"
    FIXED_PP_ST_RRT_STAR_FINAL_KEY = "fixed-pp-st-rrt-star-final"
    FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS: Tuple[str, ...] = (
        FIXED_PP_ST_RRT_STAR_FIRST_KEY,
        FIXED_PP_ST_RRT_STAR_FINAL_KEY,
    )
    CB_GCS_KEY = "cb-gcs"
    KCBS_KEY = "kcbs"
    ALL_PLANNER_KEY = "all"
    SEARCH_BASED_PLANNER_KEYS: Tuple[str, ...] = (PBS_KEY, PP_BFS_KEY, WINDOWED_PBS_KEY, WINDOWED_PP_KEY)
    BASELINE_PLANNER_KEYS: Tuple[str, ...] = (
        *FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS,
        PBS_ZETA_SIPP_KEY,
        CB_GCS_KEY,
        KCBS_KEY,
    )
    PLANNER_KEYS: Tuple[str, ...] = SEARCH_BASED_PLANNER_KEYS + BASELINE_PLANNER_KEYS
    PLANNER_SELECTION_KEYS: Tuple[str, ...] = (
        ALL_PLANNER_KEY,
        FIXED_PP_ST_RRT_STAR_KEY,
        *PLANNER_KEYS,
    )
    WINDOW_SPAN_FACTOR = 5.0
    EXECUTION_HORIZON_FACTOR = 1.0
    DYNAMIC_WINDOW_ADJUSTMENT = True
    CHILD_EXPANSION_MODE = ChildExpansionMode.NUM_CONFLICTS
    CHILD_EXPANSION_LABEL = "NumConflicts"
    CB_GCS_TIME_HORIZON = 25.0
    CB_GCS_TIME_STEP = 0.5
    MICP_ROUNDED_PATH_SCALE = 1000.0
    ZETA_SIPP_CELL_SIZE_MULTIPLIER = 1.0
    KCBS_LOW_LEVEL_SOLVE_TIME = 1.0
    KCBS_PROPAGATION_STEP_SIZE = 0.1
    KCBS_MIN_CONTROL_DURATION = 1
    KCBS_MAX_CONTROL_DURATION = 10
    KCBS_GOAL_TOLERANCE = 0.05
    KCBS_GOAL_BIAS = 0.05
    KCBS_INTERMEDIATE_STATES = True
    KCBS_NUM_THREADS = 4
    KCBS_SEED = 0

    @classmethod
    def num_agent_values(cls) -> Tuple[int, ...]:
        return tuple(range(cls.MIN_NUM_AGENTS, cls.MAX_NUM_AGENTS + 1, cls.NUM_AGENTS_STEP))

    @classmethod
    def traffic_tier(cls, num_agents: int) -> str:
        num_agents = int(num_agents)
        if num_agents not in cls.num_agent_values():
            raise ValueError(
                f"MRMP performance comparison supports {cls.MIN_NUM_AGENTS}, "
                f"{cls.MIN_NUM_AGENTS + cls.NUM_AGENTS_STEP}, ..., {cls.MAX_NUM_AGENTS} robots, "
                f"got {num_agents}."
            )
        return f"{num_agents}-robot"

    @classmethod
    def result_tiers(cls) -> Tuple[str, ...]:
        return tuple(cls.traffic_tier(num_agents) for num_agents in cls.num_agent_values())

    @classmethod
    def num_agents_from_tier(cls, tier: str) -> int:
        suffix = "-robot"
        tier = str(tier)
        if not tier.endswith(suffix):
            raise ValueError(f"MRMP performance comparison tier must end with {suffix!r}, got {tier!r}.")
        num_agents = int(tier[: -len(suffix)])
        cls.traffic_tier(num_agents)
        return num_agents

    @classmethod
    def result_directory(cls, output_root: str | Path, traffic_tier: str, domain_key: str) -> Path:
        tier = cls.traffic_tier(cls.num_agents_from_tier(traffic_tier))
        domain_key = str(domain_key)
        if domain_key not in cls.DOMAINS:
            raise ValueError(f"Unknown MRMP performance-comparison domain {domain_key!r}.")
        return Path(output_root) / tier / domain_key / "results"

    @staticmethod
    def _safe_result_name(planner_name: str) -> str:
        return planner_name.replace("/", "_")

    @classmethod
    def result_path(cls, output_root: str | Path, traffic_tier: str, domain_key: str, planner_key: str) -> Path:
        safe_name = cls._safe_result_name(cls.planner_name(planner_key))
        return cls.result_directory(output_root, traffic_tier, domain_key) / f"{safe_name}.csv"

    @classmethod
    def planner_name(cls, planner_key: str) -> str:
        low_level_name = cls.LOW_LEVEL_SPEC.name
        alpha = cls.WINDOW_SPAN_FACTOR
        beta = cls.EXECUTION_HORIZON_FACTOR
        if planner_key == cls.PBS_KEY:
            return f"PBS-{cls.CHILD_EXPANSION_LABEL} + {low_level_name}"
        if planner_key == cls.PP_BFS_KEY:
            return f"PP(order=index) + {low_level_name}"
        if planner_key == cls.WINDOWED_PBS_KEY:
            return f"Windowed PBS(alpha={alpha:g},beta={beta:g},rule=NC) + {low_level_name}"
        if planner_key == cls.WINDOWED_PP_KEY:
            return f"Windowed PP(alpha={alpha:g},beta={beta:g},order=index) + {low_level_name}"
        if planner_key == cls.FIXED_PP_ST_RRT_STAR_FIRST_KEY:
            return "Fixed PP + OMPL ST-RRT*(first)"
        if planner_key == cls.FIXED_PP_ST_RRT_STAR_FINAL_KEY:
            return "Fixed PP + OMPL ST-RRT*(final)"
        if planner_key == cls.PBS_ZETA_SIPP_KEY:
            return f"PBS-{cls.CHILD_EXPANSION_LABEL} + Zeta*-SIPP"
        if planner_key == cls.CB_GCS_KEY:
            return "CB-GCS(paper T-GCS)"
        if planner_key == cls.KCBS_KEY:
            return "OMPL K-CBS"
        raise KeyError(f"Unknown MRMP performance-comparison planner {planner_key!r}")

    @classmethod
    def planner_label(cls, planner_key: str) -> str:
        return cls.planner_name(planner_key)

    @classmethod
    def expand_planner_key(cls, planner_key: str) -> Tuple[str, ...]:
        if planner_key == cls.FIXED_PP_ST_RRT_STAR_KEY:
            return cls.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS
        return (planner_key,)

    @classmethod
    def required_heuristics(cls) -> set[str]:
        return {cls.LOW_LEVEL_SPEC.heuristic}

    @classmethod
    def micp_rounding_paths(cls, edge_count: int) -> int:
        num_edges = int(edge_count)
        if num_edges <= 1:
            raise ValueError(f"MICP(g) requires |E| > 1 to set rounded paths, got |E|={num_edges}.")
        return int(math.ceil(cls.MICP_ROUNDED_PATH_SCALE * math.log(num_edges)))


HEURISTIC_SPECS: Tuple[SearchPlannerSpec, ...] = (
    SearchPlannerSpec("Zero", ("GUB",)),
    SearchPlannerSpec("SC", ("GUB",)),
    SearchPlannerSpec("LBG", ("GUB",)),
    SearchPlannerSpec("TD", ("GUB",)),
    SearchPlannerSpec("Max", ("GUB",)),
)

SEARCH_GROUPS = {
    "heuristic": HEURISTIC_SPECS,
}
