from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlencode, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/private/tmp/stgcs_mpl")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

from environment.env import Env
from environment.grid import GridEnvironmentBuilder
from environment.iris import Iris2DEnvBuilder
from environment.maze.maze_env import make_random_2d_maze
from environment.uav_village import VillageEnv
from experiments.instance import Simple2DInstance
from visualization.viewer.solution_visualization import SolutionVisualizationService
from stgcs.bfs.domination_check import GlobalUpperBound_DC, Sampling_DC
from stgcs.bfs.heuristics import HeurShortCut
from stgcs.pbs import ChildExpansionMode, PriorityBasedSearch
from stgcs.st_planner import MPQuery, SearchPlanner
from stgcs.trajectory import STTrajectory
from stgcs.mrmp_planner import windowed_pbs
from stgcs.trajopt_postprocessing import GlobalTrajectoryOptimizer, GlobalTrajOptConfig, GlobalTrajOptResult


@dataclass(frozen=True)
class RearrangementStage:
    name: str
    queries: tuple[MPQuery, ...]


@dataclass(frozen=True)
class RearrangementStageAssignment:
    queries: tuple[MPQuery, ...]
    target_indices: tuple[int, ...]
    cost: float


@dataclass(frozen=True)
class RearrangementTask:
    source_path: Path
    name: str
    instance_id: str
    task_type: str
    robot_radius: float
    vlimit: float
    tmax: float
    stage_timeout_secs: float
    search_eps: float
    planner_coordination: str
    window_alpha: float
    window_beta: float
    dynamic_window_adjustment: bool
    child_expansion_rule: str
    child_expansion_mode: ChildExpansionMode
    environment: dict[str, Any]
    cspace: tuple[np.ndarray, ...]
    cspace_edges: tuple[tuple[int, int], ...]
    domain_lb: np.ndarray
    domain_ub: np.ndarray
    stages: tuple[RearrangementStage, ...]

    @property
    def num_agents(self) -> int:
        return len(self.stages[0].queries)

    @property
    def space_dim(self) -> int:
        return int(self.cspace[0].shape[1])

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]


class RobotRearrangementTaskDemo:
    DEMO_KEY = "mrmp"
    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
    DEFAULT_STAGE_TIMEOUT_SECS = 300
    DEFAULT_SEARCH_EPS = 1.0
    DEFAULT_ROBOT_RADIUS = 0.05
    DEFAULT_VLIMIT = 0.2
    DEFAULT_TMAX = 50.0
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8766

    PWL_PLANNER_KEY = "mrmp-pbs"
    PWL_PLANNER_NAME = "PBS + Search(SC+GUB+IPC)"
    TRAJOPT_PLANNER_KEY = f"{PWL_PLANNER_KEY}-global-trajopt"
    TRAJOPT_PLANNER_NAME = f"{PWL_PLANNER_NAME} + global trajopt"
    WINDOWED_PBS_PLANNER_KEY = "mrmp-wpbs"
    PBS_COORDINATION = "pbs"
    WINDOWED_PBS_COORDINATION = "windowed-pbs"
    SUPPORTED_COORDINATIONS = frozenset({PBS_COORDINATION, WINDOWED_PBS_COORDINATION})
    DEFAULT_WINDOW_ALPHA = 5.0
    DEFAULT_WINDOW_BETA = 1.0
    DEFAULT_DYNAMIC_WINDOW_ADJUSTMENT = True
    DEFAULT_CHILD_EXPANSION_RULE = "num_conflicts"
    CHILD_EXPANSION_RULES = {
        "lazy": ChildExpansionMode.LAZY,
        "soc": ChildExpansionMode.SOC,
        "makespan": ChildExpansionMode.MAKESPAN,
        "num_conflicts": ChildExpansionMode.NUM_CONFLICTS,
    }
    CHILD_EXPANSION_RULE_LABELS = {
        "lazy": "Lazy",
        "soc": "SOC",
        "makespan": "Makespan",
        "num_conflicts": "NumConflicts",
    }
    TASK_TYPE = "labeled_mrmp"
    REARRANGEMENT_TASK_TYPE = "rearrangement"
    FORMATION_CONTROL_TASK_TYPE = "formation_control"
    SUPPORTED_TASK_TYPES = frozenset({TASK_TYPE, REARRANGEMENT_TASK_TYPE, FORMATION_CONTROL_TASK_TYPE})
    ASSIGNMENT_STRATEGY = "fixed-query-order"
    ASSIGNMENT_SOLVER = None
    WAITED_SOLUTION_GLOB = "*waited*solution.json"
    WAITED_SOLUTION_PLANNER_SUFFIX = " + waited"

    GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR = 0.25
    GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR = 2.0
    GLOBAL_TRAJOPT_STRIDE_FACTOR = 1.0
    GLOBAL_TRAJOPT_SOLVER_MAX_ITER = 1_000_000
    GLOBAL_TRAJOPT_SOLVER_EPS = 1e-6
    GLOBAL_TRAJOPT_CLEARANCE_MARGIN = 0.0
    GLOBAL_TRAJOPT_COLLISION_TOLERANCE = 1e-4

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Solve a YAML-defined sequence of MRMP stages, "
                "export planning/global-trajopt NPZ files, and write a viewer manifest."
            )
        )
        parser.add_argument("config", type=Path, help="YAML task config.")
        parser.add_argument(
            "phase",
            nargs="?",
            choices=("serve", "planning", "trajopt", "both", "targets"),
            default="serve",
            help=(
                "`targets` plots the initial starts and all stage query targets; "
                "`planning` solves the MRMP stages; "
                "`trajopt` loads a saved planning solution when available and smooths it; "
                "`both` does both."
            ),
        )
        parser.add_argument("--output-dir", type=Path, default=cls.DEFAULT_OUTPUT_DIR)
        parser.add_argument("--viewer-output", type=Path, default=None)
        parser.add_argument("--pwl-output", type=Path, default=None)
        parser.add_argument("--trajopt-output", type=Path, default=None)
        parser.add_argument("--pwl-npz-output", type=Path, default=None)
        parser.add_argument("--trajopt-npz-output", type=Path, default=None)
        parser.add_argument(
            "--write-solution-json",
            action="store_true",
            help=(
                "Also write planning/trajopt solution JSON files. "
                "By default, solution exports are NPZ-only."
            ),
        )
        parser.add_argument(
            "--targets-output",
            type=Path,
            default=None,
            help="Optional PNG/PDF path for the query-target plot.",
        )
        parser.add_argument("--stage-timeout-secs", type=float, default=None)
        parser.add_argument("--search-eps", type=float, default=None)
        parser.add_argument(
            "--coordination",
            choices=tuple(sorted(cls.SUPPORTED_COORDINATIONS)),
            default=None,
        )
        parser.add_argument("--window-alpha", type=float, default=None)
        parser.add_argument("--window-beta", type=float, default=None)
        parser.add_argument(
            "--dynamic-window-adjustment",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            "--child-expansion-rule",
            choices=tuple(cls.CHILD_EXPANSION_RULES.keys()),
            default=None,
        )
        parser.add_argument("--trajopt-progress", action="store_true")
        parser.add_argument("--host", type=str, default=cls.DEFAULT_HOST)
        parser.add_argument("--port", type=int, default=cls.DEFAULT_PORT)
        parser.add_argument(
            "--no-serve",
            action="store_true",
            help="Only write the viewer manifest instead of serving the interactive viewer.",
        )
        parser.add_argument(
            "--no-show-targets",
            action="store_true",
            help="Write --targets-output without opening the Matplotlib window.",
        )
        return parser.parse_args()

    @classmethod
    def resolved_paths(cls, args: argparse.Namespace, task: RearrangementTask) -> dict[str, Path]:
        output_dir = Path(args.output_dir) / cls.slug(task.source_path.stem)
        return {
            "viewer_json": (
                Path(args.viewer_output)
                if args.viewer_output is not None
                else output_dir / "viewer_manifest.json"
            ),
            "pwl_json": Path(args.pwl_output) if args.pwl_output is not None else output_dir / "pwl_solution.json",
            "trajopt_json": (
                Path(args.trajopt_output)
                if args.trajopt_output is not None
                else output_dir / "trajopt_solution.json"
            ),
            "pwl_npz": (
                Path(args.pwl_npz_output)
                if args.pwl_npz_output is not None
                else output_dir / "pwl_solution.npz"
            ),
            "trajopt_npz": (
                Path(args.trajopt_npz_output)
                if args.trajopt_npz_output is not None
                else output_dir / "trajopt_solution.npz"
            ),
        }

    @classmethod
    def solver_overrides_from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for key in (
            "stage_timeout_secs",
            "search_eps",
            "coordination",
            "window_alpha",
            "window_beta",
            "dynamic_window_adjustment",
            "child_expansion_rule",
        ):
            value = getattr(args, key, None)
            if value is not None:
                overrides[key] = value
        return overrides

    @staticmethod
    def write_solution_json_from_args(args: argparse.Namespace) -> bool:
        return bool(
            getattr(args, "write_solution_json", False)
            or getattr(args, "pwl_output", None) is not None
            or getattr(args, "trajopt_output", None) is not None
        )

    @classmethod
    def load_task(
        cls,
        path: Path,
        stage_timeout_override: float | None = None,
        solver_overrides: dict[str, Any] | None = None,
    ) -> RearrangementTask:
        overrides = dict(solver_overrides or {})
        if stage_timeout_override is not None:
            overrides["stage_timeout_secs"] = stage_timeout_override

        source_path = Path(path).resolve()
        payload = cls.load_yaml_file(source_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Task config {source_path} must contain a YAML mapping.")

        name = str(payload.get("name") or source_path.stem)
        instance_id = str(
            payload.get("instance_id") or f"robot-rearrange-task-{cls.slug(name)}"
        )
        task_type = str(payload.get("task_type", cls.TASK_TYPE))
        if task_type not in cls.SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"{cls.DEMO_KEY} requires task_type in {sorted(cls.SUPPORTED_TASK_TYPES)!r}; "
                f"got {task_type!r}."
            )
        robot_radius = cls.positive_float(
            payload.get("robot_radius", cls.DEFAULT_ROBOT_RADIUS),
            "robot_radius",
        )
        vlimit = cls.positive_float(payload.get("vlimit", cls.DEFAULT_VLIMIT), "vlimit")
        tmax = cls.positive_float(payload.get("tmax", cls.DEFAULT_TMAX), "tmax")
        config_timeout = cls.positive_float(
            overrides.get(
                "stage_timeout_secs",
                payload.get("stage_timeout_secs", cls.DEFAULT_STAGE_TIMEOUT_SECS),
            ),
            "stage_timeout_secs",
        )
        stage_timeout_secs = config_timeout
        search_eps = cls.positive_float(
            overrides.get("search_eps", payload.get("search_eps", cls.DEFAULT_SEARCH_EPS)),
            "search_eps",
        )
        planner_config = cls.parse_planner_config(payload, overrides=overrides)
        environment_config = cls.parse_environment_config(payload)
        env = cls.build_env_from_config(name, environment_config, robot_radius=float(robot_radius))
        cspace = tuple(np.asarray(poly, dtype=float) for poly in env.C_Space)
        stages = cls.parse_stages(payload, vlimit=vlimit, space_dim=int(env.dim))
        task = RearrangementTask(
            source_path=source_path,
            name=name,
            instance_id=instance_id,
            task_type=task_type,
            robot_radius=robot_radius,
            vlimit=vlimit,
            tmax=tmax,
            stage_timeout_secs=stage_timeout_secs,
            search_eps=search_eps,
            planner_coordination=planner_config["coordination"],
            window_alpha=planner_config["window_alpha"],
            window_beta=planner_config["window_beta"],
            dynamic_window_adjustment=planner_config["dynamic_window_adjustment"],
            child_expansion_rule=planner_config["child_expansion_rule"],
            child_expansion_mode=planner_config["child_expansion_mode"],
            environment=environment_config,
            cspace=cspace,
            cspace_edges=tuple((int(i), int(j)) for i, j in env.edges),
            domain_lb=np.asarray(env.lb, dtype=float),
            domain_ub=np.asarray(env.ub, dtype=float),
            stages=tuple(stages),
        )
        cls.validate_task(task)
        return task

    @staticmethod
    def load_yaml_file(path: Path) -> Any:
        try:
            from pydrake.common.yaml import yaml_load_file

            return yaml_load_file(filename=str(path))
        except ModuleNotFoundError:
            pass
        except ImportError:
            pass

        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YAML parsing requires either pydrake.common.yaml or PyYAML."
            ) from exc
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def parse_planner_config(
        cls,
        payload: dict[str, Any],
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        overrides = dict(overrides or {})
        raw_planner = payload.get("planner", {})
        if raw_planner is None:
            raw_planner = {}
        if not isinstance(raw_planner, dict):
            raise ValueError("planner must be a mapping when provided.")

        coordination = str(
            overrides.get(
                "coordination",
                raw_planner.get("coordination", cls.PBS_COORDINATION),
            )
        )
        if coordination not in cls.SUPPORTED_COORDINATIONS:
            raise ValueError(
                f"planner.coordination must be one of {sorted(cls.SUPPORTED_COORDINATIONS)!r}; "
                f"got {coordination!r}."
            )
        child_expansion_rule = str(
            overrides.get(
                "child_expansion_rule",
                raw_planner.get(
                    "child_expansion_rule",
                    cls.DEFAULT_CHILD_EXPANSION_RULE,
                ),
            )
        )
        return {
            "coordination": coordination,
            "window_alpha": cls.positive_float(
                overrides.get(
                    "window_alpha",
                    raw_planner.get("window_alpha", cls.DEFAULT_WINDOW_ALPHA),
                ),
                "planner.window_alpha",
            ),
            "window_beta": cls.unit_interval_float(
                overrides.get(
                    "window_beta",
                    raw_planner.get("window_beta", cls.DEFAULT_WINDOW_BETA),
                ),
                "planner.window_beta",
            ),
            "dynamic_window_adjustment": cls.bool_value(
                overrides.get(
                    "dynamic_window_adjustment",
                    raw_planner.get(
                        "dynamic_window_adjustment",
                        cls.DEFAULT_DYNAMIC_WINDOW_ADJUSTMENT,
                    ),
                ),
                "planner.dynamic_window_adjustment",
            ),
            "child_expansion_rule": child_expansion_rule,
            "child_expansion_mode": cls.child_expansion_mode(child_expansion_rule),
        }

    @classmethod
    def child_expansion_mode(cls, rule: str) -> ChildExpansionMode:
        try:
            return cls.CHILD_EXPANSION_RULES[str(rule)]
        except KeyError as exc:
            raise ValueError(
                f"planner.child_expansion_rule must be one of {sorted(cls.CHILD_EXPANSION_RULES)!r}; "
                f"got {rule!r}."
            ) from exc

    @classmethod
    def child_expansion_rule_label(cls, rule: str) -> str:
        return cls.CHILD_EXPANSION_RULE_LABELS[str(rule)]

    @classmethod
    def parse_environment_config(cls, payload: dict[str, Any]) -> dict[str, Any]:
        raw_environment = payload.get("environment", payload.get("env"))
        if raw_environment is None:
            if "cspace" in payload:
                raw_environment = {"type": "cspace", "cspace": payload["cspace"]}
            elif "bounds" in payload:
                raw_environment = {"type": "bounds", "bounds": payload["bounds"]}
            else:
                raw_environment = {"type": "bounds", "bounds": [[0.0, 0.0], [1.0, 1.0]]}
        elif isinstance(raw_environment, str):
            raw_environment = {"type": raw_environment}
        if not isinstance(raw_environment, dict):
            raise ValueError("environment must be a mapping or a string environment type.")

        config = dict(raw_environment)
        raw_params = config.pop("params", None)
        if raw_params is not None:
            if not isinstance(raw_params, dict):
                raise ValueError("environment.params must be a mapping when provided.")
            merged = dict(raw_params)
            merged.update(config)
            config = merged
        config["type"] = cls.normalized_environment_type(str(config.get("type", "bounds")))
        return config

    @staticmethod
    def normalized_environment_type(value: str) -> str:
        key = value.strip().lower().replace("_", "-")
        aliases = {
            "box": "bounds",
            "rectangle": "bounds",
            "polygon": "cspace",
            "polygons": "cspace",
            "yaml-cspace": "cspace",
            "grid": "grid2d",
            "grid-2d": "grid2d",
            "grid-3d": "grid3d",
            "iris2d": "iris-2d",
            "iris": "iris-2d",
            "simple": "simple2d",
            "empty-square": "empty-square2d",
            "uav-village3d": "uav-village",
            "village3d": "uav-village",
            "village-3d": "uav-village",
        }
        key = aliases.get(key, key)
        supported = {
            "bounds",
            "cspace",
            "empty-square2d",
            "simple2d",
            "grid2d",
            "grid3d",
            "maze",
            "iris-2d",
            "uav-village",
        }
        if key not in supported:
            raise ValueError(f"Unsupported environment type {value!r}; expected one of {sorted(supported)!r}.")
        return key

    @classmethod
    def build_env_from_config(cls, name: str, config: dict[str, Any], robot_radius: float) -> Env:
        kind = cls.normalized_environment_type(str(config.get("type", "bounds")))
        seed = int(config.get("seed", 0))
        if kind == "cspace":
            cspace = cls.parse_cspace({"cspace": config.get("cspace")})
            return Env(str(name), cspace, robot_radius=float(robot_radius))
        if kind == "bounds":
            cspace = cls.cspace_from_bounds(config.get("bounds", [[0.0, 0.0], [1.0, 1.0]]))
            return Env(str(name), cspace, robot_radius=float(robot_radius))
        if kind == "empty-square2d":
            square_size = cls.positive_float(config.get("square_size", 10.0), "environment.square_size")
            cspace = cls.cspace_from_bounds([[0.0, 0.0], [square_size, square_size]])
            return Env(str(name), cspace, robot_radius=float(robot_radius))
        if kind == "simple2d":
            env = Simple2DInstance.build_base_env(seed=seed, robot_radius=float(robot_radius))
        elif kind == "grid2d":
            n = int(config.get("N", config.get("n", 8)))
            m = int(config.get("M", config.get("m", n)))
            num_pts_per_ply = int(config.get("num_pts_per_ply", 8))
            env = GridEnvironmentBuilder.random_2d(seed, n, m, num_pts_per_ply=num_pts_per_ply)
            env.robot_radius = float(robot_radius)
        elif kind == "grid3d":
            n = int(config.get("N", config.get("n", 4)))
            m = int(config.get("M", config.get("m", n)))
            num_pts_per_ply = int(config.get("num_pts_per_ply", 16))
            env = GridEnvironmentBuilder.random_3d(seed, n, m, num_pts_per_ply=num_pts_per_ply)
            env.robot_radius = float(robot_radius)
        elif kind == "maze":
            width = int(config.get("width", 8))
            height = int(config.get("height", width))
            env = make_random_2d_maze(width, height, seed, robot_radius=float(robot_radius))
        elif kind == "iris-2d":
            env_params = dict(config)
            env_params.pop("type", None)
            env_params.pop("seed", None)
            env_params.setdefault("robot_radius", float(robot_radius))
            env = Iris2DEnvBuilder.build_env(seed=seed, env_params=env_params)
            env.robot_radius = float(robot_radius)
        elif kind == "uav-village":
            village_side = int(config.get("village_side", config.get("side", 8)))
            village_height = cls.positive_float(config.get("village_height", config.get("height", 3.0)), "environment.village_height")
            building_every = int(config.get("building_every", 3))
            decoration_density = float(config.get("decoration_density", 0.35))
            building_clearance_margin = float(config.get("building_clearance_margin", 1e-3))
            safe_boxes, _, _ = VillageEnv.build_village_geometry(
                village_side=village_side,
                village_height=village_height,
                building_every=building_every,
                decoration_density=decoration_density,
                robot_radius=float(robot_radius),
                building_clearance_margin=building_clearance_margin,
                seed=seed,
            )
            env = Env(
                str(name),
                [np.asarray(box, dtype=float) for box in safe_boxes],
                robot_radius=float(robot_radius),
                domain_lb=np.zeros(3, dtype=float),
                domain_ub=np.asarray([village_side, village_side, village_height], dtype=float),
            )
        else:
            raise AssertionError(f"Unhandled environment type {kind!r}.")
        env.name = str(name)
        return env

    @classmethod
    def parse_cspace(cls, payload: dict[str, Any]) -> list[np.ndarray]:
        if "cspace" in payload:
            raw_cspace = payload["cspace"]
            if not isinstance(raw_cspace, list) or not raw_cspace:
                raise ValueError("cspace must be a non-empty list of convex polygon vertex lists.")
            cspace = [cls.point_matrix(raw_poly, f"cspace[{idx}]") for idx, raw_poly in enumerate(raw_cspace)]
        else:
            cspace = cls.cspace_from_bounds(payload.get("bounds", [[0.0, 0.0], [1.0, 1.0]]))
        dim = int(cspace[0].shape[1])
        if any(poly.shape[1] != dim for poly in cspace):
            raise ValueError("All cspace polytopes must have the same dimension.")
        if any(poly.shape[0] < dim + 1 for poly in cspace):
            raise ValueError("Each cspace polytope must contain at least dim + 1 vertices.")
        return cspace

    @classmethod
    def cspace_from_bounds(cls, raw_bounds: Any) -> list[np.ndarray]:
        bounds = cls.point_matrix(raw_bounds, "bounds")
        if bounds.shape[0] != 2:
            raise ValueError("bounds must be [lower, upper].")
        lb, ub = bounds
        if np.any(ub <= lb):
            raise ValueError("bounds upper corner must be strictly greater than lower corner.")
        if bounds.shape[1] == 2:
            return [
                np.asarray(
                    [
                        [lb[0], lb[1]],
                        [ub[0], lb[1]],
                        [ub[0], ub[1]],
                        [lb[0], ub[1]],
                    ],
                    dtype=float,
                )
            ]
        vertices = np.asarray(
            [
                [ub[axis] if bit else lb[axis] for axis, bit in enumerate(bits)]
                for bits in product((0, 1), repeat=bounds.shape[1])
            ],
            dtype=float,
        )
        return [vertices]

    @classmethod
    def parse_stages(cls, payload: dict[str, Any], vlimit: float, space_dim: int) -> list[RearrangementStage]:
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Task config must contain a non-empty stages list.")

        stages: list[RearrangementStage] = []
        used_names: set[str] = set()
        for stage_idx, raw_stage in enumerate(raw_stages):
            if not isinstance(raw_stage, dict):
                raise ValueError(f"stages[{stage_idx}] must be a mapping.")
            stage_name = cls.slug(str(raw_stage.get("name") or f"stage_{stage_idx + 1:03d}"))
            if stage_name in used_names:
                raise ValueError(f"Duplicate stage name {stage_name!r}.")
            used_names.add(stage_name)
            raw_queries = raw_stage.get("queries")
            if not isinstance(raw_queries, list) or not raw_queries:
                raise ValueError(f"stages[{stage_idx}].queries must be a non-empty list.")
            queries = tuple(
                cls.parse_query(raw_query, vlimit=vlimit, space_dim=space_dim, name=f"{stage_name}.queries[{query_idx}]")
                for query_idx, raw_query in enumerate(raw_queries)
            )
            stages.append(RearrangementStage(stage_name, queries))
        return stages

    @classmethod
    def parse_query(cls, raw_query: Any, vlimit: float, space_dim: int, name: str) -> MPQuery:
        if not isinstance(raw_query, dict):
            raise ValueError(f"{name} must be a mapping.")
        start = cls.point_vector(raw_query.get("start"), f"{name}.start", space_dim)
        goal = cls.point_vector(raw_query.get("goal"), f"{name}.goal", space_dim)
        query_vlimit = cls.positive_float(raw_query.get("vlimit", vlimit), f"{name}.vlimit")
        if not math.isclose(query_vlimit, float(vlimit), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{name}.vlimit={query_vlimit:g} does not match task vlimit={float(vlimit):g}."
            )
        t_start = cls.nonnegative_float(raw_query.get("t_start", 0.0), f"{name}.t_start")
        return MPQuery(
            start=start,
            goal=goal,
            t_start=t_start,
            is_stay=bool(raw_query.get("is_stay", True)),
            vlimit=float(vlimit),
        )

    @staticmethod
    def point_matrix(raw_points: Any, name: str) -> np.ndarray:
        points = np.asarray(raw_points, dtype=float)
        if points.ndim != 2 or points.shape[1] == 0:
            raise ValueError(f"{name} must be a 2D numeric array of points.")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{name} contains non-finite coordinates.")
        return points

    @staticmethod
    def point_vector(raw_point: Any, name: str, space_dim: int) -> np.ndarray:
        point = np.asarray(raw_point, dtype=float).reshape(-1)
        if point.shape != (space_dim,):
            raise ValueError(f"{name} must contain {space_dim} coordinates.")
        if not np.all(np.isfinite(point)):
            raise ValueError(f"{name} contains non-finite coordinates.")
        return point

    @staticmethod
    def positive_float(value: Any, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return numeric

    @staticmethod
    def nonnegative_float(value: Any, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        return numeric

    @staticmethod
    def unit_interval_float(value: Any, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return numeric

    @staticmethod
    def bool_value(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        raise ValueError(f"{name} must be a boolean.")

    @staticmethod
    def slug(value: str) -> str:
        chars = []
        for char in value.strip().lower():
            if char.isalnum():
                chars.append(char)
            elif char in {"-", "_", "."}:
                chars.append(char)
            elif char.isspace():
                chars.append("-")
        slug = "".join(chars).strip("-_.")
        return slug or "task"

    @classmethod
    def assignment_strategy_for_task(cls, task: RearrangementTask) -> str:
        return cls.ASSIGNMENT_STRATEGY

    @classmethod
    def assignment_solver_for_task(cls, task: RearrangementTask) -> str | None:
        return cls.ASSIGNMENT_SOLVER

    @classmethod
    def validate_task(cls, task: RearrangementTask) -> None:
        if task.num_agents <= 0:
            raise ValueError("Task must contain at least one robot query.")
        for stage in task.stages:
            if len(stage.queries) != task.num_agents:
                raise ValueError(
                    f"Stage {stage.name!r} contains {len(stage.queries)} queries; "
                    f"expected {task.num_agents}."
                )
        expected_starts = [np.asarray(query.start, dtype=float) for query in task.stages[0].queries]
        for stage in task.stages:
            for agent_idx, query in enumerate(stage.queries):
                if not np.allclose(query.start, expected_starts[agent_idx], rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"Stage chain is not contiguous for agent {agent_idx} at stage {stage.name!r}."
                    )
            expected_starts = [np.asarray(query.goal, dtype=float) for query in stage.queries]

    @classmethod
    def initial_positions(cls, task: RearrangementTask) -> list[np.ndarray]:
        return [np.asarray(query.start, dtype=float) for query in task.stages[0].queries]

    @classmethod
    def assign_stage_queries(
        cls,
        task: RearrangementTask,
        current_positions: Sequence[np.ndarray],
        stage: RearrangementStage,
    ) -> RearrangementStageAssignment:
        if len(current_positions) != len(stage.queries):
            raise ValueError(
                f"Stage assignment requires {len(stage.queries)} current positions; "
                f"got {len(current_positions)}."
            )
        assigned_queries = []
        assignment_cost = 0.0
        for agent_idx, (current_position, stage_query) in enumerate(zip(current_positions, stage.queries)):
            current = np.asarray(current_position, dtype=float)
            if not np.allclose(current, stage_query.start, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"Stage {stage.name!r} query {agent_idx} starts at "
                    f"{stage_query.start.tolist()}, expected current position {current.tolist()}."
                )
            assigned_queries.append(
                MPQuery(
                    start=current,
                    goal=np.asarray(stage_query.goal, dtype=float),
                    t_start=float(stage_query.t_start),
                    is_stay=bool(stage_query.is_stay),
                    vlimit=float(task.vlimit),
                )
            )
            assignment_cost += float(np.linalg.norm(stage_query.goal - current))
        return RearrangementStageAssignment(
            queries=tuple(assigned_queries),
            target_indices=tuple(range(task.num_agents)),
            cost=float(assignment_cost),
        )

    @classmethod
    def build_stage_assignments(cls, task: RearrangementTask) -> dict[str, RearrangementStageAssignment]:
        assignments: dict[str, RearrangementStageAssignment] = {}
        current_positions = cls.initial_positions(task)
        for stage in task.stages:
            assignment = cls.assign_stage_queries(task, current_positions, stage)
            assignments[stage.name] = assignment
            current_positions = [np.asarray(query.goal, dtype=float) for query in assignment.queries]
        return assignments

    @classmethod
    def build_env(cls, task: RearrangementTask) -> Env:
        return Env(
            name=task.name,
            CSpace=[np.asarray(poly, dtype=float) for poly in task.cspace],
            robot_radius=float(task.robot_radius),
            edges=[tuple(edge) for edge in task.cspace_edges],
            domain_lb=np.asarray(task.domain_lb, dtype=float),
            domain_ub=np.asarray(task.domain_ub, dtype=float),
        )

    @classmethod
    def build_stgcs(cls, task: RearrangementTask, env: Env):
        stgcs = env.build_STGCS(t0=0.0, tmax=float(task.tmax), vlimit=float(task.vlimit))
        stgcs.make_leaves_roots()
        return stgcs

    @classmethod
    def window_span(cls, task: RearrangementTask) -> float:
        return float(task.window_alpha) * float(task.robot_radius) / float(task.vlimit)

    @classmethod
    def pwl_planner_key(cls, task: RearrangementTask) -> str:
        if task.planner_coordination == cls.PBS_COORDINATION:
            return cls.PWL_PLANNER_KEY
        if task.planner_coordination == cls.WINDOWED_PBS_COORDINATION:
            return cls.WINDOWED_PBS_PLANNER_KEY
        raise ValueError(f"Unsupported planner coordination {task.planner_coordination!r}.")

    @classmethod
    def pwl_planner_name(cls, task: RearrangementTask) -> str:
        if task.planner_coordination == cls.PBS_COORDINATION:
            return cls.PWL_PLANNER_NAME
        if task.planner_coordination == cls.WINDOWED_PBS_COORDINATION:
            rule_label = cls.child_expansion_rule_label(task.child_expansion_rule)
            return f"wPBS-{rule_label} + Search(SC+GUB+IPC)"
        raise ValueError(f"Unsupported planner coordination {task.planner_coordination!r}.")

    @classmethod
    def trajopt_planner_key(cls, task: RearrangementTask) -> str:
        return f"{cls.pwl_planner_key(task)}-global-trajopt"

    @classmethod
    def trajopt_planner_name(cls, task: RearrangementTask) -> str:
        return f"{cls.pwl_planner_name(task)} + global trajopt"

    @classmethod
    def planner_key_for_phase(cls, task: RearrangementTask, phase: str) -> str:
        if phase == "mrmp":
            return cls.pwl_planner_key(task)
        if phase == "trajopt":
            return cls.trajopt_planner_key(task)
        raise ValueError(f"Unsupported solution phase {phase!r}.")

    @classmethod
    def planner_name_for_phase(cls, task: RearrangementTask, phase: str) -> str:
        if phase == "mrmp":
            return cls.pwl_planner_name(task)
        if phase == "trajopt":
            return cls.trajopt_planner_name(task)
        raise ValueError(f"Unsupported solution phase {phase!r}.")

    @classmethod
    def planner_metadata(cls, task: RearrangementTask) -> dict[str, Any]:
        metadata = {
            "coordination": task.planner_coordination,
            "child_expansion_rule": task.child_expansion_rule,
            "child_expansion_mode": task.child_expansion_mode.name,
            "low_level_heuristic": "SC",
            "domination_checks": ["GUB", "IPC"],
            "epsilon": float(task.search_eps),
            "vlimit": float(task.vlimit),
            "tmax": float(task.tmax),
            "stage_timeout_secs": float(task.stage_timeout_secs),
        }
        if task.planner_coordination == cls.WINDOWED_PBS_COORDINATION:
            metadata.update(
                {
                    "window_alpha": float(task.window_alpha),
                    "window_span": cls.window_span(task),
                    "window_beta": float(task.window_beta),
                    "execution_horizon_factor": float(task.window_beta),
                    "dynamic_window_adjustment": bool(task.dynamic_window_adjustment),
                }
            )
        return metadata

    @classmethod
    def low_level_planner_for_stgcs(cls, task: RearrangementTask, stgcs: Any) -> SearchPlanner:
        return SearchPlanner(
            heur=HeurShortCut(stgcs),
            dc_list=[
                GlobalUpperBound_DC(float("inf"), 0.0, float(task.search_eps)),
                Sampling_DC(),
            ],
            eps=float(task.search_eps),
            runtime_limit_secs=float(task.stage_timeout_secs),
        )

    @classmethod
    def plan_stage(
        cls,
        task: RearrangementTask,
        env: Env,
        stgcs: Any,
        st_planner: SearchPlanner,
        stage_name: str,
        queries: Sequence[MPQuery],
    ) -> tuple[list[STTrajectory], float, float]:
        if task.planner_coordination == cls.WINDOWED_PBS_COORDINATION:
            return cls.plan_stage_with_windowed_pbs(task, stgcs, st_planner, stage_name, queries)
        if task.planner_coordination != cls.PBS_COORDINATION:
            raise ValueError(f"Unsupported planner coordination {task.planner_coordination!r}.")
        pbs = PriorityBasedSearch(
            stgcs,
            st_planner,
            float(env.robot_radius),
            child_expansion_mode=task.child_expansion_mode,
        )
        sol, runtime, success = pbs.run(
            list(queries),
            timeout_secs=float(task.stage_timeout_secs),
            verbose=False,
        )
        if not success or sol is None:
            raise RuntimeError(f"Failed to find a solution for stage {stage_name!r}.")
        stage_end_time = max(float(trajectory.xT[-1]) for trajectory in sol)
        padded = [cls.pad_trajectory_to_time(trajectory, stage_end_time) for trajectory in sol]
        return padded, stage_end_time, float(runtime)

    @classmethod
    def plan_stage_with_windowed_pbs(
        cls,
        task: RearrangementTask,
        stgcs: Any,
        st_planner: SearchPlanner,
        stage_name: str,
        queries: Sequence[MPQuery],
    ) -> tuple[list[STTrajectory], float, float]:
        sol, result = windowed_pbs(
            stgcs,
            st_planner,
            queries,
            float(task.robot_radius),
            window_span=cls.window_span(task),
            execution_horizon_factor=float(task.window_beta),
            timeout_secs=float(task.stage_timeout_secs),
            dynamic_window_adjustment=bool(task.dynamic_window_adjustment),
            child_expansion_mode=task.child_expansion_mode,
            wpbs_verbose=False,
        )
        if not result.success or sol is None:
            raise RuntimeError(f"Failed to find a solution for stage {stage_name!r} with windowed PBS.")
        stage_end_time = max(float(trajectory.xT[-1]) for trajectory in sol)
        padded = [cls.pad_trajectory_to_time(trajectory, stage_end_time) for trajectory in sol]
        return padded, stage_end_time, float(result.runtime[0])

    @classmethod
    def solve_pwl(cls, task: RearrangementTask) -> dict[str, Any]:
        env = cls.build_env(task)
        stgcs = cls.build_stgcs(task, env)
        st_planner = cls.low_level_planner_for_stgcs(task, stgcs)
        chunks_by_agent: list[list[STTrajectory]] = [[] for _ in range(task.num_agents)]
        stage_trajectories: dict[str, list[STTrajectory]] = {}
        stage_queries: dict[str, tuple[MPQuery, ...]] = {}
        stage_assignment_target_indices: dict[str, tuple[int, ...]] = {}
        stage_results: list[dict[str, Any]] = []
        current_positions = cls.initial_positions(task)
        stage_offset = 0.0
        started = time.perf_counter()

        for stage in task.stages:
            print(f"Planning {stage.name} with {task.num_agents} robots...")
            stage_started = time.perf_counter()
            assignment = cls.assign_stage_queries(task, current_positions, stage)
            trajectories, stage_end_time, runtime = cls.plan_stage(
                task,
                env,
                stgcs,
                st_planner,
                stage.name,
                assignment.queries,
            )
            stage_trajectories[stage.name] = trajectories
            stage_queries[stage.name] = assignment.queries
            stage_assignment_target_indices[stage.name] = assignment.target_indices
            for agent_idx, trajectory in enumerate(trajectories):
                chunks_by_agent[agent_idx].append(cls.shift_trajectory_time(trajectory, stage_offset))
            current_positions = [np.asarray(query.goal, dtype=float) for query in assignment.queries]
            stage_offset += stage_end_time
            stage_results.append(
                {
                    "stage": stage.name,
                    "status": "SUCCESS",
                    "is_success": True,
                    "runtime": cls.finite_float_or_none(runtime),
                    "wall_time": cls.finite_float_or_none(time.perf_counter() - stage_started),
                    "makespan": cls.finite_float_or_none(stage_end_time),
                    "assignment_goal_indices": list(assignment.target_indices),
                    "assignment_cost": cls.finite_float_or_none(assignment.cost),
                }
            )
            print(f"{stage.name} solved in {runtime:.3f}s with stage makespan {stage_end_time:.3f}s.")

        full_trajectories = [STTrajectory.from_chunks(chunks) for chunks in chunks_by_agent]
        return cls.solution_payload(
            task=task,
            phase="mrmp",
            stage_trajectories=stage_trajectories,
            stage_queries=stage_queries,
            stage_assignment_target_indices=stage_assignment_target_indices,
            full_trajectories=full_trajectories,
            stage_results=stage_results,
            runtime=time.perf_counter() - started,
        )

    @classmethod
    def solve_trajopt_from_payload(
        cls,
        task: RearrangementTask,
        pwl_payload: dict[str, Any],
        progress: bool,
    ) -> dict[str, Any]:
        cls.validate_solution_payload(task, pwl_payload, expected_phase="mrmp")
        env = cls.build_env(task)
        stgcs = cls.build_stgcs(task, env)
        stage_trajectories = cls.stage_trajectories_from_payload(task, pwl_payload)
        stage_queries, stage_assignment_target_indices = cls.stage_queries_from_payload(task, pwl_payload)
        optimized_stages: dict[str, list[STTrajectory]] = {}
        stage_results: list[dict[str, Any]] = []
        chunks_by_agent: list[list[STTrajectory]] = [[] for _ in range(task.num_agents)]
        stage_offset = 0.0
        started = time.perf_counter()

        for stage in task.stages:
            print(f"Optimizing {stage.name} from stored MRMP trajectories...")
            stage_started = time.perf_counter()
            queries = stage_queries[stage.name]
            cls.validate_stage_trajectories(stage.name, queries, stage_trajectories[stage.name])
            config = cls.global_trajopt_config(task, progress)
            stored_pairwise_ok = cls.pairwise_collision_free(
                stgcs,
                stage_trajectories[stage.name],
                goal_stays=[query.is_stay for query in queries],
                robot_radius=float(task.robot_radius),
            )
            if not stored_pairwise_ok:
                raise RuntimeError(f"Stored MRMP trajectories for {stage.name!r} are not pairwise collision-free.")
            if cls.is_return_to_start_stage(task, stage):
                optimized = [trajectory.copy() for trajectory in stage_trajectories[stage.name]]
                optimized_stages[stage.name] = optimized
                stage_end_time = max(float(trajectory.xT[-1]) for trajectory in optimized)
                for agent_idx, trajectory in enumerate(optimized):
                    chunks_by_agent[agent_idx].append(cls.shift_trajectory_time(trajectory, stage_offset))
                stage_offset += stage_end_time
                stage_results.append(
                    cls.pwl_passthrough_stage_result_payload(
                        stage.name,
                        optimized,
                        config,
                        continuous_pairwise_collision_free=stored_pairwise_ok,
                        wall_time=time.perf_counter() - stage_started,
                    )
                )
                print(f"{stage.name} kept as PWL in trajopt export with stage makespan {stage_end_time:.3f}s.")
                continue
            optimized, result, config, continuous_pairwise_ok = cls.optimize_stage(
                task,
                stgcs,
                stage_trajectories[stage.name],
                queries,
                progress,
            )
            optimized_stages[stage.name] = optimized
            stage_end_time = max(float(trajectory.xT[-1]) for trajectory in optimized)
            for agent_idx, trajectory in enumerate(optimized):
                chunks_by_agent[agent_idx].append(cls.shift_trajectory_time(trajectory, stage_offset))
            stage_offset += stage_end_time
            stage_results.append(
                cls.trajopt_stage_result_payload(
                    stage.name,
                    result,
                    config,
                    stage_end_time,
                    continuous_pairwise_ok,
                    wall_time=time.perf_counter() - stage_started,
                )
            )
            print(f"{stage.name} optimized in {result.runtime:.3f}s with stage makespan {stage_end_time:.3f}s.")

        full_trajectories = [STTrajectory.from_chunks(chunks) for chunks in chunks_by_agent]
        return cls.solution_payload(
            task=task,
            phase="trajopt",
            stage_trajectories=optimized_stages,
            stage_queries=stage_queries,
            stage_assignment_target_indices=stage_assignment_target_indices,
            full_trajectories=full_trajectories,
            stage_results=stage_results,
            runtime=time.perf_counter() - started,
            source_pwl_solution=pwl_payload.get("output_path"),
        )

    @classmethod
    def optimize_stage(
        cls,
        task: RearrangementTask,
        stgcs: Any,
        trajectories: Sequence[STTrajectory],
        queries: Sequence[MPQuery],
        progress: bool,
    ) -> tuple[list[STTrajectory], GlobalTrajOptResult, GlobalTrajOptConfig, bool]:
        config = cls.global_trajopt_config(task, progress)
        result = GlobalTrajectoryOptimizer.optimize_sliding_window(
            trajectories,
            float(task.robot_radius),
            float(task.vlimit),
            config=config,
            window_span=cls.global_trajopt_window_span(task),
            stride=cls.global_trajopt_stride(task),
            stgcs=stgcs,
            env=None,
        )
        if not result.sampled_pairwise_collision_free:
            raise RuntimeError("Global trajectory optimization produced a sampled pairwise collision.")
        continuous_pairwise_ok = cls.pairwise_collision_free(
            stgcs,
            result.trajectories,
            goal_stays=[query.is_stay for query in queries],
            robot_radius=float(task.robot_radius),
        )
        stage_end_time = max(float(trajectory.xT[-1]) for trajectory in result.trajectories)
        padded = [cls.pad_trajectory_to_time(trajectory, stage_end_time) for trajectory in result.trajectories]
        return padded, result, config, continuous_pairwise_ok

    @classmethod
    def global_trajopt_time_scale(cls, task: RearrangementTask) -> float:
        return float(task.robot_radius) / float(task.vlimit)

    @classmethod
    def global_trajopt_window_span(cls, task: RearrangementTask) -> float:
        return cls.GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR * cls.global_trajopt_time_scale(task)

    @classmethod
    def global_trajopt_stride(cls, task: RearrangementTask) -> float:
        return cls.GLOBAL_TRAJOPT_STRIDE_FACTOR * cls.global_trajopt_time_scale(task)

    @classmethod
    def global_trajopt_config(cls, task: RearrangementTask, progress: bool) -> GlobalTrajOptConfig:
        time_scale = cls.global_trajopt_time_scale(task)
        return GlobalTrajOptConfig(
            sample_dt=cls.GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR * time_scale,
            clearance_margin=cls.GLOBAL_TRAJOPT_CLEARANCE_MARGIN,
            solver_max_iter=cls.GLOBAL_TRAJOPT_SOLVER_MAX_ITER,
            solver_eps_abs=cls.GLOBAL_TRAJOPT_SOLVER_EPS,
            solver_eps_rel=cls.GLOBAL_TRAJOPT_SOLVER_EPS,
            include_trajectory_knot_times=True,
            progress=bool(progress),
        )

    @staticmethod
    def pad_trajectory_to_time(trajectory: STTrajectory, target_time: float) -> STTrajectory:
        if target_time < float(trajectory.xT[-1]) - 1e-9:
            raise ValueError(
                f"Cannot pad trajectory ending at {trajectory.xT[-1]:.4f} back to {target_time:.4f}."
            )
        padded = trajectory.copy()
        if np.isclose(float(padded.xT[-1]), float(target_time)):
            return padded
        padded.vertex_path.append(padded.vertex_path[-1])
        padded.points.append(np.hstack([padded.xT, padded.xT[:-1], float(target_time)]))
        return padded

    @staticmethod
    def shift_trajectory_time(trajectory: STTrajectory, time_offset: float) -> STTrajectory:
        shifted = trajectory.copy()
        for point in shifted.points:
            point[shifted.dim - 1] += float(time_offset)
            point[-1] += float(time_offset)
        return shifted

    @classmethod
    def pairwise_collision_free(
        cls,
        stgcs: Any,
        trajectories: Sequence[STTrajectory],
        goal_stays: Sequence[bool],
        robot_radius: float,
    ) -> bool:
        checker = PriorityBasedSearch(stgcs, None, float(robot_radius))
        for agent_i in range(len(trajectories)):
            for agent_j in range(agent_i + 1, len(trajectories)):
                if checker.collision_checking(
                    trajectories[agent_i].points,
                    trajectories[agent_j].points,
                    cc_to_t0=True,
                    cc_to_tf=False,
                    cc_to_tf_a=bool(goal_stays[agent_i]),
                    cc_to_tf_b=bool(goal_stays[agent_j]),
                    tolerance=cls.GLOBAL_TRAJOPT_COLLISION_TOLERANCE,
                ):
                    return False
        return True

    @staticmethod
    def finite_float_or_none(value: float | int | None) -> float | None:
        return SolutionVisualizationService.finite_float_or_none(value)

    @staticmethod
    def trajectory_to_viewer_json(trajectory: STTrajectory | None) -> dict[str, Any]:
        return SolutionVisualizationService.trajectory_to_viewer_json(trajectory)

    @classmethod
    def trajectory_to_payload(cls, trajectory: STTrajectory) -> dict[str, Any]:
        return {
            "dim": int(trajectory.dim - 1),
            "vertex_path": list(trajectory.vertex_path),
            "points": [np.asarray(point, dtype=float).tolist() for point in trajectory.points],
            "viewer": cls.trajectory_to_viewer_json(trajectory),
        }

    @staticmethod
    def trajectory_from_payload(payload: dict[str, Any]) -> STTrajectory:
        dim = int(payload["dim"])
        vertex_path = [str(vertex) for vertex in payload["vertex_path"]]
        points = [np.asarray(point, dtype=float) for point in payload["points"]]
        return STTrajectory(vertex_path, points, dim=dim)

    @classmethod
    def query_to_viewer_json(cls, query: MPQuery, agent_idx: int) -> dict[str, Any]:
        return {
            "agent_index": int(agent_idx),
            "start": np.asarray(query.start, dtype=float).tolist(),
            "goal": np.asarray(query.goal, dtype=float).tolist(),
            "t_start": float(query.t_start),
            "is_stay": bool(query.is_stay),
            "vlimit": float(query.vlimit),
        }

    @classmethod
    def assigned_query_to_payload(cls, query: MPQuery, agent_idx: int, target_idx: int) -> dict[str, Any]:
        payload = cls.query_to_viewer_json(query, agent_idx)
        payload["target_index"] = int(target_idx)
        return payload

    @classmethod
    def stage_queries_from_payload(
        cls,
        task: RearrangementTask,
        payload: dict[str, Any],
    ) -> tuple[dict[str, tuple[MPQuery, ...]], dict[str, tuple[int, ...]]]:
        raw_stage_queries = payload.get("stage_queries")
        if not isinstance(raw_stage_queries, dict):
            raise ValueError("Solution payload does not contain assigned stage_queries.")
        queries_by_stage: dict[str, tuple[MPQuery, ...]] = {}
        target_indices_by_stage: dict[str, tuple[int, ...]] = {}
        for stage in task.stages:
            raw_queries = raw_stage_queries.get(stage.name)
            if not isinstance(raw_queries, list):
                raise ValueError(f"Solution payload does not contain assigned queries for stage {stage.name!r}.")
            if len(raw_queries) != task.num_agents:
                raise ValueError(
                    f"Stage {stage.name!r} contains {len(raw_queries)} assigned queries; "
                    f"expected {task.num_agents}."
                )
            queries: list[MPQuery] = []
            target_indices: list[int] = []
            for agent_idx, raw_query in enumerate(raw_queries):
                if not isinstance(raw_query, dict):
                    raise ValueError(f"stage_queries[{stage.name!r}][{agent_idx}] must be a mapping.")
                payload_agent_idx = int(raw_query.get("agent_index", agent_idx))
                if payload_agent_idx != agent_idx:
                    raise ValueError(
                        f"stage_queries[{stage.name!r}][{agent_idx}] has agent_index={payload_agent_idx}."
                    )
                target_idx = int(raw_query.get("target_index", -1))
                if target_idx < 0 or target_idx >= task.num_agents:
                    raise ValueError(
                        f"stage_queries[{stage.name!r}][{agent_idx}] has invalid target_index={target_idx}."
                    )
                queries.append(
                    cls.parse_query(
                        raw_query,
                        vlimit=task.vlimit,
                        space_dim=task.space_dim,
                        name=f"stage_queries[{stage.name}][{agent_idx}]",
                    )
                )
                target_indices.append(target_idx)
            queries_by_stage[stage.name] = tuple(queries)
            target_indices_by_stage[stage.name] = tuple(target_indices)
        cls.validate_assigned_stage_queries(task, queries_by_stage, target_indices_by_stage)
        return queries_by_stage, target_indices_by_stage

    @classmethod
    def validate_assigned_stage_queries(
        cls,
        task: RearrangementTask,
        queries_by_stage: dict[str, Sequence[MPQuery]],
        target_indices_by_stage: dict[str, Sequence[int]],
    ) -> None:
        expected_starts = cls.initial_positions(task)
        expected_target_order = list(range(task.num_agents))
        for stage in task.stages:
            queries = list(queries_by_stage.get(stage.name, ()))
            target_indices = [int(idx) for idx in target_indices_by_stage.get(stage.name, ())]
            if len(queries) != task.num_agents:
                raise ValueError(f"Assigned query count mismatch for stage {stage.name!r}.")
            if sorted(target_indices) != expected_target_order:
                raise ValueError(f"Assigned target indices for stage {stage.name!r} are not a permutation.")
            target_points = [np.asarray(query.goal, dtype=float) for query in stage.queries]
            for agent_idx, query in enumerate(queries):
                target_idx = target_indices[agent_idx]
                if not np.allclose(query.start, expected_starts[agent_idx], rtol=0.0, atol=1e-8):
                    raise ValueError(
                        f"Assigned query {agent_idx} for {stage.name!r} starts at "
                        f"{query.start.tolist()}, expected {expected_starts[agent_idx].tolist()}."
                    )
                if not np.allclose(query.goal, target_points[target_idx], rtol=0.0, atol=1e-8):
                    raise ValueError(
                        f"Assigned query {agent_idx} for {stage.name!r} ends at "
                        f"{query.goal.tolist()}, expected target {target_idx}={target_points[target_idx].tolist()}."
                    )
            expected_starts = [np.asarray(query.goal, dtype=float) for query in queries]

    @classmethod
    def stage_trajectories_from_payload(
        cls,
        task: RearrangementTask,
        payload: dict[str, Any],
    ) -> dict[str, list[STTrajectory]]:
        raw_stages = payload.get("stage_trajectories")
        if not isinstance(raw_stages, dict):
            raise ValueError("PWL payload does not contain stage_trajectories.")
        trajectories_by_stage: dict[str, list[STTrajectory]] = {}
        for stage in task.stages:
            raw_trajectories = raw_stages.get(stage.name)
            if not isinstance(raw_trajectories, list):
                raise ValueError(f"PWL payload does not contain trajectories for stage {stage.name!r}.")
            trajectories = [cls.trajectory_from_payload(item) for item in raw_trajectories]
            if len(trajectories) != task.num_agents:
                raise ValueError(
                    f"Stage {stage.name!r} contains {len(trajectories)} trajectories; "
                    f"expected {task.num_agents}."
                )
            trajectories_by_stage[stage.name] = trajectories
        return trajectories_by_stage

    @classmethod
    def validate_stage_trajectories(
        cls,
        stage_name: str,
        queries: Sequence[MPQuery],
        trajectories: Sequence[STTrajectory],
    ) -> None:
        if len(queries) != len(trajectories):
            raise ValueError(f"Stage {stage_name!r} query/trajectory count mismatch.")
        for idx, (query, trajectory) in enumerate(zip(queries, trajectories)):
            start = np.asarray(trajectory.x0[:-1], dtype=float)
            goal = np.asarray(trajectory.xT[:-1], dtype=float)
            if not np.allclose(start, query.start, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"Stored trajectory {idx} for {stage_name!r} starts at {start}, expected {query.start}."
                )
            if not np.allclose(goal, query.goal, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"Stored trajectory {idx} for {stage_name!r} ends at {goal}, expected {query.goal}."
                )

    @classmethod
    def is_return_to_start_stage(cls, task: RearrangementTask, stage: RearrangementStage) -> bool:
        if not task.stages or stage.name != task.stages[-1].name:
            return False
        initial_positions = cls.initial_positions(task)
        if len(stage.queries) != len(initial_positions):
            return False
        return all(
            np.allclose(query.goal, initial_positions[agent_idx], rtol=0.0, atol=1e-8)
            for agent_idx, query in enumerate(stage.queries)
        )

    @classmethod
    def pwl_passthrough_stage_result_payload(
        cls,
        stage_name: str,
        trajectories: Sequence[STTrajectory],
        config: GlobalTrajOptConfig,
        continuous_pairwise_collision_free: bool,
        wall_time: float,
    ) -> dict[str, Any]:
        stage_end_time = max(float(trajectory.xT[-1]) for trajectory in trajectories)
        sample_times = GlobalTrajectoryOptimizer.sample_times(trajectories, config)
        positions = GlobalTrajectoryOptimizer.sample_positions(trajectories, sample_times)
        return {
            "stage": stage_name,
            "status": "SUCCESS",
            "is_success": True,
            "runtime": 0.0,
            "wall_time": cls.finite_float_or_none(wall_time),
            "makespan": cls.finite_float_or_none(stage_end_time),
            "sample_dt": float(config.sample_dt),
            "num_sample_times": int(sample_times.size),
            "sampled_pairwise_collision_free": True,
            "continuous_pairwise_collision_free": bool(continuous_pairwise_collision_free),
            "min_pairwise_distance": float(GlobalTrajectoryOptimizer.min_pairwise_distance(positions)),
            "max_velocity_component": float(GlobalTrajectoryOptimizer.max_velocity_component(sample_times, positions)),
            "solver_message": "pwl_passthrough",
            "trajopt_skipped": True,
            "skip_reason": "return_to_start_stage",
        }

    @classmethod
    def trajopt_stage_result_payload(
        cls,
        stage_name: str,
        result: GlobalTrajOptResult,
        config: GlobalTrajOptConfig,
        stage_end_time: float,
        continuous_pairwise_collision_free: bool,
        wall_time: float,
    ) -> dict[str, Any]:
        return {
            "stage": stage_name,
            "status": "SUCCESS",
            "is_success": True,
            "runtime": cls.finite_float_or_none(result.runtime),
            "wall_time": cls.finite_float_or_none(wall_time),
            "makespan": cls.finite_float_or_none(stage_end_time),
            "sample_dt": float(config.sample_dt),
            "num_sample_times": int(result.times.size),
            "sampled_pairwise_collision_free": bool(result.sampled_pairwise_collision_free),
            "continuous_pairwise_collision_free": bool(continuous_pairwise_collision_free),
            "min_pairwise_distance": float(result.min_pairwise_distance),
            "max_velocity_component": float(result.max_velocity_component),
            "solver_message": result.solver_message,
        }

    @classmethod
    def solution_payload(
        cls,
        task: RearrangementTask,
        phase: str,
        stage_trajectories: dict[str, list[STTrajectory]],
        stage_queries: dict[str, Sequence[MPQuery]],
        stage_assignment_target_indices: dict[str, Sequence[int]],
        full_trajectories: Sequence[STTrajectory],
        stage_results: Sequence[dict[str, Any]],
        runtime: float,
        source_pwl_solution: Any = None,
    ) -> dict[str, Any]:
        planner_key = cls.planner_key_for_phase(task, phase)
        planner_name = cls.planner_name_for_phase(task, phase)
        cls.validate_assigned_stage_queries(task, stage_queries, stage_assignment_target_indices)
        durations = [float(trajectory.duration) for trajectory in full_trajectories]
        global_config = cls.global_trajopt_config(task, progress=False) if phase == "trajopt" else None
        return {
            "schema_version": 1,
            "ok": True,
            "instance_id": task.instance_id,
            "demo": cls.DEMO_KEY,
            "task_config_path": str(task.source_path),
            "task_name": task.name,
            "phase": phase,
            "source_pwl_solution": source_pwl_solution,
            "planner_key": planner_key,
            "planner_name": planner_name,
            "status": "SUCCESS",
            "is_success": True,
            "task_type": task.task_type,
            "assignment_strategy": cls.assignment_strategy_for_task(task),
            "assignment_solver": cls.assignment_solver_for_task(task),
            "runtime": cls.finite_float_or_none(runtime),
            "cost": cls.finite_float_or_none(sum(durations)),
            "makespan": cls.finite_float_or_none(max(durations, default=0.0)),
            "num_agents": int(task.num_agents),
            "num_completed_agents": int(task.num_agents),
            "path_alpha": 0.98,
            "body_alpha": 0.82,
            "stage_names": list(task.stage_names),
            "env": cls.env_payload(task),
            "planner": cls.planner_metadata(task),
            "global_trajopt": None if phase != "trajopt" else {
                "sample_dt_factor": cls.GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR,
                "window_span_factor": cls.GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR,
                "stride_factor": cls.GLOBAL_TRAJOPT_STRIDE_FACTOR,
                "window_span": cls.global_trajopt_window_span(task),
                "stride": cls.global_trajopt_stride(task),
                "velocity_gradient_weight": float(global_config.velocity_gradient_weight),
                "displacement_weight": float(global_config.displacement_weight),
                "displacement_jitter_weight": float(global_config.displacement_jitter_weight),
                "clearance_margin": float(global_config.clearance_margin),
                "solver_max_iter": cls.GLOBAL_TRAJOPT_SOLVER_MAX_ITER,
                "solver_eps": cls.GLOBAL_TRAJOPT_SOLVER_EPS,
            },
            "stage_results": list(stage_results),
            "stage_queries": {
                stage_name: [
                    cls.assigned_query_to_payload(
                        query,
                        agent_idx,
                        int(stage_assignment_target_indices[stage_name][agent_idx]),
                    )
                    for agent_idx, query in enumerate(stage_queries[stage_name])
                ]
                for stage_name in task.stage_names
            },
            "stage_trajectories": {
                stage_name: [
                    cls.trajectory_to_payload(trajectory)
                    for trajectory in stage_trajectories[stage_name]
                ]
                for stage_name in task.stage_names
            },
            "full_trajectories": [
                cls.trajectory_to_payload(trajectory)
                for trajectory in full_trajectories
            ],
            "trajectories": [
                {
                    **cls.trajectory_to_viewer_json(trajectory),
                    "agent_index": agent_idx,
                }
                for agent_idx, trajectory in enumerate(full_trajectories)
            ],
        }

    @classmethod
    def env_payload(cls, task: RearrangementTask) -> dict[str, Any]:
        env = cls.build_env(task)
        return {
            "name": task.name,
            "space_dim": int(task.space_dim),
            "environment": task.environment,
            "cspace": str(task.environment.get("type", "cspace")),
            "bounds": [
                np.asarray(env.lb, dtype=float).tolist(),
                np.asarray(env.ub, dtype=float).tolist(),
            ],
            "robot_radius": float(task.robot_radius),
        }

    @classmethod
    def validate_solution_payload(
        cls,
        task: RearrangementTask,
        payload: dict[str, Any],
        expected_phase: str | None = None,
    ) -> None:
        if not bool(payload.get("ok")):
            raise ValueError("Stored solution payload is not marked ok.")
        if payload.get("demo") != cls.DEMO_KEY:
            raise ValueError(f"Stored payload is for demo={payload.get('demo')!r}, not {cls.DEMO_KEY}.")
        if payload.get("instance_id") != task.instance_id:
            raise ValueError(f"Stored payload has instance_id={payload.get('instance_id')!r}.")
        if int(payload.get("num_agents", -1)) != task.num_agents:
            raise ValueError(f"Stored payload does not have {task.num_agents} agents.")
        if list(payload.get("stage_names", [])) != task.stage_names:
            raise ValueError("Stored payload stage sequence does not match this task config.")
        if expected_phase is not None and payload.get("phase") != expected_phase:
            raise ValueError(f"Stored payload phase={payload.get('phase')!r}, expected {expected_phase!r}.")
        phase = str(payload.get("phase"))
        if payload.get("planner_key") != cls.planner_key_for_phase(task, phase):
            raise ValueError("Stored payload planner key does not match this task config.")
        if payload.get("planner_name") != cls.planner_name_for_phase(task, phase):
            raise ValueError("Stored payload planner name does not match this task config.")
        if payload.get("planner") != cls.planner_metadata(task):
            raise ValueError("Stored payload planner metadata does not match this task config.")
        if payload.get("task_type") != task.task_type:
            raise ValueError(
                f"Stored payload task_type={payload.get('task_type')!r}, expected {task.task_type!r}."
            )
        if payload.get("assignment_strategy") != cls.assignment_strategy_for_task(task):
            raise ValueError("Stored payload assignment strategy does not match this demo.")
        if payload.get("assignment_solver") != cls.assignment_solver_for_task(task):
            raise ValueError("Stored payload assignment solver does not match this demo.")
        cls.stage_queries_from_payload(task, payload)
        if not cls.solution_payload_drawable(payload):
            raise ValueError("Stored payload does not contain drawable trajectories.")

    @staticmethod
    def load_payload(path: Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}.")
        payload["output_path"] = str(path)
        return payload

    @staticmethod
    def load_npz_payload(path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as values:
            if "payload_json" not in values:
                raise ValueError(f"NPZ solution {path} does not contain payload_json.")
            raw_payload = values["payload_json"].item()
        payload = json.loads(str(raw_payload))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected payload_json object in {path}.")
        payload["output_path"] = str(path)
        return payload

    @classmethod
    def load_solution_payload(cls, path: Path) -> dict[str, Any]:
        if Path(path).suffix == ".npz":
            return cls.load_npz_payload(path)
        return cls.load_payload(path)

    @classmethod
    def solution_payload_drawable(cls, payload: dict[str, Any]) -> bool:
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, list):
            return False
        return any(
            isinstance(trajectory, dict)
            and (
                bool(trajectory.get("segments"))
                or bool(trajectory.get("path"))
            )
            for trajectory in trajectories
        )

    @classmethod
    def load_saved_solution_payload(
        cls,
        task: RearrangementTask,
        path: Path,
        expected_phase: str,
    ) -> dict[str, Any] | None:
        if not Path(path).exists():
            return None
        try:
            payload = cls.load_solution_payload(Path(path))
            cls.validate_solution_payload(task, payload, expected_phase=expected_phase)
        except Exception as exc:
            print(f"Skipping saved {expected_phase} solution payload {path}: {exc}")
            return None
        return payload

    @classmethod
    def load_saved_solution_payloads(cls, task: RearrangementTask, paths: dict[str, Path]) -> list[dict[str, Any]]:
        payloads = []
        seen: set[tuple[str, str]] = set()

        def append_payload(payload: dict[str, Any] | None) -> None:
            if payload is None:
                return
            key = (str(payload.get("phase")), str(payload.get("planner_key")))
            if key in seen:
                return
            seen.add(key)
            payloads.append(payload)

        append_payload(cls.load_saved_solution_payload(task, paths["pwl_npz"], expected_phase="mrmp"))
        append_payload(cls.load_saved_solution_payload(task, paths["trajopt_npz"], expected_phase="trajopt"))
        append_payload(cls.load_saved_solution_payload(task, paths["pwl_json"], expected_phase="mrmp"))
        append_payload(cls.load_saved_solution_payload(task, paths["trajopt_json"], expected_phase="trajopt"))
        payloads.extend(cls.load_waited_solution_payloads(task, paths))
        return payloads

    @classmethod
    def load_saved_planning_payload(cls, task: RearrangementTask, paths: dict[str, Path]) -> dict[str, Any] | None:
        payload = cls.load_saved_solution_payload(task, paths["pwl_npz"], expected_phase="mrmp")
        if payload is not None:
            return payload
        return cls.load_saved_solution_payload(task, paths["pwl_json"], expected_phase="mrmp")

    @classmethod
    def load_waited_solution_payloads(cls, task: RearrangementTask, paths: dict[str, Path]) -> list[dict[str, Any]]:
        output_dir = Path(paths["viewer_json"]).parent
        standard_paths = {
            Path(paths["pwl_json"]).resolve(),
            Path(paths["trajopt_json"]).resolve(),
            Path(paths["viewer_json"]).resolve(),
        }
        payloads = []
        for path in sorted(output_dir.glob(cls.WAITED_SOLUTION_GLOB)):
            if Path(path).resolve() in standard_paths:
                continue
            payload = cls.load_waited_solution_payload(task, path)
            if payload is not None:
                payloads.append(payload)
        return payloads

    @classmethod
    def load_waited_solution_payload(cls, task: RearrangementTask, path: Path) -> dict[str, Any] | None:
        try:
            payload = cls.load_payload(Path(path))
            if payload.get("phase") not in {"mrmp", "trajopt"}:
                raise ValueError(f"Unsupported waited solution phase={payload.get('phase')!r}.")
            cls.validate_solution_payload(task, payload)
            return cls.waited_solution_manifest_payload(Path(path), payload)
        except Exception as exc:
            print(f"Skipping waited solution payload {path}: {exc}")
            return None

    @classmethod
    def waited_solution_manifest_payload(cls, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        manifest_payload = dict(payload)
        planner_name = str(manifest_payload.get("planner_name") or Path(path).stem)
        if cls.WAITED_SOLUTION_PLANNER_SUFFIX.strip().lower() not in planner_name.lower():
            manifest_payload["planner_name"] = f"{planner_name}{cls.WAITED_SOLUTION_PLANNER_SUFFIX}"
        manifest_payload["waited_solution_path"] = str(Path(path))
        return manifest_payload

    @staticmethod
    def write_json(output_path: Path, payload: dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output_path"] = str(output_path)

    @classmethod
    def write_npz(cls, output_path: Path, payload: dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stage_names = [str(stage_name) for stage_name in payload["stage_names"]]
        arrays: dict[str, Any] = {
            "phase": np.asarray(str(payload["phase"]), dtype=str),
            "task_name": np.asarray(str(payload["task_name"]), dtype=str),
            "instance_id": np.asarray(str(payload["instance_id"]), dtype=str),
            "stage_names": np.asarray(stage_names, dtype=str),
            "num_agents": np.asarray(int(payload["num_agents"]), dtype=int),
            "payload_json": np.asarray(json.dumps(payload), dtype=str),
        }
        for stage_name in stage_names:
            assigned_queries = payload["stage_queries"][stage_name]
            arrays[f"{stage_name}_assignment_target_indices"] = np.asarray(
                [int(query["target_index"]) for query in assigned_queries],
                dtype=int,
            )
            arrays[f"{stage_name}_assigned_starts"] = np.asarray(
                [query["start"] for query in assigned_queries],
                dtype=float,
            )
            arrays[f"{stage_name}_assigned_goals"] = np.asarray(
                [query["goal"] for query in assigned_queries],
                dtype=float,
            )
            for agent_idx, trajectory_payload in enumerate(payload["stage_trajectories"][stage_name]):
                arrays[f"{stage_name}_vertex_path_{agent_idx}"] = np.asarray(
                    trajectory_payload["vertex_path"],
                    dtype=str,
                )
                arrays[f"{stage_name}_points_{agent_idx}"] = np.asarray(
                    trajectory_payload["points"],
                    dtype=float,
                )
                arrays[f"{stage_name}_dim_{agent_idx}"] = np.asarray(
                    trajectory_payload["dim"],
                    dtype=int,
                )
        for agent_idx, trajectory_payload in enumerate(payload["full_trajectories"]):
            prefix = f"full_{agent_idx}"
            arrays[f"{prefix}_vertex_path"] = np.asarray(trajectory_payload["vertex_path"], dtype=str)
            arrays[f"{prefix}_points"] = np.asarray(trajectory_payload["points"], dtype=float)
            arrays[f"{prefix}_dim"] = np.asarray(trajectory_payload["dim"], dtype=int)
        np.savez(output_path, **arrays)
        payload["output_path"] = str(output_path)

    @classmethod
    def write_solution_exports(
        cls,
        paths: dict[str, Path],
        payload: dict[str, Any],
        *,
        write_solution_json: bool = False,
    ) -> None:
        phase = str(payload.get("phase"))
        if phase == "mrmp":
            json_path = paths["pwl_json"]
            npz_path = paths["pwl_npz"]
        elif phase == "trajopt":
            json_path = paths["trajopt_json"]
            npz_path = paths["trajopt_npz"]
        else:
            raise ValueError(f"Unsupported solution payload phase: {phase!r}")

        cls.write_npz(npz_path, payload)
        if write_solution_json:
            cls.write_json(json_path, payload)

    @classmethod
    def independent_metrics(cls, task: RearrangementTask) -> tuple[float, float]:
        per_agent = np.zeros(task.num_agents, dtype=float)
        for assignment in cls.build_stage_assignments(task).values():
            for agent_idx, query in enumerate(assignment.queries):
                per_agent[agent_idx] += float(np.linalg.norm(query.goal - query.start) / float(query.vlimit))
        return float(per_agent.sum()), float(per_agent.max(initial=0.0))

    @classmethod
    def build_viewer_manifest(
        cls,
        task: RearrangementTask,
        saved_solutions: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        env = cls.build_env(task)
        stgcs = cls.build_stgcs(task, env)
        independent_cost_sum, independent_makespan = cls.independent_metrics(task)
        first_stage_assignment = cls.assign_stage_queries(task, cls.initial_positions(task), task.stages[0])
        time_horizon = 0.0
        for solution in saved_solutions or []:
            value = solution.get("makespan")
            if value is not None and math.isfinite(float(value)):
                time_horizon = max(time_horizon, float(value))
        instance = {
            "instance_id": task.instance_id,
            "base_instance_id": f"base-{task.instance_id}",
            "problem_kind": "mrmp",
            "domain": task.name,
            "space_dim": int(task.space_dim),
            "spatial_seed": 0,
            "env_params": {
                "task_type": task.task_type,
                "assignment_strategy": cls.assignment_strategy_for_task(task),
                "assignment_solver": cls.assignment_solver_for_task(task),
                "task_config_path": str(task.source_path),
                "environment": task.environment,
                "num_stages": len(task.stages),
                "stage_names": list(task.stage_names),
                "vlimit": float(task.vlimit),
                "tmax": float(task.tmax),
            },
            "traffic_family": "yaml-rearrangement",
            "traffic_tier": f"{task.num_agents}-robot-{len(task.stages)}-stage",
            "stgcs_num_vertices": int(stgcs.G.number_of_nodes()),
            "stgcs_num_edges": int(stgcs.G.number_of_edges()),
            "dimension": int(task.space_dim),
            "robot_radius": float(env.robot_radius),
            "bounds": {
                "min": np.asarray(env.lb, dtype=float).tolist(),
                "max": np.asarray(env.ub, dtype=float).tolist(),
            },
            "spatial_sets": [
                {
                    "id": f"v{idx}",
                    "vertices": np.asarray(poly, dtype=float).tolist(),
                }
                for idx, poly in enumerate(task.cspace)
            ],
            "static_obstacles": [],
            "time_horizon": float(time_horizon),
            "time_range": [0.0, float(time_horizon)],
            "queries": [
                cls.query_to_viewer_json(query, agent_idx)
                for agent_idx, query in enumerate(first_stage_assignment.queries)
            ],
            "endpoint_markers": [],
            "hide_query_labels": True,
            "dynamic_obstacles": [],
            "num_agents": int(task.num_agents),
            "independent_cost_sum": float(independent_cost_sum),
            "independent_makespan": float(independent_makespan),
            "independent_runtime": 0.0,
            "num_conflicting_pairs": 0,
            "num_conflicting_agents": 0,
            "conflict_largest_component": 0,
            "conflict_density": 0.0,
        }
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": cls.DEMO_KEY,
            "instance_count": 1,
            "instances": [instance],
            "solutions": list(saved_solutions or []),
        }

    @classmethod
    def query_target_configurations(cls, task: RearrangementTask) -> list[tuple[str, list[np.ndarray]]]:
        configurations = [
            (
                "initial starts",
                [np.asarray(query.start, dtype=float) for query in task.stages[0].queries],
            )
        ]
        for stage in task.stages:
            configurations.append(
                (
                    f"{stage.name} targets",
                    [np.asarray(query.goal, dtype=float) for query in stage.queries],
                )
            )
        return configurations

    @classmethod
    def plot_query_targets(
        cls,
        task: RearrangementTask,
        output_path: Path | None = None,
        show: bool = True,
    ) -> None:
        if task.space_dim != 2:
            raise ValueError("Query target plotting currently supports 2D tasks only.")
        import matplotlib.pyplot as plt

        env = cls.build_env(task)
        configurations = cls.query_target_configurations(task)
        num_panels = len(configurations)
        cols = min(3, num_panels)
        rows = int(math.ceil(num_panels / cols))
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(4.0 * cols, 4.0 * rows))
        flat_axes = axes.ravel()

        for ax, (title, points) in zip(flat_axes, configurations):
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(float(env.lb[0]), float(env.ub[0]))
            ax.set_ylim(float(env.lb[1]), float(env.ub[1]))
            ax.grid(True, alpha=0.25)
            for poly in task.cspace:
                closed = np.vstack([poly, poly[0]])
                ax.plot(closed[:, 0], closed[:, 1], color="0.35", linewidth=1.0)
            points_array = np.asarray(points, dtype=float)
            ax.scatter(points_array[:, 0], points_array[:, 1], s=80, c="#1f77b4", edgecolors="black", zorder=3)
            for agent_idx, point in enumerate(points_array):
                ax.text(
                    float(point[0]),
                    float(point[1]),
                    str(agent_idx),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                    zorder=4,
                )

        for ax in flat_axes[num_panels:]:
            ax.axis("off")
        fig.suptitle(f"{task.name}: query target configurations", y=0.995)
        fig.tight_layout()
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=200)
            print(f"Wrote query target plot: {target}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    @classmethod
    def planner_options(cls, task: RearrangementTask) -> list[dict[str, Any]]:
        return [
            {
                "key": cls.pwl_planner_key(task),
                "name": cls.pwl_planner_name(task),
                "button_label": "MRMP planner",
                "space_dims": [int(task.space_dim)],
            },
            {
                "key": cls.trajopt_planner_key(task),
                "name": cls.trajopt_planner_name(task),
                "button_label": "MRMP + trajopt",
                "space_dims": [int(task.space_dim)],
            },
        ]

    @classmethod
    def planner_payload(cls, task: RearrangementTask) -> dict[str, Any]:
        return {
            "ok": True,
            "default_budget": float(task.stage_timeout_secs),
            "planners": {
                "mrmp": cls.planner_options(task),
                "st_heuristic_ablation": [],
                "base": [],
            },
        }

    @classmethod
    def solve_for_planner(
        cls,
        task: RearrangementTask,
        planner_key: str,
        paths: dict[str, Path],
        trajopt_progress: bool = False,
        write_solution_json: bool = False,
        planning_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if planner_key == cls.pwl_planner_key(task):
            pwl_payload = cls.solve_pwl(task)
            cls.write_solution_exports(
                paths,
                pwl_payload,
                write_solution_json=bool(write_solution_json),
            )
            return pwl_payload
        if planner_key != cls.trajopt_planner_key(task):
            raise ValueError(f"Unknown planner_key: {planner_key!r}")

        pwl_payload = planning_payload
        if pwl_payload is not None:
            cls.validate_solution_payload(task, pwl_payload, expected_phase="mrmp")
        else:
            pwl_payload = cls.load_saved_planning_payload(task, paths)
        if pwl_payload is None:
            pwl_payload = cls.solve_pwl(task)
            cls.write_solution_exports(
                paths,
                pwl_payload,
                write_solution_json=bool(write_solution_json),
            )
        trajopt_payload = cls.solve_trajopt_from_payload(task, pwl_payload, bool(trajopt_progress))
        cls.write_solution_exports(
            paths,
            trajopt_payload,
            write_solution_json=bool(write_solution_json),
        )
        return trajopt_payload

    @classmethod
    def serve_manifest(
        cls,
        task: RearrangementTask,
        payload: dict[str, Any],
        paths: dict[str, Path],
        host: str,
        port: int,
    ) -> None:
        payload_bytes = json.dumps(payload).encode("utf-8")
        planner_payload_bytes = json.dumps(cls.planner_payload(task)).encode("utf-8")
        viewer_path = Path("visualization/viewer/instance_manifest_viewer.html")
        viewer_mtime_ns = (ROOT / viewer_path).stat().st_mtime_ns
        viewer_url = (
            f"/{viewer_path.as_posix()}?"
            f"{urlencode({'manifest': '/api/manifest', 'v': str(viewer_mtime_ns)})}"
        )

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *handler_args, **handler_kwargs):
                super().__init__(*handler_args, directory=str(ROOT), **handler_kwargs)

            def end_headers(self) -> None:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def do_HEAD(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", viewer_url)
                    self.end_headers()
                    return
                return super().do_HEAD()

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/api/manifest":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload_bytes)))
                    self.end_headers()
                    self.wfile.write(payload_bytes)
                    return
                if parsed.path == "/api/solution-planners":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(planner_payload_bytes)))
                    self.end_headers()
                    self.wfile.write(planner_payload_bytes)
                    return
                if parsed.path == "/":
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", viewer_url)
                    self.end_headers()
                    return
                return super().do_GET()

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/solution":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                    request = json.loads(body.decode("utf-8"))
                    if request.get("instance_id") != task.instance_id:
                        raise ValueError(f"Unknown instance_id: {request.get('instance_id')!r}")
                    planner_key = str(request.get("planner_key", cls.pwl_planner_key(task)))
                    response_payload = cls.solve_for_planner(
                        task,
                        planner_key,
                        paths,
                        trajopt_progress=False,
                    )
                    status = HTTPStatus.OK
                except Exception as exc:
                    response_payload = {"ok": False, "error": str(exc)}
                    status = HTTPStatus.BAD_REQUEST
                response = json.dumps(response_payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer((host, port), Handler)
        print(
            json.dumps(
                {
                    "viewer_output": str(paths["viewer_json"]),
                    "planning_npz_output": str(paths["pwl_npz"]),
                    "trajopt_npz_output": str(paths["trajopt_npz"]),
                    "instance_id": task.instance_id,
                    "url": f"http://{host}:{port}{viewer_url}",
                },
                indent=2,
            ),
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()

    @classmethod
    def write_viewer_manifest(
        cls,
        task: RearrangementTask,
        paths: dict[str, Path],
        saved_solutions: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if saved_solutions is None:
            saved_solutions = cls.load_saved_solution_payloads(task, paths)
        else:
            saved_solutions = list(saved_solutions)
            for solution in saved_solutions:
                cls.validate_solution_payload(task, solution)
        payload = cls.build_viewer_manifest(task, saved_solutions=saved_solutions)
        cls.write_json(paths["viewer_json"], payload)
        return payload

    @classmethod
    def resolved_paths_from_values(
        cls,
        task: RearrangementTask,
        *,
        output_dir: Path | str | None = None,
        viewer_output: Path | str | None = None,
        pwl_output: Path | str | None = None,
        trajopt_output: Path | str | None = None,
        pwl_npz_output: Path | str | None = None,
        trajopt_npz_output: Path | str | None = None,
    ) -> dict[str, Path]:
        args = argparse.Namespace(
            output_dir=cls.DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir),
            viewer_output=None if viewer_output is None else Path(viewer_output),
            pwl_output=None if pwl_output is None else Path(pwl_output),
            trajopt_output=None if trajopt_output is None else Path(trajopt_output),
            pwl_npz_output=None if pwl_npz_output is None else Path(pwl_npz_output),
            trajopt_npz_output=None if trajopt_npz_output is None else Path(trajopt_npz_output),
        )
        return cls.resolved_paths(args, task)

    @classmethod
    def run_phase(
        cls,
        task: RearrangementTask,
        paths: dict[str, Path],
        *,
        phase: str = "serve",
        targets_output: Path | str | None = None,
        show_targets: bool = False,
        trajopt_progress: bool = False,
        write_solution_json: bool = False,
        planning_payload: dict[str, Any] | None = None,
        saved_solutions: Sequence[dict[str, Any]] | None = None,
        serve: bool = False,
        host: str | None = None,
        port: int | None = None,
    ) -> dict[str, Any]:
        phase_key = str(phase)
        if phase_key == "targets":
            cls.plot_query_targets(
                task,
                output_path=None if targets_output is None else Path(targets_output),
                show=bool(show_targets),
            )
            return {
                "phase": phase_key,
                "instance_id": task.instance_id,
                "targets_output": None if targets_output is None else str(Path(targets_output)),
            }

        if phase_key == "planning":
            pwl_payload = cls.solve_for_planner(
                task,
                cls.pwl_planner_key(task),
                paths,
                write_solution_json=bool(write_solution_json),
            )
            cls.write_viewer_manifest(task, paths, saved_solutions=[pwl_payload])
            result = {
                "phase": phase_key,
                "instance_id": task.instance_id,
                "planning_npz": str(paths["pwl_npz"]),
                "viewer_manifest": str(paths["viewer_json"]),
            }
            if write_solution_json:
                result["planning_json"] = str(paths["pwl_json"])
            return result

        if phase_key == "trajopt":
            trajopt_payload = cls.solve_for_planner(
                task,
                cls.trajopt_planner_key(task),
                paths,
                trajopt_progress=bool(trajopt_progress),
                write_solution_json=bool(write_solution_json),
                planning_payload=planning_payload,
            )
            manifest_solutions = []
            if planning_payload is not None:
                manifest_solutions.append(planning_payload)
            else:
                saved_planning_payload = cls.load_saved_planning_payload(task, paths)
                if saved_planning_payload is not None:
                    manifest_solutions.append(saved_planning_payload)
            manifest_solutions.append(trajopt_payload)
            cls.write_viewer_manifest(task, paths, saved_solutions=manifest_solutions)
            result = {
                "phase": phase_key,
                "instance_id": task.instance_id,
                "trajopt_npz": str(paths["trajopt_npz"]),
                "viewer_manifest": str(paths["viewer_json"]),
            }
            if write_solution_json:
                result["trajopt_json"] = str(paths["trajopt_json"])
            return result

        if phase_key == "both":
            pwl_payload = cls.solve_for_planner(
                task,
                cls.pwl_planner_key(task),
                paths,
                write_solution_json=bool(write_solution_json),
            )
            trajopt_payload = cls.solve_for_planner(
                task,
                cls.trajopt_planner_key(task),
                paths,
                trajopt_progress=bool(trajopt_progress),
                write_solution_json=bool(write_solution_json),
                planning_payload=pwl_payload,
            )
            cls.write_viewer_manifest(task, paths, saved_solutions=[pwl_payload, trajopt_payload])
            result = {
                "phase": phase_key,
                "instance_id": task.instance_id,
                "planning_npz": str(paths["pwl_npz"]),
                "trajopt_npz": str(paths["trajopt_npz"]),
                "viewer_manifest": str(paths["viewer_json"]),
            }
            if write_solution_json:
                result["planning_json"] = str(paths["pwl_json"])
                result["trajopt_json"] = str(paths["trajopt_json"])
            return result

        if phase_key != "serve":
            raise ValueError("phase must be one of: serve, targets, planning, trajopt, both")

        payload = cls.write_viewer_manifest(task, paths, saved_solutions=saved_solutions)
        if serve:
            cls.serve_manifest(
                task,
                payload,
                paths,
                host=cls.DEFAULT_HOST if host is None else str(host),
                port=cls.DEFAULT_PORT if port is None else int(port),
            )
        return {
            "phase": phase_key,
            "instance_id": task.instance_id,
            "viewer_manifest": str(paths["viewer_json"]),
            "num_saved_solutions": len(payload.get("solutions", [])),
        }

    @classmethod
    def run_pipeline(
        cls,
        task: RearrangementTask,
        paths: dict[str, Path],
        *,
        trajopt_progress: bool = False,
        write_solution_json: bool = False,
        serve: bool = False,
        host: str | None = None,
        port: int | None = None,
    ) -> dict[str, Any]:
        pwl_payload = cls.solve_for_planner(
            task,
            cls.pwl_planner_key(task),
            paths,
            write_solution_json=bool(write_solution_json),
        )
        trajopt_payload = cls.solve_for_planner(
            task,
            cls.trajopt_planner_key(task),
            paths,
            trajopt_progress=bool(trajopt_progress),
            write_solution_json=bool(write_solution_json),
            planning_payload=pwl_payload,
        )
        payload = cls.write_viewer_manifest(task, paths, saved_solutions=[pwl_payload, trajopt_payload])
        if serve:
            cls.serve_manifest(
                task,
                payload,
                paths,
                host=cls.DEFAULT_HOST if host is None else str(host),
                port=cls.DEFAULT_PORT if port is None else int(port),
            )
        result = {
            "phase": "pipeline",
            "instance_id": task.instance_id,
            "planning_npz": str(paths["pwl_npz"]),
            "trajopt_npz": str(paths["trajopt_npz"]),
            "viewer_manifest": str(paths["viewer_json"]),
            "num_saved_solutions": len(payload.get("solutions", [])),
        }
        if write_solution_json:
            result["planning_json"] = str(paths["pwl_json"])
            result["trajopt_json"] = str(paths["trajopt_json"])
        return result

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        task = cls.load_task(Path(args.config), solver_overrides=cls.solver_overrides_from_args(args))
        paths = cls.resolved_paths(args, task)
        write_solution_json = cls.write_solution_json_from_args(args)

        if args.phase == "targets":
            cls.plot_query_targets(
                task,
                output_path=args.targets_output,
                show=not bool(args.no_show_targets),
            )
            return

        if args.phase == "planning":
            pwl_payload = cls.solve_for_planner(
                task,
                cls.pwl_planner_key(task),
                paths,
                write_solution_json=write_solution_json,
            )
            cls.write_viewer_manifest(task, paths, saved_solutions=[pwl_payload])
            print(f"Wrote planning npz: {paths['pwl_npz']}")
            if write_solution_json:
                print(f"Wrote planning JSON: {paths['pwl_json']}")
            print(f"Wrote viewer manifest: {paths['viewer_json']}")
            return

        if args.phase == "trajopt":
            trajopt_payload = cls.solve_for_planner(
                task,
                cls.trajopt_planner_key(task),
                paths,
                trajopt_progress=bool(args.trajopt_progress),
                write_solution_json=write_solution_json,
            )
            manifest_solutions = []
            saved_planning_payload = cls.load_saved_planning_payload(task, paths)
            if saved_planning_payload is not None:
                manifest_solutions.append(saved_planning_payload)
            manifest_solutions.append(trajopt_payload)
            cls.write_viewer_manifest(task, paths, saved_solutions=manifest_solutions)
            print(f"Wrote trajopt npz: {paths['trajopt_npz']}")
            if write_solution_json:
                print(f"Wrote trajopt JSON: {paths['trajopt_json']}")
            print(f"Wrote viewer manifest: {paths['viewer_json']}")
            return

        if args.phase == "both":
            cls.run_pipeline(
                task,
                paths,
                trajopt_progress=bool(args.trajopt_progress),
                write_solution_json=write_solution_json,
            )
            print(f"Wrote planning npz: {paths['pwl_npz']}")
            print(f"Wrote trajopt npz: {paths['trajopt_npz']}")
            if write_solution_json:
                print(f"Wrote planning JSON: {paths['pwl_json']}")
                print(f"Wrote trajopt JSON: {paths['trajopt_json']}")
            print(f"Wrote viewer manifest: {paths['viewer_json']}")
            return

        payload = cls.write_viewer_manifest(task, paths)
        print(f"Wrote viewer manifest: {paths['viewer_json']}")
        if args.no_serve:
            print(
                json.dumps(
                    {
                        "viewer_output": str(paths["viewer_json"]),
                        "instance_id": task.instance_id,
                        "num_saved_solutions": len(payload.get("solutions", [])),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
        cls.serve_manifest(task, payload, paths, host=str(args.host), port=int(args.port))


MRMPDemo = RobotRearrangementTaskDemo


if __name__ == "__main__":
    MRMPDemo.main()
