from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stgcs-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stgcs-xdg-cache")

import numpy as np

from benchmark.environment.uav_village import VillageEnv
from demos.trajopt.optimization import GlobalTrajOptConfig, GlobalTrajOptResult, GlobalTrajectoryOptimizer
from stgcs.bfs.dominance_check import GlobalUpperBoundDominanceCheck, PositionBasedDominanceCheck
from stgcs.bfs.heuristics import MaxHeuristic
from stgcs.mrmp_planner import windowed_pbs
from stgcs.pbs import ChildExpansionMode, PriorityBasedSearch
from stgcs.st_planner import MPQuery, SearchPlanner
from stgcs.trajectory import STTrajectory
from stgcs.windowed_coordination import WindowedCoordinationReturn
from visualization.village_3d import Village3DVisualizationMixin
from visualization.viewer_utils import print_json


class DemoR48Village(VillageEnv, Village3DVisualizationMixin):
    DEFAULT_BASE_ROOT = ROOT / "data/stgcs_base"
    DEFAULT_MANIFEST_PATH = ROOT / "data/instances/mrmp/demo_r48_village/manifest.json"
    DEFAULT_SOLUTION_DIR = ROOT / "data/solutions/demo_r48_village"
    DEFAULT_SOLUTION_OUTPUT = DEFAULT_SOLUTION_DIR / "mrmp_solution.json"
    DEFAULT_TRAJOPT_OUTPUT = DEFAULT_SOLUTION_DIR / "trajopt_solution.json"
    DEFAULT_MRMP_CACHE = DEFAULT_SOLUTION_DIR / "mrmp_cache.json"
    DEFAULT_TRAJOPT_CACHE = DEFAULT_SOLUTION_DIR / "trajopt_cache.json"
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8780

    DEFAULT_VILLAGE_SIDE = 8
    DEFAULT_VILLAGE_HEIGHT = 3.0
    DEFAULT_BUILDING_EVERY = 3
    DEFAULT_DECORATION_DENSITY = 0.35
    DEFAULT_NUM_UAVS = 48
    DEFAULT_SEED = 1000
    DEFAULT_ROBOT_RADIUS = 0.2
    DEFAULT_BUILDING_CLEARANCE_MARGIN = 1e-3
    DEFAULT_VLIMIT = 1.0
    DEFAULT_TMAX = 1000.0
    DEFAULT_SOLUTION_BUDGET = 300.0
    WINDOW_ALPHA = 5.0
    EPSILON = 1000.0
    CHILD_EXPANSION_MODE = ChildExpansionMode.LAZY

    BASE_DOMAIN_KEY = "fastpathplanning-village3d"
    BASE_INSTANCE_ID = "base-fastpathplanning-village3d-s8-d0p35-bm0p001-bc1-csmerge2-seed01000"
    INSTANCE_ID = "mrmp-fastpathplanning-village3d-random-s8-d0p35-bm0p001-bc1-csmerge2-seed01000-48-uav"
    TRAFFIC_FAMILY = "fastpathplanning-village-random-exchange"
    EXPECTED_NUM_SAFE_BOXES = 119
    EXPECTED_NUM_STATIC_OBSTACLES = 204
    EXPECTED_STGCS_NUM_EDGES = 728

    MRMP_PLANNER_KEY = "fastpathplanning-village-wpbs"
    MRMP_PLANNER_NAME = "wPBS + h_max + GUB + delta_pos"
    PLANNER_KEY = f"{MRMP_PLANNER_KEY}-global-trajopt"
    PLANNER_NAME = f"{MRMP_PLANNER_NAME} + global trajopt"

    DEFAULT_GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR = 0.6
    DEFAULT_GLOBAL_TRAJOPT_MIN_SAMPLE_DT_FACTOR = 0.07
    DEFAULT_GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR = 3.0
    DEFAULT_GLOBAL_TRAJOPT_STRIDE_FACTOR = 1.0
    DEFAULT_GLOBAL_TRAJOPT_VELOCITY_GRADIENT_WEIGHT = 0.05
    DEFAULT_GLOBAL_TRAJOPT_DISPLACEMENT_WEIGHT = 0.2
    DEFAULT_GLOBAL_TRAJOPT_DISPLACEMENT_JITTER_WEIGHT = 1e-4
    DEFAULT_GLOBAL_TRAJOPT_CLEARANCE_MARGIN = 0.0
    DEFAULT_GLOBAL_TRAJOPT_SPATIAL_CONTAINMENT_TOLERANCE = 1e-5
    DEFAULT_GLOBAL_TRAJOPT_SPATIAL_CONTAINMENT_SOFT_MARGIN = 1e-3
    DEFAULT_GLOBAL_TRAJOPT_CONSTRAINT_SLACK_PENALTY = 1e4
    DEFAULT_GLOBAL_TRAJOPT_CONSTRAINT_SLACK_LIMIT = 1.0
    DEFAULT_MRMP_ENVIRONMENT_VALIDATION_TOLERANCE = 1e-8
    DEFAULT_GLOBAL_TRAJOPT_VALIDATION_TOLERANCE = 1e-4
    DEFAULT_GLOBAL_TRAJOPT_INCLUDE_INTERVAL_COLLISION_CONSTRAINTS = False
    DEFAULT_GLOBAL_TRAJOPT_INCLUDE_TRAJECTORY_KNOT_TIMES = True
    DEFAULT_GLOBAL_TRAJOPT_REFINEMENT_PASSES = 1
    DEFAULT_GLOBAL_TRAJOPT_SOLVER_MAX_ITER = 10_000
    DEFAULT_GLOBAL_TRAJOPT_SOLVER_EPS = 1e-4
    GLOBAL_TRAJOPT_COLLISION_TOLERANCE = 1e-6

    MRMP_CACHE_SCHEMA_VERSION = 1
    TRAJOPT_CACHE_SCHEMA_VERSION = 1

    DEFAULT_THIRD_PERSON_VIDEO_AGENT_INDICES = (8, 22)
    DEFAULT_THIRD_PERSON_VIDEO_BASENAME = "third_person_uav{agent_index:02d}.mp4"
    CSPACE_ALPHA = 0.14
    STATIC_OBSTACLE_ALPHA = 0.9
    UAV_MATERIALS = {
        "body_color": [0.92, 0.05, 0.04, 0.92],
        "arm_color": [0.92, 0.05, 0.04, 0.95],
        "propeller_color": [0.50, 0.50, 0.50, 0.88],
        "outline_color": [0.18, 0.04, 0.04, 1.0],
    }
    UAV_VISUAL_SCALE = 1.0
    VIEWER_LAYER_DEFAULTS = {
        "spatial_sets_fill": False,
        "spatial_sets_outline": False,
    }

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Solve and view the 48-UAV fastpathplanning village MRMP demo."
        )
        parser.add_argument("--base-root", type=Path, default=cls.DEFAULT_BASE_ROOT)
        parser.add_argument("--manifest", type=Path, default=cls.DEFAULT_MANIFEST_PATH)
        parser.add_argument("--solution-output", type=Path, default=cls.DEFAULT_SOLUTION_OUTPUT)
        parser.add_argument("--trajopt-output", type=Path, default=cls.DEFAULT_TRAJOPT_OUTPUT)
        parser.add_argument("--mrmp-cache", type=Path, default=cls.DEFAULT_MRMP_CACHE)
        parser.add_argument("--trajopt-cache", type=Path, default=cls.DEFAULT_TRAJOPT_CACHE)
        parser.add_argument("--solution-budget", type=float, default=cls.DEFAULT_SOLUTION_BUDGET)
        parser.add_argument("--window-alpha", type=float, default=cls.WINDOW_ALPHA)
        parser.add_argument("--epsilon", type=float, default=cls.EPSILON)
        parser.add_argument("--trajopt", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--force-solve", action="store_true")
        parser.add_argument("--force-trajopt", action="store_true")
        parser.add_argument("--no-serve", action="store_true")
        parser.add_argument("--host", type=str, default=cls.DEFAULT_HOST)
        parser.add_argument("--port", type=int, default=cls.DEFAULT_PORT)
        return parser.parse_args()

    @staticmethod
    def validate_args(args: argparse.Namespace) -> None:
        if not math.isfinite(float(args.solution_budget)) or float(args.solution_budget) <= 0.0:
            raise ValueError(f"Solution budget must be finite and positive, got {args.solution_budget!r}.")
        if not math.isfinite(float(args.window_alpha)) or float(args.window_alpha) <= 0.0:
            raise ValueError(f"Window alpha must be finite and positive, got {args.window_alpha!r}.")
        if not math.isfinite(float(args.epsilon)) or float(args.epsilon) <= 0.0:
            raise ValueError(f"Epsilon must be finite and positive, got {args.epsilon!r}.")

    @classmethod
    def build_demo_state(cls) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        safe_boxes, static_obstacles, geometry_metadata = cls.build_village_geometry(
            village_side=cls.DEFAULT_VILLAGE_SIDE,
            village_height=cls.DEFAULT_VILLAGE_HEIGHT,
            building_every=cls.DEFAULT_BUILDING_EVERY,
            decoration_density=cls.DEFAULT_DECORATION_DENSITY,
            robot_radius=cls.DEFAULT_ROBOT_RADIUS,
            building_clearance_margin=cls.DEFAULT_BUILDING_CLEARANCE_MARGIN,
            seed=cls.DEFAULT_SEED,
        )
        record = cls.build_record(
            safe_boxes=safe_boxes,
            village_side=cls.DEFAULT_VILLAGE_SIDE,
            village_height=cls.DEFAULT_VILLAGE_HEIGHT,
            building_every=cls.DEFAULT_BUILDING_EVERY,
            decoration_density=cls.DEFAULT_DECORATION_DENSITY,
            building_clearance_margin=cls.DEFAULT_BUILDING_CLEARANCE_MARGIN,
            num_uavs=cls.DEFAULT_NUM_UAVS,
            robot_radius=cls.DEFAULT_ROBOT_RADIUS,
            vlimit=cls.DEFAULT_VLIMIT,
            seed=cls.DEFAULT_SEED,
            geometry_metadata=geometry_metadata,
        )
        record["stgcs_num_edges"] = cls.EXPECTED_STGCS_NUM_EDGES
        cls.validate_demo_record(record, geometry_metadata, len(static_obstacles))
        return safe_boxes, static_obstacles, geometry_metadata, record

    @classmethod
    def validate_demo_record(
        cls,
        record: dict[str, Any],
        geometry_metadata: dict[str, Any],
        num_static_obstacles: int,
    ) -> None:
        if record["instance_id"] != cls.INSTANCE_ID:
            raise ValueError(f"Unexpected instance_id {record['instance_id']!r}.")
        if record["base_instance_id"] != cls.BASE_INSTANCE_ID:
            raise ValueError(f"Unexpected base_instance_id {record['base_instance_id']!r}.")
        if record["traffic_family"] != cls.TRAFFIC_FAMILY:
            raise ValueError(f"Unexpected traffic_family {record['traffic_family']!r}.")
        if int(record["num_agents"]) != cls.DEFAULT_NUM_UAVS:
            raise ValueError(f"Expected {cls.DEFAULT_NUM_UAVS} UAVs, got {record['num_agents']}.")
        if int(geometry_metadata["num_safe_boxes"]) != cls.EXPECTED_NUM_SAFE_BOXES:
            raise ValueError(
                f"Expected {cls.EXPECTED_NUM_SAFE_BOXES} safe boxes, got {geometry_metadata['num_safe_boxes']}."
            )
        if int(num_static_obstacles) != cls.EXPECTED_NUM_STATIC_OBSTACLES:
            raise ValueError(
                f"Expected {cls.EXPECTED_NUM_STATIC_OBSTACLES} static obstacles, got {num_static_obstacles}."
            )
        if int(record["stgcs_num_edges"]) != cls.EXPECTED_STGCS_NUM_EDGES:
            raise ValueError(
                f"Expected {cls.EXPECTED_STGCS_NUM_EDGES} ST-GCS edges, got {record['stgcs_num_edges']}."
            )

    @staticmethod
    def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(target)

    @classmethod
    def validate_solution_payload(
        cls,
        payload: dict[str, Any],
        record: dict[str, Any],
        expected_planner_key: str,
        budget: float,
    ) -> None:
        if not bool(payload.get("ok")):
            raise ValueError("Saved solution payload is not ok.")
        if str(payload.get("instance_id")) != str(record["instance_id"]):
            raise ValueError(
                f"Saved solution belongs to {payload.get('instance_id')!r}, expected {record['instance_id']!r}."
            )
        if str(payload.get("planner_key")) != str(expected_planner_key):
            raise ValueError(
                f"Saved solution planner is {payload.get('planner_key')!r}, expected {expected_planner_key!r}."
            )
        if abs(float(payload.get("budget")) - float(budget)) > 1e-9:
            raise ValueError(f"Saved solution budget does not match {float(budget):g}.")
        if not bool(payload.get("is_success")):
            raise ValueError("Saved solution is not successful.")
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != int(record["num_agents"]):
            count = 0 if not isinstance(trajectories, list) else len(trajectories)
            raise ValueError(f"Saved solution has {count} trajectories, expected {record['num_agents']}.")

    @classmethod
    def load_solution_payload(
        cls,
        path: Path,
        record: dict[str, Any],
        expected_planner_key: str,
        budget: float,
    ) -> dict[str, Any] | None:
        target = Path(path)
        if not target.exists():
            return None
        payload = json.loads(target.read_text())
        cls.validate_solution_payload(payload, record, expected_planner_key, budget)
        print(f"Reusing saved solution: {os.path.relpath(target, ROOT)}")
        return payload

    @classmethod
    def build_low_level_planner(
        cls,
        instance: SimpleNamespace,
        budget: float,
        epsilon: float,
    ) -> SearchPlanner:
        return SearchPlanner(
            dc_list=[GlobalUpperBoundDominanceCheck(float("inf"), 0.0, float(epsilon)), PositionBasedDominanceCheck()],
            heur=MaxHeuristic.from_instance(instance),
            eps=float(epsilon),
            runtime_limit_secs=float(budget),
        )

    @classmethod
    def mrmp_cache_signature(
        cls,
        record: dict[str, Any],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        window_alpha: float,
        window_span: float,
        epsilon: float,
    ) -> dict[str, Any]:
        return {
            "instance_id": record["instance_id"],
            "base_instance_id": record["base_instance_id"],
            "num_agents": int(record["num_agents"]),
            "village_side": int(record["village_side"]),
            "village_height": float(record["village_height"]),
            "building_every": int(record["building_every"]),
            "decoration_density": float(record["decoration_density"]),
            "building_clearance_margin": float(record["building_clearance_margin"]),
            "building_cspace_model": str(record["building_cspace_model"]),
            "cspace_simplification_model": str(record["cspace_simplification_model"]),
            "spatial_seed": int(record["spatial_seed"]),
            "robot_radius": float(robot_radius),
            "vlimit": float(vlimit),
            "tmax": float(tmax),
            "window_alpha": float(window_alpha),
            "window_span": float(window_span),
            "epsilon": float(epsilon),
            "planner_key": cls.MRMP_PLANNER_KEY,
            "coordination": "windowed-pbs",
            "child_expansion_mode": cls.CHILD_EXPANSION_MODE.name,
            "dynamic_window_adjustment": True,
            "queries": record["queries"],
        }

    @staticmethod
    def trajectory_to_cache_json(solution: STTrajectory) -> dict[str, Any]:
        return {
            "dimension": int(solution.dim) - 1,
            "vertex_path": [str(name) for name in solution.vertex_path],
            "points": [np.asarray(point, dtype=float).tolist() for point in solution.points],
        }

    @staticmethod
    def trajectory_from_cache_json(payload: dict[str, Any]) -> STTrajectory:
        dim = int(payload["dimension"])
        vertex_path = [str(name) for name in payload["vertex_path"]]
        points = [np.asarray(point, dtype=float) for point in payload["points"]]
        if len(vertex_path) != len(points):
            raise ValueError("Cached trajectory has inconsistent vertex_path and point counts.")
        expected_point_size = 2 * (dim + 1)
        for point in points:
            if point.shape != (expected_point_size,):
                raise ValueError(
                    f"Cached trajectory point has shape {point.shape}, expected {(expected_point_size,)}."
                )
        return STTrajectory(vertex_path, points, dim=dim)

    @staticmethod
    def windowed_result_to_cache_json(result: WindowedCoordinationReturn) -> dict[str, Any]:
        return {
            "success": bool(result.success),
            "mp": np.asarray(result.mp, dtype=float).tolist(),
            "gcs": np.asarray(result.gcs, dtype=float).tolist(),
            "gub": np.asarray(result.gub, dtype=float).tolist(),
            "search": np.asarray(result.search, dtype=float).tolist(),
            "cr": np.asarray(result.cr, dtype=float).tolist(),
            "dc": np.asarray(result.dc, dtype=float).tolist(),
            "dc_cr": np.asarray(result.dc_cr, dtype=float).tolist(),
            "ecd": np.asarray(result.ecd, dtype=float).tolist(),
            "cc": np.asarray(result.cc, dtype=float).tolist(),
            "wpbs": np.asarray(result.wpbs, dtype=float).tolist(),
            "runtime": np.asarray(result.runtime, dtype=float).tolist(),
            "pbs_popped_nodes": int(result.pbs_popped_nodes),
            "pbs_generated_children": int(result.pbs_generated_children),
            "pbs_update_calls": int(result.pbs_update_calls),
        }

    @staticmethod
    def windowed_result_from_cache_json(
        payload: dict[str, Any],
        solutions: list[STTrajectory],
    ) -> WindowedCoordinationReturn:
        zeros = [0.0, 0.0]
        return WindowedCoordinationReturn(
            success=bool(payload["success"]),
            solutions=[solution.copy() for solution in solutions],
            mp=np.asarray(payload.get("mp", zeros), dtype=float),
            gcs=np.asarray(payload.get("gcs", zeros), dtype=float),
            gub=np.asarray(payload.get("gub", zeros), dtype=float),
            search=np.asarray(payload.get("search", zeros), dtype=float),
            cr=np.asarray(payload.get("cr", zeros), dtype=float),
            dc=np.asarray(payload.get("dc", zeros), dtype=float),
            dc_cr=np.asarray(payload.get("dc_cr", zeros), dtype=float),
            ecd=np.asarray(payload.get("ecd", zeros), dtype=float),
            cc=np.asarray(payload.get("cc", zeros), dtype=float),
            wpbs=np.asarray(payload.get("wpbs", zeros), dtype=float),
            runtime=np.asarray(payload["runtime"], dtype=float),
            pbs_popped_nodes=int(payload["pbs_popped_nodes"]),
            pbs_generated_children=int(payload["pbs_generated_children"]),
            pbs_update_calls=int(payload["pbs_update_calls"]),
        )

    @classmethod
    def load_mrmp_cache(
        cls,
        cache_path: Path,
        record: dict[str, Any],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        window_alpha: float,
        window_span: float,
        epsilon: float,
    ) -> tuple[list[STTrajectory], WindowedCoordinationReturn] | None:
        target = Path(cache_path)
        if not target.exists():
            return None
        payload = json.loads(target.read_text())
        if int(payload.get("schema_version", -1)) != cls.MRMP_CACHE_SCHEMA_VERSION:
            return None
        expected_signature = cls.mrmp_cache_signature(
            record,
            robot_radius,
            vlimit,
            tmax,
            window_alpha,
            window_span,
            epsilon,
        )
        if payload.get("signature") != expected_signature:
            return None

        solutions = [cls.trajectory_from_cache_json(item) for item in payload["trajectories"]]
        if len(solutions) != int(record["num_agents"]):
            raise ValueError(
                f"Cached MRMP trajectory count {len(solutions)} does not match num_agents={record['num_agents']}."
            )
        result = cls.windowed_result_from_cache_json(payload["result"], solutions)
        if not result.success:
            raise ValueError("Cached MRMP trajectory result is not successful.")
        return solutions, result

    @classmethod
    def save_mrmp_cache(
        cls,
        cache_path: Path,
        record: dict[str, Any],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        window_alpha: float,
        window_span: float,
        epsilon: float,
        solutions: list[STTrajectory],
        result: WindowedCoordinationReturn,
    ) -> None:
        if not result.success:
            return
        payload = {
            "schema_version": cls.MRMP_CACHE_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "signature": cls.mrmp_cache_signature(
                record,
                robot_radius,
                vlimit,
                tmax,
                window_alpha,
                window_span,
                epsilon,
            ),
            "result": cls.windowed_result_to_cache_json(result),
            "trajectories": [cls.trajectory_to_cache_json(solution) for solution in solutions],
        }
        cls.write_json_atomic(Path(cache_path), payload)

    @classmethod
    def global_trajopt_time_scale(cls, robot_radius: float, vlimit: float) -> float:
        return float(robot_radius) / float(vlimit)

    @classmethod
    def global_trajopt_window_span(cls, robot_radius: float, vlimit: float) -> float:
        return cls.DEFAULT_GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR * cls.global_trajopt_time_scale(robot_radius, vlimit)

    @classmethod
    def global_trajopt_stride(cls, robot_radius: float, vlimit: float) -> float:
        return cls.DEFAULT_GLOBAL_TRAJOPT_STRIDE_FACTOR * cls.global_trajopt_time_scale(robot_radius, vlimit)

    @classmethod
    def global_trajopt_config(cls, robot_radius: float, vlimit: float) -> GlobalTrajOptConfig:
        time_scale = cls.global_trajopt_time_scale(robot_radius, vlimit)
        return GlobalTrajOptConfig(
            sample_dt=cls.DEFAULT_GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR * time_scale,
            min_sample_dt=cls.DEFAULT_GLOBAL_TRAJOPT_MIN_SAMPLE_DT_FACTOR * time_scale,
            velocity_gradient_weight=cls.DEFAULT_GLOBAL_TRAJOPT_VELOCITY_GRADIENT_WEIGHT,
            displacement_weight=cls.DEFAULT_GLOBAL_TRAJOPT_DISPLACEMENT_WEIGHT,
            displacement_jitter_weight=cls.DEFAULT_GLOBAL_TRAJOPT_DISPLACEMENT_JITTER_WEIGHT,
            clearance_margin=cls.DEFAULT_GLOBAL_TRAJOPT_CLEARANCE_MARGIN,
            spatial_containment_tolerance=cls.DEFAULT_GLOBAL_TRAJOPT_SPATIAL_CONTAINMENT_TOLERANCE,
            spatial_containment_soft_margin=cls.DEFAULT_GLOBAL_TRAJOPT_SPATIAL_CONTAINMENT_SOFT_MARGIN,
            constraint_slack_penalty=cls.DEFAULT_GLOBAL_TRAJOPT_CONSTRAINT_SLACK_PENALTY,
            constraint_slack_limit=cls.DEFAULT_GLOBAL_TRAJOPT_CONSTRAINT_SLACK_LIMIT,
            include_interval_collision_constraints=cls.DEFAULT_GLOBAL_TRAJOPT_INCLUDE_INTERVAL_COLLISION_CONSTRAINTS,
            include_trajectory_knot_times=cls.DEFAULT_GLOBAL_TRAJOPT_INCLUDE_TRAJECTORY_KNOT_TIMES,
            solver_max_iter=cls.DEFAULT_GLOBAL_TRAJOPT_SOLVER_MAX_ITER,
            solver_eps_abs=cls.DEFAULT_GLOBAL_TRAJOPT_SOLVER_EPS,
            solver_eps_rel=cls.DEFAULT_GLOBAL_TRAJOPT_SOLVER_EPS,
            progress=True,
        )

    @classmethod
    def global_trajopt_cache_signature(
        cls,
        record: dict[str, Any],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        mrmp_window_alpha: float,
        mrmp_window_span: float,
        epsilon: float,
        global_config: GlobalTrajOptConfig,
        global_window_span: float,
        global_stride: float,
    ) -> dict[str, Any]:
        signature = cls.mrmp_cache_signature(
            record,
            robot_radius,
            vlimit,
            tmax,
            mrmp_window_alpha,
            mrmp_window_span,
            epsilon,
        )
        signature["mrmp_planner_key"] = signature.pop("planner_key")
        signature.update(
            {
                "planner_key": cls.PLANNER_KEY,
                "global_trajopt_mode": "sliding_window",
                "global_trajopt_sample_dt": float(global_config.sample_dt),
                "global_trajopt_min_sample_dt": float(global_config.min_sample_dt),
                "global_trajopt_window_span": float(global_window_span),
                "global_trajopt_stride": float(global_stride),
                "global_trajopt_velocity_gradient_weight": float(global_config.velocity_gradient_weight),
                "global_trajopt_displacement_weight": float(global_config.displacement_weight),
                "global_trajopt_displacement_jitter_weight": float(global_config.displacement_jitter_weight),
                "global_trajopt_clearance_margin": float(global_config.clearance_margin),
                "global_trajopt_spatial_containment_tolerance": float(global_config.spatial_containment_tolerance),
                "global_trajopt_spatial_containment_soft_margin": float(
                    global_config.spatial_containment_soft_margin
                ),
                "global_trajopt_constraint_slack_penalty": float(global_config.constraint_slack_penalty),
                "global_trajopt_constraint_slack_limit": float(global_config.constraint_slack_limit),
                "global_trajopt_validation_tolerance": float(cls.DEFAULT_GLOBAL_TRAJOPT_VALIDATION_TOLERANCE),
                "global_trajopt_include_interval_collision_constraints": bool(
                    global_config.include_interval_collision_constraints
                ),
                "global_trajopt_refinement_passes": int(cls.DEFAULT_GLOBAL_TRAJOPT_REFINEMENT_PASSES),
                "global_trajopt_fix_terminal_states": bool(global_config.fix_terminal_states),
                "global_trajopt_include_trajectory_knot_times": bool(global_config.include_trajectory_knot_times),
                "global_trajopt_solver_max_iter": int(global_config.solver_max_iter),
                "global_trajopt_solver_eps_abs": float(global_config.solver_eps_abs),
                "global_trajopt_solver_eps_rel": float(global_config.solver_eps_rel),
                "global_trajopt_round_time_decimals": int(global_config.round_time_decimals),
            }
        )
        return signature

    @classmethod
    def load_global_trajopt_cache(
        cls,
        cache_path: Path,
        record: dict[str, Any],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        mrmp_window_alpha: float,
        mrmp_window_span: float,
        epsilon: float,
    ) -> tuple[list[STTrajectory], dict[str, Any]] | None:
        target = Path(cache_path)
        if not target.exists():
            return None
        payload = json.loads(target.read_text())
        if int(payload.get("schema_version", -1)) != cls.TRAJOPT_CACHE_SCHEMA_VERSION:
            return None

        global_config = cls.global_trajopt_config(robot_radius, vlimit)
        global_window_span = cls.global_trajopt_window_span(robot_radius, vlimit)
        global_stride = cls.global_trajopt_stride(robot_radius, vlimit)
        expected_signature = cls.global_trajopt_cache_signature(
            record,
            robot_radius,
            vlimit,
            tmax,
            mrmp_window_alpha,
            mrmp_window_span,
            epsilon,
            global_config,
            global_window_span,
            global_stride,
        )
        if payload.get("signature") != expected_signature:
            return None

        trajectories = [cls.trajectory_from_cache_json(item) for item in payload["trajectories"]]
        if len(trajectories) != int(record["num_agents"]):
            raise ValueError(
                f"Cached trajopt trajectory count {len(trajectories)} does not match "
                f"num_agents={record['num_agents']}."
            )
        return trajectories, payload

    @classmethod
    def save_global_trajopt_cache(
        cls,
        cache_path: Path,
        record: dict[str, Any],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        mrmp_window_alpha: float,
        mrmp_window_span: float,
        epsilon: float,
        global_result: GlobalTrajOptResult,
        global_config: GlobalTrajOptConfig,
        global_window_span: float,
        global_stride: float,
        continuous_pairwise_collision_free: bool | None,
        continuous_environment_collision_free: bool | None,
    ) -> None:
        payload = {
            "schema_version": cls.TRAJOPT_CACHE_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "signature": cls.global_trajopt_cache_signature(
                record,
                robot_radius,
                vlimit,
                tmax,
                mrmp_window_alpha,
                mrmp_window_span,
                epsilon,
                global_config,
                global_window_span,
                global_stride,
            ),
            "global_trajopt": cls.global_trajopt_result_to_json(
                record,
                global_result,
                global_config,
                global_window_span,
                global_stride,
                continuous_pairwise_collision_free,
                continuous_environment_collision_free,
            ),
            "trajectories": [cls.trajectory_to_cache_json(solution) for solution in global_result.trajectories],
        }
        cls.write_json_atomic(Path(cache_path), payload)

    @staticmethod
    def global_trajopt_result_from_saved_json(
        payload: dict[str, Any],
        trajectories: list[STTrajectory],
    ) -> GlobalTrajOptResult:
        global_payload = payload["global_trajopt"]
        return GlobalTrajOptResult(
            trajectories=[trajectory.copy() for trajectory in trajectories],
            times=np.empty(int(global_payload.get("num_sample_times", 0)), dtype=float),
            original_positions=np.empty((0,), dtype=float),
            optimized_positions=np.empty((0,), dtype=float),
            displacements=np.empty((0,), dtype=float),
            runtime=float(global_payload["runtime"]),
            sampled_pairwise_collision_free=bool(global_payload["sampled_pairwise_collision_free"]),
            sampled_environment_collision_free=global_payload.get("sampled_environment_collision_free"),
            continuous_environment_collision_free=global_payload.get("continuous_environment_collision_free"),
            min_pairwise_distance=float(global_payload["min_pairwise_distance"]),
            max_velocity_component=float(global_payload["max_velocity_component"]),
            solver_message=str(global_payload.get("solver_message", "")),
        )

    @classmethod
    def pairwise_collision_free(
        cls,
        stgcs: Any,
        trajectories: list[STTrajectory],
        robot_radius: float,
        goal_stays: list[bool],
        tolerance: float = 1e-4,
    ) -> bool:
        checker = PriorityBasedSearch(stgcs, None, float(robot_radius))
        for agent_i, agent_j in combinations(range(len(trajectories)), 2):
            if checker.collision_checking(
                trajectories[agent_i].points,
                trajectories[agent_j].points,
                cc_to_t0=True,
                cc_to_tf=False,
                cc_to_tf_a=bool(goal_stays[agent_i]),
                cc_to_tf_b=bool(goal_stays[agent_j]),
                tolerance=float(tolerance),
            ):
                return False
        return True

    @classmethod
    def environment_collision_free(
        cls,
        env: Any,
        trajectories: list[STTrajectory],
        tolerance: float | None = None,
    ) -> bool:
        for trajectory in trajectories:
            for point in trajectory.points:
                values = np.asarray(point, dtype=float).reshape(-1)
                dim = int(trajectory.dim)
                if values.size != 2 * dim:
                    return False
                if tolerance is None:
                    in_cspace = env.is_segment_in_CSpace(values[: dim - 1], values[dim: 2 * dim - 1])
                else:
                    in_cspace = env.is_segment_in_CSpace(
                        values[: dim - 1],
                        values[dim: 2 * dim - 1],
                        tol=float(tolerance),
                    )
                if not in_cspace:
                    return False
        return True

    @classmethod
    def optimize_mrmp_trajectories(
        cls,
        instance: SimpleNamespace,
        trajectories: list[STTrajectory],
        queries: list[MPQuery],
    ) -> tuple[GlobalTrajOptResult, GlobalTrajOptConfig, bool, bool]:
        robot_radius = float(instance.env.robot_radius)
        vlimit = float(instance.stgcs.vlimit)
        config = cls.global_trajopt_config(robot_radius, vlimit)
        reference_trajectories = trajectories
        total_runtime = 0.0
        solver_messages: list[str] = []
        last_error: Exception | None = None
        for pass_idx in range(1, cls.DEFAULT_GLOBAL_TRAJOPT_REFINEMENT_PASSES + 1):
            result = GlobalTrajectoryOptimizer.optimize_sliding_window(
                reference_trajectories,
                robot_radius,
                vlimit,
                config=config,
                window_span=cls.global_trajopt_window_span(robot_radius, vlimit),
                stride=cls.global_trajopt_stride(robot_radius, vlimit),
                stgcs=instance.stgcs,
                env=None,
            )
            total_runtime += float(result.runtime)
            solver_messages.append(f"pass {pass_idx}: {result.solver_message}")
            if not result.sampled_pairwise_collision_free:
                raise RuntimeError("Global trajectory optimization produced a sampled pairwise collision.")
            if result.sampled_environment_collision_free is False:
                raise RuntimeError("Global trajectory optimization produced a sampled environment collision.")

            result = replace(result, runtime=total_runtime, solver_message="; ".join(solver_messages))
            try:
                continuous_pairwise_ok, continuous_environment_ok = cls.validate_trajopt_result(
                    instance,
                    result,
                    queries,
                )
                return result, config, continuous_pairwise_ok, continuous_environment_ok
            except RuntimeError as exc:
                last_error = exc
                reference_trajectories = result.trajectories

        if last_error is not None:
            raise last_error
        raise RuntimeError("Global trajectory optimization did not run any refinement pass.")

    @classmethod
    def validate_trajopt_result(
        cls,
        instance: SimpleNamespace,
        result: GlobalTrajOptResult,
        queries: list[MPQuery],
    ) -> tuple[bool, bool]:
        continuous_pairwise_ok = cls.pairwise_collision_free(
            instance.stgcs,
            result.trajectories,
            instance.env.robot_radius,
            goal_stays=[query.is_stay for query in queries],
            tolerance=cls.GLOBAL_TRAJOPT_COLLISION_TOLERANCE,
        )
        continuous_environment_ok = cls.environment_collision_free(
            instance.env,
            result.trajectories,
            tolerance=cls.DEFAULT_GLOBAL_TRAJOPT_VALIDATION_TOLERANCE,
        )
        return continuous_pairwise_ok, continuous_environment_ok

    @classmethod
    def solve_mrmp_record(
        cls,
        record: dict[str, Any],
        safe_boxes: list[np.ndarray],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        budget: float,
        window_alpha: float,
        epsilon: float,
        mrmp_cache_path: Path | None = None,
    ) -> SimpleNamespace:
        started = time.perf_counter()
        queries = [cls.query_to_mp_query(query) for query in record["queries"]]
        window_span = float(window_alpha) * float(robot_radius) / float(vlimit)
        cache_hit = False
        cached = None
        if mrmp_cache_path is not None:
            cached = cls.load_mrmp_cache(
                Path(mrmp_cache_path),
                record,
                float(robot_radius),
                float(vlimit),
                float(tmax),
                float(window_alpha),
                window_span,
                float(epsilon),
            )

        if cached is None:
            instance = cls.build_planning_instance(
                record,
                safe_boxes,
                robot_radius,
                vlimit,
                tmax,
                build_heuristics=True,
            )
            planner = cls.build_low_level_planner(instance, budget, epsilon)
            solutions, result = windowed_pbs(
                instance.stgcs,
                planner,
                queries,
                float(robot_radius),
                window_span=window_span,
                timeout_secs=float(budget),
                dynamic_window_adjustment=True,
                child_expansion_mode=cls.CHILD_EXPANSION_MODE,
            )
        else:
            solutions, result = cached
            cache_hit = True
            instance = cls.build_planning_instance(
                record,
                safe_boxes,
                robot_radius,
                vlimit,
                tmax,
                build_heuristics=False,
            )

        pairwise_ok = False
        environment_ok = False
        if result.success:
            pairwise_ok = cls.pairwise_collision_free(
                instance.stgcs,
                solutions,
                float(robot_radius),
                goal_stays=[query.is_stay for query in queries],
            )
            environment_ok = cls.environment_collision_free(
                instance.env,
                solutions,
                tolerance=cls.DEFAULT_MRMP_ENVIRONMENT_VALIDATION_TOLERANCE,
            )
            if not pairwise_ok:
                raise RuntimeError("Planner returned trajectories with a pairwise collision.")
            if not environment_ok:
                raise RuntimeError("Planner returned a trajectory segment outside the generated C-space.")
            if not cache_hit and mrmp_cache_path is not None:
                cls.save_mrmp_cache(
                    Path(mrmp_cache_path),
                    record,
                    float(robot_radius),
                    float(vlimit),
                    float(tmax),
                    float(window_alpha),
                    window_span,
                    float(epsilon),
                    solutions,
                    result,
                )

        payload = cls.solution_to_viewer_payload(
            record,
            solutions,
            result,
            budget,
            window_span,
            epsilon,
            planner_key=cls.MRMP_PLANNER_KEY,
            planner_name=cls.MRMP_PLANNER_NAME,
        )
        payload["wall_time"] = cls.finite_float_or_none(time.perf_counter() - started)
        payload["environment_collision_free"] = bool(environment_ok)
        payload["pairwise_collision_free"] = bool(pairwise_ok)
        payload["mrmp_cache_path"] = None if mrmp_cache_path is None else str(Path(mrmp_cache_path))
        payload["mrmp_cache_hit"] = bool(cache_hit)
        return SimpleNamespace(
            instance=instance,
            queries=queries,
            solutions=solutions,
            result=result,
            window_span=window_span,
            pairwise_collision_free=pairwise_ok,
            environment_collision_free=environment_ok,
            mrmp_cache_hit=cache_hit,
            payload=payload,
        )

    @classmethod
    def solve_trajopt_from_mrmp(
        cls,
        record: dict[str, Any],
        mrmp: SimpleNamespace,
        robot_radius: float,
        vlimit: float,
        tmax: float,
        budget: float,
        window_alpha: float,
        epsilon: float,
        trajopt_cache_path: Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        global_window_span = cls.global_trajopt_window_span(float(robot_radius), float(vlimit))
        global_stride = cls.global_trajopt_stride(float(robot_radius), float(vlimit))
        global_config = cls.global_trajopt_config(float(robot_radius), float(vlimit))
        cache_hit = False
        cached = None
        if trajopt_cache_path is not None:
            cached = cls.load_global_trajopt_cache(
                Path(trajopt_cache_path),
                record,
                float(robot_radius),
                float(vlimit),
                float(tmax),
                float(window_alpha),
                float(mrmp.window_span),
                float(epsilon),
            )

        if cached is None:
            global_result, global_config, continuous_pairwise_ok, continuous_environment_ok = (
                cls.optimize_mrmp_trajectories(mrmp.instance, mrmp.solutions, mrmp.queries)
            )
            if trajopt_cache_path is not None:
                cls.save_global_trajopt_cache(
                    Path(trajopt_cache_path),
                    record,
                    float(robot_radius),
                    float(vlimit),
                    float(tmax),
                    float(window_alpha),
                    float(mrmp.window_span),
                    float(epsilon),
                    global_result,
                    global_config,
                    global_window_span,
                    global_stride,
                    continuous_pairwise_ok,
                    continuous_environment_ok,
                )
        else:
            trajopt_trajectories, trajopt_cache_payload = cached
            global_result = cls.global_trajopt_result_from_saved_json(
                trajopt_cache_payload,
                trajopt_trajectories,
            )
            continuous_pairwise_ok, continuous_environment_ok = cls.validate_trajopt_result(
                mrmp.instance,
                global_result,
                mrmp.queries,
            )
            cache_hit = True

        payload = cls.solution_to_viewer_payload(
            record,
            mrmp.solutions,
            mrmp.result,
            budget,
            mrmp.window_span,
            epsilon,
            planner_key=cls.PLANNER_KEY,
            planner_name=cls.PLANNER_NAME,
            global_result=global_result,
            global_config=global_config,
            global_window_span=global_window_span,
            global_stride=global_stride,
            continuous_pairwise_collision_free=continuous_pairwise_ok,
            continuous_environment_collision_free=continuous_environment_ok,
        )
        payload["wall_time"] = cls.finite_float_or_none(
            float(mrmp.payload.get("wall_time") or 0.0) + time.perf_counter() - started
        )
        payload["environment_collision_free"] = bool(continuous_environment_ok)
        payload["pairwise_collision_free"] = bool(continuous_pairwise_ok)
        payload["mrmp_cache_path"] = mrmp.payload.get("mrmp_cache_path")
        payload["mrmp_cache_hit"] = bool(getattr(mrmp, "mrmp_cache_hit", False))
        payload["trajopt_cache_path"] = None if trajopt_cache_path is None else str(Path(trajopt_cache_path))
        payload["trajopt_cache_hit"] = bool(cache_hit)
        return payload

    @classmethod
    def solve_record(
        cls,
        record: dict[str, Any],
        safe_boxes: list[np.ndarray],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        budget: float,
        window_alpha: float,
        epsilon: float,
        planner_key: str | None = None,
        mrmp_cache_path: Path | None = None,
        trajopt_output_path: Path | None = None,
    ) -> dict[str, Any]:
        selected_planner_key = cls.PLANNER_KEY if planner_key is None else str(planner_key)
        if selected_planner_key not in {cls.MRMP_PLANNER_KEY, cls.PLANNER_KEY}:
            raise ValueError(f"Unknown planner_key: {selected_planner_key!r}")

        mrmp = cls.solve_mrmp_record(
            record,
            safe_boxes,
            robot_radius,
            vlimit,
            tmax,
            budget,
            window_alpha,
            epsilon,
            mrmp_cache_path=mrmp_cache_path,
        )
        if selected_planner_key == cls.MRMP_PLANNER_KEY or not bool(mrmp.result.success):
            return mrmp.payload

        return cls.solve_trajopt_from_mrmp(
            record,
            mrmp,
            robot_radius,
            vlimit,
            tmax,
            budget,
            window_alpha,
            epsilon,
            trajopt_cache_path=trajopt_output_path,
        )

    @classmethod
    def run(cls) -> None:
        args = cls.parse_args()
        cls.validate_args(args)
        safe_boxes, static_obstacles, geometry_metadata, record = cls.build_demo_state()
        budget = float(args.solution_budget)
        solution_output = Path(args.solution_output)
        trajopt_output = Path(args.trajopt_output)

        mrmp_payload = None
        mrmp = None
        if not bool(args.force_solve):
            mrmp_payload = cls.load_solution_payload(
                solution_output,
                record,
                cls.MRMP_PLANNER_KEY,
                budget,
            )
        if mrmp_payload is None:
            print(f"Running {record['instance_id']} with {cls.MRMP_PLANNER_NAME}")
            mrmp = cls.solve_mrmp_record(
                record,
                safe_boxes,
                robot_radius=cls.DEFAULT_ROBOT_RADIUS,
                vlimit=cls.DEFAULT_VLIMIT,
                tmax=cls.DEFAULT_TMAX,
                budget=budget,
                window_alpha=float(args.window_alpha),
                epsilon=float(args.epsilon),
                mrmp_cache_path=Path(args.mrmp_cache),
            )
            mrmp_payload = mrmp.payload
            cls.validate_solution_payload(mrmp_payload, record, cls.MRMP_PLANNER_KEY, budget)
            cls.write_json_atomic(solution_output, mrmp_payload)

        trajopt_payload = None
        if bool(args.trajopt):
            if not bool(args.force_trajopt):
                trajopt_payload = cls.load_solution_payload(
                    trajopt_output,
                    record,
                    cls.PLANNER_KEY,
                    budget,
                )
            if trajopt_payload is None:
                if mrmp is None:
                    mrmp = cls.solve_mrmp_record(
                        record,
                        safe_boxes,
                        robot_radius=cls.DEFAULT_ROBOT_RADIUS,
                        vlimit=cls.DEFAULT_VLIMIT,
                        tmax=cls.DEFAULT_TMAX,
                        budget=budget,
                        window_alpha=float(args.window_alpha),
                        epsilon=float(args.epsilon),
                        mrmp_cache_path=Path(args.mrmp_cache),
                    )
                print(f"Running trajopt for {record['instance_id']}")
                trajopt_payload = cls.solve_trajopt_from_mrmp(
                    record,
                    mrmp,
                    robot_radius=cls.DEFAULT_ROBOT_RADIUS,
                    vlimit=cls.DEFAULT_VLIMIT,
                    tmax=cls.DEFAULT_TMAX,
                    budget=budget,
                    window_alpha=float(args.window_alpha),
                    epsilon=float(args.epsilon),
                    trajopt_cache_path=Path(args.trajopt_cache),
                )
                cls.validate_solution_payload(trajopt_payload, record, cls.PLANNER_KEY, budget)
                cls.write_json_atomic(trajopt_output, trajopt_payload)

        payload = cls.build_viewer_manifest(
            record=record,
            safe_boxes=safe_boxes,
            static_obstacles=static_obstacles,
            robot_radius=cls.DEFAULT_ROBOT_RADIUS,
            geometry_metadata=geometry_metadata,
        )
        payload["instances"][0]["viewer_layer_defaults"] = dict(cls.VIEWER_LAYER_DEFAULTS)
        payload["solutions"] = [mrmp_payload]
        if trajopt_payload is not None:
            payload["solutions"].append(trajopt_payload)

        summary = {
            "instance_id": record["instance_id"],
            "base_manifest": os.path.relpath(
                Path(args.base_root) / cls.BASE_DOMAIN_KEY / "manifest.json",
                ROOT,
            ),
            "manifest": os.path.relpath(Path(args.manifest), ROOT),
            "solution": os.path.relpath(solution_output, ROOT),
            "success": bool(mrmp_payload["is_success"]),
            "runtime": mrmp_payload["runtime"],
            "sum_of_costs": mrmp_payload["cost"],
            "makespan": mrmp_payload["makespan"],
            "completed_agents": int(mrmp_payload["num_completed_agents"]),
            "num_safe_boxes": int(geometry_metadata["num_safe_boxes"]),
            "num_static_obstacles": int(geometry_metadata["num_static_obstacles"]),
        }
        if trajopt_payload is not None:
            summary.update(
                {
                    "trajopt_solution": os.path.relpath(trajopt_output, ROOT),
                    "trajopt_runtime": trajopt_payload["runtime"],
                    "trajopt_sum_of_costs": trajopt_payload["cost"],
                    "trajopt_makespan": trajopt_payload["makespan"],
                }
            )
        print_json(summary)

        if not bool(args.no_serve):
            cls.serve_manifest(
                payload,
                record,
                safe_boxes,
                Path(args.solution_output),
                host=str(args.host),
                port=int(args.port),
                robot_radius=cls.DEFAULT_ROBOT_RADIUS,
                vlimit=cls.DEFAULT_VLIMIT,
                tmax=cls.DEFAULT_TMAX,
                solution_budget=budget,
                window_alpha=float(args.window_alpha),
                epsilon=float(args.epsilon),
                mrmp_cache_path=Path(args.mrmp_cache),
                trajopt_output_path=Path(args.trajopt_cache),
            )


if __name__ == "__main__":
    DemoR48Village.run()
