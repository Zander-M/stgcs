from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import numpy as np

from stgcs.trajectory import STTrajectory
from demos.trajopt.optimization import GlobalTrajOptConfig, GlobalTrajOptResult
from stgcs.windowed_coordination import WindowedCoordinationReturn
from visualization.viewer.static_assets import ViewerStaticAssets


class Village3DVisualizationMixin:
    """Viewer manifest, serving, and export helpers for the 3D multi-UAV demo."""

    @staticmethod
    def finite_float_or_none(value: float | int | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def trajectory_to_viewer_json(solution: STTrajectory | None) -> dict[str, Any]:
        if solution is None or solution.dim <= 1 or not solution.points:
            return {
                "dimension": 0 if solution is None else max(int(solution.dim) - 1, 0),
                "segments": [],
                "path": [],
                "time_range": [0.0, 0.0],
            }

        segments: list[dict[str, Any]] = []
        path: list[list[float]] = []
        min_time = math.inf
        max_time = -math.inf
        for point in solution.points:
            values = np.asarray(point, dtype=float).reshape(-1)
            dim = int(solution.dim)
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

        return {
            "dimension": int(solution.dim) - 1,
            "segments": segments,
            "path": path,
            "time_range": [float(min_time), float(max_time)],
        }

    @classmethod
    def global_trajopt_result_to_json(
        cls,
        record: dict[str, Any],
        global_result: GlobalTrajOptResult,
        global_config: GlobalTrajOptConfig,
        global_window_span: float,
        global_stride: float,
        continuous_pairwise_collision_free: bool | None,
        continuous_environment_collision_free: bool | None,
    ) -> dict[str, Any]:
        return {
            "mode": "sliding_window",
            "sample_dt": float(global_config.sample_dt),
            "window_span": float(global_window_span),
            "stride": float(global_stride),
            "min_sample_dt": float(global_config.min_sample_dt),
            "velocity_gradient_weight": float(global_config.velocity_gradient_weight),
            "displacement_weight": float(global_config.displacement_weight),
            "displacement_jitter_weight": float(global_config.displacement_jitter_weight),
            "clearance_margin": float(global_config.clearance_margin),
            "spatial_containment_tolerance": float(global_config.spatial_containment_tolerance),
            "spatial_containment_soft_margin": float(global_config.spatial_containment_soft_margin),
            "constraint_slack_penalty": float(global_config.constraint_slack_penalty),
            "constraint_slack_limit": float(global_config.constraint_slack_limit),
            "validation_tolerance": float(cls.DEFAULT_GLOBAL_TRAJOPT_VALIDATION_TOLERANCE),
            "include_interval_collision_constraints": bool(
                global_config.include_interval_collision_constraints
            ),
            "refinement_passes": int(cls.DEFAULT_GLOBAL_TRAJOPT_REFINEMENT_PASSES),
            "fix_terminal_states": bool(global_config.fix_terminal_states),
            "include_trajectory_knot_times": bool(global_config.include_trajectory_knot_times),
            "solver_eps_abs": float(global_config.solver_eps_abs),
            "solver_eps_rel": float(global_config.solver_eps_rel),
            "runtime": cls.finite_float_or_none(global_result.runtime),
            "num_sample_times": int(global_result.times.size),
            "sampled_pairwise_collision_free": bool(global_result.sampled_pairwise_collision_free),
            "continuous_pairwise_collision_free": continuous_pairwise_collision_free,
            "sampled_environment_collision_free": global_result.sampled_environment_collision_free,
            "continuous_environment_collision_free": continuous_environment_collision_free,
            "min_pairwise_distance": float(global_result.min_pairwise_distance),
            "max_velocity_component": float(global_result.max_velocity_component),
            "velocity_limit": float(record["queries"][0]["vlimit"]) if record["queries"] else None,
            "solver_message": global_result.solver_message,
        }

    @classmethod
    def solution_to_viewer_payload(
        cls,
        record: dict[str, Any],
        solutions: list[STTrajectory],
        result: WindowedCoordinationReturn,
        budget: float,
        window_span: float,
        epsilon: float,
        *,
        planner_key: str | None = None,
        planner_name: str | None = None,
        global_result: GlobalTrajOptResult | None = None,
        global_config: GlobalTrajOptConfig | None = None,
        global_window_span: float | None = None,
        global_stride: float | None = None,
        continuous_pairwise_collision_free: bool | None = None,
        continuous_environment_collision_free: bool | None = None,
    ) -> dict[str, Any]:
        selected_planner_key = cls.MRMP_PLANNER_KEY if planner_key is None else planner_key
        selected_planner_name = cls.MRMP_PLANNER_NAME if planner_name is None else planner_name
        is_success = bool(result.success)
        output_solutions = global_result.trajectories if is_success and global_result is not None else solutions
        trajectories = [
            cls.trajectory_to_viewer_json(solution)
            for solution in output_solutions
        ] if is_success else []
        durations = [float(solution.duration) for solution in output_solutions] if is_success else []
        mrmp_trajectories = [
            cls.trajectory_to_viewer_json(solution)
            for solution in solutions
        ] if is_success else []
        mrmp_durations = [float(solution.duration) for solution in solutions] if is_success else []
        mrmp_runtime = float(result.runtime[0])
        global_runtime = 0.0 if global_result is None else float(global_result.runtime)
        return {
            "ok": True,
            "instance_id": record["instance_id"],
            "planner_key": selected_planner_key,
            "planner_name": selected_planner_name,
            "budget": cls.finite_float_or_none(budget),
            "status": "SUCCESS" if is_success else "FAIL",
            "is_success": is_success,
            "runtime": cls.finite_float_or_none(mrmp_runtime + global_runtime),
            "cost": cls.finite_float_or_none(sum(durations) if is_success else math.inf),
            "makespan": cls.finite_float_or_none(max(durations, default=math.inf) if is_success else math.inf),
            "num_agents": int(record["num_agents"]),
            "num_completed_agents": len(trajectories),
            "path_alpha": 0.98,
            "body_alpha": 0.82,
            "body_model": "uav",
            "uav_materials": cls.UAV_MATERIALS,
            "mrmp_planner_key": cls.MRMP_PLANNER_KEY,
            "mrmp_planner_name": cls.MRMP_PLANNER_NAME,
            "mrmp_runtime": cls.finite_float_or_none(mrmp_runtime),
            "mrmp_cost": cls.finite_float_or_none(sum(mrmp_durations) if is_success else math.inf),
            "mrmp_makespan": cls.finite_float_or_none(
                max(mrmp_durations, default=math.inf) if is_success else math.inf
            ),
            "mrmp_trajectory": mrmp_trajectories[0] if mrmp_trajectories else cls.trajectory_to_viewer_json(None),
            "mrmp_trajectories": mrmp_trajectories,
            "window_alpha": float(window_span * record["queries"][0]["vlimit"] / record["robot_radius"]),
            "window_span": float(window_span),
            "dynamic_window_adjustment": True,
            "epsilon": float(epsilon),
            "low_level_heuristic": "Max",
            "domination_checks": ["GUB", "IPC"],
            "coordination": "windowed-pbs",
            "child_expansion_mode": cls.CHILD_EXPANSION_MODE.name,
            "pbs_calls": int(result.wpbs[0]),
            "pbs_runtime": cls.finite_float_or_none(float(result.wpbs[1])),
            "pbs_popped_nodes": int(result.pbs_popped_nodes),
            "pbs_generated_children": int(result.pbs_generated_children),
            "pbs_update_calls": int(result.pbs_update_calls),
            "mp_calls": int(result.mp[0]),
            "mp_runtime": cls.finite_float_or_none(float(result.mp[1])),
            "ecd_calls": int(result.ecd[0]),
            "ecd_runtime": cls.finite_float_or_none(float(result.ecd[1])),
            "cc_calls": int(result.cc[0]),
            "cc_runtime": cls.finite_float_or_none(float(result.cc[1])),
            "trajectory": trajectories[0] if trajectories else cls.trajectory_to_viewer_json(None),
            "trajectories": trajectories,
            "global_trajopt": (
                None
                if (
                    global_result is None
                    or global_config is None
                    or global_window_span is None
                    or global_stride is None
                )
                else cls.global_trajopt_result_to_json(
                    record,
                    global_result,
                    global_config,
                    global_window_span,
                    global_stride,
                    continuous_pairwise_collision_free,
                    continuous_environment_collision_free,
                )
            ),
        }

    @classmethod
    def build_viewer_manifest(
        cls,
        record: dict[str, Any],
        safe_boxes: list[np.ndarray],
        static_obstacles: list[dict[str, Any]],
        robot_radius: float,
        geometry_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        bounds = {
            "min": [0.0, 0.0, 0.0],
            "max": [
                float(record["village_side"]),
                float(record["village_side"]),
                float(record["village_height"]),
            ],
        }
        instance = {
            "instance_id": record["instance_id"],
            "base_instance_id": record["base_instance_id"],
            "problem_kind": "mrmp",
            "domain": "fastpathplanning-village3d",
            "space_dim": 3,
            "spatial_seed": int(record["spatial_seed"]),
            "env_params": {
                "village_side": int(record["village_side"]),
                "village_height": float(record["village_height"]),
                "building_every": int(record["building_every"]),
                "decoration_density": float(record["decoration_density"]),
                "building_clearance_margin": float(record["building_clearance_margin"]),
                "building_cspace_model": str(record["building_cspace_model"]),
                "cspace_simplification_model": str(record["cspace_simplification_model"]),
                "source_repo": "cvxgrp/fastpathplanning",
            },
            "traffic_family": record["traffic_family"],
            "traffic_tier": record["traffic_tier"],
            "stgcs_num_vertices": int(record["stgcs_num_vertices"]),
            "stgcs_num_edges": int(record["stgcs_num_edges"]),
            "dimension": 3,
            "robot_radius": float(robot_radius),
            "agent_model": "uav",
            "uav_materials": cls.UAV_MATERIALS,
            "uav_visual_scale": cls.UAV_VISUAL_SCALE,
            "bounds": bounds,
            "spatial_sets": [
                {
                    "id": f"v{idx}",
                    "vertices": np.asarray(vertices, dtype=float).tolist(),
                    "faces": cls.box_faces(),
                    "edges": cls.box_edges(),
                }
                for idx, vertices in enumerate(safe_boxes)
            ],
            "static_obstacles": static_obstacles,
            "time_horizon": 0.0,
            "time_range": [0.0, 0.0],
            "queries": record["queries"],
            "dynamic_obstacles": [],
            "num_agents": int(record["num_agents"]),
            "independent_cost_sum": float(record["independent_cost_sum"]),
            "independent_makespan": float(record["independent_makespan"]),
            "independent_runtime": float(record["independent_runtime"]),
            "num_conflicting_pairs": int(record["num_conflicting_pairs"]),
            "num_conflicting_agents": int(record["num_conflicting_agents"]),
            "conflict_largest_component": int(record["conflict_largest_component"]),
            "conflict_density": float(record["conflict_density"]),
            "cspace_alpha": cls.CSPACE_ALPHA,
            "static_obstacle_alpha": cls.STATIC_OBSTACLE_ALPHA,
            "geometry_metadata": geometry_metadata,
        }
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": "mrmp",
            "instance_count": 1,
            "instances": [instance],
        }

    @staticmethod
    def write_manifest(output_path: Path, payload: dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def meshcat_viewer_class(cls) -> Any:
        from visualization.village_3d_meshcat import Village3DMeshcatViewer

        return Village3DMeshcatViewer

    @classmethod
    def third_person_video_output_path(cls, manifest_path: Path, agent_index: int) -> Path:
        return Path(manifest_path).parent / cls.DEFAULT_THIRD_PERSON_VIDEO_BASENAME.format(
            agent_index=int(agent_index)
        )

    @classmethod
    def export_third_person_videos(cls, manifest_path: Path) -> list[Path]:
        viewer_cls = cls.meshcat_viewer_class()
        output_paths: list[Path] = []
        for agent_index in cls.DEFAULT_THIRD_PERSON_VIDEO_AGENT_INDICES:
            output_path = cls.third_person_video_output_path(manifest_path, agent_index)
            export_args = SimpleNamespace(
                manifest=Path(manifest_path),
                animate_mode="trajopt",
                camera_mode="third_person",
                third_person_agent_index=int(agent_index),
                third_person_lookahead_seconds=viewer_cls.DEFAULT_THIRD_PERSON_LOOKAHEAD_SECONDS,
                third_person_camera_distance=viewer_cls.DEFAULT_THIRD_PERSON_CAMERA_DISTANCE,
                third_person_camera_up_offset=viewer_cls.DEFAULT_THIRD_PERSON_CAMERA_UP_OFFSET,
                third_person_camera_target_forward_offset=(
                    viewer_cls.DEFAULT_THIRD_PERSON_CAMERA_TARGET_FORWARD_OFFSET
                ),
                third_person_camera_target_up_offset=(
                    viewer_cls.DEFAULT_THIRD_PERSON_CAMERA_TARGET_UP_OFFSET
                ),
                third_person_camera_fov=viewer_cls.DEFAULT_THIRD_PERSON_CAMERA_FOV,
                framerate=viewer_cls.DEFAULT_FRAMERATE,
                video_output=output_path,
                video_width=viewer_cls.DEFAULT_VIDEO_WIDTH,
                video_height=viewer_cls.DEFAULT_VIDEO_HEIGHT,
                video_bitrate=viewer_cls.DEFAULT_VIDEO_BITRATE,
            )
            output_paths.append(viewer_cls.export_video_from_args(export_args))
        return output_paths

    @classmethod
    def planner_options(cls) -> list[dict[str, Any]]:
        return [
            {
                "key": cls.MRMP_PLANNER_KEY,
                "name": cls.MRMP_PLANNER_NAME,
                "button_label": "MRMP planner",
                "space_dims": [3],
            },
            {
                "key": cls.PLANNER_KEY,
                "name": cls.PLANNER_NAME,
                "button_label": "MRMP + trajopt",
                "space_dims": [3],
            },
        ]

    @classmethod
    def planner_name(cls, planner_key: str) -> str:
        for option in cls.planner_options():
            if option["key"] == planner_key:
                return str(option["name"])
        raise ValueError(f"Unknown planner_key: {planner_key!r}")

    @classmethod
    def planner_payload(cls, solution_budget: float, window_alpha: float, epsilon: float) -> dict[str, Any]:
        return {
            "ok": True,
            "default_budget": float(solution_budget),
            "default_window_alpha": float(window_alpha),
            "default_window_beta": 1.0,
            "default_epsilon": float(epsilon),
            "planners": {
                "mrmp": cls.planner_options(),
                "st_heuristic_ablation": [],
                "base": [],
            },
        }

    @classmethod
    def serve_manifest(
        cls,
        payload: dict[str, Any],
        record: dict[str, Any],
        safe_boxes: list[np.ndarray],
        output_path: Path,
        host: str,
        port: int,
        robot_radius: float,
        vlimit: float,
        tmax: float,
        solution_budget: float,
        window_alpha: float,
        epsilon: float,
        mrmp_cache_path: Path | None,
        trajopt_output_path: Path | None,
    ) -> None:
        payload_bytes = json.dumps(payload).encode("utf-8")
        planner_payload_bytes = json.dumps(
            cls.planner_payload(solution_budget, window_alpha, epsilon)
        ).encode("utf-8")
        viewer_url = ViewerStaticAssets.viewer_url()

        class Handler(SimpleHTTPRequestHandler):
            def end_headers(self) -> None:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def do_HEAD(self) -> None:
                parsed = urlparse(self.path)
                if ViewerStaticAssets.is_viewer_path(parsed.path):
                    ViewerStaticAssets.write_html_response(self)
                    return
                if parsed.path == "/":
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", viewer_url)
                    self.end_headers()
                    return
                return super().do_HEAD()

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if ViewerStaticAssets.is_viewer_path(parsed.path):
                    ViewerStaticAssets.write_html_response(self)
                    return
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
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/solution":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                    request = json.loads(body.decode("utf-8"))
                    if request.get("instance_id") != record["instance_id"]:
                        raise ValueError(f"Unknown instance_id: {request.get('instance_id')!r}")
                    planner_key = str(request.get("planner_key", cls.PLANNER_KEY))
                    cls.planner_name(planner_key)
                    budget = float(request.get("budget", solution_budget))
                    request_window_alpha = float(request.get("window_alpha", window_alpha))
                    request_epsilon = float(request.get("epsilon", epsilon))
                    if budget <= 0.0:
                        raise ValueError(f"Budget must be positive, got {budget}.")
                    if request_window_alpha <= 0.0:
                        raise ValueError(f"window_alpha must be positive, got {request_window_alpha}.")
                    if request_epsilon <= 0.0:
                        raise ValueError(f"epsilon must be positive, got {request_epsilon}.")
                    response_payload = cls.solve_record(
                        record=record.copy(),
                        safe_boxes=safe_boxes,
                        robot_radius=float(robot_radius),
                        vlimit=float(vlimit),
                        tmax=float(tmax),
                        budget=budget,
                        window_alpha=request_window_alpha,
                        epsilon=request_epsilon,
                        planner_key=planner_key,
                        mrmp_cache_path=mrmp_cache_path,
                        trajopt_output_path=trajopt_output_path,
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
                    "output": str(output_path),
                    "instance_id": payload["instances"][0]["instance_id"],
                    "planner": cls.PLANNER_NAME,
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
