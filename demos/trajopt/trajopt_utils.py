from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.manifests.mrmp import MRMPBenchmarkRecord
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from benchmark.planners.mrmp import SearchPlannerSpec
from experiments.mrmp_runners.common import MRMPExperiment
from stgcs.pbs import ChildExpansionMode, PriorityBasedSearch
from stgcs.trajectory import STTrajectory
from demos.trajopt.optimization import GlobalTrajOptConfig, GlobalTrajectoryOptimizer
from visualization.viewer.solution_visualization import SolutionVisualizationService, ViewerMRMPSolutionRun


STORED_PWL_TIME_ORDER_TOLERANCE = 1e-6
STORED_PWL_POSITION_CONTINUITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MRMPPlannerSettings:
    child_expansion_rules: Mapping[str, ChildExpansionMode]
    dynamic_window_adjustment: bool


@dataclass(frozen=True)
class TrajoptSettings:
    sample_dt_factor: float
    window_span_factor: float
    stride_factor: float
    velocity_gradient_weight: float
    displacement_weight: float
    displacement_jitter_weight: float
    solver_max_iter: int
    solver_eps: float
    clearance_margin: float
    include_trajectory_knot_times: bool
    collision_tolerance: float


def validate_solution_payload(
    payload: dict[str, Any],
    record: MRMPBenchmarkRecord,
    expected_planner_key: str,
    budget: float,
) -> None:
    if not bool(payload.get("ok")):
        raise ValueError("Saved solution payload is not ok.")
    if str(payload.get("instance_id")) != record.instance_id:
        raise ValueError(
            f"Saved solution belongs to {payload.get('instance_id')!r}, expected {record.instance_id!r}."
        )
    if str(payload.get("planner_key")) != expected_planner_key:
        raise ValueError(
            f"Saved solution planner is {payload.get('planner_key')!r}, expected {expected_planner_key!r}."
        )
    if abs(float(payload.get("budget")) - float(budget)) > 1e-9:
        raise ValueError(f"Saved solution budget does not match {float(budget):g}.")
    if not bool(payload.get("is_success")):
        completed = int(payload.get("num_completed_agents", 0))
        raise ValueError(
            f"Saved MRMP solution is not successful: completed {completed}/{int(record.num_agents)} agents."
        )
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != int(record.num_agents):
        count = 0 if not isinstance(trajectories, list) else len(trajectories)
        raise ValueError(f"Saved MRMP solution has {count} trajectories, expected {int(record.num_agents)}.")


def run_mrmp(
    record: MRMPBenchmarkRecord,
    base_manifest_path: Path,
    base_record,
    low_level_spec: SearchPlannerSpec,
    planner_key: str,
    planner_name: str,
    planner_settings: MRMPPlannerSettings,
    budget: float,
    window_alpha: float,
    window_beta: float,
    child_expansion_rule: str,
) -> dict[str, Any]:
    instance = queries = None
    try:
        instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
        BaseOfflineHeuristicStore.prepare_instance_for_search(
            instance,
            base_manifest_path,
            base_record,
            required_heuristics={low_level_spec.heuristic},
            online_td_timeout_secs=float(budget),
        )
        solutions, entry = MRMPExperiment.run_windowed_pbs_spec_with_solutions(
            instance,
            queries,
            low_level_spec,
            budget=float(budget),
            window_span_factor=float(window_alpha),
            dynamic_window_adjustment=bool(planner_settings.dynamic_window_adjustment),
            child_expansion_mode=planner_settings.child_expansion_rules[child_expansion_rule],
            execution_horizon_factor=float(window_beta),
        )
        payload = SolutionVisualizationService.mrmp_run_to_viewer_json(
            record,
            ViewerMRMPSolutionRun(
                planner_key=planner_key,
                planner_name=planner_name,
                budget=float(budget),
                status="SUCCESS" if entry.is_success else "FAIL",
                entry=entry,
                solutions=solutions,
            ),
        )
        payload.update(
            {
                "window_alpha": float(window_alpha),
                "window_beta": float(window_beta),
                "epsilon": float(low_level_spec.epsilon),
                "child_expansion_rule": str(child_expansion_rule),
                "dynamic_window_adjustment": bool(planner_settings.dynamic_window_adjustment),
            }
        )
        return payload
    finally:
        del queries
        del instance
        gc.collect()


def trajectory_from_viewer_json(payload: dict[str, Any]) -> STTrajectory:
    dim = int(payload.get("dimension", 0))
    segments = payload.get("segments")
    if dim <= 0 or not isinstance(segments, list) or not segments:
        raise ValueError("Stored PWL trajectory must contain drawable segments.")

    vertex_path: list[str] = []
    points: list[np.ndarray] = []
    expected_spatial_shape = (dim,)
    last_time = -math.inf
    last_goal: np.ndarray | None = None
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError("Stored PWL trajectory segment must be a JSON object.")
        start = np.asarray(segment["start"], dtype=float).reshape(-1)
        goal = np.asarray(segment["goal"], dtype=float).reshape(-1)
        if start.shape != expected_spatial_shape or goal.shape != expected_spatial_shape:
            raise ValueError(
                f"Stored PWL trajectory segment has shape {start.shape}->{goal.shape}, "
                f"expected {expected_spatial_shape}."
            )
        t_start = float(segment["t_start"])
        t_end = float(segment["t_end"])
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            raise ValueError("Stored PWL trajectory contains a non-finite spatial coordinate.")
        if (
            not math.isfinite(t_start)
            or not math.isfinite(t_end)
            or t_end + STORED_PWL_TIME_ORDER_TOLERANCE < t_start
        ):
            raise ValueError("Stored PWL trajectory contains an invalid time interval.")
        if t_end < t_start:
            t_end = t_start
        if t_start + STORED_PWL_TIME_ORDER_TOLERANCE < last_time:
            raise ValueError("Stored PWL trajectory segments are not time ordered.")
        if t_start < last_time:
            if last_goal is None or not np.allclose(
                start,
                last_goal,
                rtol=0.0,
                atol=STORED_PWL_POSITION_CONTINUITY_TOLERANCE,
            ):
                raise ValueError("Stored PWL trajectory has a tiny time overlap but is spatially discontinuous.")
            t_start = last_time
        vertex_path.append(f"stored-pwl-segment-{idx}")
        points.append(np.hstack([start, t_start, goal, t_end]))
        last_time = t_end
        last_goal = goal
    return STTrajectory(vertex_path, points, dim=dim)


def pwl_trajectories_from_solution_payload(
    payload: dict[str, Any],
    expected_num_agents: int,
) -> list[STTrajectory]:
    trajectory_payloads = payload.get("pwl_trajectories")
    if not isinstance(trajectory_payloads, list) or not trajectory_payloads:
        trajectory_payloads = payload.get("trajectories")
    if not isinstance(trajectory_payloads, list) or not trajectory_payloads:
        raise ValueError("Stored PWL solution payload does not contain trajectories.")

    trajectories = [trajectory_from_viewer_json(trajectory_payload) for trajectory_payload in trajectory_payloads]
    if len(trajectories) != int(expected_num_agents):
        raise ValueError(
            f"Stored PWL trajectory count {len(trajectories)} does not match "
            f"num_agents={int(expected_num_agents)}."
        )
    return trajectories


def global_trajopt_time_scale(robot_radius: float, vlimit: float) -> float:
    return float(robot_radius) / float(vlimit)


def global_trajopt_window_span(settings: TrajoptSettings, robot_radius: float, vlimit: float) -> float:
    return float(settings.window_span_factor) * global_trajopt_time_scale(robot_radius, vlimit)


def global_trajopt_stride(settings: TrajoptSettings, robot_radius: float, vlimit: float) -> float:
    return float(settings.stride_factor) * global_trajopt_time_scale(robot_radius, vlimit)


def global_trajopt_config(settings: TrajoptSettings, robot_radius: float, vlimit: float) -> GlobalTrajOptConfig:
    time_scale = global_trajopt_time_scale(robot_radius, vlimit)
    return GlobalTrajOptConfig(
        sample_dt=float(settings.sample_dt_factor) * time_scale,
        velocity_gradient_weight=float(settings.velocity_gradient_weight),
        displacement_weight=float(settings.displacement_weight),
        displacement_jitter_weight=float(settings.displacement_jitter_weight),
        clearance_margin=float(settings.clearance_margin),
        solver_max_iter=int(settings.solver_max_iter),
        solver_eps_abs=float(settings.solver_eps),
        solver_eps_rel=float(settings.solver_eps),
        include_trajectory_knot_times=bool(settings.include_trajectory_knot_times),
        progress=True,
    )


def pairwise_collision_free(
    stgcs: Any,
    trajectories: list[STTrajectory],
    robot_radius: float,
    goal_stays: list[bool],
    collision_tolerance: float,
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
            tolerance=float(collision_tolerance),
        ):
            return False
    return True


def trajectory_knot_times(trajectory: STTrajectory) -> np.ndarray:
    knot_times: list[float] = []
    for idx in range(trajectory.size):
        knot_times.append(float(trajectory.xA(idx)[-1]))
        knot_times.append(float(trajectory.xB(idx)[-1]))
    return np.asarray(sorted(set(round(time_value, 9) for time_value in knot_times)), dtype=float)


def global_trajopt_uniform_times(
    trajectories: Sequence[STTrajectory],
    config: GlobalTrajOptConfig,
) -> np.ndarray:
    t0 = min(float(trajectory.x0[-1]) for trajectory in trajectories)
    tf = max(float(trajectory.xT[-1]) for trajectory in trajectories)
    count = max(1, int(math.ceil((tf - t0) / float(config.sample_dt))))
    return np.linspace(t0, tf, count + 1)


def reference_times_for_global_trajopt(
    env: Any,
    stgcs: Any | None,
    trajectory: STTrajectory,
    uniform_times: np.ndarray,
) -> np.ndarray:
    x0_time = float(trajectory.x0[-1])
    xT_time = float(trajectory.xT[-1])
    selected = [
        float(time_value)
        for time_value in uniform_times
        if time_value >= x0_time - 1e-9 and time_value <= xT_time + 1e-9
    ]
    selected.extend([x0_time, xT_time])
    knot_times = trajectory_knot_times(trajectory)

    changed = True
    while changed:
        changed = False
        ordered = np.asarray(sorted(set(round(float(time_value), 9) for time_value in selected)), dtype=float)
        expanded = list(ordered)
        for left_time, right_time in zip(ordered[:-1], ordered[1:]):
            left = trajectory.lerp(float(left_time))
            right = trajectory.lerp(float(right_time))
            if stgcs is not None:
                needs_refinement = not reference_segment_has_spatial_gcs_cell(
                    stgcs,
                    trajectory,
                    float(left_time),
                    float(right_time),
                )
            else:
                needs_refinement = env.collision_checking_seg(
                    left[:-1],
                    right[:-1],
                    float(left_time),
                    float(right_time),
                )
            if not needs_refinement:
                continue
            interior_knots = knot_times[
                (knot_times > float(left_time) + 1e-9)
                & (knot_times < float(right_time) - 1e-9)
            ]
            if interior_knots.size > 0:
                expanded.extend(float(time_value) for time_value in interior_knots)
                changed = True
        selected = expanded

    return np.asarray(sorted(set(round(float(time_value), 9) for time_value in selected)), dtype=float)


def reference_segment_has_spatial_gcs_cell(
    stgcs: Any,
    trajectory: STTrajectory,
    left_time: float,
    right_time: float,
) -> bool:
    left = trajectory.lerp(float(left_time))
    right = trajectory.lerp(float(right_time))
    mid_time = 0.5 * (float(left_time) + float(right_time))
    probes = [left, right, trajectory.lerp(mid_time)]
    try:
        source_segment_idx = trajectory.find_segment_index(mid_time)
        vertex_name = trajectory.vertex_path[source_segment_idx]
        vertex = stgcs.get_vertex(vertex_name)
        if vertex is not None and all(vertex.st_hpoly.PointInSet(point, 1e-7) for point in probes):
            return True
    except KeyError:
        pass

    for candidate_name in stgcs.G.nodes:
        if candidate_name in {"source", "target"} or str(candidate_name).startswith("sub-target"):
            continue
        candidate = stgcs.get_vertex(candidate_name)
        if candidate is not None and all(candidate.st_hpoly.PointInSet(point, 1e-7) for point in probes):
            return True
    return False


def global_trajopt_reference_trajectories(
    instance: Any,
    trajectories: Sequence[STTrajectory],
    config: GlobalTrajOptConfig,
) -> list[STTrajectory]:
    uniform_times = global_trajopt_uniform_times(trajectories, config)
    stgcs = (
        instance.stgcs
        if GlobalTrajectoryOptimizer._has_spatial_gcs_containment(instance.stgcs)
        else None
    )
    reference_trajectories: list[STTrajectory] = []
    for trajectory in trajectories:
        times = reference_times_for_global_trajopt(
            instance.env,
            stgcs,
            trajectory,
            uniform_times,
        )
        states = [trajectory.lerp(float(time_value)) for time_value in times]
        points = [
            np.hstack([states[idx], states[idx + 1]])
            for idx in range(len(states) - 1)
            if float(states[idx + 1][-1] - states[idx][-1]) > 1e-9
        ]
        vertex_path = [
            trajectory.vertex_path[
                trajectory.find_segment_index(0.5 * (float(times[idx]) + float(times[idx + 1])))
            ]
            for idx in range(len(points))
        ]
        reference_trajectories.append(STTrajectory(vertex_path, points, int(trajectory.dim - 1)))
    return reference_trajectories


def global_trajopt_environment(instance: Any) -> Any | None:
    if GlobalTrajectoryOptimizer._has_spatial_gcs_containment(instance.stgcs):
        return None
    return instance.env


def optimize_pwl_trajectories(
    instance: Any,
    trajectories: list[STTrajectory],
    queries: Sequence[Any],
    settings: TrajoptSettings,
) -> tuple[Any, GlobalTrajOptConfig, bool]:
    robot_radius = float(instance.env.robot_radius)
    vlimit = float(instance.stgcs.vlimit)
    config = global_trajopt_config(settings, robot_radius, vlimit)
    reference_trajectories = global_trajopt_reference_trajectories(
        instance,
        trajectories,
        config,
    )
    result = GlobalTrajectoryOptimizer.optimize_sliding_window(
        reference_trajectories,
        robot_radius,
        vlimit,
        config=config,
        window_span=global_trajopt_window_span(settings, robot_radius, vlimit),
        stride=global_trajopt_stride(settings, robot_radius, vlimit),
        stgcs=instance.stgcs,
        env=global_trajopt_environment(instance),
    )
    if not result.sampled_pairwise_collision_free:
        raise RuntimeError("Global trajectory optimization produced a sampled pairwise collision.")
    continuous_pairwise_ok = pairwise_collision_free(
        instance.stgcs,
        result.trajectories,
        instance.env.robot_radius,
        goal_stays=[query.is_stay for query in queries],
        collision_tolerance=settings.collision_tolerance,
    )
    return result, config, continuous_pairwise_ok


def run_trajopt(
    record: MRMPBenchmarkRecord,
    base_record,
    mrmp_payload: dict[str, Any],
    trajopt_planner_key: str,
    trajopt_planner_name: str,
    settings: TrajoptSettings,
) -> dict[str, Any]:
    trajectories = pwl_trajectories_from_solution_payload(mrmp_payload, expected_num_agents=int(record.num_agents))
    instance = queries = None
    try:
        instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
        robot_radius = float(instance.env.robot_radius)
        vlimit = float(instance.stgcs.vlimit)
        window_span = global_trajopt_window_span(settings, robot_radius, vlimit)
        stride = global_trajopt_stride(settings, robot_radius, vlimit)
        result, config, continuous_pairwise_ok = optimize_pwl_trajectories(
            instance,
            trajectories,
            queries,
            settings,
        )
    finally:
        del queries
        del instance
        gc.collect()

    durations = [float(trajectory.duration) for trajectory in result.trajectories]
    payload = dict(mrmp_payload)
    payload.update(
        {
            "planner_key": trajopt_planner_key,
            "planner_name": trajopt_planner_name,
            "runtime": SolutionVisualizationService.finite_float_or_none(
                float(mrmp_payload["runtime"]) + float(result.runtime)
            ),
            "cost": SolutionVisualizationService.finite_float_or_none(sum(durations)),
            "makespan": SolutionVisualizationService.finite_float_or_none(max(durations, default=math.inf)),
            "trajectories": [
                {
                    **SolutionVisualizationService.trajectory_to_viewer_json(trajectory),
                    "agent_index": agent_idx,
                }
                for agent_idx, trajectory in enumerate(result.trajectories)
            ],
            "source_planner_key": mrmp_payload["planner_key"],
            "source_runtime": mrmp_payload["runtime"],
            "global_trajopt": {
                "mode": "sliding_window",
                "sample_dt": float(config.sample_dt),
                "window_span": float(window_span),
                "stride": float(stride),
                "min_sample_dt": float(config.min_sample_dt),
                "velocity_gradient_weight": float(config.velocity_gradient_weight),
                "displacement_weight": float(config.displacement_weight),
                "displacement_jitter_weight": float(config.displacement_jitter_weight),
                "clearance_margin": float(config.clearance_margin),
                "fix_terminal_states": bool(config.fix_terminal_states),
                "include_trajectory_knot_times": bool(config.include_trajectory_knot_times),
                "solver_eps_abs": float(config.solver_eps_abs),
                "solver_eps_rel": float(config.solver_eps_rel),
                "runtime": SolutionVisualizationService.finite_float_or_none(result.runtime),
                "num_sample_times": int(result.times.size),
                "sampled_pairwise_collision_free": bool(result.sampled_pairwise_collision_free),
                "continuous_pairwise_collision_free": continuous_pairwise_ok,
                "sampled_environment_collision_free": result.sampled_environment_collision_free,
                "continuous_environment_collision_free": result.continuous_environment_collision_free,
                "min_pairwise_distance": float(result.min_pairwise_distance),
                "max_velocity_component": float(result.max_velocity_component),
                "velocity_limit": float(record.queries[0].vlimit) if record.queries else None,
                "solver_message": result.solver_message,
            },
        }
    )
    return payload
