from __future__ import annotations

from dataclasses import dataclass
import math
import time
from itertools import combinations
from typing import Sequence

import numpy as np
from pydrake.all import CommonSolverOption, MathematicalProgram, OsqpSolver, SolverOptions
from scipy import sparse

from environment.obstacle import StaticPolygon, StaticSphere
from stgcs.trajectory import STTrajectory


@dataclass(frozen=True)
class GlobalTrajOptConfig:
    sample_dt: float = 0.02
    min_sample_dt: float = 1e-4
    velocity_gradient_weight: float = 0.05
    displacement_weight: float = 0.2
    displacement_jitter_weight: float = 1e-4
    clearance_margin: float = 1e-6
    spatial_containment_tolerance: float = 0.0
    spatial_containment_soft_margin: float = 0.0
    constraint_slack_penalty: float = 0.0
    constraint_slack_limit: float = math.inf
    include_interval_collision_constraints: bool = False
    fix_terminal_states: bool = True
    include_trajectory_knot_times: bool = True
    solver_max_iter: int = 100000
    solver_eps_abs: float = 1e-6
    solver_eps_rel: float = 1e-6
    solver_print_to_console: bool = False
    progress: bool = False
    round_time_decimals: int = 9


@dataclass(frozen=True)
class GlobalTrajOptResult:
    trajectories: list[STTrajectory]
    times: np.ndarray
    original_positions: np.ndarray
    optimized_positions: np.ndarray
    displacements: np.ndarray
    runtime: float
    sampled_pairwise_collision_free: bool
    sampled_environment_collision_free: bool | None
    continuous_environment_collision_free: bool | None
    min_pairwise_distance: float
    max_velocity_component: float
    solver_message: str


@dataclass(frozen=True)
class _LinearConstraintMatrix:
    A: sparse.csc_matrix
    lb: np.ndarray
    ub: np.ndarray
    soft: np.ndarray | None = None


LinearConstraintRows = list[tuple[np.ndarray, np.ndarray]]


class GlobalTrajectoryOptimizer:
    @classmethod
    def optimize(
        cls,
        trajectories: Sequence[STTrajectory],
        robot_radius: float,
        velocity_limit: float,
        config: GlobalTrajOptConfig | None = None,
        *,
        stgcs: object | None = None,
        env: object | None = None,
    ) -> GlobalTrajOptResult:
        if config is None:
            config = GlobalTrajOptConfig()
        cls._validate_inputs(trajectories, robot_radius, velocity_limit, config, stgcs, env)

        start_time = time.perf_counter()
        times = cls.sample_times(trajectories, config)
        original_positions = cls.sample_positions(trajectories, times)
        optimized_positions, solver_message = cls._optimize_positions_on_samples(
            trajectories,
            times,
            original_positions,
            robot_radius,
            velocity_limit,
            config,
            stgcs,
            env,
        )

        return cls._build_result(
            trajectories,
            times,
            original_positions,
            optimized_positions,
            robot_radius,
            config,
            start_time,
            solver_message,
            env,
        )

    @classmethod
    def optimize_sliding_window(
        cls,
        trajectories: Sequence[STTrajectory],
        robot_radius: float,
        velocity_limit: float,
        config: GlobalTrajOptConfig | None = None,
        *,
        window_span: float,
        stride: float,
        stgcs: object | None = None,
        env: object | None = None,
    ) -> GlobalTrajOptResult:
        if config is None:
            config = GlobalTrajOptConfig()
        cls._validate_inputs(trajectories, robot_radius, velocity_limit, config, stgcs, env)
        cls._validate_sliding_window_params(window_span, stride)

        start_time = time.perf_counter()
        times = cls.sample_times(trajectories, config)
        original_positions = cls.sample_positions(trajectories, times)
        optimized_positions = original_positions.copy()
        solver_messages: list[str] = []
        window_ranges = cls._sliding_window_ranges(times, float(window_span), float(stride))
        cls._log_progress(
            config,
            "sliding-window start: "
            f"agents={len(trajectories)} samples={times.size} "
            f"window_span={float(window_span):.6g} stride={float(stride):.6g} "
            f"windows={len(window_ranges)}",
        )

        for window_idx, (start_idx, window_end_idx, commit_end_idx) in enumerate(window_ranges, start=1):
            window_start_time = time.perf_counter()
            window_slice = slice(start_idx, window_end_idx + 1)
            local_times = times[window_slice]
            local_reference_positions = optimized_positions[:, window_slice, :]
            cls._log_progress(
                config,
                f"window {window_idx}/{len(window_ranges)} start: "
                f"sample_idx=[{start_idx},{window_end_idx}] "
                f"time=[{float(local_times[0]):.6g},{float(local_times[-1]):.6g}] "
                f"samples={local_times.size} vars={local_reference_positions.size} "
                f"commit_time={float(times[commit_end_idx]):.6g}",
            )

            try:
                local_optimized_positions, solver_message = cls._optimize_positions_on_samples(
                    trajectories,
                    local_times,
                    local_reference_positions,
                    robot_radius,
                    velocity_limit,
                    config,
                    stgcs,
                    env,
                    fixed_time_indices=(0, local_times.size - 1),
                )
            except Exception as exc:
                cls._log_progress(
                    config,
                    f"window {window_idx}/{len(window_ranges)} failed after "
                    f"{time.perf_counter() - window_start_time:.2f}s: {type(exc).__name__}: {exc}",
                )
                raise

            local_commit_end_idx = int(commit_end_idx - start_idx)
            optimized_positions[:, start_idx : commit_end_idx + 1, :] = local_optimized_positions[
                :,
                : local_commit_end_idx + 1,
                :,
            ]
            solver_messages.append(solver_message)
            cls._log_progress(
                config,
                f"window {window_idx}/{len(window_ranges)} done: "
                f"runtime={time.perf_counter() - window_start_time:.2f}s solver={solver_message}",
            )

        result = cls._build_result(
            trajectories,
            times,
            original_positions,
            optimized_positions,
            robot_radius,
            config,
            start_time,
            cls._format_sliding_window_solver_message(solver_messages),
            env,
        )
        try:
            cls._enforce_sampled_collision_free(result)
        except Exception as exc:
            cls._log_progress(
                config,
                f"sliding-window failed validation after {result.runtime:.2f}s: {type(exc).__name__}: {exc}",
            )
            raise
        cls._log_progress(
            config,
            "sliding-window done: "
            f"runtime={result.runtime:.2f}s "
            f"min_pairwise_distance={result.min_pairwise_distance:.6g} "
            f"max_velocity_component={result.max_velocity_component:.6g}",
        )
        return result

    @classmethod
    def _optimize_positions_on_samples(
        cls,
        trajectories: Sequence[STTrajectory],
        times: np.ndarray,
        reference_positions: np.ndarray,
        robot_radius: float,
        velocity_limit: float,
        config: GlobalTrajOptConfig,
        stgcs: object | None,
        env: object | None,
        fixed_time_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, str]:
        x0 = np.zeros(reference_positions.size, dtype=float)
        linear_constraints = cls._linear_constraints(
            trajectories,
            times,
            reference_positions,
            robot_radius,
            velocity_limit,
            config,
            stgcs,
            env,
            fixed_time_indices=fixed_time_indices,
        )
        Q, b = cls._quadratic_objective(times, reference_positions, config)
        displacement_x, solver_message = cls._solve_quadratic_program(
            x0,
            Q,
            b,
            linear_constraints,
            config,
        )
        displacements = displacement_x.reshape(reference_positions.shape)
        return reference_positions + displacements, solver_message

    @classmethod
    def _build_result(
        cls,
        source_trajectories: Sequence[STTrajectory],
        times: np.ndarray,
        original_positions: np.ndarray,
        optimized_positions: np.ndarray,
        robot_radius: float,
        config: GlobalTrajOptConfig,
        start_time: float,
        solver_message: str,
        env: object | None,
    ) -> GlobalTrajOptResult:
        displacements = optimized_positions - original_positions
        optimized_trajectories = cls.to_st_trajectories(
            source_trajectories,
            times,
            optimized_positions,
        )
        min_distance = cls.min_pairwise_distance(optimized_positions)
        max_velocity = cls.max_velocity_component(times, optimized_positions)
        min_sep = 2.0 * float(robot_radius) + float(config.clearance_margin)
        sampled_environment_ok = None
        continuous_environment_ok = None
        if env is not None:
            sampled_environment_ok = cls.sampled_environment_collision_free(env, times, optimized_positions)
            continuous_environment_ok = cls.environment_collision_free(env, optimized_trajectories)

        return GlobalTrajOptResult(
            trajectories=optimized_trajectories,
            times=times,
            original_positions=original_positions,
            optimized_positions=optimized_positions,
            displacements=displacements,
            runtime=time.perf_counter() - start_time,
            sampled_pairwise_collision_free=bool(min_distance + 1e-9 >= min_sep),
            sampled_environment_collision_free=sampled_environment_ok,
            continuous_environment_collision_free=continuous_environment_ok,
            min_pairwise_distance=float(min_distance),
            max_velocity_component=float(max_velocity),
            solver_message=solver_message,
        )

    @staticmethod
    def _validate_sliding_window_params(window_span: float, stride: float) -> None:
        if not math.isfinite(float(window_span)) or float(window_span) <= 0.0:
            raise ValueError("window_span must be finite and positive.")
        if not math.isfinite(float(stride)) or float(stride) <= 0.0:
            raise ValueError("stride must be finite and positive.")
        if float(stride) >= float(window_span):
            raise ValueError("stride must be smaller than window_span to provide overlap.")

    @classmethod
    def _sliding_window_ranges(
        cls,
        times: np.ndarray,
        window_span: float,
        stride: float,
    ) -> list[tuple[int, int, int]]:
        ranges: list[tuple[int, int, int]] = []
        start_idx = 0
        while start_idx < times.size - 1:
            window_end_idx = cls._window_end_index(times, start_idx, float(window_span))
            commit_end_idx = min(
                cls._window_end_index(times, start_idx, float(stride)),
                window_end_idx,
            )
            if commit_end_idx <= start_idx:
                commit_end_idx = min(start_idx + 1, window_end_idx)
            ranges.append((start_idx, window_end_idx, commit_end_idx))
            start_idx = commit_end_idx
        return ranges

    @staticmethod
    def _log_progress(config: GlobalTrajOptConfig, message: str) -> None:
        if bool(config.progress):
            print(f"[global_trajopt] {message}", flush=True)

    @staticmethod
    def _window_end_index(times: np.ndarray, start_idx: int, span: float) -> int:
        if start_idx >= times.size - 1:
            return int(start_idx)
        target_time = float(times[start_idx]) + float(span)
        end_idx = int(np.searchsorted(times, target_time + 1e-12, side="right") - 1)
        end_idx = max(end_idx, int(start_idx) + 1)
        return min(end_idx, int(times.size) - 1)

    @staticmethod
    def _format_sliding_window_solver_message(solver_messages: Sequence[str]) -> str:
        if not solver_messages:
            return "sliding_window_windows=0"
        unique_messages = sorted(set(str(message) for message in solver_messages))
        return (
            f"sliding_window_windows={len(solver_messages)}; "
            f"solver_messages={', '.join(unique_messages)}"
        )

    @staticmethod
    def _enforce_sampled_collision_free(result: GlobalTrajOptResult) -> None:
        if not result.sampled_pairwise_collision_free:
            raise RuntimeError("Sliding-window trajectory optimization produced a sampled pairwise collision.")
        if result.sampled_environment_collision_free is False:
            raise RuntimeError("Sliding-window trajectory optimization produced a sampled environment collision.")

    @staticmethod
    def _validate_inputs(
        trajectories: Sequence[STTrajectory],
        robot_radius: float,
        velocity_limit: float,
        config: GlobalTrajOptConfig,
        stgcs: object | None = None,
        env: object | None = None,
    ) -> None:
        if not trajectories:
            raise ValueError("Expected at least one trajectory.")
        if any(trajectory.size == 0 for trajectory in trajectories):
            raise ValueError("Cannot optimize empty trajectories.")
        space_dims = {int(trajectory.dim - 1) for trajectory in trajectories}
        if len(space_dims) != 1:
            raise ValueError("All trajectories must have the same spatial dimension.")
        space_dim = next(iter(space_dims))
        if stgcs is not None and int(getattr(stgcs, "dimension")) != space_dim:
            raise ValueError(
                f"ST-GCS dimension {int(getattr(stgcs, 'dimension'))} does not match "
                f"trajectory spatial dimension {space_dim}."
            )
        if env is not None:
            env_dim = getattr(env, "dim", None)
            if env_dim is not None and int(env_dim) != space_dim:
                raise ValueError(
                    f"Environment dimension {int(env_dim)} does not match trajectory spatial dimension {space_dim}."
                )
            env_radius = getattr(env, "robot_radius", None)
            if env_radius is not None and not math.isclose(
                float(env_radius),
                float(robot_radius),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"robot_radius={float(robot_radius)} does not match env.robot_radius={float(env_radius)}."
                )
        if not math.isfinite(float(robot_radius)) or float(robot_radius) < 0.0:
            raise ValueError("robot_radius must be finite and nonnegative.")
        if not math.isfinite(float(velocity_limit)) or float(velocity_limit) <= 0.0:
            raise ValueError("velocity_limit must be finite and positive.")
        if not math.isfinite(float(config.sample_dt)) or float(config.sample_dt) <= 0.0:
            raise ValueError("sample_dt must be finite and positive.")
        if not math.isfinite(float(config.min_sample_dt)) or float(config.min_sample_dt) <= 0.0:
            raise ValueError("min_sample_dt must be finite and positive.")
        if float(config.velocity_gradient_weight) < 0.0:
            raise ValueError("velocity_gradient_weight must be nonnegative.")
        if float(config.displacement_weight) < 0.0:
            raise ValueError("displacement_weight must be nonnegative.")
        if float(config.displacement_jitter_weight) < 0.0:
            raise ValueError("displacement_jitter_weight must be nonnegative.")
        if float(config.spatial_containment_tolerance) < 0.0:
            raise ValueError("spatial_containment_tolerance must be nonnegative.")
        if (
            not math.isfinite(float(config.spatial_containment_soft_margin))
            or float(config.spatial_containment_soft_margin) < 0.0
        ):
            raise ValueError("spatial_containment_soft_margin must be finite and nonnegative.")
        if not math.isfinite(float(config.constraint_slack_penalty)) or float(config.constraint_slack_penalty) < 0.0:
            raise ValueError("constraint_slack_penalty must be finite and nonnegative.")
        slack_limit = float(config.constraint_slack_limit)
        if math.isnan(slack_limit) or slack_limit <= 0.0:
            raise ValueError("constraint_slack_limit must be positive.")
        if int(config.solver_max_iter) <= 0:
            raise ValueError("solver_max_iter must be positive.")
        if not math.isfinite(float(config.solver_eps_abs)) or float(config.solver_eps_abs) <= 0.0:
            raise ValueError("solver_eps_abs must be finite and positive.")
        if not math.isfinite(float(config.solver_eps_rel)) or float(config.solver_eps_rel) <= 0.0:
            raise ValueError("solver_eps_rel must be finite and positive.")

    @classmethod
    def sample_times(
        cls,
        trajectories: Sequence[STTrajectory],
        config: GlobalTrajOptConfig | None = None,
    ) -> np.ndarray:
        if config is None:
            config = GlobalTrajOptConfig()
        if not trajectories:
            raise ValueError("Expected at least one trajectory.")

        t0 = min(float(trajectory.x0[-1]) for trajectory in trajectories)
        tf = max(float(trajectory.xT[-1]) for trajectory in trajectories)
        if tf < t0:
            raise ValueError("Invalid trajectory time range.")

        count = max(1, int(math.ceil((tf - t0) / float(config.sample_dt))))
        times = [np.linspace(t0, tf, count + 1)]
        if config.include_trajectory_knot_times:
            knot_times = []
            for trajectory in trajectories:
                for idx in range(trajectory.size):
                    knot_times.append(float(trajectory.xA(idx)[-1]))
                    knot_times.append(float(trajectory.xB(idx)[-1]))
            times.append(np.asarray(knot_times, dtype=float))

        samples = np.concatenate(times)
        samples = samples[(samples >= t0 - 1e-9) & (samples <= tf + 1e-9)]
        samples = np.unique(np.round(samples, decimals=int(config.round_time_decimals)))
        min_interval = min(float(config.min_sample_dt), float(config.sample_dt))
        return cls._coalesce_sample_times(samples, min_interval)

    @staticmethod
    def _coalesce_sample_times(samples: np.ndarray, min_interval: float) -> np.ndarray:
        if samples.size <= 1:
            return samples

        ordered = np.asarray(samples, dtype=float)
        coalesced = [float(ordered[0])]
        for sample in ordered[1:]:
            sample = float(sample)
            if sample - coalesced[-1] >= float(min_interval):
                coalesced.append(sample)

        final_sample = float(ordered[-1])
        if coalesced[-1] < final_sample:
            if final_sample - coalesced[-1] < float(min_interval):
                coalesced[-1] = final_sample
            else:
                coalesced.append(final_sample)
        return np.asarray(coalesced, dtype=float)

    @staticmethod
    def sample_positions(
        trajectories: Sequence[STTrajectory],
        times: np.ndarray,
    ) -> np.ndarray:
        if not trajectories:
            raise ValueError("Expected at least one trajectory.")
        space_dim = int(trajectories[0].dim - 1)
        positions = np.empty((len(trajectories), len(times), space_dim), dtype=float)
        for agent_idx, trajectory in enumerate(trajectories):
            for time_idx, sample_time in enumerate(times):
                positions[agent_idx, time_idx] = trajectory.lerp(float(sample_time))[:space_dim]
        return positions

    @classmethod
    def _linear_constraints(
        cls,
        trajectories: Sequence[STTrajectory],
        times: np.ndarray,
        original_positions: np.ndarray,
        robot_radius: float,
        velocity_limit: float,
        config: GlobalTrajOptConfig,
        stgcs: object | None,
        env: object | None,
        fixed_time_indices: Sequence[int] | None = None,
    ) -> _LinearConstraintMatrix | None:
        rows: LinearConstraintRows = []
        lower: list[float] = []
        upper: list[float] = []
        soft: list[bool] = []
        nvars = original_positions.size
        has_static_obstacles = env is not None and bool(getattr(env, "O_Static", []) or [])
        has_spatial_gcs_containment = cls._has_spatial_gcs_containment(stgcs)

        cls._add_terminal_fix_rows(
            trajectories,
            times,
            original_positions,
            config,
            nvars,
            rows,
            lower,
            upper,
            soft,
        )
        cls._add_sample_fix_rows(
            original_positions,
            fixed_time_indices,
            nvars,
            rows,
            lower,
            upper,
            soft,
        )
        cls._add_velocity_bound_rows(
            times,
            original_positions,
            velocity_limit,
            nvars,
            rows,
            lower,
            upper,
            soft,
        )
        if has_spatial_gcs_containment:
            cls._add_spatial_gcs_containment_rows(
                stgcs,
                trajectories,
                times,
                original_positions,
                config.spatial_containment_tolerance,
                config.spatial_containment_soft_margin,
                nvars,
                rows,
                lower,
                upper,
                soft,
            )
        elif has_static_obstacles:
            cls._add_domain_bound_rows(
                env,
                original_positions,
                nvars,
                rows,
                lower,
                upper,
                soft,
            )
            cls._add_static_obstacle_avoidance_rows(
                stgcs,
                env,
                trajectories,
                times,
                original_positions,
                robot_radius,
                config,
                nvars,
                rows,
                lower,
                upper,
                soft,
            )
        cls._add_collision_avoidance_rows(
            times,
            original_positions,
            robot_radius,
            velocity_limit,
            config,
            nvars,
            rows,
            lower,
            upper,
            soft,
            fixed_time_indices=fixed_time_indices,
        )
        if bool(config.include_interval_collision_constraints):
            cls._add_interval_collision_avoidance_rows(
                times,
                original_positions,
                robot_radius,
                velocity_limit,
                config,
                nvars,
                rows,
                lower,
                upper,
                soft,
                fixed_time_indices=fixed_time_indices,
            )

        if not rows:
            return None
        return _LinearConstraintMatrix(
            cls._sparse_constraint_matrix(rows, nvars),
            np.asarray(lower),
            np.asarray(upper),
            np.asarray(soft, dtype=bool),
        )

    @staticmethod
    def _append_linear_constraint_row(
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        indices: Sequence[int],
        coefficients: Sequence[float],
        lb: float,
        ub: float,
        soft: list[bool] | None = None,
        is_soft: bool = False,
    ) -> None:
        index_array = np.asarray(indices, dtype=int).reshape(-1)
        coefficient_array = np.asarray(coefficients, dtype=float).reshape(-1)
        if index_array.shape != coefficient_array.shape:
            raise ValueError(
                f"Linear constraint row has {index_array.size} indices but {coefficient_array.size} coefficients."
            )
        nonzero = np.abs(coefficient_array) > 1e-15
        rows.append((index_array[nonzero], coefficient_array[nonzero]))
        lower.append(float(lb))
        upper.append(float(ub))
        if soft is not None:
            soft.append(bool(is_soft))

    @staticmethod
    def _sparse_constraint_matrix(rows: LinearConstraintRows, nvars: int) -> sparse.csc_matrix:
        row_indices: list[int] = []
        col_indices: list[int] = []
        data: list[float] = []
        for row_idx, (indices, coefficients) in enumerate(rows):
            row_indices.extend([row_idx] * int(indices.size))
            col_indices.extend(int(index) for index in indices)
            data.extend(float(value) for value in coefficients)
        return sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(rows), int(nvars)),
            dtype=float,
        ).tocsc()

    @staticmethod
    def _has_spatial_gcs_containment(stgcs: object | None) -> bool:
        if stgcs is None or not hasattr(stgcs, "G") or not hasattr(stgcs, "get_vertex"):
            return False
        for vertex_name in stgcs.G.nodes:
            if vertex_name in {"source", "target"} or str(vertex_name).startswith("sub-target"):
                continue
            try:
                vertex = stgcs.get_vertex(vertex_name)
            except KeyError:
                continue
            if vertex is not None and getattr(vertex, "st_hpoly", None) is not None:
                return True
        return False

    @classmethod
    def _add_sample_fix_rows(
        cls,
        original_positions: np.ndarray,
        fixed_time_indices: Sequence[int] | None,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
    ) -> None:
        if fixed_time_indices is None:
            return

        _, num_times, space_dim = original_positions.shape
        for time_idx in sorted(set(int(idx) for idx in fixed_time_indices)):
            if time_idx < 0 or time_idx >= num_times:
                raise ValueError(f"fixed time index {time_idx} is outside [0, {num_times}).")
            for agent_idx in range(original_positions.shape[0]):
                for axis in range(space_dim):
                    cls._append_linear_constraint_row(
                        rows,
                        lower,
                        upper,
                        [cls._var_index(original_positions, agent_idx, time_idx, axis)],
                        [1.0],
                        0.0,
                        0.0,
                        soft=soft,
                    )

    @classmethod
    def _add_terminal_fix_rows(
        cls,
        trajectories: Sequence[STTrajectory],
        times: np.ndarray,
        original_positions: np.ndarray,
        config: GlobalTrajOptConfig,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
    ) -> None:
        if not config.fix_terminal_states:
            return

        _, _, space_dim = original_positions.shape
        for agent_idx, trajectory in enumerate(trajectories):
            t_start = float(trajectory.x0[-1])
            t_final = float(trajectory.xT[-1])
            for time_idx, sample_time in enumerate(times):
                if sample_time > t_start + 1e-9 and sample_time < t_final - 1e-9:
                    continue
                for axis in range(space_dim):
                    cls._append_linear_constraint_row(
                        rows,
                        lower,
                        upper,
                        [cls._var_index(original_positions, agent_idx, time_idx, axis)],
                        [1.0],
                        0.0,
                        0.0,
                        soft=soft,
                    )

    @classmethod
    def _add_velocity_bound_rows(
        cls,
        times: np.ndarray,
        original_positions: np.ndarray,
        velocity_limit: float,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
    ) -> None:
        num_agents, num_times, space_dim = original_positions.shape
        for agent_idx in range(num_agents):
            for time_idx in range(num_times - 1):
                dt = float(times[time_idx + 1] - times[time_idx])
                if dt <= 0.0:
                    raise ValueError("Sample times must be strictly increasing.")
                for axis in range(space_dim):
                    indices = [
                        cls._var_index(original_positions, agent_idx, time_idx + 1, axis),
                        cls._var_index(original_positions, agent_idx, time_idx, axis),
                    ]
                    coefficients = [1.0 / dt, -1.0 / dt]
                    offset = (
                        original_positions[agent_idx, time_idx + 1, axis]
                        - original_positions[agent_idx, time_idx, axis]
                    ) / dt
                    cls._append_linear_constraint_row(
                        rows,
                        lower,
                        upper,
                        indices,
                        coefficients,
                        float(-velocity_limit - offset),
                        float(velocity_limit - offset),
                        soft=soft,
                    )

    @classmethod
    def _add_spatial_gcs_containment_rows(
        cls,
        stgcs: object | None,
        trajectories: Sequence[STTrajectory],
        times: np.ndarray,
        original_positions: np.ndarray,
        tolerance: float,
        soft_margin: float,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
    ) -> None:
        if stgcs is None:
            return

        _, num_times, space_dim = original_positions.shape
        for agent_idx, trajectory in enumerate(trajectories):
            for time_idx in range(num_times - 1):
                mid_time = 0.5 * (float(times[time_idx]) + float(times[time_idx + 1]))
                source_segment_idx = trajectory.find_segment_index(mid_time)
                hpoly = cls._trajectory_segment_hpoly(stgcs, trajectory, source_segment_idx)
                A = np.asarray(hpoly.A(), dtype=float)
                b = np.asarray(hpoly.b(), dtype=float).reshape(-1)
                if A.shape[1] != space_dim + 1:
                    raise ValueError(
                        f"ST-GCS cell dimension {A.shape[1]} does not match expected {space_dim + 1}."
                    )
                for endpoint_time_idx in (time_idx, time_idx + 1):
                    sample_time = float(times[endpoint_time_idx])
                    original = original_positions[agent_idx, endpoint_time_idx]
                    for normal, bound in zip(A, b):
                        spatial_normal = np.asarray(normal[:space_dim], dtype=float)
                        if np.linalg.norm(spatial_normal) <= 1e-12:
                            continue
                        indices = [
                            cls._var_index(original_positions, agent_idx, endpoint_time_idx, axis)
                            for axis in range(space_dim)
                        ]
                        upper_bound = float(
                            bound
                            - normal[space_dim] * sample_time
                            - spatial_normal @ original
                            + float(tolerance)
                        )
                        cls._append_linear_constraint_row(
                            rows,
                            lower,
                            upper,
                            indices,
                            spatial_normal,
                            -np.inf,
                            upper_bound,
                            soft=soft,
                            is_soft=upper_bound <= float(soft_margin),
                        )

    @classmethod
    def _add_domain_bound_rows(
        cls,
        env: object | None,
        original_positions: np.ndarray,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
    ) -> None:
        if env is None or not hasattr(env, "lb") or not hasattr(env, "ub"):
            return

        lb = np.asarray(getattr(env, "lb"), dtype=float).reshape(-1)
        ub = np.asarray(getattr(env, "ub"), dtype=float).reshape(-1)
        num_agents, num_times, space_dim = original_positions.shape
        if lb.shape != (space_dim,) or ub.shape != (space_dim,):
            raise ValueError(f"Environment bounds must have shape {(space_dim,)}, got {lb.shape} and {ub.shape}.")

        for agent_idx in range(num_agents):
            for time_idx in range(num_times):
                for axis in range(space_dim):
                    original = float(original_positions[agent_idx, time_idx, axis])
                    cls._append_linear_constraint_row(
                        rows,
                        lower,
                        upper,
                        [cls._var_index(original_positions, agent_idx, time_idx, axis)],
                        [1.0],
                        float(lb[axis] - original),
                        float(ub[axis] - original),
                        soft=soft,
                    )

    @classmethod
    def _add_static_obstacle_avoidance_rows(
        cls,
        stgcs: object | None,
        env: object | None,
        trajectories: Sequence[STTrajectory],
        times: np.ndarray,
        original_positions: np.ndarray,
        robot_radius: float,
        config: GlobalTrajOptConfig,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
    ) -> None:
        if env is None:
            return
        obstacles = list(getattr(env, "O_Static", []) or [])
        if not obstacles:
            return

        _, num_times, space_dim = original_positions.shape
        margin = float(config.clearance_margin)
        for agent_idx, trajectory in enumerate(trajectories):
            for time_idx in range(num_times - 1):
                mid_time = 0.5 * (float(times[time_idx]) + float(times[time_idx + 1]))
                source_segment_idx = trajectory.find_segment_index(mid_time)
                movement_bounds = cls._trajectory_segment_spatial_bounds(stgcs, trajectory, source_segment_idx)
                p0 = original_positions[agent_idx, time_idx]
                p1 = original_positions[agent_idx, time_idx + 1]
                for obstacle in obstacles:
                    obstacle_bounds = cls._static_obstacle_aabb(obstacle, robot_radius, margin)
                    if (
                        movement_bounds is not None
                        and obstacle_bounds is not None
                        and not cls._aabb_intersects(movement_bounds, obstacle_bounds)
                    ):
                        continue
                    normal, bound = cls._static_obstacle_separating_halfspace(
                        obstacle,
                        p0,
                        p1,
                        robot_radius,
                        margin,
                    )
                    if normal.shape != (space_dim,):
                        raise ValueError(
                            f"Static obstacle separating normal has shape {normal.shape}, expected {(space_dim,)}."
                        )
                    for endpoint_time_idx in (time_idx, time_idx + 1):
                        indices = [
                            cls._var_index(original_positions, agent_idx, endpoint_time_idx, axis)
                            for axis in range(space_dim)
                        ]
                        cls._append_linear_constraint_row(
                            rows,
                            lower,
                            upper,
                            indices,
                            normal,
                            float(bound - normal @ original_positions[agent_idx, endpoint_time_idx]),
                            np.inf,
                            soft=soft,
                            is_soft=True,
                        )

    @classmethod
    def _add_collision_avoidance_rows(
        cls,
        times: np.ndarray,
        original_positions: np.ndarray,
        robot_radius: float,
        velocity_limit: float,
        config: GlobalTrajOptConfig,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
        fixed_time_indices: Sequence[int] | None = None,
    ) -> None:
        num_agents, num_times, space_dim = original_positions.shape
        min_sep = 2.0 * float(robot_radius) + float(config.clearance_margin)
        displacement_bounds = cls._sample_displacement_l2_bounds(
            times,
            original_positions,
            velocity_limit,
            fixed_time_indices,
        )
        for time_idx in range(num_times):
            for agent_i, agent_j in combinations(range(num_agents), 2):
                diff = original_positions[agent_i, time_idx] - original_positions[agent_j, time_idx]
                reference_distance = float(np.linalg.norm(diff))
                if displacement_bounds is not None:
                    max_closing_distance = (
                        float(displacement_bounds[agent_i, time_idx])
                        + float(displacement_bounds[agent_j, time_idx])
                    )
                    if reference_distance > min_sep + max_closing_distance + 1e-12:
                        continue
                normal = cls._separation_normal(original_positions, agent_i, agent_j, time_idx)
                indices: list[int] = []
                coefficients: list[float] = []
                for axis in range(space_dim):
                    indices.append(cls._var_index(original_positions, agent_i, time_idx, axis))
                    coefficients.append(float(normal[axis]))
                    indices.append(cls._var_index(original_positions, agent_j, time_idx, axis))
                    coefficients.append(float(-normal[axis]))
                cls._append_linear_constraint_row(
                    rows,
                    lower,
                    upper,
                    indices,
                    coefficients,
                    float(min_sep - normal @ diff),
                    np.inf,
                    soft=soft,
                    is_soft=True,
                )

    @classmethod
    def _add_interval_collision_avoidance_rows(
        cls,
        times: np.ndarray,
        original_positions: np.ndarray,
        robot_radius: float,
        velocity_limit: float,
        config: GlobalTrajOptConfig,
        nvars: int,
        rows: LinearConstraintRows,
        lower: list[float],
        upper: list[float],
        soft: list[bool],
        fixed_time_indices: Sequence[int] | None = None,
    ) -> None:
        num_agents, num_times, space_dim = original_positions.shape
        min_sep = 2.0 * float(robot_radius) + float(config.clearance_margin)
        displacement_bounds = cls._sample_displacement_l2_bounds(
            times,
            original_positions,
            velocity_limit,
            fixed_time_indices,
        )
        for time_idx in range(num_times - 1):
            for agent_i, agent_j in combinations(range(num_agents), 2):
                rel0 = original_positions[agent_i, time_idx] - original_positions[agent_j, time_idx]
                rel1 = original_positions[agent_i, time_idx + 1] - original_positions[agent_j, time_idx + 1]
                delta = rel1 - rel0
                denom = float(delta @ delta)
                if denom <= 1e-12:
                    s = 0.5
                else:
                    s = float(np.clip(-(rel0 @ delta) / denom, 0.0, 1.0))
                if s <= 1e-9 or s >= 1.0 - 1e-9:
                    continue

                rel = (1.0 - s) * rel0 + s * rel1
                norm = float(np.linalg.norm(rel))
                if displacement_bounds is not None:
                    max_closing_distance = (
                        max(
                            float(displacement_bounds[agent_i, time_idx]),
                            float(displacement_bounds[agent_i, time_idx + 1]),
                        )
                        + max(
                            float(displacement_bounds[agent_j, time_idx]),
                            float(displacement_bounds[agent_j, time_idx + 1]),
                        )
                    )
                    if norm > min_sep + max_closing_distance + 1e-12:
                        continue
                if norm > 1e-12:
                    normal = rel / norm
                else:
                    normal = cls._separation_normal(original_positions, agent_i, agent_j, time_idx)

                indices: list[int] = []
                coefficients: list[float] = []
                for axis in range(space_dim):
                    indices.append(cls._var_index(original_positions, agent_i, time_idx, axis))
                    coefficients.append(float((1.0 - s) * normal[axis]))
                    indices.append(cls._var_index(original_positions, agent_i, time_idx + 1, axis))
                    coefficients.append(float(s * normal[axis]))
                    indices.append(cls._var_index(original_positions, agent_j, time_idx, axis))
                    coefficients.append(float(-(1.0 - s) * normal[axis]))
                    indices.append(cls._var_index(original_positions, agent_j, time_idx + 1, axis))
                    coefficients.append(float(-s * normal[axis]))

                cls._append_linear_constraint_row(
                    rows,
                    lower,
                    upper,
                    indices,
                    coefficients,
                    float(min_sep - normal @ rel),
                    np.inf,
                    soft=soft,
                    is_soft=True,
                )

    @staticmethod
    def _sample_displacement_l2_bounds(
        times: np.ndarray,
        original_positions: np.ndarray,
        velocity_limit: float,
        fixed_time_indices: Sequence[int] | None,
    ) -> np.ndarray | None:
        if fixed_time_indices is None:
            return None

        num_agents, num_times, space_dim = original_positions.shape
        fixed_indices = sorted(set(int(index) for index in fixed_time_indices))
        if not fixed_indices:
            return None
        if fixed_indices[0] < 0 or fixed_indices[-1] >= num_times:
            raise ValueError(f"fixed time indices must be inside [0, {num_times}).")

        edge_axis_bounds = np.empty((num_agents, max(0, num_times - 1), space_dim), dtype=float)
        for time_idx in range(num_times - 1):
            dt = float(times[time_idx + 1] - times[time_idx])
            if dt <= 0.0:
                raise ValueError("Sample times must be strictly increasing.")
            original_velocity = (
                original_positions[:, time_idx + 1, :]
                - original_positions[:, time_idx, :]
            ) / dt
            edge_axis_bounds[:, time_idx, :] = dt * (
                float(velocity_limit) + np.abs(original_velocity)
            )

        axis_bounds = np.full((num_agents, num_times, space_dim), np.inf, dtype=float)
        for fixed_idx in fixed_indices:
            axis_bounds[:, fixed_idx, :] = 0.0

            cumulative = np.zeros((num_agents, space_dim), dtype=float)
            for time_idx in range(fixed_idx + 1, num_times):
                cumulative = cumulative + edge_axis_bounds[:, time_idx - 1, :]
                axis_bounds[:, time_idx, :] = np.minimum(axis_bounds[:, time_idx, :], cumulative)

            cumulative = np.zeros((num_agents, space_dim), dtype=float)
            for time_idx in range(fixed_idx - 1, -1, -1):
                cumulative = cumulative + edge_axis_bounds[:, time_idx, :]
                axis_bounds[:, time_idx, :] = np.minimum(axis_bounds[:, time_idx, :], cumulative)

        if np.any(~np.isfinite(axis_bounds)):
            return None
        return np.linalg.norm(axis_bounds, axis=2)

    @staticmethod
    def _separation_normal(
        original_positions: np.ndarray,
        agent_i: int,
        agent_j: int,
        time_idx: int,
    ) -> np.ndarray:
        diff = original_positions[agent_i, time_idx] - original_positions[agent_j, time_idx]
        norm = float(np.linalg.norm(diff))
        if norm > 1e-12:
            return diff / norm

        num_times = original_positions.shape[1]
        for offset in range(1, num_times):
            for candidate_idx in (time_idx - offset, time_idx + offset):
                if candidate_idx < 0 or candidate_idx >= num_times:
                    continue
                diff = original_positions[agent_i, candidate_idx] - original_positions[agent_j, candidate_idx]
                norm = float(np.linalg.norm(diff))
                if norm > 1e-12:
                    return diff / norm

        normal = np.zeros(original_positions.shape[2], dtype=float)
        normal[0] = 1.0
        return normal

    @classmethod
    def _trajectory_segment_hpoly(cls, stgcs: object, trajectory: STTrajectory, segment_idx: int):
        vertex_name = trajectory.vertex_path[segment_idx]
        try:
            vertex = stgcs.get_vertex(vertex_name)
            if vertex is not None:
                return vertex.st_hpoly
        except KeyError:
            pass
        return cls._recover_hpoly_for_trajectory_segment(stgcs, trajectory, segment_idx)

    @staticmethod
    def _recover_hpoly_for_trajectory_segment(stgcs: object, trajectory: STTrajectory, segment_idx: int):
        start = trajectory.xA(segment_idx)
        goal = trajectory.xB(segment_idx)
        probes = [start, goal, 0.5 * (start + goal)]
        for candidate_name in stgcs.G.nodes:
            if candidate_name in {"source", "target"} or str(candidate_name).startswith("sub-target"):
                continue
            candidate = stgcs.get_vertex(candidate_name)
            if candidate is not None and all(candidate.st_hpoly.PointInSet(point, 1e-7) for point in probes):
                return candidate.st_hpoly
        raise KeyError(f"Could not recover an ST-GCS cell containing segment {trajectory.vertex_path[segment_idx]!r}.")

    @staticmethod
    def _trajectory_segment_spatial_bounds(
        stgcs: object | None,
        trajectory: STTrajectory,
        segment_idx: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if stgcs is None:
            return None
        try:
            vertex = stgcs.get_vertex(trajectory.vertex_path[segment_idx])
        except KeyError:
            return None
        if vertex is None or not getattr(vertex, "space_itvls", None):
            return None
        lb = np.asarray([itvl.start for itvl in vertex.space_itvls], dtype=float)
        ub = np.asarray([itvl.end for itvl in vertex.space_itvls], dtype=float)
        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            return None
        return lb, ub

    @staticmethod
    def _static_obstacle_aabb(
        obstacle: object,
        robot_radius: float,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        padding = float(robot_radius) + float(margin)
        if isinstance(obstacle, StaticSphere):
            radius = float(obstacle.radius) + padding
            center = np.asarray(obstacle.pos, dtype=float)
            return center - radius, center + radius
        if isinstance(obstacle, StaticPolygon):
            vertices = np.asarray(obstacle.vertices, dtype=float)
            return np.min(vertices, axis=0) - padding, np.max(vertices, axis=0) + padding
        return None

    @staticmethod
    def _aabb_intersects(
        lhs: tuple[np.ndarray, np.ndarray],
        rhs: tuple[np.ndarray, np.ndarray],
        tol: float = 1e-12,
    ) -> bool:
        lhs_lb, lhs_ub = lhs
        rhs_lb, rhs_ub = rhs
        return bool(np.all(lhs_lb <= rhs_ub + tol) and np.all(rhs_lb <= lhs_ub + tol))

    @classmethod
    def _static_obstacle_separating_halfspace(
        cls,
        obstacle: object,
        p0: np.ndarray,
        p1: np.ndarray,
        robot_radius: float,
        margin: float,
    ) -> tuple[np.ndarray, float]:
        if isinstance(obstacle, StaticSphere):
            return cls._sphere_separating_halfspace(obstacle, p0, p1, robot_radius, margin)
        if isinstance(obstacle, StaticPolygon):
            return cls._polygon_separating_halfspace(obstacle, p0, p1, robot_radius, margin)
        raise NotImplementedError(f"Unsupported static obstacle type {type(obstacle).__name__}.")

    @staticmethod
    def _sphere_separating_halfspace(
        obstacle: StaticSphere,
        p0: np.ndarray,
        p1: np.ndarray,
        robot_radius: float,
        margin: float,
    ) -> tuple[np.ndarray, float]:
        center = np.asarray(obstacle.pos, dtype=float)
        direction = p1 - p0
        denom = float(direction @ direction)
        if denom <= 1e-12:
            closest = np.asarray(p0, dtype=float)
        else:
            alpha = float(np.clip(((center - p0) @ direction) / denom, 0.0, 1.0))
            closest = p0 + alpha * direction
        diff = closest - center
        dist = float(np.linalg.norm(diff))
        required = float(obstacle.radius) + float(robot_radius) + float(margin)
        if dist <= required + 1e-12:
            raise RuntimeError("Cannot linearize static sphere avoidance from a colliding original segment.")
        normal = diff / dist
        return normal, float(normal @ center + required)

    @staticmethod
    def _polygon_separating_halfspace(
        obstacle: StaticPolygon,
        p0: np.ndarray,
        p1: np.ndarray,
        robot_radius: float,
        margin: float,
    ) -> tuple[np.ndarray, float]:
        if p0.size != 2 or p1.size != 2:
            raise NotImplementedError("StaticPolygon avoidance expects 2D segment endpoints.")
        A, b = GlobalTrajectoryOptimizer._inflated_polygon_halfspaces(
            obstacle,
            robot_radius,
            margin,
        )
        vertices = GlobalTrajectoryOptimizer._convex_polygon_vertices_from_halfspaces(A, b)
        segment_point, obstacle_point = GlobalTrajectoryOptimizer._segment_polygon_closest_points(
            p0,
            p1,
            vertices,
        )
        diff = segment_point - obstacle_point
        dist = float(np.linalg.norm(diff))
        if dist <= 1e-10:
            contact_halfspace = GlobalTrajectoryOptimizer._polygon_contact_face_halfspace(A, b, p0, p1)
            if contact_halfspace is not None:
                return contact_halfspace
            raise RuntimeError("Cannot linearize static polygon avoidance from a colliding original segment.")
        normal = diff / dist
        return normal, float(normal @ obstacle_point)

    @staticmethod
    def _inflated_polygon_halfspaces(
        obstacle: StaticPolygon,
        robot_radius: float,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        hpoly = obstacle.hpoly
        A = np.asarray(hpoly.A(), dtype=float)
        b = np.asarray(hpoly.b(), dtype=float).reshape(-1)
        norms = np.linalg.norm(A, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("Cannot linearize a polygon obstacle with a near-zero face normal.")
        A = A / norms[:, None]
        b = b / norms
        b = b + float(robot_radius) * np.linalg.norm(A, ord=1, axis=1) + float(margin)
        return A, b

    @staticmethod
    def _convex_polygon_vertices_from_halfspaces(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        if A.shape[1] != 2:
            raise NotImplementedError("StaticPolygon avoidance expects 2D halfspaces.")
        vertices: list[np.ndarray] = []
        for row_i in range(A.shape[0]):
            for row_j in range(row_i + 1, A.shape[0]):
                matrix = A[[row_i, row_j]]
                if abs(float(np.linalg.det(matrix))) <= 1e-12:
                    continue
                point = np.linalg.solve(matrix, b[[row_i, row_j]])
                if np.all(A @ point <= b + 1e-9):
                    vertices.append(point)
        if len(vertices) < 3:
            raise RuntimeError("Could not recover inflated polygon vertices.")

        unique = GlobalTrajectoryOptimizer._unique_points(vertices)
        if unique.shape[0] < 3:
            raise RuntimeError("Could not recover inflated polygon vertices.")
        center = np.mean(unique, axis=0)
        angles = np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0])
        return unique[np.argsort(angles)]

    @staticmethod
    def _unique_points(points: Sequence[np.ndarray], tol: float = 1e-9) -> np.ndarray:
        unique: list[np.ndarray] = []
        for point in points:
            candidate = np.asarray(point, dtype=float)
            if not any(np.linalg.norm(candidate - existing) <= tol for existing in unique):
                unique.append(candidate)
        return np.asarray(unique, dtype=float)

    @staticmethod
    def _point_to_segment_closest_point(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
        direction = seg_b - seg_a
        denom = float(direction @ direction)
        if denom <= 1e-12:
            return np.asarray(seg_a, dtype=float)
        alpha = float(np.clip(((point - seg_a) @ direction) / denom, 0.0, 1.0))
        return seg_a + alpha * direction

    @staticmethod
    def _polygon_contact_face_halfspace(
        A: np.ndarray,
        b: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        tolerance: float = 1e-7,
    ) -> tuple[np.ndarray, float] | None:
        endpoint_slacks = np.vstack([A @ p0 - b, A @ p1 - b])
        face_scores = np.min(endpoint_slacks, axis=0)
        face_idx = int(np.argmax(face_scores))
        if float(face_scores[face_idx]) < -float(tolerance):
            return None
        return np.asarray(A[face_idx], dtype=float), float(b[face_idx])

    @classmethod
    def _segment_polygon_closest_points(
        cls,
        seg_a: np.ndarray,
        seg_b: np.ndarray,
        polygon_vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        best_distance = math.inf
        best_segment_point: np.ndarray | None = None
        best_polygon_point: np.ndarray | None = None

        def update(segment_point: np.ndarray, polygon_point: np.ndarray) -> None:
            nonlocal best_distance, best_segment_point, best_polygon_point
            distance = float(np.linalg.norm(segment_point - polygon_point))
            if distance < best_distance:
                best_distance = distance
                best_segment_point = np.asarray(segment_point, dtype=float)
                best_polygon_point = np.asarray(polygon_point, dtype=float)

        for point in (seg_a, seg_b):
            for vertex_idx in range(polygon_vertices.shape[0]):
                edge_a = polygon_vertices[vertex_idx]
                edge_b = polygon_vertices[(vertex_idx + 1) % polygon_vertices.shape[0]]
                update(point, cls._point_to_segment_closest_point(point, edge_a, edge_b))

        for vertex in polygon_vertices:
            update(cls._point_to_segment_closest_point(vertex, seg_a, seg_b), vertex)

        if best_segment_point is None or best_polygon_point is None:
            raise RuntimeError("Could not compute segment-polygon closest points.")
        return best_segment_point, best_polygon_point

    @classmethod
    def _quadratic_objective(
        cls,
        times: np.ndarray,
        original_positions: np.ndarray,
        config: GlobalTrajOptConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        nvars = original_positions.size
        Q = np.zeros((nvars, nvars), dtype=float)
        b = np.zeros(nvars, dtype=float)

        displacement_weight = float(config.displacement_weight)
        if displacement_weight != 0.0:
            Q.flat[:: nvars + 1] += 2.0 * displacement_weight

        def add_square_entries(
            indices: Sequence[int],
            coefficients: Sequence[float],
            offset: float,
            weight: float,
        ) -> None:
            if weight == 0.0:
                return
            index_array = np.asarray(indices, dtype=int)
            coefficient_array = np.asarray(coefficients, dtype=float)
            scale = 2.0 * float(weight)
            Q[np.ix_(index_array, index_array)] += scale * np.outer(coefficient_array, coefficient_array)
            b[index_array] += scale * float(offset) * coefficient_array

        velocity_gradient_weight = float(config.velocity_gradient_weight)
        num_agents, num_times, space_dim = original_positions.shape
        if velocity_gradient_weight != 0.0 and num_times >= 4:
            for agent_idx in range(num_agents):
                for time_idx in range(2, num_times - 1):
                    denom_curr = float(times[time_idx + 1] - times[time_idx - 1])
                    denom_prev = float(times[time_idx] - times[time_idx - 2])
                    for axis in range(space_dim):
                        indices = [
                            cls._var_index(original_positions, agent_idx, time_idx + 1, axis),
                            cls._var_index(original_positions, agent_idx, time_idx - 1, axis),
                            cls._var_index(original_positions, agent_idx, time_idx, axis),
                            cls._var_index(original_positions, agent_idx, time_idx - 2, axis),
                        ]
                        coefficients = [
                            1.0 / denom_curr,
                            -1.0 / denom_curr,
                            -1.0 / denom_prev,
                            1.0 / denom_prev,
                        ]
                        offset = (
                            (
                                original_positions[agent_idx, time_idx + 1, axis]
                                - original_positions[agent_idx, time_idx - 1, axis]
                            )
                            / denom_curr
                            - (
                                original_positions[agent_idx, time_idx, axis]
                                - original_positions[agent_idx, time_idx - 2, axis]
                            )
                            / denom_prev
                        )
                        add_square_entries(indices, coefficients, offset, velocity_gradient_weight)

        displacement_jitter_weight = float(config.displacement_jitter_weight)
        if displacement_jitter_weight != 0.0 and num_times >= 3:
            for agent_idx in range(num_agents):
                for time_idx in range(1, num_times - 1):
                    dt_prev = float(times[time_idx] - times[time_idx - 1])
                    dt_next = float(times[time_idx + 1] - times[time_idx])
                    for axis in range(space_dim):
                        indices = [
                            cls._var_index(original_positions, agent_idx, time_idx - 1, axis),
                            cls._var_index(original_positions, agent_idx, time_idx, axis),
                            cls._var_index(original_positions, agent_idx, time_idx + 1, axis),
                        ]
                        coefficients = [
                            1.0 / dt_prev,
                            -1.0 / dt_prev - 1.0 / dt_next,
                            1.0 / dt_next,
                        ]
                        add_square_entries(indices, coefficients, 0.0, displacement_jitter_weight)

        Q[:, :] = 0.5 * (Q + Q.T)
        Q.flat[:: nvars + 1] += 1e-10
        return Q, b

    @classmethod
    def _solve_quadratic_program(
        cls,
        x0: np.ndarray,
        Q: np.ndarray,
        b: np.ndarray,
        linear_constraints: _LinearConstraintMatrix | None,
        config: GlobalTrajOptConfig,
    ) -> tuple[np.ndarray, str]:
        prog, x, slack, num_soft_rows = cls._build_quadratic_program(
            x0,
            Q,
            b,
            linear_constraints,
            config,
            soften_constraints=False,
        )
        result = cls._solve_with_osqp(prog, config)
        if result.is_success():
            return np.asarray(result.GetSolution(x), dtype=float), str(result.get_solution_result())

        hard_result = str(result.get_solution_result())
        if (
            linear_constraints is None
            or float(config.constraint_slack_penalty) <= 0.0
            or linear_constraints.soft is None
            or not bool(np.any(linear_constraints.soft))
        ):
            raise RuntimeError(f"Global trajectory QP failed: {hard_result}")

        prog, x, slack, num_soft_rows = cls._build_quadratic_program(
            x0,
            Q,
            b,
            linear_constraints,
            config,
            soften_constraints=True,
        )
        result = cls._solve_with_osqp(prog, config)
        if not result.is_success():
            raise RuntimeError(
                "Global trajectory QP failed: "
                f"{result.get_solution_result()} after soft retry from {hard_result}"
            )

        solver_message = f"{result.get_solution_result()}; constraint_soft_retry_from={hard_result}"
        if slack is not None:
            solver_message = cls._append_slack_diagnostics(
                solver_message,
                result,
                slack,
                num_soft_rows,
            )
        return np.asarray(result.GetSolution(x), dtype=float), solver_message

    @classmethod
    def _build_quadratic_program(
        cls,
        x0: np.ndarray,
        Q: np.ndarray,
        b: np.ndarray,
        linear_constraints: _LinearConstraintMatrix | None,
        config: GlobalTrajOptConfig,
        soften_constraints: bool,
    ) -> tuple[MathematicalProgram, np.ndarray, np.ndarray | None, int]:
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(x0.size, "global_disp")
        slack = None
        num_soft_rows = 0
        if linear_constraints is not None:
            if soften_constraints and float(config.constraint_slack_penalty) > 0.0:
                slack, num_soft_rows = cls._add_softened_linear_constraints(
                    prog,
                    x,
                    linear_constraints,
                    float(config.constraint_slack_penalty),
                    float(config.constraint_slack_limit),
                )
            else:
                prog.AddLinearConstraint(
                    linear_constraints.A,
                    linear_constraints.lb,
                    linear_constraints.ub,
                    x,
                )
        prog.AddQuadraticCost(Q, b, x)
        prog.SetInitialGuess(x, x0)
        if slack is not None:
            prog.SetInitialGuess(slack, np.zeros(slack.size, dtype=float))
        return prog, x, slack, num_soft_rows

    @staticmethod
    def _solve_with_osqp(
        prog: MathematicalProgram,
        config: GlobalTrajOptConfig,
    ):
        solver_options = SolverOptions()
        solver_options.SetOption(
            CommonSolverOption.kPrintToConsole,
            int(bool(config.solver_print_to_console)),
        )
        solver_options.SetOption(OsqpSolver.id(), "max_iter", int(config.solver_max_iter))
        solver_options.SetOption(OsqpSolver.id(), "eps_abs", float(config.solver_eps_abs))
        solver_options.SetOption(OsqpSolver.id(), "eps_rel", float(config.solver_eps_rel))
        return OsqpSolver().Solve(prog, None, solver_options)

    @staticmethod
    def _append_slack_diagnostics(
        solver_message: str,
        result: object,
        slack: np.ndarray,
        num_soft_rows: int,
    ) -> str:
        slack_values = np.maximum(np.asarray(result.GetSolution(slack), dtype=float), 0.0)
        return (
            f"{solver_message}; constraint_soft_rows={int(num_soft_rows)}; "
            f"constraint_slack_max={float(np.max(slack_values, initial=0.0)):.6g}; "
            f"constraint_slack_l1={float(np.sum(slack_values)):.6g}"
        )

    @classmethod
    def _add_softened_linear_constraints(
        cls,
        prog: MathematicalProgram,
        x: np.ndarray,
        linear_constraints: _LinearConstraintMatrix,
        slack_penalty: float,
        slack_limit: float,
    ) -> tuple[np.ndarray | None, int]:
        lower = np.asarray(linear_constraints.lb, dtype=float)
        upper = np.asarray(linear_constraints.ub, dtype=float)
        if linear_constraints.soft is None:
            soft_candidates = np.zeros(lower.shape, dtype=bool)
        else:
            soft_candidates = np.asarray(linear_constraints.soft, dtype=bool).reshape(-1)
            if soft_candidates.shape != lower.shape:
                raise ValueError(
                    f"Soft constraint mask has shape {soft_candidates.shape}, expected {lower.shape}."
                )

        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        soft_mask = soft_candidates & (finite_lower | finite_upper)
        hard_mask = ~soft_mask
        if np.any(hard_mask):
            prog.AddLinearConstraint(
                linear_constraints.A[hard_mask],
                lower[hard_mask],
                upper[hard_mask],
                x,
            )

        if not np.any(soft_mask):
            return None, 0

        soft_rows = np.flatnonzero(soft_mask)
        soft_A = linear_constraints.A[soft_rows]
        soft_lower = lower[soft_rows]
        soft_upper = upper[soft_rows]
        slack = prog.NewContinuousVariables(int(soft_rows.size), "constraint_slack")
        prog.AddBoundingBoxConstraint(
            np.zeros(slack.size, dtype=float),
            np.full(slack.size, float(slack_limit), dtype=float),
            slack,
        )
        prog.AddQuadraticCost(
            2.0 * float(slack_penalty) * np.eye(slack.size, dtype=float),
            np.zeros(slack.size, dtype=float),
            slack,
        )

        variables = np.concatenate((x, slack))
        finite_soft_lower = np.isfinite(soft_lower)
        if np.any(finite_soft_lower):
            lower_row_indices = np.flatnonzero(finite_soft_lower)
            lower_slack_matrix = sparse.coo_matrix(
                (
                    np.ones(lower_row_indices.size, dtype=float),
                    (np.arange(lower_row_indices.size), lower_row_indices),
                ),
                shape=(lower_row_indices.size, slack.size),
                dtype=float,
            ).tocsc()
            lower_matrix = sparse.hstack(
                [soft_A[finite_soft_lower], lower_slack_matrix],
                format="csc",
            )
            prog.AddLinearConstraint(
                lower_matrix,
                soft_lower[finite_soft_lower],
                np.full(lower_row_indices.size, np.inf, dtype=float),
                variables,
            )

        finite_soft_upper = np.isfinite(soft_upper)
        if np.any(finite_soft_upper):
            upper_row_indices = np.flatnonzero(finite_soft_upper)
            upper_slack_matrix = sparse.coo_matrix(
                (
                    -np.ones(upper_row_indices.size, dtype=float),
                    (np.arange(upper_row_indices.size), upper_row_indices),
                ),
                shape=(upper_row_indices.size, slack.size),
                dtype=float,
            ).tocsc()
            upper_matrix = sparse.hstack(
                [soft_A[finite_soft_upper], upper_slack_matrix],
                format="csc",
            )
            prog.AddLinearConstraint(
                upper_matrix,
                np.full(upper_row_indices.size, -np.inf, dtype=float),
                soft_upper[finite_soft_upper],
                variables,
            )

        return slack, int(soft_rows.size)

    @classmethod
    def to_st_trajectories(
        cls,
        source_trajectories: Sequence[STTrajectory],
        times: np.ndarray,
        positions: np.ndarray,
    ) -> list[STTrajectory]:
        if len(times) < 2:
            raise ValueError("Need at least two sample times to build STTrajectory outputs.")

        optimized: list[STTrajectory] = []
        for agent_idx, source in enumerate(source_trajectories):
            points: list[np.ndarray] = []
            vertex_path: list[str] = []
            for time_idx, (left_time, right_time) in enumerate(zip(times[:-1], times[1:])):
                start = np.hstack([positions[agent_idx, time_idx], float(left_time)])
                goal = np.hstack([positions[agent_idx, time_idx + 1], float(right_time)])
                points.append(np.hstack([start, goal]))
                mid_time = 0.5 * (float(left_time) + float(right_time))
                vertex_path.append(source.vertex_path[source.find_segment_index(mid_time)])
            optimized.append(STTrajectory(vertex_path, points, int(source.dim - 1)))
        return optimized

    @staticmethod
    def min_pairwise_distance(positions: np.ndarray) -> float:
        if positions.shape[0] < 2:
            return np.inf
        min_distance = np.inf
        for time_idx in range(positions.shape[1]):
            for agent_i, agent_j in combinations(range(positions.shape[0]), 2):
                distance = float(np.linalg.norm(positions[agent_i, time_idx] - positions[agent_j, time_idx]))
                min_distance = min(min_distance, distance)
        return float(min_distance)

    @staticmethod
    def max_velocity_component(times: np.ndarray, positions: np.ndarray) -> float:
        if len(times) < 2:
            return 0.0
        max_velocity = 0.0
        for time_idx in range(len(times) - 1):
            dt = float(times[time_idx + 1] - times[time_idx])
            velocity = (positions[:, time_idx + 1, :] - positions[:, time_idx, :]) / dt
            max_velocity = max(max_velocity, float(np.max(np.abs(velocity))))
        return float(max_velocity)

    @classmethod
    def sampled_environment_collision_free(
        cls,
        env: object,
        times: np.ndarray,
        positions: np.ndarray,
        tolerance: float = 1e-9,
    ) -> bool:
        num_agents, num_times, _ = positions.shape
        for agent_idx in range(num_agents):
            for time_idx in range(num_times):
                position = positions[agent_idx, time_idx]
                if not cls._point_in_env_domain(env, position, tolerance):
                    return False
                if env.collision_checking_seg(position, position, float(times[time_idx]), float(times[time_idx])):
                    return False
        return True

    @classmethod
    def environment_collision_free(
        cls,
        env: object,
        trajectories: Sequence[STTrajectory],
        tolerance: float = 1e-9,
    ) -> bool:
        for trajectory in trajectories:
            for segment_idx in range(trajectory.size):
                start = trajectory.xA(segment_idx)
                goal = trajectory.xB(segment_idx)
                if not cls._point_in_env_domain(env, start[:-1], tolerance):
                    return False
                if not cls._point_in_env_domain(env, goal[:-1], tolerance):
                    return False
                if env.collision_checking_seg(start[:-1], goal[:-1], float(start[-1]), float(goal[-1])):
                    return False
        return True

    @staticmethod
    def _point_in_env_domain(env: object, point: np.ndarray, tolerance: float) -> bool:
        if not hasattr(env, "lb") or not hasattr(env, "ub"):
            return True
        lb = np.asarray(getattr(env, "lb"), dtype=float)
        ub = np.asarray(getattr(env, "ub"), dtype=float)
        return bool(np.all(point >= lb - tolerance) and np.all(point <= ub + tolerance))

    @staticmethod
    def _var_index(
        positions: np.ndarray,
        agent_idx: int,
        time_idx: int,
        axis: int,
    ) -> int:
        _, num_times, space_dim = positions.shape
        return ((int(agent_idx) * num_times + int(time_idx)) * space_dim) + int(axis)
