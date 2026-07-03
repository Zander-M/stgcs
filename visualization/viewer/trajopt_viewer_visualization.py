from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from experiments.mrmp.manifest import MRMPBenchmarkRecord
from visualization.viewer.build_viewer_manifest import ViewerManifestBuilder
from visualization.viewer.solution_visualization import SolutionVisualizationService
from stgcs.trajectory import STTrajectory
from stgcs.trajopt_postprocessing import GlobalTrajOptConfig, GlobalTrajOptResult


class TrajOptViewerVisualization:
    @staticmethod
    def finite_float_or_none(value: float | int | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    @classmethod
    def trajectory_metrics(cls, trajectories: Sequence[STTrajectory]) -> tuple[float, float]:
        durations = [float(trajectory.duration) for trajectory in trajectories]
        return float(sum(durations)), float(max(durations, default=0.0))

    @classmethod
    def solution_payload(
        cls,
        record: MRMPBenchmarkRecord,
        trajectories: Sequence[STTrajectory],
        *,
        planner_key: str,
        planner_name: str,
        runtime: float | None,
        budget: float | None,
        path_alpha: float,
        body_alpha: float,
    ) -> dict[str, Any]:
        trajectory_payloads = [
            SolutionVisualizationService.trajectory_to_viewer_json(trajectory)
            for trajectory in trajectories
        ]
        sum_of_costs, makespan = cls.trajectory_metrics(trajectories)
        return {
            "ok": True,
            "instance_id": record.instance_id,
            "planner_key": planner_key,
            "planner_name": planner_name,
            "budget": cls.finite_float_or_none(budget),
            "status": "SUCCESS",
            "is_success": True,
            "runtime": cls.finite_float_or_none(runtime),
            "cost": cls.finite_float_or_none(sum_of_costs),
            "makespan": cls.finite_float_or_none(makespan),
            "num_agents": int(record.num_agents),
            "num_completed_agents": len(trajectory_payloads),
            "trajectory": (
                trajectory_payloads[0]
                if trajectory_payloads
                else SolutionVisualizationService.trajectory_to_viewer_json(None)
            ),
            "trajectories": trajectory_payloads,
            "path_alpha": float(path_alpha),
            "body_alpha": float(body_alpha),
        }

    @classmethod
    def build_global_comparison_manifest(
        cls,
        record: MRMPBenchmarkRecord,
        *,
        source_manifest_path: str | Path,
        base_root: str | Path,
        original_trajectories: Sequence[STTrajectory],
        global_result: GlobalTrajOptResult,
        global_config: GlobalTrajOptConfig,
        budget: float | None,
        pbs_runtime: float | None,
        velocity_limit: float,
    ) -> dict[str, Any]:
        payload = ViewerManifestBuilder.build_manifest(
            [record],
            source_manifest_path=source_manifest_path,
            instance_ids=[record.instance_id],
            limit=None,
            base_root=base_root,
        )
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["solutions"] = [
            cls.solution_payload(
                record,
                original_trajectories,
                planner_key="pbs-piecewise-linear",
                planner_name="PBS piecewise-linear",
                runtime=pbs_runtime,
                budget=budget,
                path_alpha=0.28,
                body_alpha=0.28,
            ),
            cls.solution_payload(
                record,
                global_result.trajectories,
                planner_key="pbs-global-trajopt",
                planner_name="PBS + global sampled trajopt",
                runtime=global_result.runtime,
                budget=budget,
                path_alpha=0.98,
                body_alpha=0.82,
            ),
        ]
        payload["trajopt"] = {
            "mode": "global_sampled_state",
            "sample_dt": float(global_config.sample_dt),
            "min_sample_dt": float(global_config.min_sample_dt),
            "velocity_gradient_weight": float(global_config.velocity_gradient_weight),
            "displacement_weight": float(global_config.displacement_weight),
            "displacement_jitter_weight": float(global_config.displacement_jitter_weight),
            "clearance_margin": float(global_config.clearance_margin),
            "fix_terminal_states": bool(global_config.fix_terminal_states),
            "include_trajectory_knot_times": bool(global_config.include_trajectory_knot_times),
            "solver_eps_abs": float(global_config.solver_eps_abs),
            "solver_eps_rel": float(global_config.solver_eps_rel),
            "num_sample_times": int(global_result.times.size),
            "sampled_pairwise_collision_free": bool(global_result.sampled_pairwise_collision_free),
            "sampled_environment_collision_free": global_result.sampled_environment_collision_free,
            "continuous_environment_collision_free": global_result.continuous_environment_collision_free,
            "min_pairwise_distance": float(global_result.min_pairwise_distance),
            "max_velocity_component": float(global_result.max_velocity_component),
            "velocity_limit": float(velocity_limit),
            "solver_message": global_result.solver_message,
        }
        return payload

    @classmethod
    def write_global_comparison_manifest(
        cls,
        output_path: str | Path,
        record: MRMPBenchmarkRecord,
        *,
        source_manifest_path: str | Path,
        base_root: str | Path,
        original_trajectories: Sequence[STTrajectory],
        global_result: GlobalTrajOptResult,
        global_config: GlobalTrajOptConfig,
        budget: float | None,
        pbs_runtime: float | None,
        velocity_limit: float,
    ) -> Path:
        target = Path(output_path)
        payload = cls.build_global_comparison_manifest(
            record,
            source_manifest_path=source_manifest_path,
            base_root=base_root,
            original_trajectories=original_trajectories,
            global_result=global_result,
            global_config=global_config,
            budget=budget,
            pbs_runtime=pbs_runtime,
            velocity_limit=velocity_limit,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target
