from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demos.mrmp import (  # noqa: E402
    RearrangementStage,
    RearrangementStageAssignment,
    RearrangementTask,
    RobotRearrangementTaskDemo,
)
from stgcs.st_planner import MPQuery  # noqa: E402


class UnlabeledMRMPTaskDemo(RobotRearrangementTaskDemo):
    """MRMP wrapper for unlabeled stage goals assigned by the Hungarian algorithm."""

    DEMO_KEY = "unlabeled_mrmp"
    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
    ASSIGNMENT_STRATEGY = "hungarian-nearest-goal"
    ASSIGNMENT_SOLVER = "scipy.optimize.linear_sum_assignment"
    UNLABELED_TASK_TYPE = "unlabeled_mrmp"
    HUNGARIAN_TASK_TYPE = "hungarian_assignment"
    TASK_TYPE = UNLABELED_TASK_TYPE
    SUPPORTED_TASK_TYPES = RobotRearrangementTaskDemo.SUPPORTED_TASK_TYPES | frozenset(
        {UNLABELED_TASK_TYPE, HUNGARIAN_TASK_TYPE}
    )

    @classmethod
    def parse_initial_starts(cls, payload: dict[str, Any], space_dim: int) -> list[np.ndarray]:
        raw_starts = payload.get("starts", payload.get("start_positions"))
        if raw_starts is None:
            raw_stages = payload.get("stages")
            if not isinstance(raw_stages, list) or not raw_stages:
                raise ValueError("Unlabeled MRMP config must contain starts or a non-empty stages list.")
            first_queries = raw_stages[0].get("queries") if isinstance(raw_stages[0], dict) else None
            if not isinstance(first_queries, list) or not first_queries:
                raise ValueError("Unlabeled MRMP config must contain starts when the first stage has no queries.")
            raw_starts = [query["start"] for query in first_queries]
        if not isinstance(raw_starts, list) or not raw_starts:
            raise ValueError("starts must be a non-empty list of start positions.")
        return [
            cls.point_vector(raw_start, f"starts[{idx}]", space_dim)
            for idx, raw_start in enumerate(raw_starts)
        ]

    @classmethod
    def parse_goal_spec(cls, raw_goal: Any, vlimit: float, space_dim: int, name: str) -> MPQuery:
        t_start = 0.0
        is_stay = True
        goal_vlimit = float(vlimit)
        if isinstance(raw_goal, dict):
            if "goal" in raw_goal:
                raw_point = raw_goal["goal"]
            elif "position" in raw_goal:
                raw_point = raw_goal["position"]
            elif "point" in raw_goal:
                raw_point = raw_goal["point"]
            else:
                raise ValueError(f"{name} must contain goal, position, or point.")
            t_start = cls.nonnegative_float(raw_goal.get("t_start", 0.0), f"{name}.t_start")
            is_stay = bool(raw_goal.get("is_stay", True))
            goal_vlimit = cls.positive_float(raw_goal.get("vlimit", vlimit), f"{name}.vlimit")
            if not np.isclose(goal_vlimit, float(vlimit), rtol=1e-9, atol=1e-12):
                raise ValueError(f"{name}.vlimit={goal_vlimit:g} does not match task vlimit={float(vlimit):g}.")
        else:
            raw_point = raw_goal
        goal = cls.point_vector(raw_point, name, space_dim)
        return MPQuery(
            start=np.zeros(space_dim, dtype=float),
            goal=goal,
            t_start=float(t_start),
            is_stay=bool(is_stay),
            vlimit=float(vlimit),
        )

    @classmethod
    def raw_stage_goals(cls, raw_stage: dict[str, Any], stage_idx: int) -> list[Any]:
        raw_goals = raw_stage.get("goals", raw_stage.get("goal_positions"))
        if raw_goals is not None:
            if not isinstance(raw_goals, list) or not raw_goals:
                raise ValueError(f"stages[{stage_idx}].goals must be a non-empty list.")
            return raw_goals
        raw_queries = raw_stage.get("queries")
        if isinstance(raw_queries, list) and raw_queries:
            return [{"goal": query["goal"], "t_start": query.get("t_start", 0.0), "is_stay": query.get("is_stay", True)}
                    for query in raw_queries]
        raise ValueError(f"stages[{stage_idx}] must contain goals or queries.")

    @classmethod
    def parse_stages(cls, payload: dict[str, Any], vlimit: float, space_dim: int) -> list[RearrangementStage]:
        starts = cls.parse_initial_starts(payload, space_dim)
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
            raw_goals = cls.raw_stage_goals(raw_stage, stage_idx)
            if len(raw_goals) != len(starts):
                raise ValueError(
                    f"stages[{stage_idx}] contains {len(raw_goals)} goals; expected {len(starts)}."
                )
            goals = tuple(
                cls.parse_goal_spec(raw_goal, vlimit, space_dim, f"{stage_name}.goals[{goal_idx}]")
                for goal_idx, raw_goal in enumerate(raw_goals)
            )
            if stage_idx == 0:
                goals = tuple(
                    MPQuery(
                        start=starts[agent_idx],
                        goal=query.goal,
                        t_start=float(query.t_start),
                        is_stay=bool(query.is_stay),
                        vlimit=float(vlimit),
                    )
                    for agent_idx, query in enumerate(goals)
                )
            stages.append(RearrangementStage(stage_name, goals))
        return stages

    @classmethod
    def validate_task(cls, task: RearrangementTask) -> None:
        if task.num_agents <= 0:
            raise ValueError("Task must contain at least one robot start.")
        for stage in task.stages:
            if len(stage.queries) != task.num_agents:
                raise ValueError(
                    f"Stage {stage.name!r} contains {len(stage.queries)} goals; expected {task.num_agents}."
                )

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
        current = np.asarray(current_positions, dtype=float)
        goals = np.asarray([query.goal for query in stage.queries], dtype=float)
        if current.shape != goals.shape:
            raise ValueError(f"Current positions shape {current.shape} does not match goals shape {goals.shape}.")

        cost_matrix = np.linalg.norm(current[:, None, :] - goals[None, :, :], axis=2)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        assigned_queries: list[MPQuery | None] = [None for _ in range(task.num_agents)]
        target_indices: list[int | None] = [None for _ in range(task.num_agents)]
        assignment_cost = 0.0
        for row, col in zip(row_indices, col_indices):
            target_spec = stage.queries[int(col)]
            assigned_queries[int(row)] = MPQuery(
                start=current[int(row)].copy(),
                goal=np.asarray(target_spec.goal, dtype=float),
                t_start=float(target_spec.t_start),
                is_stay=bool(target_spec.is_stay),
                vlimit=float(task.vlimit),
            )
            target_indices[int(row)] = int(col)
            assignment_cost += float(cost_matrix[int(row), int(col)])

        if any(query is None for query in assigned_queries) or any(idx is None for idx in target_indices):
            raise RuntimeError("Hungarian assignment did not assign every robot.")
        return RearrangementStageAssignment(
            queries=tuple(query for query in assigned_queries if query is not None),
            target_indices=tuple(int(idx) for idx in target_indices if idx is not None),
            cost=float(assignment_cost),
        )

    @classmethod
    def is_return_to_start_stage(cls, task: RearrangementTask, stage: RearrangementStage) -> bool:
        del task, stage
        return False


if __name__ == "__main__":
    UnlabeledMRMPTaskDemo.main()
