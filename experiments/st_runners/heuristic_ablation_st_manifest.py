from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from benchmark.manifests.st_planning import (
    STHeuristicAblationRecord,
    STHeuristicAblationResultEntry,
    STPlanningManifestStore,
    STQuerySpec,
)
from benchmark.environment.obstacle import DynamicSphere
from benchmark.base import BaseInstanceFactory
from experiments.st_runners.heuristic_ablation_selection import (
    HeuristicAblationSelectionEntry,
    HeuristicAblationSelector,
)
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from benchmark.instance import Instance
from experiments.mrmp_runners.common import MRMPExperiment
from benchmark.planners.mrmp import SearchPlannerSpec
from stgcs.interval import Interval
from stgcs.st_planner import MPQuery, STPlanStatus


class STHeuristicAblationResultStore:
    RESULT_FIELDS = (
        "name",
        "is_success",
        "runtime",
        "cost",
        "num_expanded_nodes",
        "num_generated_nodes",
        "instance_id",
        "budget",
        "group",
        "source_domain",
        "base_instance_id",
        "obstacle_count",
        "stgcs_num_vertices",
        "stgcs_num_edges",
    )

    @classmethod
    def result_key(cls, instance_id: str, budget: float) -> tuple[str, float]:
        return str(instance_id), float(budget)

    @staticmethod
    def planner_output_path(output_root: str | Path, planner_name: str) -> Path:
        target_dir = Path(output_root)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{planner_name.replace('/', '_')}.csv"

    @classmethod
    def load_result_rows(cls, path: str | Path) -> list[Dict[str, object]]:
        target = Path(path)
        if not target.exists():
            return []
        rows: list[Dict[str, object]] = []
        with target.open(newline="") as fh:
            reader = csv.reader(fh)
            expected_width = len(cls.RESULT_FIELDS)
            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) != expected_width:
                    raise ValueError(
                        f"{target}:{line_number}: expected {expected_width} ST result fields, got {len(row)}."
                    )
                rows.append(
                    {
                        "name": row[0],
                        "is_success": row[1].lower() == "true",
                        "runtime": float(row[2]) if row[2] != "inf" else math.inf,
                        "cost": float(row[3]) if row[3] != "inf" else math.inf,
                        "num_expanded_nodes": int(float(row[4])),
                        "num_generated_nodes": int(float(row[5])),
                        "instance_id": row[6],
                        "budget": float(row[7]),
                        "group": row[8],
                        "source_domain": row[9],
                        "base_instance_id": row[10],
                        "obstacle_count": int(float(row[11])),
                        "stgcs_num_vertices": int(float(row[12])),
                        "stgcs_num_edges": int(float(row[13])),
                    }
                )
        return rows

    @classmethod
    def completed_result_keys(cls, output_path: Path) -> set[tuple[str, float]]:
        return {
            cls.result_key(str(row["instance_id"]), float(row["budget"]))
            for row in cls.load_result_rows(output_path)
        }

    @classmethod
    def append_result_row(
        cls,
        output_path: str | Path,
        planner_name: str,
        record: STHeuristicAblationRecord,
        budget: float,
        entry: STHeuristicAblationResultEntry,
    ) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", newline="") as fh:
            writer = csv.writer(fh)
            cls.write_result_row(writer, planner_name, record, budget, entry)

    @classmethod
    def write_result_rows(
        cls,
        output_path: str | Path,
        planner_name: str,
        rows: Sequence[tuple[STHeuristicAblationRecord, STHeuristicAblationResultEntry]],
        budget: float,
    ) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as fh:
            writer = csv.writer(fh)
            for record, entry in rows:
                cls.write_result_row(writer, planner_name, record, budget, entry)

    @staticmethod
    def write_result_row(
        writer,
        planner_name: str,
        record: STHeuristicAblationRecord,
        budget: float,
        entry: STHeuristicAblationResultEntry,
    ) -> None:
        writer.writerow(
            [
                planner_name,
                entry.is_success,
                entry.runtime,
                entry.cost,
                entry.num_expanded_nodes,
                entry.num_generated_nodes,
                record.instance_id,
                float(budget),
                record.group,
                record.source_domain,
                record.base_instance_id,
                len(record.dynamic_obstacles),
                record.stgcs_num_vertices,
                record.stgcs_num_edges,
            ]
        )


class STHeuristicAblationManifestBuilder:
    TMAX = STPlanningManifestStore.TMAX
    VLIMIT = STPlanningManifestStore.VLIMIT
    MAX_ATTEMPTS = 32
    MOTION_ONLY_SCREEN_SPEC = SearchPlannerSpec("h_mot", ("GUB",))
    TRIPLET_RELAXATION_SCREEN_SPEC = SearchPlannerSpec("h_tri", ("GUB",))
    INTERFACE_TO_SET_TABLE_SCREEN_SPEC = SearchPlannerSpec("h_tab", ("GUB",))
    DEFAULT_SCREEN_BUDGET = 600.0
    OBSTACLE_COUNTS = {
        HeuristicAblationSelector.GENERAL_OPEN: 1,
        HeuristicAblationSelector.IRIS2D: 1,
    }
    OBSTACLE_INTERVALS = {
        HeuristicAblationSelector.GENERAL_OPEN: ((1.0, 4.0),),
        HeuristicAblationSelector.IRIS2D: ((1.0, 5.0),),
    }
    INSTANCE_ID_GROUP_LABELS = {
        HeuristicAblationSelector.GENERAL_OPEN: "general-open",
    }

    @staticmethod
    def save_manifest(path: Path, records: Sequence[STHeuristicAblationRecord]) -> None:
        STPlanningManifestStore.save_manifest(path, records)

    @staticmethod
    def load_manifest(path: Path) -> list[STHeuristicAblationRecord]:
        return STPlanningManifestStore.load_manifest(path)

    @classmethod
    def query_for_entry(cls, entry: HeuristicAblationSelectionEntry, instance: Instance) -> MPQuery:
        centers = cls.cspace_centers(instance)
        start, goal = cls.farthest_center_pair(centers)
        return MPQuery(
            np.asarray(start, dtype=float),
            np.asarray(goal, dtype=float),
            0.0,
            True,
            cls.VLIMIT,
        )

    @staticmethod
    def cspace_centers(instance: Instance) -> list[np.ndarray]:
        return [
            np.mean(np.asarray(cspace, dtype=float), axis=0)
            for cspace in instance.env.C_Space
        ]

    @staticmethod
    def unique_centers(centers: Sequence[np.ndarray]) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        for point in centers:
            if not any(np.allclose(point, existing, rtol=1e-9, atol=1e-9) for existing in unique):
                unique.append(point)
        return unique

    @staticmethod
    def farthest_center_pair(centers: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        best_pair: tuple[np.ndarray, np.ndarray] | None = None
        best_dist = -1.0
        for i, lhs in enumerate(centers):
            for rhs in centers[i + 1:]:
                dist = float(np.linalg.norm(lhs - rhs))
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (lhs, rhs)
        if best_pair is None:
            raise ValueError("Need at least two C-space centers to construct a query.")
        return best_pair

    @classmethod
    def obstacle_count(cls, group: str) -> int:
        return int(cls.OBSTACLE_COUNTS[group])

    @classmethod
    def obstacle_intervals(cls, group: str) -> tuple[tuple[float, float], ...]:
        intervals = cls.OBSTACLE_INTERVALS[group]
        if len(intervals) != cls.obstacle_count(group):
            raise ValueError(f"Obstacle interval count does not match obstacle count for group {group!r}.")
        return intervals

    @staticmethod
    def normalized_distance(point: np.ndarray, other: np.ndarray, extent: np.ndarray) -> float:
        return float(np.linalg.norm((point - other) / np.maximum(extent, 1e-9)))

    @classmethod
    def ranked_obstacle_pairs(
        cls,
        instance: Instance,
        query: MPQuery,
        min_pair_count: int = 1,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        centers = cls.unique_centers(cls.cspace_centers(instance))
        lb = np.asarray(instance.env.lb, dtype=float)
        ub = np.asarray(instance.env.ub, dtype=float)
        center = 0.5 * (lb + ub)
        extent = np.maximum(ub - lb, 1e-9)
        clearance = 4.0 * float(instance.env.robot_radius)
        usable = [
            point for point in centers
            if np.linalg.norm(point - query.start) > clearance
            and np.linalg.norm(point - query.goal) > clearance
        ]
        usable_pair_count = len(usable) * (len(usable) - 1) // 2
        if len(usable) < 2 or usable_pair_count < int(min_pair_count):
            usable = centers

        scored_pairs: list[tuple[tuple[float, float, int, int], tuple[np.ndarray, np.ndarray]]] = []
        for i, lhs in enumerate(usable):
            for j, rhs in enumerate(usable[i + 1:], start=i + 1):
                midpoint = 0.5 * (lhs + rhs)
                centrality = cls.normalized_distance(midpoint, center, extent)
                length = float(np.linalg.norm(lhs - rhs))
                scored_pairs.append(((centrality, -length, i, j), (lhs, rhs)))
        if not scored_pairs:
            raise ValueError("Need at least two usable centers to construct dynamic obstacles.")
        return [pair for _, pair in sorted(scored_pairs, key=lambda item: item[0])]

    @classmethod
    def obstacle_specs(
        cls,
        entry: HeuristicAblationSelectionEntry,
        instance: Instance,
        query: MPQuery,
        attempt: int,
    ) -> list[Dict[str, Any]]:
        specs: list[Dict[str, Any]] = []
        obstacle_count = cls.obstacle_count(entry.group)
        pairs = cls.ranked_obstacle_pairs(instance, query, min_pair_count=obstacle_count)
        stride = max(1, len(pairs) // max(obstacle_count, 1))
        flip_period = 1 << obstacle_count
        pair_attempt = int(attempt) // flip_period
        flip_mask = int(attempt) % flip_period
        for obs_idx, (t_start, t_end) in enumerate(cls.obstacle_intervals(entry.group)):
            pair_idx = (pair_attempt + obs_idx * stride) % len(pairs)
            start, goal = pairs[pair_idx]
            if (flip_mask >> obs_idx) & 1:
                start, goal = goal, start
            obstacle = DynamicSphere(
                np.asarray(start, dtype=float),
                np.asarray(goal, dtype=float),
                radius=float(instance.env.robot_radius),
                itvl=Interval(float(t_start), float(t_end)),
            )
            specs.append(Instance.dynamic_obstacle_to_spec(obstacle))
        return specs

    @classmethod
    def instance_id(cls, entry: HeuristicAblationSelectionEntry) -> str:
        group_label = cls.INSTANCE_ID_GROUP_LABELS.get(
            entry.group,
            entry.group,
        )
        return f"st-heur-{group_label}-{entry.record.instance_id}"

    @classmethod
    def build_record(cls, entry: HeuristicAblationSelectionEntry) -> STHeuristicAblationRecord:
        base_instance = BaseInstanceFactory.from_record(entry.record, compute_heuristics=False)
        query = cls.query_for_entry(entry, base_instance)
        last_reason = ""
        for attempt in range(cls.MAX_ATTEMPTS):
            specs = cls.obstacle_specs(entry, base_instance, query, attempt=attempt)
            env = base_instance.env.copy()
            env.O_Dynamic = Instance.dynamic_obstacles_from_specs(specs)
            stgcs = Instance.build_stgcs_from_env(env, tmax=cls.TMAX, vlimit=cls.VLIMIT)
            valid, reason = stgcs.validate_query(query)
            if valid:
                return STHeuristicAblationRecord(
                    instance_id=cls.instance_id(entry),
                    group=entry.group,
                    source_domain=entry.source_domain,
                    base_instance_id=entry.record.instance_id,
                    query=STQuerySpec.from_query(query),
                    dynamic_obstacles=specs,
                    stgcs_num_vertices=int(stgcs.G.number_of_nodes()),
                    stgcs_num_edges=int(stgcs.G.number_of_edges()),
                )
            last_reason = "" if reason is None else reason
        raise ValueError(
            f"Unable to build a valid ST heuristic-ablation record for {entry.record.instance_id}: {last_reason}"
        )

    @classmethod
    def build_manifest(
        cls,
        entries: Sequence[HeuristicAblationSelectionEntry],
    ) -> list[STHeuristicAblationRecord]:
        records: list[STHeuristicAblationRecord] = []
        for entry in entries:
            record = cls.build_record(entry)
            records.append(record)
            print(
                f"{record.instance_id} | group={record.group} | obstacles={len(record.dynamic_obstacles)} "
                f"| sets={record.stgcs_num_vertices}"
            )
        return records

    @classmethod
    def group_counts(cls, records: Sequence[STHeuristicAblationRecord]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in records:
            counts[record.group] = counts.get(record.group, 0) + 1
        return counts

    @staticmethod
    def merge_records_by_instance_id(
        existing: Sequence[STHeuristicAblationRecord],
        incoming: Sequence[STHeuristicAblationRecord],
    ) -> list[STHeuristicAblationRecord]:
        return STPlanningManifestStore.merge_records_by_instance_id(existing, incoming)

    @classmethod
    def reconstruct_screen_instance(
        cls,
        record: STHeuristicAblationRecord,
        base_root: str | Path,
    ):
        return STPlanningManifestStore.reconstruct_instance(record, base_root)

    @classmethod
    def run_h_tab_screen(
        cls,
        record: STHeuristicAblationRecord,
        base_root: str | Path,
        budget: float,
    ) -> STHeuristicAblationResultEntry:
        return cls.run_screen_spec(
            record,
            base_root=base_root,
            budget=budget,
            spec=cls.INTERFACE_TO_SET_TABLE_SCREEN_SPEC,
        )

    @classmethod
    def run_screen_spec(
        cls,
        record: STHeuristicAblationRecord,
        base_root: str | Path,
        budget: float,
        spec: SearchPlannerSpec,
    ) -> STHeuristicAblationResultEntry:
        instance, query, base_manifest_path, base_record = cls.reconstruct_screen_instance(record, base_root)
        BaseOfflineHeuristicStore.prepare_instance_for_search(
            instance,
            base_manifest_path,
            base_record,
            required_heuristics={spec.heuristic},
            online_h_tab_timeout_secs=budget,
        )
        planner = MRMPExperiment._build_low_level_planner(
            instance,
            spec,
            runtime_limit_secs=float(budget),
        )
        sol, runtime, status = planner.plan(instance.stgcs, query)
        algorithm = planner.last_search_algorithm
        return STHeuristicAblationResultEntry(
            is_success=status != STPlanStatus.FAIL,
            runtime=float(runtime),
            cost=float("inf") if sol is None else float(sol.duration),
            num_expanded_nodes=0 if algorithm is None else int(algorithm.n_expanded),
            num_generated_nodes=0 if algorithm is None else int(algorithm.n_generated),
        )

    @classmethod
    def run_screen_specs(
        cls,
        record: STHeuristicAblationRecord,
        base_root: str | Path,
        budget: float,
        specs: Sequence[SearchPlannerSpec],
    ) -> Dict[str, STHeuristicAblationResultEntry]:
        return {
            spec.name: cls.run_screen_spec(record, base_root=base_root, budget=budget, spec=spec)
            for spec in specs
        }

    @classmethod
    def screen_records_with_h_tab(
        cls,
        entries: Sequence[HeuristicAblationSelectionEntry],
        target_group_counts: Dict[str, int],
        base_root: str | Path,
        budget: float,
    ) -> list[tuple[STHeuristicAblationRecord, Dict[str, STHeuristicAblationResultEntry]]]:
        kept: list[tuple[STHeuristicAblationRecord, Dict[str, STHeuristicAblationResultEntry]]] = []
        kept_counts = {group: 0 for group in target_group_counts}
        for entry in entries:
            target = int(target_group_counts.get(entry.group, 0))
            if target <= 0 or kept_counts.get(entry.group, 0) >= target:
                continue
            try:
                record = cls.build_record(entry)
                result = cls.run_h_tab_screen(record, base_root=base_root, budget=float(budget))
                results_by_planner = {cls.INTERFACE_TO_SET_TABLE_SCREEN_SPEC.name: result}
                passed = result.is_success
            except Exception as exc:
                print(f"{entry.record.instance_id} | group={entry.group} | screen=False | error={exc}")
                continue
            h_tab_result = results_by_planner[cls.INTERFACE_TO_SET_TABLE_SCREEN_SPEC.name]
            if not passed:
                print(
                    f"{record.instance_id} | group={record.group} | screen=False "
                    f"| runtime={h_tab_result.runtime:.3f}"
                )
                continue
            kept.append((record, results_by_planner))
            kept_counts[record.group] = kept_counts.get(record.group, 0) + 1
            print(
                f"{record.instance_id} | group={record.group} | screen=True "
                f"| kept={kept_counts[record.group]}/{target} | runtime={h_tab_result.runtime:.3f}"
            )
            if all(kept_counts.get(group, 0) >= int(count) for group, count in target_group_counts.items()):
                break
        missing = {
            group: int(count) - kept_counts.get(group, 0)
            for group, count in target_group_counts.items()
            if kept_counts.get(group, 0) < int(count)
        }
        if missing:
            raise RuntimeError(f"h_tab screening did not find enough successful records: missing {missing}.")
        return kept


class STHeuristicAblationManifestCLI:
    DEFAULT_OUTPUT = Path("data/instances/st_planning/manifest.json")
    DEFAULT_RESULT_OUTPUT_ROOT = Path("data/results/st_planning/heuristic_ablation")

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Build deterministic dynamic-obstacle ST instances for the BFS heuristic ablation."
        )
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--group-count", type=int, default=HeuristicAblationSelector.DEFAULT_GROUP_COUNT)
        parser.add_argument("--total-count", type=int, default=None)
        parser.add_argument("--output", type=Path, default=cls.DEFAULT_OUTPUT)
        parser.add_argument("--screen-with-h-tab", action="store_true")
        parser.add_argument(
            "--screen-budget",
            type=float,
            default=STHeuristicAblationManifestBuilder.DEFAULT_SCREEN_BUDGET,
        )
        parser.add_argument("--candidate-multiplier", type=int, default=3)
        parser.add_argument("--result-output-root", type=Path, default=None)
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        if args.total_count is not None and int(args.total_count) <= 0:
            raise ValueError(f"--total-count must be positive, got {args.total_count}")
        if int(args.group_count) <= 0:
            raise ValueError(f"--group-count must be positive, got {args.group_count}")
        if args.screen_with_h_tab and (not math.isfinite(float(args.screen_budget)) or float(args.screen_budget) <= 0.0):
            raise ValueError(f"--screen-budget must be positive, got {args.screen_budget}")
        if args.screen_with_h_tab and int(args.candidate_multiplier) <= 0:
            raise ValueError(f"--candidate-multiplier must be positive, got {args.candidate_multiplier}")

        records_by_domain = HeuristicAblationSelector.load_records_by_domain(Path(args.base_root).resolve())
        target_counts = HeuristicAblationSelector.target_group_counts(
            group_count=int(args.group_count),
            total_count=None if args.total_count is None else int(args.total_count),
        )
        if args.screen_with_h_tab:
            candidate_counts = {
                group: count * int(args.candidate_multiplier)
                for group, count in target_counts.items()
            }
            entries = HeuristicAblationSelector.build_selection_for_group_counts(
                records_by_domain,
                candidate_counts,
            )
            screened_rows = STHeuristicAblationManifestBuilder.screen_records_with_h_tab(
                entries,
                target_counts,
                base_root=Path(args.base_root).resolve(),
                budget=float(args.screen_budget),
            )
            records = [record for record, _ in screened_rows]
            result_output_root = (
                cls.DEFAULT_RESULT_OUTPUT_ROOT
                if args.result_output_root is None
                else Path(args.result_output_root)
            )
            completed_by_planner: Dict[str, set[tuple[str, float]]] = {}
            wrote_rows = 0
            for record, results_by_planner in screened_rows:
                for planner_name, result in results_by_planner.items():
                    result_path = STHeuristicAblationResultStore.planner_output_path(
                        result_output_root,
                        planner_name,
                    )
                    if planner_name not in completed_by_planner:
                        completed_by_planner[planner_name] = STHeuristicAblationResultStore.completed_result_keys(
                            result_path
                        )
                    key = STHeuristicAblationResultStore.result_key(record.instance_id, float(args.screen_budget))
                    if key in completed_by_planner[planner_name]:
                        continue
                    STHeuristicAblationResultStore.append_result_row(
                        result_path,
                        planner_name,
                        record,
                        float(args.screen_budget),
                        result,
                    )
                    completed_by_planner[planner_name].add(key)
                    wrote_rows += 1
            print(f"Appended {wrote_rows} screening result rows to {Path(result_output_root).resolve()}")
        else:
            entries = HeuristicAblationSelector.build_selection_for_group_counts(
                records_by_domain,
                target_counts,
            )
            records = STHeuristicAblationManifestBuilder.build_manifest(entries)
        output_path = Path(args.output)
        if output_path.exists():
            existing_records = STHeuristicAblationManifestBuilder.load_manifest(output_path)
            records = STHeuristicAblationManifestBuilder.merge_records_by_instance_id(existing_records, records)
        STHeuristicAblationManifestBuilder.save_manifest(output_path, records)
        print(
            f"Wrote {len(records)} ST heuristic-ablation records to {output_path.resolve()} "
            f"with groups {STHeuristicAblationManifestBuilder.group_counts(records)}"
        )


if __name__ == "__main__":
    STHeuristicAblationManifestCLI.main()
