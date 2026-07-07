from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmark.environment.obstacle import DynamicSphere
from benchmark.base import BaseInstanceFactory
from experiments.st_runners.heuristic_ablation_st_manifest import (
    STHeuristicAblationManifestBuilder,
    STHeuristicAblationRecord,
    STQuerySpec,
)
from benchmark.manifests.base import BaseBenchmarkRecord, load_manifest
from benchmark.instance import Instance
from stgcs.interval import Interval
from stgcs.st_planner import MPQuery


@dataclass(frozen=True)
class STDominationStressCandidate:
    record: STHeuristicAblationRecord


class STDominationStressManifestBuilder:
    DEFAULT_BASE_ROOT = Path("data/stgcs_base")
    DEFAULT_OUTPUT = Path("data/instances/st_planning/stress_test/manifest.json")
    DEFAULT_DOMAIN_KEYS = ("grid2d", "maze", "iris-2d")
    DEFAULT_COUNT_PER_DOMAIN = 15
    DEFAULT_COUNT = DEFAULT_COUNT_PER_DOMAIN * len(DEFAULT_DOMAIN_KEYS)
    DEFAULT_MIN_EDGES = 500
    DEFAULT_MAX_EDGES = 1500
    DEFAULT_EDGE_TOLERANCE = 0.0
    DEFAULT_BASE_CANDIDATE_LIMIT = 120
    DEFAULT_ATTEMPTS_PER_COUNT = 1
    DEFAULT_OBSTACLE_COUNTS = (1, 2, 3, 4)
    DEFAULT_MIN_GRID_SIZE = 3

    GROUP_BY_DOMAIN = {
        "grid2d": "grid2d",
        "grid3d": "grid3d",
        "maze": "maze",
        "iris-2d": "iris2d",
    }

    @staticmethod
    def edge_targets(count: int, min_edges: int, max_edges: int) -> tuple[int, ...]:
        target_count = int(count)
        low = int(min_edges)
        high = int(max_edges)
        if target_count <= 0:
            raise ValueError(f"count must be positive, got {count}.")
        if low <= 0 or high <= 0:
            raise ValueError(f"edge targets must be positive, got [{min_edges}, {max_edges}].")
        if low > high:
            raise ValueError(f"min_edges must be <= max_edges, got [{min_edges}, {max_edges}].")
        if target_count == 1:
            return (low,)
        return tuple(
            int(round(low + (high - low) * idx / (target_count - 1)))
            for idx in range(target_count)
        )

    @staticmethod
    def edge_bounds(min_edges: int, max_edges: int, edge_tolerance: float) -> tuple[int, int]:
        low = int(min_edges)
        high = int(max_edges)
        tolerance = float(edge_tolerance)
        if low <= 0 or high <= 0:
            raise ValueError(f"edge bounds must be positive, got [{min_edges}, {max_edges}].")
        if low > high:
            raise ValueError(f"min_edges must be <= max_edges, got [{min_edges}, {max_edges}].")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"edge_tolerance must be finite and non-negative, got {edge_tolerance}.")
        accepted_min_edges = max(1, int(math.floor(low * (1.0 - tolerance))))
        accepted_max_edges = max(accepted_min_edges, int(math.ceil(high * (1.0 + tolerance))))
        return accepted_min_edges, accepted_max_edges

    @staticmethod
    def grid_size(record: BaseBenchmarkRecord) -> int:
        return int(record.env_params.get("size", record.env_params.get("N", 0)))

    @classmethod
    def base_bucket(cls, domain_key: str, record: BaseBenchmarkRecord) -> tuple[int, ...]:
        if domain_key == "grid2d":
            return (cls.grid_size(record),)
        if domain_key == "maze":
            return (
                int(record.env_params.get("width", 0)),
                int(record.env_params.get("height", 0)),
            )
        if domain_key == "iris-2d":
            return (int(record.env_params.get("m", 0)),)
        raise ValueError(f"Unsupported stress-test domain key {domain_key!r}.")

    @classmethod
    def base_record_is_eligible(cls, domain_key: str, record: BaseBenchmarkRecord) -> bool:
        if int(record.stgcs_num_vertices) < 2:
            return False
        if domain_key == "grid2d":
            return cls.grid_size(record) >= cls.DEFAULT_MIN_GRID_SIZE
        return True

    @classmethod
    def load_base_records(
        cls,
        base_root: Path,
        domain_key: str,
        base_candidate_limit: int,
    ) -> list[BaseBenchmarkRecord]:
        if int(base_candidate_limit) <= 0:
            raise ValueError(f"base_candidate_limit must be positive, got {base_candidate_limit}.")
        manifest_path = Path(base_root) / domain_key / "manifest.json"
        records = [
            record for record in load_manifest(manifest_path)
            if cls.base_record_is_eligible(domain_key, record)
        ]
        records_by_bucket: dict[tuple[int, ...], list[BaseBenchmarkRecord]] = {}
        for record in records:
            records_by_bucket.setdefault(cls.base_bucket(domain_key, record), []).append(record)

        selected: list[BaseBenchmarkRecord] = []
        buckets = sorted(records_by_bucket)
        per_bucket = int(math.ceil(int(base_candidate_limit) / max(len(buckets), 1)))
        for bucket in buckets:
            selected.extend(
                sorted(
                    records_by_bucket[bucket],
                    key=lambda record: (
                        -int(record.stgcs_num_edges),
                        -int(record.stgcs_num_vertices),
                        int(record.spatial_seed),
                        str(record.instance_id),
                    ),
                )[:per_bucket]
            )
        return selected[: int(base_candidate_limit)]

    @classmethod
    def load_grid2d_base_records(cls, base_root: Path, base_candidate_limit: int) -> list[BaseBenchmarkRecord]:
        return cls.load_base_records(base_root, "grid2d", base_candidate_limit)

    @staticmethod
    def base_instance_from_record(record: BaseBenchmarkRecord):
        return BaseInstanceFactory.build(
            domain=record.domain,
            env_params=record.env_params,
            spatial_seed=record.spatial_seed,
            space_dim=record.space_dim,
            compute_heuristics=False,
        )

    @staticmethod
    def query_for_instance(instance) -> MPQuery:
        centers = STHeuristicAblationManifestBuilder.cspace_centers(instance)
        start, goal = STHeuristicAblationManifestBuilder.farthest_center_pair(centers)
        return MPQuery(
            np.asarray(start, dtype=float),
            np.asarray(goal, dtype=float),
            0.0,
            True,
            STHeuristicAblationManifestBuilder.VLIMIT,
        )

    @classmethod
    def obstacle_interval(
        cls,
        query,
        obstacle_index: int,
        obstacle_count: int,
        attempt: int,
    ) -> Interval:
        direct_time = float(np.linalg.norm(query.goal - query.start) / max(float(query.vlimit), 1e-9))
        active_window = max(4.0, min(20.0, direct_time * 1.5))
        phase_count = max(1, min(int(obstacle_count), 8))
        phase = (int(obstacle_index) + int(attempt)) % phase_count
        t_start = 0.25 + active_window * float(phase) / float(phase_count + 1)
        duration = max(2.0, min(8.0, direct_time * 0.45))
        return Interval(float(t_start), float(t_start + duration))

    @classmethod
    def obstacle_specs(
        cls,
        instance,
        query,
        obstacle_count: int,
        attempt: int,
    ) -> list[dict]:
        pairs = STHeuristicAblationManifestBuilder.ranked_obstacle_pairs(instance, query)
        stride = max(1, len(pairs) // max(int(obstacle_count), 1))
        specs: list[dict] = []
        for obs_idx in range(int(obstacle_count)):
            pair_idx = (int(attempt) + obs_idx * stride) % len(pairs)
            start, goal = pairs[pair_idx]
            if (obs_idx + attempt) % 2 == 1:
                start, goal = goal, start
            obstacle = DynamicSphere(
                np.asarray(start, dtype=float),
                np.asarray(goal, dtype=float),
                radius=float(instance.env.robot_radius),
                itvl=cls.obstacle_interval(query, obs_idx, int(obstacle_count), int(attempt)),
            )
            specs.append(Instance.dynamic_obstacle_to_spec(obstacle))
        return specs

    @classmethod
    def instance_id(
        cls,
        domain: str,
        base_record: BaseBenchmarkRecord,
        obstacle_count: int,
        attempt: int,
    ) -> str:
        domain_label = domain.replace("-", "")
        return f"st-stress-{domain_label}-{base_record.instance_id}-o{int(obstacle_count):02d}-a{int(attempt):02d}"

    @classmethod
    def build_record(
        cls,
        domain: str,
        base_record: BaseBenchmarkRecord,
        obstacle_count: int,
        attempt: int,
    ) -> STHeuristicAblationRecord | None:
        base_instance = cls.base_instance_from_record(base_record)
        query = cls.query_for_instance(base_instance)
        return cls.build_record_from_instance(domain, base_record, base_instance, query, obstacle_count, attempt)

    @classmethod
    def build_record_from_instance(
        cls,
        domain: str,
        base_record: BaseBenchmarkRecord,
        base_instance,
        query: MPQuery,
        obstacle_count: int,
        attempt: int,
    ) -> STHeuristicAblationRecord | None:
        specs = cls.obstacle_specs(base_instance, query, obstacle_count, attempt)
        env = base_instance.env.copy()
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(specs)
        stgcs = Instance.build_stgcs_from_env(
            env,
            tmax=STHeuristicAblationManifestBuilder.TMAX,
            vlimit=STHeuristicAblationManifestBuilder.VLIMIT,
        )
        valid, _ = stgcs.validate_query(query)
        if not valid:
            return None
        return STHeuristicAblationRecord(
            instance_id=cls.instance_id(domain, base_record, obstacle_count, attempt),
            group=cls.GROUP_BY_DOMAIN[domain],
            source_domain=domain,
            base_instance_id=base_record.instance_id,
            query=STQuerySpec.from_query(query),
            dynamic_obstacles=specs,
            stgcs_num_vertices=int(stgcs.G.number_of_nodes()),
            stgcs_num_edges=int(stgcs.G.number_of_edges()),
        )

    @staticmethod
    def edge_gap(candidate: STDominationStressCandidate, target_edges: int) -> tuple[int, int, str]:
        edges = int(candidate.record.stgcs_num_edges)
        return abs(edges - int(target_edges)), edges, candidate.record.instance_id

    @classmethod
    def candidate_pool(
        cls,
        base_root: Path,
        min_edges: int,
        max_edges: int,
        edge_tolerance: float,
        obstacle_counts: Sequence[int],
        attempts_per_count: int,
        base_candidate_limit: int,
        domain_keys: Sequence[str] | None = None,
    ) -> list[STDominationStressCandidate]:
        accepted_min_edges, accepted_max_edges = cls.edge_bounds(min_edges, max_edges, edge_tolerance)
        candidates: list[STDominationStressCandidate] = []
        for domain_key in tuple(cls.DEFAULT_DOMAIN_KEYS if domain_keys is None else domain_keys):
            for base_record in cls.load_base_records(base_root, domain_key, base_candidate_limit):
                try:
                    base_instance = cls.base_instance_from_record(base_record)
                    query = cls.query_for_instance(base_instance)
                except ValueError:
                    continue
                for obstacle_count in obstacle_counts:
                    for attempt in range(int(attempts_per_count)):
                        record = cls.build_record_from_instance(
                            domain_key,
                            base_record,
                            base_instance,
                            query,
                            int(obstacle_count),
                            attempt,
                        )
                        if record is None:
                            continue
                        if not (accepted_min_edges <= int(record.stgcs_num_edges) <= accepted_max_edges):
                            continue
                        candidates.append(STDominationStressCandidate(record=record))
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.record.source_domain,
                int(candidate.record.stgcs_num_edges),
                candidate.record.base_instance_id,
                candidate.record.instance_id,
            ),
        )

    @classmethod
    def domain_counts(cls, count: int, domain_keys: Sequence[str]) -> dict[str, int]:
        total_count = int(count)
        domains = tuple(str(domain_key) for domain_key in domain_keys)
        if total_count <= 0:
            raise ValueError(f"count must be positive, got {count}.")
        if not domains:
            raise ValueError("At least one stress-test domain is required.")
        if total_count % len(domains) != 0:
            raise ValueError(
                f"count={total_count} must split evenly across {len(domains)} stress-test domains."
            )
        per_domain = total_count // len(domains)
        return {domain_key: per_domain for domain_key in domains}

    @classmethod
    def domain_edge_targets(
        cls,
        count: int,
        domain_keys: Sequence[str],
        min_edges: int,
        max_edges: int,
    ) -> dict[str, tuple[int, ...]]:
        return {
            domain_key: cls.edge_targets(domain_count, min_edges, max_edges)
            for domain_key, domain_count in cls.domain_counts(count, domain_keys).items()
        }

    @classmethod
    def existing_counts_by_domain(
        cls,
        records: Sequence[STHeuristicAblationRecord],
        domain_keys: Sequence[str],
    ) -> Counter[str]:
        domains = set(domain_keys)
        seen_instance_ids: set[str] = set()
        counts: Counter[str] = Counter()
        for record in records:
            if record.instance_id in seen_instance_ids:
                raise ValueError(f"Duplicate existing stress-test instance_id {record.instance_id!r}.")
            seen_instance_ids.add(record.instance_id)
            if record.source_domain not in domains:
                raise ValueError(
                    f"Existing stress-test record {record.instance_id!r} has unsupported "
                    f"source_domain {record.source_domain!r}; expected one of {tuple(domain_keys)!r}."
                )
            counts[record.source_domain] += 1
        return counts

    @classmethod
    def validated_existing_counts_by_domain(
        cls,
        existing_records: Sequence[STHeuristicAblationRecord],
        count: int,
        domain_keys: Sequence[str],
    ) -> tuple[dict[str, int], Counter[str]]:
        target_counts = cls.domain_counts(count, domain_keys)
        existing_counts = cls.existing_counts_by_domain(existing_records, domain_keys)
        for domain_key in domain_keys:
            target_count = int(target_counts[domain_key])
            existing_count = int(existing_counts[domain_key])
            if existing_count > target_count:
                raise ValueError(
                    f"Existing manifest already has {existing_count} {domain_key} records, "
                    f"which exceeds target count {target_count}."
                )
        return target_counts, existing_counts

    @classmethod
    def append_edge_targets(
        cls,
        existing_records: Sequence[STHeuristicAblationRecord],
        target_count: int,
        append_count: int,
        min_edges: int,
        max_edges: int,
    ) -> tuple[int, ...]:
        if int(append_count) <= 0:
            return ()
        targets = cls.edge_targets(target_count, min_edges, max_edges)
        if int(append_count) > len(targets):
            raise ValueError(
                f"append_count={append_count} cannot exceed target_count={target_count}."
            )
        existing_edges = tuple(int(record.stgcs_num_edges) for record in existing_records)
        if not existing_edges:
            return targets[: int(append_count)]
        ranked_targets = sorted(
            targets,
            key=lambda target_edges: (
                -min(abs(int(edge_count) - int(target_edges)) for edge_count in existing_edges),
                int(target_edges),
            ),
        )
        return tuple(sorted(ranked_targets[: int(append_count)]))

    @classmethod
    def build_candidate_with_cache(
        cls,
        domain_key: str,
        base_record: BaseBenchmarkRecord,
        obstacle_count: int,
        attempt: int,
        context_cache: dict[tuple[str, str], tuple[object, MPQuery] | None],
        candidate_cache: dict[tuple[str, str, int, int], STDominationStressCandidate | None],
    ) -> STDominationStressCandidate | None:
        key = (domain_key, base_record.instance_id, int(obstacle_count), int(attempt))
        if key in candidate_cache:
            return candidate_cache[key]

        context_key = (domain_key, base_record.instance_id)
        if context_key not in context_cache:
            try:
                base_instance = cls.base_instance_from_record(base_record)
                query = cls.query_for_instance(base_instance)
                context_cache[context_key] = (base_instance, query)
            except ValueError:
                context_cache[context_key] = None

        context = context_cache[context_key]
        if context is None:
            candidate_cache[key] = None
            return None

        base_instance, query = context
        record = cls.build_record_from_instance(
            domain_key,
            base_record,
            base_instance,
            query,
            int(obstacle_count),
            int(attempt),
        )
        candidate = None if record is None else STDominationStressCandidate(record=record)
        candidate_cache[key] = candidate
        return candidate

    @classmethod
    def find_candidate_for_target(
        cls,
        domain_key: str,
        target_edges: int,
        base_records: Sequence[BaseBenchmarkRecord],
        obstacle_counts: Sequence[int],
        attempts_per_count: int,
        accepted_min_edges: int,
        accepted_max_edges: int,
        used_instance_ids: set[str],
        context_cache: dict[tuple[str, str], tuple[object, MPQuery] | None],
        candidate_cache: dict[tuple[str, str, int, int], STDominationStressCandidate | None],
        early_stop_gap: int,
    ) -> STDominationStressCandidate | None:
        best: STDominationStressCandidate | None = None
        for base_record in base_records:
            for obstacle_count in obstacle_counts:
                for attempt in range(int(attempts_per_count)):
                    candidate = cls.build_candidate_with_cache(
                        domain_key,
                        base_record,
                        int(obstacle_count),
                        int(attempt),
                        context_cache,
                        candidate_cache,
                    )
                    if candidate is None:
                        continue
                    edge_count = int(candidate.record.stgcs_num_edges)
                    if not (int(accepted_min_edges) <= edge_count <= int(accepted_max_edges)):
                        continue
                    if candidate.record.instance_id in used_instance_ids:
                        continue
                    if best is None or cls.edge_gap(candidate, target_edges) < cls.edge_gap(best, target_edges):
                        best = candidate
                        if cls.edge_gap(best, target_edges)[0] <= int(early_stop_gap):
                            return best
        return best

    @staticmethod
    def save_progress(output_path: Path | None, records: Sequence[STHeuristicAblationRecord]) -> None:
        if output_path is not None:
            sorted_records = sorted(
                records,
                key=lambda record: (
                    int(record.stgcs_num_edges),
                    str(record.source_domain),
                    str(record.base_instance_id),
                    str(record.instance_id),
                ),
            )
            STHeuristicAblationManifestBuilder.save_manifest(Path(output_path), sorted_records)

    @classmethod
    def select_records_for_targets(
        cls,
        targets: Sequence[int],
        candidates: Sequence[STDominationStressCandidate],
        output_path: Path | None,
    ) -> list[STHeuristicAblationRecord]:
        records: list[STHeuristicAblationRecord] = []
        used_instance_ids: set[str] = set()
        cls.save_progress(output_path, records)
        for target_edges in targets:
            eligible = [
                candidate for candidate in candidates
                if candidate.record.instance_id not in used_instance_ids
            ]
            if not eligible:
                raise RuntimeError(
                    f"No unused stress-test candidate remains for target {target_edges} edges."
                )
            candidate = min(eligible, key=lambda item: cls.edge_gap(item, int(target_edges)))
            records.append(candidate.record)
            used_instance_ids.add(candidate.record.instance_id)
            cls.save_progress(output_path, records)
            print(
                f"{candidate.record.instance_id} | target_edges={int(target_edges)} "
                f"| obstacles={len(candidate.record.dynamic_obstacles)} "
                f"| sets={candidate.record.stgcs_num_vertices} "
                f"| edges={candidate.record.stgcs_num_edges} "
                f"| saved={len(records)}/{len(targets)}"
            )
        return records

    @classmethod
    def build_manifest(
        cls,
        base_root: Path,
        count: int,
        min_edges: int,
        max_edges: int,
        edge_tolerance: float,
        obstacle_counts: Sequence[int],
        attempts_per_count: int,
        base_candidate_limit: int,
        output_path: Path | None = None,
    ) -> list[STHeuristicAblationRecord]:
        domain_keys = cls.DEFAULT_DOMAIN_KEYS
        targets_by_domain = cls.domain_edge_targets(count, domain_keys, min_edges, max_edges)
        total_targets = sum(len(targets) for targets in targets_by_domain.values())
        cls.save_progress(output_path, [])
        accepted_min_edges, accepted_max_edges = cls.edge_bounds(min_edges, max_edges, edge_tolerance)
        base_records_by_domain = {
            domain_key: cls.load_base_records(base_root, domain_key, base_candidate_limit)
            for domain_key in domain_keys
        }
        target_steps = (
            abs(rhs - lhs)
            for targets in targets_by_domain.values()
            for lhs, rhs in zip(targets[:-1], targets[1:])
        )
        target_step = max(target_steps, default=0)
        early_stop_gap = max(25, int(math.ceil(target_step / 2.0)))
        records: list[STHeuristicAblationRecord] = []
        used_instance_ids: set[str] = set()
        context_cache: dict[tuple[str, str], tuple[object, MPQuery] | None] = {}
        candidate_cache: dict[tuple[str, str, int, int], STDominationStressCandidate | None] = {}

        for target_index in range(max((len(targets) for targets in targets_by_domain.values()), default=0)):
            for domain_key in domain_keys:
                targets = targets_by_domain[domain_key]
                if target_index >= len(targets):
                    continue
                target_edges = int(targets[target_index])
                candidate = cls.find_candidate_for_target(
                    domain_key,
                    target_edges,
                    base_records_by_domain[domain_key],
                    obstacle_counts,
                    int(attempts_per_count),
                    accepted_min_edges,
                    accepted_max_edges,
                    used_instance_ids,
                    context_cache,
                    candidate_cache,
                    early_stop_gap,
                )
                if candidate is None:
                    raise RuntimeError(
                        f"No unused {domain_key} stress-test candidate found for target {target_edges} edges "
                        f"in the accepted edge range [{accepted_min_edges}, {accepted_max_edges}]."
                    )
                records.append(candidate.record)
                used_instance_ids.add(candidate.record.instance_id)
                cls.save_progress(output_path, records)
                print(
                    f"{candidate.record.instance_id} | domain={domain_key} | target_edges={target_edges} "
                    f"| obstacles={len(candidate.record.dynamic_obstacles)} "
                    f"| sets={candidate.record.stgcs_num_vertices} "
                    f"| edges={candidate.record.stgcs_num_edges} "
                    f"| saved={len(records)}/{total_targets}"
                )
        return sorted(
            records,
            key=lambda record: (
                int(record.stgcs_num_edges),
                str(record.source_domain),
                str(record.base_instance_id),
                str(record.instance_id),
            ),
        )

    @classmethod
    def extend_manifest(
        cls,
        base_root: Path,
        existing_records: Sequence[STHeuristicAblationRecord],
        count: int,
        min_edges: int,
        max_edges: int,
        edge_tolerance: float,
        obstacle_counts: Sequence[int],
        attempts_per_count: int,
        base_candidate_limit: int,
        output_path: Path | None = None,
    ) -> list[STHeuristicAblationRecord]:
        domain_keys = cls.DEFAULT_DOMAIN_KEYS
        target_counts, existing_counts = cls.validated_existing_counts_by_domain(
            existing_records,
            count,
            domain_keys,
        )
        existing_by_domain = {
            domain_key: [
                record for record in existing_records
                if record.source_domain == domain_key
            ]
            for domain_key in domain_keys
        }
        append_targets_by_domain: dict[str, tuple[int, ...]] = {}
        for domain_key in domain_keys:
            target_count = int(target_counts[domain_key])
            existing_count = int(existing_counts[domain_key])
            append_targets_by_domain[domain_key] = cls.append_edge_targets(
                existing_by_domain[domain_key],
                target_count,
                target_count - existing_count,
                min_edges,
                max_edges,
            )

        total_new_targets = sum(len(targets) for targets in append_targets_by_domain.values())
        if total_new_targets == 0:
            return list(existing_records)

        accepted_min_edges, accepted_max_edges = cls.edge_bounds(min_edges, max_edges, edge_tolerance)
        base_records_by_domain = {
            domain_key: cls.load_base_records(base_root, domain_key, base_candidate_limit)
            for domain_key in domain_keys
        }
        all_targets_by_domain = cls.domain_edge_targets(count, domain_keys, min_edges, max_edges)
        target_steps = (
            abs(rhs - lhs)
            for targets in all_targets_by_domain.values()
            for lhs, rhs in zip(targets[:-1], targets[1:])
        )
        target_step = max(target_steps, default=0)
        early_stop_gap = max(25, int(math.ceil(target_step / 2.0)))
        records = list(existing_records)
        used_instance_ids = {record.instance_id for record in existing_records}
        context_cache: dict[tuple[str, str], tuple[object, MPQuery] | None] = {}
        candidate_cache: dict[tuple[str, str, int, int], STDominationStressCandidate | None] = {}
        saved_count = 0

        for target_index in range(max((len(targets) for targets in append_targets_by_domain.values()), default=0)):
            for domain_key in domain_keys:
                targets = append_targets_by_domain[domain_key]
                if target_index >= len(targets):
                    continue
                target_edges = int(targets[target_index])
                candidate = cls.find_candidate_for_target(
                    domain_key,
                    target_edges,
                    base_records_by_domain[domain_key],
                    obstacle_counts,
                    int(attempts_per_count),
                    accepted_min_edges,
                    accepted_max_edges,
                    used_instance_ids,
                    context_cache,
                    candidate_cache,
                    early_stop_gap,
                )
                if candidate is None:
                    raise RuntimeError(
                        f"No unused {domain_key} stress-test candidate found for appended target "
                        f"{target_edges} edges in the accepted edge range "
                        f"[{accepted_min_edges}, {accepted_max_edges}]."
                    )
                records.append(candidate.record)
                used_instance_ids.add(candidate.record.instance_id)
                saved_count += 1
                if output_path is not None:
                    STHeuristicAblationManifestBuilder.save_manifest(Path(output_path), records)
                print(
                    f"{candidate.record.instance_id} | domain={domain_key} | target_edges={target_edges} "
                    f"| obstacles={len(candidate.record.dynamic_obstacles)} "
                    f"| sets={candidate.record.stgcs_num_vertices} "
                    f"| edges={candidate.record.stgcs_num_edges} "
                    f"| appended={saved_count}/{total_new_targets}"
                )
        return records


class STDominationStressManifestCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Build an ST domination-check stress-test manifest from existing data/stgcs_base records "
                "by adding dynamic obstacles until the constructed ST-GCS is near the target size."
            )
        )
        parser.add_argument("--base-root", type=Path, default=STDominationStressManifestBuilder.DEFAULT_BASE_ROOT)
        parser.add_argument("--output", type=Path, default=STDominationStressManifestBuilder.DEFAULT_OUTPUT)
        parser.add_argument(
            "--count",
            "--k",
            dest="count",
            type=int,
            default=STDominationStressManifestBuilder.DEFAULT_COUNT,
        )
        parser.add_argument("--min-edges", type=int, default=STDominationStressManifestBuilder.DEFAULT_MIN_EDGES)
        parser.add_argument("--max-edges", type=int, default=STDominationStressManifestBuilder.DEFAULT_MAX_EDGES)
        parser.add_argument(
            "--edge-tolerance",
            type=float,
            default=STDominationStressManifestBuilder.DEFAULT_EDGE_TOLERANCE,
        )
        parser.add_argument(
            "--obstacle-counts",
            type=int,
            nargs="+",
            default=STDominationStressManifestBuilder.DEFAULT_OBSTACLE_COUNTS,
        )
        parser.add_argument(
            "--attempts-per-count",
            type=int,
            default=STDominationStressManifestBuilder.DEFAULT_ATTEMPTS_PER_COUNT,
        )
        parser.add_argument(
            "--base-candidate-limit",
            type=int,
            default=STDominationStressManifestBuilder.DEFAULT_BASE_CANDIDATE_LIMIT,
        )
        parser.add_argument(
            "--append-existing",
            action="store_true",
            help=(
                "Force extension of the existing --output manifest to --count records. "
                "Existing partial manifests are resumed by default."
            ),
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        output_path = Path(args.output)
        count = int(args.count)
        existing_records = None
        if output_path.exists():
            existing_records = STHeuristicAblationManifestBuilder.load_manifest(output_path)
            STDominationStressManifestBuilder.validated_existing_counts_by_domain(
                existing_records,
                count,
                STDominationStressManifestBuilder.DEFAULT_DOMAIN_KEYS,
            )

        if existing_records is not None and (args.append_existing or len(existing_records) < count):
            original_count = len(existing_records)
            records = STDominationStressManifestBuilder.extend_manifest(
                base_root=Path(args.base_root),
                existing_records=existing_records,
                count=count,
                min_edges=int(args.min_edges),
                max_edges=int(args.max_edges),
                edge_tolerance=float(args.edge_tolerance),
                obstacle_counts=tuple(int(value) for value in args.obstacle_counts),
                attempts_per_count=int(args.attempts_per_count),
                base_candidate_limit=int(args.base_candidate_limit),
                output_path=output_path,
            )
            print(
                f"Appended {len(records) - original_count} ST domination stress-test records "
                f"to {output_path.resolve()}"
            )
        elif existing_records is not None:
            records = list(existing_records)
            print(
                f"Existing ST domination stress-test manifest already has "
                f"{len(records)}/{count} records at {output_path.resolve()}"
            )
        elif args.append_existing:
            existing_records = STHeuristicAblationManifestBuilder.load_manifest(Path(args.output))
            records = STDominationStressManifestBuilder.extend_manifest(
                base_root=Path(args.base_root),
                existing_records=existing_records,
                count=count,
                min_edges=int(args.min_edges),
                max_edges=int(args.max_edges),
                edge_tolerance=float(args.edge_tolerance),
                obstacle_counts=tuple(int(value) for value in args.obstacle_counts),
                attempts_per_count=int(args.attempts_per_count),
                base_candidate_limit=int(args.base_candidate_limit),
                output_path=output_path,
            )
            print(
                f"Appended {len(records) - len(existing_records)} ST domination stress-test records "
                f"to {output_path.resolve()}"
            )
        else:
            records = STDominationStressManifestBuilder.build_manifest(
                base_root=Path(args.base_root),
                count=count,
                min_edges=int(args.min_edges),
                max_edges=int(args.max_edges),
                edge_tolerance=float(args.edge_tolerance),
                obstacle_counts=tuple(int(value) for value in args.obstacle_counts),
                attempts_per_count=int(args.attempts_per_count),
                base_candidate_limit=int(args.base_candidate_limit),
                output_path=output_path,
            )
            print(f"Wrote {len(records)} ST domination stress-test records to {output_path.resolve()}")


if __name__ == "__main__":
    STDominationStressManifestCLI.main()
