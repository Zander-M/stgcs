from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Dict, Sequence

from benchmark.base import BaseInstanceFactory, BaseManifestStore
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from benchmark.manifests.base import BaseBenchmarkRecord, load_manifest
from stgcs.bfs.heuristics import InterfaceToSetCostTableHeuristic


class BaseOfflineHeuristicCLI:
    _BASE_RECORD_KEYS = frozenset(
        {
            "instance_id",
            "domain",
            "space_dim",
            "spatial_seed",
            "env_params",
            "supports_sampling",
        }
    )
    _ST_HEURISTIC_ABLATION_KEYS = frozenset({"source_domain", "base_instance_id"})

    @staticmethod
    def _requested_heuristics(h_tab_timeout_secs: float | None) -> set[str]:
        requested = {
            BaseOfflineHeuristicStore.MOTION_ONLY_NAME,
            BaseOfflineHeuristicStore.TRIPLET_RELAXATION_NAME,
        }
        if h_tab_timeout_secs is not None:
            requested.add(BaseOfflineHeuristicStore.INTERFACE_TO_SET_COST_TABLE_NAME)
        return requested

    @classmethod
    def _is_base_record_payload(cls, payload: object) -> bool:
        return isinstance(payload, dict) and cls._BASE_RECORD_KEYS.issubset(payload)

    @classmethod
    def _is_st_heuristic_ablation_payload(cls, payload: object) -> bool:
        return isinstance(payload, dict) and cls._ST_HEURISTIC_ABLATION_KEYS.issubset(payload)

    @staticmethod
    def _default_base_root_for_st_manifest(manifest_path: Path) -> Path:
        manifest_path = Path(manifest_path).resolve()
        parts = manifest_path.parts
        for idx, part in enumerate(parts[:-1]):
            if part == "data" and idx + 1 < len(parts) and parts[idx + 1] == "instances":
                return Path(*parts[: idx + 1]) / "stgcs_base"
        if manifest_path.parent.name == "st_planning":
            return manifest_path.parent.parent / "stgcs_base"
        if manifest_path.parent.parent.name == "st_planning":
            return manifest_path.parent.parent.parent / "stgcs_base"
        raise ValueError(
            f"Unable to infer base root from ST manifest {manifest_path}. "
            "Pass --base-root explicitly."
        )

    @classmethod
    def _base_root_for_st_manifest(cls, manifest_path: Path, base_root: Path | None) -> Path:
        if base_root is not None:
            return Path(base_root).resolve()
        return cls._default_base_root_for_st_manifest(manifest_path).resolve()

    @classmethod
    def _load_work_items(
        cls,
        manifest_path: Path,
        base_root: Path | None,
    ) -> list[tuple[Path, BaseBenchmarkRecord]]:
        manifest_path = Path(manifest_path).resolve()
        payload = json.loads(manifest_path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected {manifest_path} to contain a list of records.")
        if not payload:
            return []

        if all(cls._is_base_record_payload(item) for item in payload):
            return [(manifest_path, record) for record in load_manifest(manifest_path)]

        if all(cls._is_st_heuristic_ablation_payload(item) for item in payload):
            resolved_base_root = cls._base_root_for_st_manifest(manifest_path, base_root)
            seen: set[tuple[Path, str]] = set()
            work_items: list[tuple[Path, BaseBenchmarkRecord]] = []
            for item in payload:
                domain_key = str(item["source_domain"])
                base_instance_id = str(item["base_instance_id"])
                base_manifest_path = BaseManifestStore.manifest_path(resolved_base_root, domain_key=domain_key)
                key = (base_manifest_path.resolve(), base_instance_id)
                if key in seen:
                    continue
                seen.add(key)
                work_items.append(
                    (
                        base_manifest_path,
                        BaseManifestStore.record_by_id(base_manifest_path, base_instance_id),
                    )
                )
            return work_items

        raise ValueError(
            f"Unsupported manifest schema in {manifest_path}. Expected a base manifest or an ST "
            "heuristic-ablation manifest with source_domain/base_instance_id fields."
        )

    @classmethod
    def _work_items_to_update(
        cls,
        work_items: Sequence[tuple[Path, BaseBenchmarkRecord]],
        h_tab_timeout_secs: float | None,
        force: bool,
    ) -> list[tuple[Path, BaseBenchmarkRecord]]:
        if force:
            return list(work_items)

        required_heuristics = cls._requested_heuristics(h_tab_timeout_secs)
        pending: list[tuple[Path, BaseBenchmarkRecord]] = []
        for manifest_path, record in work_items:
            manifest_path = Path(manifest_path).resolve()
            if not BaseOfflineHeuristicStore.has_complete_caches(
                manifest_path,
                record,
                required_heuristics,
            ):
                pending.append((manifest_path, record))
        return pending

    @classmethod
    def _records_to_update(
        cls,
        manifest_path: Path,
        records: Sequence[BaseBenchmarkRecord],
        h_tab_timeout_secs: float | None,
        force: bool,
    ) -> list[BaseBenchmarkRecord]:
        if force:
            return list(records)

        work_items = [(Path(manifest_path).resolve(), record) for record in records]
        return [record for _, record in cls._work_items_to_update(work_items, h_tab_timeout_secs, force=False)]

    @staticmethod
    def _build_record_caches(
        manifest_path: str,
        record_payload: Dict[str, object],
        h_tab_timeout_secs: float | None,
        force: bool,
    ) -> tuple[str, Dict[str, float]]:
        manifest_file = Path(manifest_path).resolve()
        record = BaseBenchmarkRecord.from_dict(record_payload).with_manifest_path(manifest_file)
        instance = BaseInstanceFactory.from_record(record, compute_heuristics=False)
        times: Dict[str, float] = {}

        times[BaseOfflineHeuristicStore.MOTION_ONLY_NAME] = float(
            BaseOfflineHeuristicStore.ensure_motion_only(instance).computation_time
        )

        h_tri_path = BaseOfflineHeuristicStore.triplet_relaxation_backbone_path(manifest_file, record)
        if force:
            h_tri_path.unlink(missing_ok=True)
        if h_tri_path.exists():
            BaseOfflineHeuristicStore.load_cached_triplet_relaxation(instance, manifest_file, record)
        else:
            BaseOfflineHeuristicStore.ensure_triplet_relaxation(instance)
            h_tri_path.parent.mkdir(parents=True, exist_ok=True)
            instance.triplet_relaxation_heuristic.save_backbone(str(h_tri_path))
        times[BaseOfflineHeuristicStore.TRIPLET_RELAXATION_NAME] = float(
            instance.triplet_relaxation_heuristic.backbone_computation_time
        )

        if h_tab_timeout_secs is not None:
            h_tab_path = BaseOfflineHeuristicStore.interface_to_set_table_cache_path(manifest_file, record)
            if force:
                h_tab_path.unlink(missing_ok=True)
            if h_tab_path.exists():
                instance.interface_to_set_cost_table_heuristic = InterfaceToSetCostTableHeuristic.load(str(h_tab_path))
            else:
                base_gcs = instance.stgcs.get_gcs_instance().gcs
                h_tab_heuristic = InterfaceToSetCostTableHeuristic(
                    instance.stgcs,
                    base_gcs,
                    instance.stgcs.vlimit,
                    instance.triplet_relaxation_heuristic,
                    timeout_secs=float(h_tab_timeout_secs),
                )
                if not h_tab_heuristic._successful:
                    raise RuntimeError(
                        f"h_tab precomputation timed out for base {record.instance_id} at {float(h_tab_timeout_secs):.1f}s."
                    )
                h_tab_path.parent.mkdir(parents=True, exist_ok=True)
                h_tab_heuristic.save(str(h_tab_path))
                instance.interface_to_set_cost_table_heuristic = h_tab_heuristic
            times[BaseOfflineHeuristicStore.INTERFACE_TO_SET_COST_TABLE_NAME] = float(
                instance.interface_to_set_cost_table_heuristic.computation_time
            )

        return record.instance_id, times

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Build shared base h_tri/h_tab heuristic caches."
        )
        parser.add_argument("manifest", type=Path)
        parser.add_argument(
            "--base-root",
            type=Path,
            default=None,
            help="Base manifest root used when the input is an ST heuristic-ablation manifest.",
        )
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--h-tab-timeout-secs", type=float, default=None)
        parser.add_argument("--force", action="store_true")
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        manifest_path = Path(args.manifest).resolve()
        if args.h_tab_timeout_secs is not None and float(args.h_tab_timeout_secs) <= 0.0:
            raise ValueError(f"--h-tab-timeout-secs must be > 0, got {args.h_tab_timeout_secs}")
        work_items = cls._load_work_items(manifest_path, args.base_root)
        if not work_items:
            print(f"No base records found in {manifest_path}")
            return

        h_tab_timeout_secs = None if args.h_tab_timeout_secs is None else float(args.h_tab_timeout_secs)
        work_items_to_update = cls._work_items_to_update(
            work_items,
            h_tab_timeout_secs,
            bool(args.force),
        )
        skipped_count = len(work_items) - len(work_items_to_update)
        if skipped_count:
            print(f"skipped {skipped_count} already complete base instances")
        if not work_items_to_update:
            requested = ", ".join(sorted(cls._requested_heuristics(h_tab_timeout_secs)))
            print(f"{manifest_path}: all {len(work_items)} base instances already have {requested} caches")
            return

        completed: Dict[str, Dict[str, float]] = {}
        failures: Dict[str, str] = {}
        worker_count = max(1, int(args.workers))

        if worker_count == 1:
            for item_manifest_path, record in work_items_to_update:
                try:
                    instance_id, times = cls._build_record_caches(
                        str(item_manifest_path),
                        record.to_dict(),
                        h_tab_timeout_secs,
                        bool(args.force),
                    )
                    completed[instance_id] = times
                    print(f"updated {instance_id}")
                except Exception as exc:
                    failures[record.instance_id] = str(exc)
        else:
            try:
                executor_context = ProcessPoolExecutor(max_workers=worker_count)
            except PermissionError:
                executor_context = ThreadPoolExecutor(max_workers=worker_count)
            with executor_context as executor:
                futures = {
                    executor.submit(
                        cls._build_record_caches,
                        str(item_manifest_path),
                        record.to_dict(),
                        h_tab_timeout_secs,
                        bool(args.force),
                    ): record.instance_id
                    for item_manifest_path, record in work_items_to_update
                }
                for future in as_completed(futures):
                    instance_id = futures[future]
                    try:
                        built_id, times = future.result()
                        completed[built_id] = times
                        print(f"updated {built_id}")
                    except Exception as exc:
                        failures[instance_id] = str(exc)

        print(
            f"{manifest_path}: updated caches for {len(completed)} of {len(work_items_to_update)} pending base "
            f"instances ({skipped_count} already complete, {len(work_items)} total)"
        )
        if failures:
            for instance_id in sorted(failures):
                print(f"{instance_id}: {failures[instance_id]}")
            raise RuntimeError(f"Failed to build caches for {len(failures)} base instances.")


if __name__ == "__main__":
    BaseOfflineHeuristicCLI.main()
