from __future__ import annotations

from pathlib import Path
from typing import Sequence

from benchmark.manifests.base import BaseBenchmarkRecord
from benchmark.instance import Instance
from stgcs.bfs.heuristics import TripletRelaxationHeuristic, MotionOnlyHeuristic, InterfaceToSetCostTableHeuristic


class BaseOfflineHeuristicStore:
    ZERO_NAME = "h_zero"
    MOTION_ONLY_NAME = "h_mot"
    TRIPLET_RELAXATION_NAME = "h_tri"
    INTERFACE_TO_SET_COST_TABLE_NAME = "h_tab"
    MAX_NAME = "h_max"

    _TRIPLET_RELAXATION_BACKBONE_FILENAME = "h_tri_backbone.pkl"
    _INTERFACE_TO_SET_COST_TABLE_FILENAME = "h_tab.pkl"

    @staticmethod
    def _info(message: str) -> None:
        print(f"[stgcs-base] {message}")

    @classmethod
    def heuristics_dir(cls, manifest_path: str | Path) -> Path:
        return Path(manifest_path).resolve().parent / "heuristics"

    @classmethod
    def cache_root(cls, manifest_path: str | Path) -> Path:
        return cls.heuristics_dir(manifest_path)

    @classmethod
    def base_dir(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> Path:
        return cls.cache_root(manifest_path) / record.instance_id

    @classmethod
    def triplet_relaxation_backbone_path(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> Path:
        return cls.base_dir(manifest_path, record) / cls._TRIPLET_RELAXATION_BACKBONE_FILENAME

    @classmethod
    def interface_to_set_table_cache_path(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> Path:
        return cls.base_dir(manifest_path, record) / cls._INTERFACE_TO_SET_COST_TABLE_FILENAME

    @staticmethod
    def _has_nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    @classmethod
    def has_cached_triplet_relaxation(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> bool:
        return cls._has_nonempty_file(cls.triplet_relaxation_backbone_path(manifest_path, record))

    @classmethod
    def has_cached_interface_to_set_table(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> bool:
        return cls._has_nonempty_file(cls.interface_to_set_table_cache_path(manifest_path, record))

    @classmethod
    def filter_records_with_cached_h_tab(
        cls,
        manifest_path: str | Path,
        records: Sequence[BaseBenchmarkRecord],
    ) -> list[BaseBenchmarkRecord]:
        return [record for record in records if cls.has_cached_interface_to_set_table(manifest_path, record)]

    @classmethod
    def has_complete_caches(
        cls,
        manifest_path: str | Path,
        record: BaseBenchmarkRecord,
        required_heuristics: set[str],
    ) -> bool:
        required_heuristics = cls.expand_required_heuristics(required_heuristics)
        if (
            cls.TRIPLET_RELAXATION_NAME in required_heuristics
            and not cls.has_cached_triplet_relaxation(manifest_path, record)
        ):
            return False
        if (
            cls.INTERFACE_TO_SET_COST_TABLE_NAME in required_heuristics
            and not cls.has_cached_interface_to_set_table(manifest_path, record)
        ):
            return False
        return True

    @classmethod
    def expand_required_heuristics(cls, heuristic_names: set[str]) -> set[str]:
        expanded = set(heuristic_names)
        if cls.MAX_NAME in expanded:
            expanded.remove(cls.MAX_NAME)
            expanded.update(
                {
                    cls.MOTION_ONLY_NAME,
                    cls.TRIPLET_RELAXATION_NAME,
                    cls.INTERFACE_TO_SET_COST_TABLE_NAME,
                }
            )
        return expanded

    @classmethod
    def ensure_motion_only(cls, instance: Instance) -> MotionOnlyHeuristic:
        if instance.motion_only_heuristic is None:
            instance.motion_only_heuristic = MotionOnlyHeuristic(instance.stgcs)
        return instance.motion_only_heuristic

    @classmethod
    def ensure_triplet_relaxation(cls, instance: Instance) -> TripletRelaxationHeuristic:
        if instance.triplet_relaxation_heuristic is None:
            base_gcs = instance.stgcs.get_gcs_instance().gcs
            instance.triplet_relaxation_heuristic = TripletRelaxationHeuristic(
                instance.stgcs,
                base_gcs,
                use_update=True,
            )
        return instance.triplet_relaxation_heuristic

    @classmethod
    def load_cached_triplet_relaxation(
        cls,
        instance: Instance,
        manifest_path: str | Path,
        record: BaseBenchmarkRecord,
    ) -> None:
        backbone_path = cls.triplet_relaxation_backbone_path(manifest_path, record)
        if backbone_path.exists():
            backbone = TripletRelaxationHeuristic.load_backbone(str(backbone_path))
            instance.triplet_relaxation_heuristic = TripletRelaxationHeuristic.from_backbone(
                instance.stgcs,
                backbone,
                use_update=True,
                show_progress=False,
            )
            return

        cls._info(
            f"{record.instance_id}: h_tri cache missing at {backbone_path}; constructing online and saving it."
        )
        cls.ensure_triplet_relaxation(instance)
        backbone_path.parent.mkdir(parents=True, exist_ok=True)
        instance.triplet_relaxation_heuristic.save_backbone(str(backbone_path))

    @classmethod
    def load_cached_interface_to_set_table(
        cls,
        instance: Instance,
        manifest_path: str | Path,
        record: BaseBenchmarkRecord,
        online_timeout_secs: float | None = None,
    ) -> None:
        h_tab_path = cls.interface_to_set_table_cache_path(manifest_path, record)
        if h_tab_path.exists():
            instance.interface_to_set_cost_table_heuristic = InterfaceToSetCostTableHeuristic.load(str(h_tab_path))
            return
        if online_timeout_secs is None:
            raise FileNotFoundError(
                f"Offline h_tab cache missing for base {record.instance_id}. Expected {h_tab_path}. "
                "Pass online_h_tab_timeout_secs to construct it online."
            )
        timeout_secs = float(online_timeout_secs)
        if timeout_secs <= 0.0:
            raise ValueError(f"online_h_tab_timeout_secs must be positive, got {online_timeout_secs}.")
        cls._info(
            f"{record.instance_id}: h_tab cache missing at {h_tab_path}; constructing online with "
            f"timeout={timeout_secs:g}s and saving it."
        )
        if instance.triplet_relaxation_heuristic is None:
            cls.load_cached_triplet_relaxation(instance, manifest_path, record)
        base_gcs = instance.stgcs.get_gcs_instance().gcs
        h_tab_heuristic = InterfaceToSetCostTableHeuristic(
            instance.stgcs,
            base_gcs,
            instance.stgcs.vlimit,
            instance.triplet_relaxation_heuristic,
            timeout_secs=timeout_secs,
        )
        if not h_tab_heuristic._successful:
            raise RuntimeError(
                f"h_tab online construction timed out for base {record.instance_id} at {timeout_secs:.1f}s."
            )
        h_tab_path.parent.mkdir(parents=True, exist_ok=True)
        h_tab_heuristic.save(str(h_tab_path))
        instance.interface_to_set_cost_table_heuristic = h_tab_heuristic

    @classmethod
    def prepare_instance_for_search(
        cls,
        instance: Instance,
        manifest_path: str | Path,
        record: BaseBenchmarkRecord,
        required_heuristics: set[str],
        online_h_tab_timeout_secs: float | None = None,
    ) -> None:
        required_heuristics = cls.expand_required_heuristics(required_heuristics)
        if cls.MOTION_ONLY_NAME in required_heuristics:
            cls.ensure_motion_only(instance)
        if (
            cls.TRIPLET_RELAXATION_NAME in required_heuristics
            or cls.INTERFACE_TO_SET_COST_TABLE_NAME in required_heuristics
        ):
            cls.load_cached_triplet_relaxation(instance, manifest_path, record)
        if cls.INTERFACE_TO_SET_COST_TABLE_NAME in required_heuristics:
            cls.load_cached_interface_to_set_table(
                instance,
                manifest_path,
                record,
                online_timeout_secs=online_h_tab_timeout_secs,
            )
