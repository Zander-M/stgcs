from __future__ import annotations

from pathlib import Path
from typing import Sequence

from benchmark.manifests.base import BaseBenchmarkRecord
from benchmark.instance import Instance
from stgcs.bfs.heuristics import HeurLowerBoundGraph, HeurShortCut, HeurTrueDistance


class BaseOfflineHeuristicStore:
    SC_NAME = "SC"
    LBG_NAME = "LBG"
    TD_NAME = "TD"
    MAX_NAME = "Max"

    _LBG_BACKBONE_FILENAME = "lbg_backbone.pkl"
    _TD_FILENAME = "td.pkl"

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
    def lbg_backbone_path(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> Path:
        return cls.base_dir(manifest_path, record) / cls._LBG_BACKBONE_FILENAME

    @classmethod
    def td_cache_path(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> Path:
        return cls.base_dir(manifest_path, record) / cls._TD_FILENAME

    @staticmethod
    def _has_nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    @classmethod
    def has_cached_lbg(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> bool:
        return cls._has_nonempty_file(cls.lbg_backbone_path(manifest_path, record))

    @classmethod
    def has_cached_td(cls, manifest_path: str | Path, record: BaseBenchmarkRecord) -> bool:
        return cls._has_nonempty_file(cls.td_cache_path(manifest_path, record))

    @classmethod
    def filter_records_with_cached_td(
        cls,
        manifest_path: str | Path,
        records: Sequence[BaseBenchmarkRecord],
    ) -> list[BaseBenchmarkRecord]:
        return [record for record in records if cls.has_cached_td(manifest_path, record)]

    @classmethod
    def has_complete_caches(
        cls,
        manifest_path: str | Path,
        record: BaseBenchmarkRecord,
        required_heuristics: set[str],
    ) -> bool:
        required_heuristics = cls.expand_required_heuristics(required_heuristics)
        if cls.LBG_NAME in required_heuristics and not cls.has_cached_lbg(manifest_path, record):
            return False
        if cls.TD_NAME in required_heuristics and not cls.has_cached_td(manifest_path, record):
            return False
        return True

    @classmethod
    def expand_required_heuristics(cls, heuristic_names: set[str]) -> set[str]:
        expanded = set(heuristic_names)
        if cls.MAX_NAME in expanded:
            expanded.remove(cls.MAX_NAME)
            expanded.update({cls.SC_NAME, cls.LBG_NAME, cls.TD_NAME})
        return expanded

    @classmethod
    def ensure_sc(cls, instance: Instance) -> HeurShortCut:
        if instance.sc_heur is None:
            instance.sc_heur = HeurShortCut(instance.stgcs)
        return instance.sc_heur

    @classmethod
    def ensure_lbg(cls, instance: Instance) -> HeurLowerBoundGraph:
        if instance.lbg is None:
            base_gcs = instance.stgcs.get_gcs_instance().gcs
            instance.lbg = HeurLowerBoundGraph(instance.stgcs, base_gcs, use_update=True)
        return instance.lbg

    @classmethod
    def load_cached_lbg(cls, instance: Instance, manifest_path: str | Path, record: BaseBenchmarkRecord) -> None:
        backbone_path = cls.lbg_backbone_path(manifest_path, record)
        if backbone_path.exists():
            backbone = HeurLowerBoundGraph.load_backbone(str(backbone_path))
            instance.lbg = HeurLowerBoundGraph.from_backbone(
                instance.stgcs,
                backbone,
                use_update=True,
                show_progress=False,
            )
            return

        cls._info(
            f"{record.instance_id}: LBG cache missing at {backbone_path}; constructing online and saving it."
        )
        cls.ensure_lbg(instance)
        backbone_path.parent.mkdir(parents=True, exist_ok=True)
        instance.lbg.save_backbone(str(backbone_path))

    @classmethod
    def load_cached_td(
        cls,
        instance: Instance,
        manifest_path: str | Path,
        record: BaseBenchmarkRecord,
        online_timeout_secs: float | None = None,
    ) -> None:
        td_path = cls.td_cache_path(manifest_path, record)
        if td_path.exists():
            instance.td_heur = HeurTrueDistance.load(str(td_path))
            return
        if online_timeout_secs is None:
            raise FileNotFoundError(
                f"Offline TD cache missing for base {record.instance_id}. Expected {td_path}. "
                "Pass online_td_timeout_secs to construct it online."
            )
        timeout_secs = float(online_timeout_secs)
        if timeout_secs <= 0.0:
            raise ValueError(f"online_td_timeout_secs must be positive, got {online_timeout_secs}.")
        cls._info(
            f"{record.instance_id}: TD cache missing at {td_path}; constructing online with "
            f"timeout={timeout_secs:g}s and saving it."
        )
        if instance.lbg is None:
            cls.load_cached_lbg(instance, manifest_path, record)
        base_gcs = instance.stgcs.get_gcs_instance().gcs
        td_heur = HeurTrueDistance(
            instance.stgcs,
            base_gcs,
            instance.stgcs.vlimit,
            instance.lbg,
            timeout_secs=timeout_secs,
        )
        if not td_heur._successful:
            raise RuntimeError(
                f"TD online construction timed out for base {record.instance_id} at {timeout_secs:.1f}s."
            )
        td_path.parent.mkdir(parents=True, exist_ok=True)
        td_heur.save(str(td_path))
        instance.td_heur = td_heur

    @classmethod
    def prepare_instance_for_search(
        cls,
        instance: Instance,
        manifest_path: str | Path,
        record: BaseBenchmarkRecord,
        required_heuristics: set[str],
        online_td_timeout_secs: float | None = None,
    ) -> None:
        required_heuristics = cls.expand_required_heuristics(required_heuristics)
        if cls.SC_NAME in required_heuristics:
            cls.ensure_sc(instance)
        if cls.LBG_NAME in required_heuristics or cls.TD_NAME in required_heuristics:
            cls.load_cached_lbg(instance, manifest_path, record)
        if cls.TD_NAME in required_heuristics:
            cls.load_cached_td(
                instance,
                manifest_path,
                record,
                online_timeout_secs=online_td_timeout_secs,
            )
