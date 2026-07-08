from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from benchmark.manifests.base import BaseBenchmarkRecord, load_manifest


@dataclass(frozen=True)
class HeuristicAblationSelectionEntry:
    group: str
    source_domain: str
    record: BaseBenchmarkRecord

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "source_domain": self.source_domain,
            "record": self.record.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "HeuristicAblationSelectionEntry":
        return HeuristicAblationSelectionEntry(
            group=str(data["group"]),
            source_domain=str(data["source_domain"]),
            record=BaseBenchmarkRecord.from_dict(data["record"]),
        )


class HeuristicAblationSelector:
    GENERAL_OPEN = "grid2d"
    IRIS2D = "iris2d"

    DEFAULT_GROUP_COUNT = 15
    GENERAL_OPEN_GRID_SIZE = 5
    GROUP_ORDER = (GENERAL_OPEN, IRIS2D)
    DOMAIN_KEYS = ("grid2d", "iris-2d")

    GROUP_DESCRIPTIONS: Dict[str, str] = {
        GENERAL_OPEN: "Random-grid cases where h_mot, h_tri, and h_tab should be close.",
        IRIS2D: "IRIS-grown cluttered free-space cases.",
    }

    @classmethod
    def manifest_path(cls, base_root: Path, domain_key: str) -> Path:
        return Path(base_root) / domain_key / "manifest.json"

    @classmethod
    def load_records_by_domain(cls, base_root: Path) -> Dict[str, list[BaseBenchmarkRecord]]:
        return {
            domain_key: load_manifest(cls.manifest_path(base_root, domain_key))
            for domain_key in cls.DOMAIN_KEYS
        }

    @staticmethod
    def split_count(total: int, parts: int) -> tuple[int, ...]:
        if total < 0:
            raise ValueError(f"Selection count must be non-negative, got {total}.")
        if parts <= 0:
            raise ValueError(f"Selection parts must be positive, got {parts}.")
        base = total // parts
        remainder = total % parts
        return tuple(base + (1 if idx < remainder else 0) for idx in range(parts))

    @staticmethod
    def grid_size(record: BaseBenchmarkRecord) -> int:
        return int(record.env_params.get("size", record.env_params.get("N", 0)))

    @staticmethod
    def edge_density(record: BaseBenchmarkRecord) -> float:
        vertices = max(int(record.stgcs_num_vertices), 1)
        return float(record.stgcs_num_edges) / float(vertices)

    @staticmethod
    def base_scale_key(record: BaseBenchmarkRecord) -> tuple[int, int, int, str]:
        return (
            int(record.stgcs_num_vertices),
            int(record.stgcs_num_edges),
            int(record.spatial_seed),
            str(record.instance_id),
        )

    @classmethod
    def choose_records(
        cls,
        records: Sequence[BaseBenchmarkRecord],
        count: int,
        predicate: Callable[[BaseBenchmarkRecord], bool],
        sort_key: Callable[[BaseBenchmarkRecord], object],
    ) -> list[BaseBenchmarkRecord]:
        candidates = sorted((record for record in records if predicate(record)), key=sort_key)
        if len(candidates) < count:
            raise ValueError(f"Need {count} records, found {len(candidates)}.")
        return candidates[:count]

    @classmethod
    def make_entry(
        cls,
        group: str,
        source_domain: str,
        record: BaseBenchmarkRecord,
    ) -> HeuristicAblationSelectionEntry:
        return HeuristicAblationSelectionEntry(
            group=group,
            source_domain=source_domain,
            record=record,
        )

    @classmethod
    def select_general_open(
        cls,
        records_by_domain: Dict[str, Sequence[BaseBenchmarkRecord]],
        group_count: int,
    ) -> list[HeuristicAblationSelectionEntry]:
        grid2d = cls.choose_records(
            records_by_domain["grid2d"],
            group_count,
            predicate=lambda record: cls.grid_size(record) == cls.GENERAL_OPEN_GRID_SIZE,
            sort_key=lambda record: (
                int(record.stgcs_num_vertices),
                int(record.stgcs_num_edges),
                record.spatial_seed,
                record.instance_id,
            ),
        )
        return [
            cls.make_entry(cls.GENERAL_OPEN, "grid2d", record)
            for record in grid2d
        ]

    @classmethod
    def select_iris2d(
        cls,
        records_by_domain: Dict[str, Sequence[BaseBenchmarkRecord]],
        group_count: int,
    ) -> list[HeuristicAblationSelectionEntry]:
        records = cls.choose_records(
            records_by_domain["iris-2d"],
            group_count,
            predicate=lambda record: record.domain == "iris-2d",
            sort_key=cls.base_scale_key,
        )
        return [
            cls.make_entry(cls.IRIS2D, "iris-2d", record)
            for record in records
        ]

    @classmethod
    def build_selection(
        cls,
        records_by_domain: Dict[str, Sequence[BaseBenchmarkRecord]],
        group_count: int = DEFAULT_GROUP_COUNT,
    ) -> list[HeuristicAblationSelectionEntry]:
        return cls.build_selection_for_group_counts(
            records_by_domain,
            {group: int(group_count) for group in cls.GROUP_ORDER},
        )

    @classmethod
    def build_selection_for_group_counts(
        cls,
        records_by_domain: Dict[str, Sequence[BaseBenchmarkRecord]],
        group_counts: Dict[str, int],
    ) -> list[HeuristicAblationSelectionEntry]:
        return [
            *cls.select_general_open(records_by_domain, int(group_counts.get(cls.GENERAL_OPEN, 0))),
            *cls.select_iris2d(records_by_domain, int(group_counts.get(cls.IRIS2D, 0))),
        ]

    @classmethod
    def target_group_counts(
        cls,
        group_count: int,
        total_count: int | None = None,
    ) -> Dict[str, int]:
        if total_count is None:
            return {group: int(group_count) for group in cls.GROUP_ORDER}
        return {
            group: count
            for group, count in zip(cls.GROUP_ORDER, cls.split_count(int(total_count), len(cls.GROUP_ORDER)))
        }
