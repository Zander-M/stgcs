from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Sequence

from experiments.build_base_manifest import BaseManifestBuilder, ManifestLock
from benchmark.base import BaseInstanceFactory
from benchmark.manifests.base import BaseBenchmarkRecord, load_manifest, save_manifest
from experiments.compute_heuristics import BaseOfflineHeuristicCLI
from benchmark.offline_heuristics import BaseOfflineHeuristicStore


class HeuristicComputationTimeScalingBuilder(BaseManifestBuilder):
    DEFAULT_OUTPUT_ROOT = Path("data/stgcs_base/heur_computation_time_scaling")
    DEFAULT_N_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    DEFAULT_M_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    DEFAULT_SAMPLES_PER_SHAPE = 1
    DEFAULT_CANDIDATE_MULTIPLIER = 3
    DEFAULT_STOP_AFTER_FAILED_SHAPES = 3
    DEFAULT_TD_TIMEOUT_SECS = 10_000.0

    @staticmethod
    def instance_id(n: int, m: int, seed: int) -> str:
        return f"base-heur-scaling-grid2d-{n}x{m}-seed{seed:05d}"

    @staticmethod
    def grid_env_params(n: int, m: int) -> dict[str, int]:
        return {"N": int(n), "M": int(m)}

    @staticmethod
    def grid_shape(record: BaseBenchmarkRecord) -> tuple[int, int]:
        return int(record.env_params["N"]), int(record.env_params["M"])

    @staticmethod
    def grid_shapes(n_values: Sequence[int], m_values: Sequence[int]) -> list[tuple[int, int]]:
        shapes = {(int(n), int(m)) for n in n_values for m in m_values}
        return sorted(
            shapes,
            key=lambda item: (item[0] * item[1], item[0] + item[1], item[0], item[1]),
        )

    @classmethod
    def build_grid2d_record(cls, n: int, m: int, seed: int) -> BaseBenchmarkRecord:
        env_params = cls.grid_env_params(n, m)
        instance = BaseInstanceFactory.build(
            domain="grid",
            env_params=env_params,
            spatial_seed=int(seed),
            space_dim=2,
            compute_heuristics=False,
        )
        return cls._record(
            instance_id=cls.instance_id(n, m, seed),
            domain="grid",
            space_dim=2,
            spatial_seed=int(seed),
            env_params=env_params,
            instance=instance,
        )

    @staticmethod
    def records_by_shape(records: Sequence[BaseBenchmarkRecord]) -> dict[tuple[int, int], list[BaseBenchmarkRecord]]:
        grouped: dict[tuple[int, int], list[BaseBenchmarkRecord]] = {}
        for record in records:
            shape = HeuristicComputationTimeScalingBuilder.grid_shape(record)
            grouped.setdefault(shape, []).append(record)
        return grouped

    @staticmethod
    def cleanup_failed_candidate(manifest_path: Path, record: BaseBenchmarkRecord) -> None:
        shutil.rmtree(BaseOfflineHeuristicStore.base_dir(manifest_path, record), ignore_errors=True)

    @classmethod
    def compute_td_timing(
        cls,
        manifest_path: Path,
        record: BaseBenchmarkRecord,
        td_timeout_secs: float,
    ) -> dict[str, float]:
        _, times = BaseOfflineHeuristicCLI._build_record_caches(
            str(manifest_path),
            record.to_dict(),
            float(td_timeout_secs),
            False,
        )
        if BaseOfflineHeuristicStore.TD_NAME not in times:
            raise RuntimeError(f"TD timing was not recorded for {record.instance_id}.")
        return times

    @classmethod
    def build_or_extend(
        cls,
        output_root: Path,
        n_values: Sequence[int],
        m_values: Sequence[int],
        samples_per_shape: int,
        seed_start: int,
        candidate_multiplier: int,
        stop_after_failed_shapes: int,
        td_timeout_secs: float,
    ) -> list[BaseBenchmarkRecord]:
        output_dir = Path(output_root)
        manifest_path = output_dir / "manifest.json"
        lock_path = output_dir / ".heur_scaling_manifest.lock"
        output_dir.mkdir(parents=True, exist_ok=True)

        if samples_per_shape <= 0:
            raise ValueError(f"samples_per_shape must be positive, got {samples_per_shape}.")
        if candidate_multiplier <= 0:
            raise ValueError(f"candidate_multiplier must be positive, got {candidate_multiplier}.")
        if stop_after_failed_shapes < 0:
            raise ValueError(f"stop_after_failed_shapes must be non-negative, got {stop_after_failed_shapes}.")
        if td_timeout_secs <= 0.0:
            raise ValueError(f"td_timeout_secs must be positive, got {td_timeout_secs}.")

        with ManifestLock(lock_path):
            records = load_manifest(manifest_path) if manifest_path.exists() else []
            seen_ids = {record.instance_id for record in records}
            grouped = cls.records_by_shape(records)
            max_attempts = int(samples_per_shape) * int(candidate_multiplier)
            consecutive_failed_shapes = 0

            for n, m in cls.grid_shapes(n_values, m_values):
                kept = list(grouped.get((n, m), []))
                shape_label = f"{n}x{m}"
                if len(kept) >= int(samples_per_shape):
                    print(f"shape={shape_label}: already has {len(kept)}/{samples_per_shape} records")
                    consecutive_failed_shapes = 0
                    continue

                for attempt in range(max_attempts):
                    if len(kept) >= int(samples_per_shape):
                        break
                    seed = int(seed_start) + attempt
                    instance_id = cls.instance_id(n, m, seed)
                    if instance_id in seen_ids:
                        continue
                    record: BaseBenchmarkRecord | None = None
                    try:
                        record = cls.build_grid2d_record(n=n, m=m, seed=seed)
                        print(
                            f"{record.instance_id}: graph=({record.stgcs_num_vertices}V,"
                            f" {record.stgcs_num_edges}E), running TD"
                        )
                        times = cls.compute_td_timing(
                            manifest_path=manifest_path,
                            record=record,
                            td_timeout_secs=float(td_timeout_secs),
                        )
                    except Exception as exc:
                        if record is None:
                            print(f"{instance_id}: skip, failed to build base ({exc})")
                        else:
                            cls.cleanup_failed_candidate(manifest_path, record)
                            print(f"{record.instance_id}: skip, TD failed ({exc})")
                        continue

                    records.append(record)
                    kept.append(record)
                    grouped.setdefault((n, m), []).append(record)
                    seen_ids.add(record.instance_id)
                    save_manifest(manifest_path, records)
                    print(
                        f"{record.instance_id}: kept {len(kept)}/{samples_per_shape} "
                        f"for shape={shape_label}, TD={times[BaseOfflineHeuristicStore.TD_NAME]:.3f}s"
                    )

                if len(kept) < int(samples_per_shape):
                    print(
                        f"shape={shape_label}: kept {len(kept)}/{samples_per_shape} "
                        f"records after {max_attempts} attempts"
                    )
                    consecutive_failed_shapes += 1
                    if (
                        int(stop_after_failed_shapes) > 0
                        and consecutive_failed_shapes >= int(stop_after_failed_shapes)
                    ):
                        print(
                            f"Stopping after {consecutive_failed_shapes} consecutive "
                            f"TD-infeasible shapes."
                        )
                        break
                else:
                    consecutive_failed_shapes = 0

            save_manifest(manifest_path, records)

        return records


class HeuristicComputationTimeScalingCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Incrementally build a rectangular grid2d base manifest for offline heuristic scaling. "
                "A candidate is kept only if TD finishes within the requested timeout."
            )
        )
        parser.add_argument(
            "--output-root",
            type=Path,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_OUTPUT_ROOT,
        )
        parser.add_argument(
            "--n-values",
            type=int,
            nargs="+",
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_N_VALUES,
        )
        parser.add_argument(
            "--m-values",
            type=int,
            nargs="+",
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_M_VALUES,
        )
        parser.add_argument(
            "--samples-per-shape",
            type=int,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_SAMPLES_PER_SHAPE,
        )
        parser.add_argument("--seed-start", type=int, default=0)
        parser.add_argument(
            "--candidate-multiplier",
            type=int,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_CANDIDATE_MULTIPLIER,
        )
        parser.add_argument(
            "--stop-after-failed-shapes",
            type=int,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_STOP_AFTER_FAILED_SHAPES,
            help="Stop after this many consecutive shapes fail to produce a TD-feasible sample; use 0 to disable.",
        )
        parser.add_argument(
            "--td-timeout-secs",
            type=float,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_TD_TIMEOUT_SECS,
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        records = HeuristicComputationTimeScalingBuilder.build_or_extend(
            output_root=args.output_root,
            n_values=tuple(int(value) for value in args.n_values),
            m_values=tuple(int(value) for value in args.m_values),
            samples_per_shape=int(args.samples_per_shape),
            seed_start=int(args.seed_start),
            candidate_multiplier=int(args.candidate_multiplier),
            stop_after_failed_shapes=int(args.stop_after_failed_shapes),
            td_timeout_secs=float(args.td_timeout_secs),
        )
        print(f"Wrote {len(records)} records to {Path(args.output_root) / 'manifest.json'}")


if __name__ == "__main__":
    HeuristicComputationTimeScalingCLI.main()
