from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from environment.env import Env
from experiments.base.manifest import BaseBenchmarkRecord, load_manifest
from experiments.base.serialization import SerializedBaseInstanceStore
from experiments.instance import GridInstance, Iris2DInstance, MazeInstance, Simple2DInstance


class BaseInstanceFactory:
    @staticmethod
    def _info(message: str) -> None:
        print(f"[stgcs-base] {message}")

    @staticmethod
    def _build_empty_square2d_env(env_params: Dict[str, Any]) -> Env:
        square_size = float(env_params.get("square_size", 10.0))
        robot_radius = float(env_params.get("robot_radius", Simple2DInstance.DEFAULT_ROBOT_RADIUS))
        if square_size <= 0.0:
            raise ValueError(f"empty-square2d square_size must be positive, got {square_size}.")
        if robot_radius <= 0.0:
            raise ValueError(f"empty-square2d robot_radius must be positive, got {robot_radius}.")
        vertices = np.asarray(
            [
                [0.0, 0.0],
                [square_size, 0.0],
                [square_size, square_size],
                [0.0, square_size],
            ],
            dtype=float,
        )
        return Env(
            name="empty-square2d",
            CSpace=[vertices],
            robot_radius=robot_radius,
        )

    @classmethod
    def _build_empty_square2d_instance(
        cls,
        env_params: Dict[str, Any],
        spatial_seed: int,
        compute_heuristics: bool,
    ):
        vlimit = float(env_params.get("vlimit", 1.0))
        tmax = float(env_params.get("tmax", 1000.0))
        env = cls._build_empty_square2d_env(env_params)
        instance = Simple2DInstance.from_env(
            seed=spatial_seed,
            env=env,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
        )
        instance.env_params = dict(env_params)
        return instance

    @staticmethod
    def build(
        domain: str,
        env_params: Dict[str, Any],
        spatial_seed: int,
        space_dim: int,
        compute_heuristics: bool = True,
    ):
        if domain == "grid":
            return GridInstance.generate_base(
                seed=spatial_seed,
                N=int(env_params["N"]),
                M=int(env_params["M"]),
                compute_heuristics=compute_heuristics,
                space_dim=space_dim,
            )
        if domain == "maze":
            vlimit = float(env_params.get("vlimit", 1.0))
            robot_radius = float(env_params.get("robot_radius", 0.25))
            return MazeInstance.generate_base(
                seed=spatial_seed,
                width=int(env_params["width"]),
                height=int(env_params["height"]),
                vlimit=vlimit,
                robot_radius=robot_radius,
                compute_heuristics=compute_heuristics,
            )
        if domain == "simple2d":
            vlimit = float(env_params.get("vlimit", 1.0))
            robot_radius = float(env_params.get("robot_radius", Simple2DInstance.DEFAULT_ROBOT_RADIUS))
            return Simple2DInstance.generate_base(
                seed=spatial_seed,
                vlimit=vlimit,
                robot_radius=robot_radius,
                compute_heuristics=compute_heuristics,
            )
        if domain == "empty-square2d":
            return BaseInstanceFactory._build_empty_square2d_instance(
                env_params=env_params,
                spatial_seed=spatial_seed,
                compute_heuristics=compute_heuristics,
            )
        if domain == "iris-2d":
            return Iris2DInstance.generate_base(
                seed=spatial_seed,
                env_params=env_params,
                compute_heuristics=compute_heuristics,
            )
        raise ValueError(f"Unsupported domain {domain!r}")

    @classmethod
    def load_cached_instance(cls, record: BaseBenchmarkRecord):
        return SerializedBaseInstanceStore.load_instance(record)

    @classmethod
    def write_cached_instance(cls, record: BaseBenchmarkRecord, instance) -> None:
        SerializedBaseInstanceStore.write_instance(record, instance)

    @staticmethod
    def _ensure_requested_heuristics(instance, compute_heuristics: bool):
        if compute_heuristics and getattr(instance, "sc_heur", None) is None:
            instance.compute_heuristics()
        return instance

    @classmethod
    def from_record(cls, record: BaseBenchmarkRecord, compute_heuristics: bool = True):
        cached_instance = cls.load_cached_instance(record)
        if cached_instance is not None:
            return cls._ensure_requested_heuristics(cached_instance, compute_heuristics)

        object_path = SerializedBaseInstanceStore.object_path(record)
        if object_path is not None:
            cls._info(
                f"{record.instance_id}: serialized base object missing at {object_path}; "
                "constructing online and saving it."
            )
        instance = cls.build(
            domain=record.domain,
            env_params=record.env_params,
            spatial_seed=record.spatial_seed,
            space_dim=record.space_dim,
            compute_heuristics=False if record.instance_cache_path is not None else compute_heuristics,
        )
        cls.write_cached_instance(record, instance)
        return cls._ensure_requested_heuristics(instance, compute_heuristics)


class BaseManifestStore:
    _records_by_manifest: Dict[Path, Dict[str, BaseBenchmarkRecord]] = {}

    @staticmethod
    def manifest_path(source: str | Path, domain_key: str | None = None) -> Path:
        source_path = Path(source).resolve()
        if source_path.suffix == ".json":
            return source_path
        if domain_key is None:
            raise ValueError("domain_key is required when resolving a base manifest from a directory.")
        if source_path.name == domain_key:
            return source_path / "manifest.json"
        return source_path / domain_key / "manifest.json"

    @staticmethod
    def default_base_root(benchmark_manifest_path: str | Path) -> Path:
        manifest_path = Path(benchmark_manifest_path).resolve()
        if manifest_path.parent.parent.name == "mrmp":
            return manifest_path.parent.parent.parent / "stgcs_base"
        if manifest_path.parent.parent.name == "base":
            return manifest_path.parent.parent
        raise ValueError(f"Unable to infer base root from {manifest_path}")

    @staticmethod
    def domain_key_for_benchmark_manifest(benchmark_manifest_path: str | Path) -> str:
        return Path(benchmark_manifest_path).resolve().parent.name

    @classmethod
    def manifest_path_for_benchmark(
        cls,
        benchmark_manifest_path: str | Path,
        base_root: str | Path | None = None,
    ) -> Path:
        domain_key = cls.domain_key_for_benchmark_manifest(benchmark_manifest_path)
        root = cls.default_base_root(benchmark_manifest_path) if base_root is None else Path(base_root).resolve()
        return cls.manifest_path(root, domain_key=domain_key)

    @classmethod
    def records_by_id(cls, manifest_path: str | Path) -> Dict[str, BaseBenchmarkRecord]:
        target = Path(manifest_path).resolve()
        if target not in cls._records_by_manifest:
            cls._records_by_manifest[target] = {
                record.instance_id: record
                for record in load_manifest(target)
            }
        return cls._records_by_manifest[target]

    @classmethod
    def record_by_id(cls, manifest_path: str | Path, instance_id: str) -> BaseBenchmarkRecord:
        try:
            return cls.records_by_id(manifest_path)[instance_id]
        except KeyError as exc:
            raise KeyError(f"Unknown base instance_id {instance_id!r} in {Path(manifest_path).resolve()}") from exc
