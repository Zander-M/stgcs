from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from pydrake.all import RandomGenerator
from benchmark.base import BaseInstanceFactory
from benchmark.manifests.base import BaseBenchmarkRecord
from benchmark.offline_heuristics import BaseOfflineHeuristicStore
from experiments.mrmp_runners.common import MRMPExperiment
from benchmark.manifests.mrmp import MRMPBenchmarkRecord
from benchmark.manifests.mrmp import QuerySpec
from benchmark.planners.mrmp import MRMPPerformanceComparison, PBSExpansionAblation
from stgcs.mrmp_planner import MRMPQuery
from stgcs.pbs import PriorityBasedSearch
from stgcs.st_planner import MPQuery, STPlanStatus
from stgcs.geometry_utils import HPolyhedronSampler


RecordCallback = Callable[[MRMPBenchmarkRecord, Sequence[MRMPBenchmarkRecord]], None]
StatusCallback = Callable[[str, Dict[str, Any]], None]


@dataclass(frozen=True)
class ConflictStats:
    num_conflicting_pairs: int
    num_conflicting_agents: int
    largest_component: int
    density: float


class MRMPManifestBuilder:
    DIFFICULTY_MODE_TEMPLATE = "template"
    DIFFICULTY_MODE_SOLVER = "solver"
    TIERS = ("small", "medium", "large")
    SIDE_BAND_RATIO = 0.18
    CENTER_BAND_RATIO = 0.22
    SWITCH_SPREAD_SEPARATION_MULTIPLIER = 1.0
    MAX_QUERY_ATTEMPTS = 300
    MAX_RECORD_ATTEMPTS = 24
    MAX_FAILED_ATTEMPTS = 500
    ENDPOINT_SEPARATION_SCALE_BY_DIM = {
        2: 2.85,
        3: 3.5,
    }
    ENDPOINT_SEPARATION_FAMILIES = frozenset(("merge-flow", "random", "switch-spread", "intersection"))
    MIN_POINT_SEPARATION_SCALE = 2.0
    MIN_POINT_SEPARATION_DIAG_RATIO = 0.02
    MAX_PAIR_ATTEMPTS = 40
    RECT_ATTEMPTS_PER_INDEX = 12
    AGENT_COUNTS = {
        "grid2d": {"small": 8, "medium": 12, "large": 16},
        "grid3d": {"small": 6, "medium": 9, "large": 12},
        "maze": {"small": 8, "medium": 12, "large": 16},
    }
    INDEPENDENT_REFERENCE_BUDGET = {
        "grid2d": 5.0,
        "grid3d": 15.0,
        "iris-2d": 5.0,
        "maze": 5.0,
    }
    MAX_CONFLICT_DENSITY_BY_DOMAIN: Dict[str, float] = {}

    @classmethod
    def tier_agent_count(cls, domain_key: str, tier: str) -> int:
        return int(cls.AGENT_COUNTS[domain_key][tier])

    @classmethod
    def _pair_targets(cls, total: int, families: Sequence[str]) -> Dict[Tuple[str, str], int]:
        pairs = [(tier, family) for tier in cls.TIERS for family in families]
        base = total // len(pairs)
        remainder = total % len(pairs)
        targets = {pair: base for pair in pairs}
        for pair in pairs[:remainder]:
            targets[pair] += 1
        return targets

    @classmethod
    def _next_pair(
        cls,
        quotas: Dict[Tuple[str, str], int],
        families: Sequence[str],
        ordinal: int,
    ) -> Tuple[str, str]:
        pairs = [(tier, family) for tier in cls.TIERS for family in families]
        for offset in range(len(pairs)):
            pair = pairs[(ordinal + offset) % len(pairs)]
            if quotas[pair] > 0:
                return pair
        raise RuntimeError("No remaining MRMP pair quota is available.")

    @classmethod
    def _candidate_pairs(
        cls,
        quotas: Dict[Tuple[str, str], int],
        families: Sequence[str],
        ordinal: int,
    ) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            pair
            for pair in cls._ordered_candidate_pairs(families, ordinal)
            if quotas[pair] > 0
        )

    @classmethod
    def _ordered_candidate_pairs(
        cls,
        families: Sequence[str],
        ordinal: int,
    ) -> Tuple[Tuple[str, str], ...]:
        pairs = [(tier, family) for tier in cls.TIERS for family in families]
        return tuple(
            pairs[(ordinal + offset) % len(pairs)]
            for offset in range(len(pairs))
        )

    @classmethod
    def _num_agents(
        cls,
        base_instance,
        tier: str,
        env_params: Dict[str, int],
    ) -> int:
        del env_params
        return cls.tier_agent_count(cls.record_domain_key(base_instance), tier)

    @classmethod
    def independent_reference_budget(cls, domain_key: str) -> float:
        return float(cls.INDEPENDENT_REFERENCE_BUDGET[domain_key])

    @classmethod
    def validate_difficulty_mode(cls, difficulty_mode: str) -> str:
        if difficulty_mode not in {cls.DIFFICULTY_MODE_TEMPLATE, cls.DIFFICULTY_MODE_SOLVER}:
            raise ValueError(f"Unsupported MRMP difficulty mode {difficulty_mode!r}")
        return difficulty_mode

    @classmethod
    def validate_min_conflicting_pairs(cls, min_conflicting_pairs: int, difficulty_mode: str) -> int:
        min_conflicting_pairs = int(min_conflicting_pairs)
        if min_conflicting_pairs < 0:
            raise ValueError(f"min_conflicting_pairs must be non-negative, got {min_conflicting_pairs}")
        if min_conflicting_pairs > 0 and difficulty_mode != cls.DIFFICULTY_MODE_SOLVER:
            raise ValueError("min_conflicting_pairs requires solver difficulty mode.")
        return min_conflicting_pairs

    @staticmethod
    def _emit_status(
        on_status: StatusCallback | None,
        event: str,
        **payload: Any,
    ) -> None:
        if on_status is not None:
            on_status(event, payload)

    @classmethod
    def _point_separation(cls, env) -> float:
        bbox_diag = float(np.linalg.norm(env.ub - env.lb))
        return max(cls.MIN_POINT_SEPARATION_SCALE * env.robot_radius, cls.MIN_POINT_SEPARATION_DIAG_RATIO * bbox_diag)

    @classmethod
    def _endpoint_separation_scale(cls, env) -> float:
        dim = int(env.dim)
        if dim not in cls.ENDPOINT_SEPARATION_SCALE_BY_DIM:
            raise ValueError(f"Unsupported endpoint-separation dimension {dim!r}.")
        return float(cls.ENDPOINT_SEPARATION_SCALE_BY_DIM[dim])

    @classmethod
    def _endpoint_separation_tol(cls, env, family: str) -> float:
        if family not in cls.ENDPOINT_SEPARATION_FAMILIES:
            return 0.0
        return cls._endpoint_separation_scale(env) * float(env.robot_radius)

    @staticmethod
    def _normalized(point: np.ndarray, env) -> np.ndarray:
        extent = np.maximum(env.ub - env.lb, 1e-9)
        return (point - env.lb) / extent

    @classmethod
    def _in_zone(cls, point: np.ndarray, env, zones: Sequence[str]) -> bool:
        normalized = cls._normalized(point, env)
        center_low = 0.5 - cls.CENTER_BAND_RATIO / 2.0
        center_high = 0.5 + cls.CENTER_BAND_RATIO / 2.0
        for zone in zones:
            if zone == "left" and normalized[0] > cls.SIDE_BAND_RATIO:
                return False
            if zone == "right" and normalized[0] < 1.0 - cls.SIDE_BAND_RATIO:
                return False
            if zone == "bottom" and normalized[1] > cls.SIDE_BAND_RATIO:
                return False
            if zone == "top" and normalized[1] < 1.0 - cls.SIDE_BAND_RATIO:
                return False
            if zone == "front" and normalized[2] > cls.SIDE_BAND_RATIO:
                return False
            if zone == "back" and normalized[2] < 1.0 - cls.SIDE_BAND_RATIO:
                return False
            if zone == "mid_x" and not (center_low <= normalized[0] <= center_high):
                return False
            if zone == "mid_y" and not (center_low <= normalized[1] <= center_high):
                return False
            if zone == "mid_z" and not (center_low <= normalized[2] <= center_high):
                return False
        return True

    @staticmethod
    def _far_enough(candidate: np.ndarray, existing_points: Sequence[np.ndarray], sep_tol: float) -> bool:
        return all(np.linalg.norm(candidate - point) >= sep_tol for point in existing_points)

    @staticmethod
    def _rect_bounds(rect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.min(rect, axis=0), np.max(rect, axis=0)

    @classmethod
    def _zone_interval(cls, env, zone: str) -> Tuple[int, float, float]:
        extent = np.maximum(env.ub - env.lb, 1e-9)
        center_low = 0.5 - cls.CENTER_BAND_RATIO / 2.0
        center_high = 0.5 + cls.CENTER_BAND_RATIO / 2.0
        if zone == "left":
            return 0, float(env.lb[0]), float(env.lb[0] + cls.SIDE_BAND_RATIO * extent[0])
        if zone == "right":
            return 0, float(env.ub[0] - cls.SIDE_BAND_RATIO * extent[0]), float(env.ub[0])
        if zone == "bottom":
            return 1, float(env.lb[1]), float(env.lb[1] + cls.SIDE_BAND_RATIO * extent[1])
        if zone == "top":
            return 1, float(env.ub[1] - cls.SIDE_BAND_RATIO * extent[1]), float(env.ub[1])
        if zone == "front":
            return 2, float(env.lb[2]), float(env.lb[2] + cls.SIDE_BAND_RATIO * extent[2])
        if zone == "back":
            return 2, float(env.ub[2] - cls.SIDE_BAND_RATIO * extent[2]), float(env.ub[2])
        if zone == "mid_x":
            return 0, float(env.lb[0] + center_low * extent[0]), float(env.lb[0] + center_high * extent[0])
        if zone == "mid_y":
            return 1, float(env.lb[1] + center_low * extent[1]), float(env.lb[1] + center_high * extent[1])
        if zone == "mid_z":
            return 2, float(env.lb[2] + center_low * extent[2]), float(env.lb[2] + center_high * extent[2])
        raise ValueError(f"Unsupported zone {zone!r}")

    @classmethod
    def _rect_matches_zone(cls, rect: np.ndarray, env, zone: str) -> bool:
        rect_lb, rect_ub = cls._rect_bounds(rect)
        axis, zone_low, zone_high = cls._zone_interval(env, zone)
        return bool(rect_ub[axis] >= zone_low and rect_lb[axis] <= zone_high)

    @classmethod
    def _zone_rect_indices(cls, env, zones: Sequence[str]) -> Tuple[int, ...]:
        cache_name = "_mrmp_zone_rect_cache"
        cache: Dict[Tuple[str, ...], Tuple[int, ...]] = getattr(env, cache_name, {})
        key = tuple(zones)
        if key not in cache:
            matches = []
            for idx, rect in enumerate(env.C_Space):
                rect_array = np.asarray(rect, dtype=float)
                if all(cls._rect_matches_zone(rect_array, env, zone) for zone in zones):
                    matches.append(idx)
            cache[key] = tuple(matches)
            setattr(env, cache_name, cache)
        return cache[key]

    @classmethod
    def _rect_measure(
        cls,
        env,
        rect_idx: int,
        zones: Sequence[str] | None = None,
    ) -> float:
        rect_lb, rect_ub = cls._rect_bounds(np.asarray(env.C_Space[rect_idx], dtype=float))
        sample_lb = rect_lb.copy()
        sample_ub = rect_ub.copy()
        for zone in zones or ():
            axis, zone_low, zone_high = cls._zone_interval(env, zone)
            sample_lb[axis] = max(sample_lb[axis], zone_low)
            sample_ub[axis] = min(sample_ub[axis], zone_high)
        if np.any(sample_ub <= sample_lb):
            return 0.0
        return max(float(np.prod(np.maximum(sample_ub - sample_lb, 1e-9))), 1e-9)

    @classmethod
    def _rect_weights(
        cls,
        env,
        rect_indices: Sequence[int],
        zones: Sequence[str] | None = None,
    ) -> np.ndarray:
        measures = []
        for rect_idx in rect_indices:
            measures.append(cls._rect_measure(env, int(rect_idx), zones=zones))
        weights = np.asarray(measures, dtype=float)
        if float(np.sum(weights)) <= 0.0:
            return np.full(len(rect_indices), 1.0 / len(rect_indices), dtype=float)
        return weights / np.sum(weights)

    @classmethod
    def _rect_sample_bounds(
        cls,
        env,
        rect_idx: int,
        zones: Sequence[str] | None = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rect_lb, rect_ub = cls._rect_bounds(np.asarray(env.C_Space[rect_idx], dtype=float))
        sample_lb = rect_lb.copy()
        sample_ub = rect_ub.copy()
        for zone in zones or ():
            axis, zone_low, zone_high = cls._zone_interval(env, zone)
            sample_lb[axis] = max(sample_lb[axis], zone_low)
            sample_ub[axis] = min(sample_ub[axis], zone_high)
        if np.any(sample_ub <= sample_lb):
            return None
        return sample_lb, sample_ub

    @staticmethod
    def _box_measure(bounds: Tuple[np.ndarray, np.ndarray]) -> float:
        lb, ub = bounds
        if np.any(ub <= lb):
            return 0.0
        return float(np.prod(ub - lb))

    @classmethod
    def _overlapping_exclusion_boxes(
        cls,
        bounds: Tuple[np.ndarray, np.ndarray],
        existing_points: Sequence[np.ndarray],
        exclusion_tol: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if exclusion_tol <= 0.0 or len(existing_points) == 0:
            return []

        bounds_lb, bounds_ub = bounds
        overlap_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
        for point in existing_points:
            point_arr = np.asarray(point, dtype=float)
            exc_lb = point_arr - exclusion_tol
            exc_ub = point_arr + exclusion_tol
            if np.any(exc_ub <= bounds_lb) or np.any(bounds_ub <= exc_lb):
                continue
            overlap_boxes.append((exc_lb, exc_ub))
        return overlap_boxes

    @classmethod
    def _subtract_box(
        cls,
        bounds: Tuple[np.ndarray, np.ndarray],
        exclusion: Tuple[np.ndarray, np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        bounds_lb, bounds_ub = bounds
        exc_lb, exc_ub = exclusion
        overlap_lb = np.maximum(bounds_lb, exc_lb)
        overlap_ub = np.minimum(bounds_ub, exc_ub)
        if np.any(overlap_ub <= overlap_lb):
            return [(bounds_lb.copy(), bounds_ub.copy())]

        pieces: List[Tuple[np.ndarray, np.ndarray]] = []
        core_lb = bounds_lb.copy()
        core_ub = bounds_ub.copy()
        for axis in range(bounds_lb.shape[0]):
            if core_lb[axis] < overlap_lb[axis]:
                piece_lb = core_lb.copy()
                piece_ub = core_ub.copy()
                piece_ub[axis] = overlap_lb[axis]
                pieces.append((piece_lb, piece_ub))
                core_lb[axis] = overlap_lb[axis]
            if overlap_ub[axis] < core_ub[axis]:
                piece_lb = core_lb.copy()
                piece_ub = core_ub.copy()
                piece_lb[axis] = overlap_ub[axis]
                pieces.append((piece_lb, piece_ub))
                core_ub[axis] = overlap_ub[axis]

        return [piece for piece in pieces if cls._box_measure(piece) > 0.0]

    @classmethod
    def _subtract_exclusion_boxes(
        cls,
        bounds: Tuple[np.ndarray, np.ndarray],
        exclusion_boxes: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        remaining = [(bounds[0].copy(), bounds[1].copy())]
        for exclusion in exclusion_boxes:
            next_remaining: List[Tuple[np.ndarray, np.ndarray]] = []
            for candidate in remaining:
                next_remaining.extend(cls._subtract_box(candidate, exclusion))
            remaining = next_remaining
            if len(remaining) == 0:
                break
        return remaining

    @classmethod
    def _sample_carved_rect_point(
        cls,
        env,
        rect_indices: Sequence[int],
        rng: np.random.RandomState,
        existing_points: Sequence[np.ndarray],
        exclusion_tol: float,
        zones: Sequence[str] | None = None,
    ) -> Optional[np.ndarray]:
        remaining_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
        for rect_idx in rect_indices:
            bounds = cls._rect_sample_bounds(env, int(rect_idx), zones=zones)
            if bounds is None:
                continue
            exclusion_boxes = cls._overlapping_exclusion_boxes(bounds, existing_points, exclusion_tol)
            remaining_boxes.extend(cls._subtract_exclusion_boxes(bounds, exclusion_boxes))

        if len(remaining_boxes) == 0:
            return None

        weights = np.asarray([cls._box_measure(box) for box in remaining_boxes], dtype=float)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return None
        box_idx = int(rng.choice(len(remaining_boxes), p=weights / total_weight))
        sample_lb, sample_ub = remaining_boxes[box_idx]
        return rng.uniform(sample_lb, sample_ub)

    @classmethod
    def _sample_zone_rect_point(
        cls,
        env,
        rect_idx: int,
        zones: Sequence[str],
        rng: np.random.RandomState,
    ) -> Optional[np.ndarray]:
        bounds = cls._rect_sample_bounds(env, rect_idx, zones=zones)
        if bounds is None:
            return None
        sample_lb, sample_ub = bounds
        candidate = rng.uniform(sample_lb, sample_ub)
        if not env._CSpace_hpoly[rect_idx].PointInSet(candidate):
            return None
        if not cls._in_zone(candidate, env, zones):
            return None
        return candidate

    @classmethod
    def _sample_point_with_predicate(
        cls,
        env,
        rng: np.random.RandomState,
        predicate: Callable[[np.ndarray], bool],
        existing_points: Sequence[np.ndarray],
        sep_tol: float,
    ) -> Optional[np.ndarray]:
        for _ in range(cls.MAX_QUERY_ATTEMPTS):
            sample_seed = int(rng.randint(0, 2**31 - 1))
            candidate = np.asarray(env.sample_CSpace(seed=sample_seed), dtype=float)
            if predicate(candidate) and cls._far_enough(candidate, existing_points, sep_tol):
                return candidate
        return None

    @classmethod
    def _sample_zone_point(
        cls,
        env,
        rng: np.random.RandomState,
        zones: Sequence[str],
        existing_points: Sequence[np.ndarray],
        sep_tol: float,
        direct_exclusion_tol: float = 0.0,
    ) -> Optional[np.ndarray]:
        rect_indices = cls._zone_rect_indices(env, zones)
        if direct_exclusion_tol > 0.0:
            return cls._sample_carved_rect_point(
                env=env,
                rect_indices=rect_indices,
                rng=rng,
                existing_points=existing_points,
                exclusion_tol=direct_exclusion_tol,
                zones=zones,
            )
        if len(rect_indices) > 0:
            weights = cls._rect_weights(env, rect_indices, zones=zones)
            max_attempts = max(cls.MAX_QUERY_ATTEMPTS, cls.RECT_ATTEMPTS_PER_INDEX * len(rect_indices))
            for _ in range(max_attempts):
                rect_choice = int(rng.choice(len(rect_indices), p=weights))
                rect_idx = int(rect_indices[rect_choice])
                candidate = cls._sample_zone_rect_point(env, rect_idx, zones, rng)
                if candidate is None:
                    continue
                if cls._far_enough(candidate, existing_points, sep_tol):
                    return candidate
        return cls._sample_point_with_predicate(
            env=env,
            rng=rng,
            predicate=lambda point: cls._in_zone(point, env, zones),
            existing_points=existing_points,
            sep_tol=sep_tol,
        )

    @classmethod
    def _sample_rect_point(
        cls,
        env,
        rect_indices: Sequence[int],
        rng: np.random.RandomState,
        existing_points: Sequence[np.ndarray],
        sep_tol: float,
        predicate: Callable[[np.ndarray], bool] | None = None,
        direct_exclusion_tol: float = 0.0,
    ) -> Optional[np.ndarray]:
        if len(rect_indices) == 0:
            return None
        if direct_exclusion_tol > 0.0 and predicate is None:
            return cls._sample_carved_rect_point(
                env=env,
                rect_indices=rect_indices,
                rng=rng,
                existing_points=existing_points,
                exclusion_tol=direct_exclusion_tol,
            )
        weights = cls._rect_weights(env, rect_indices)
        max_attempts = max(cls.MAX_QUERY_ATTEMPTS, cls.RECT_ATTEMPTS_PER_INDEX * len(rect_indices))
        for _ in range(max_attempts):
            rect_choice = int(rng.choice(len(rect_indices), p=weights))
            rect_idx = int(rect_indices[rect_choice])
            sample_seed = int(rng.randint(0, 2**31 - 1))
            candidate = np.asarray(
                HPolyhedronSampler.uniform_sample(
                    env._CSpace_hpoly[rect_idx],
                    RandomGenerator(sample_seed),
                    context=f"MRMPManifestBuilder._sample_rect_point[{rect_idx}]",
                ),
                dtype=float,
            )
            if predicate is not None and not predicate(candidate):
                continue
            if cls._far_enough(candidate, existing_points, sep_tol):
                return candidate
        return None

    @staticmethod
    def _query(start: np.ndarray, goal: np.ndarray, vlimit: float) -> MPQuery:
        return MPQuery(start=start, goal=goal, t_start=0.0, is_stay=True, vlimit=vlimit)

    @classmethod
    def _sample_zone_query(
        cls,
        env,
        rng: np.random.RandomState,
        start_zones: Sequence[str],
        goal_zones: Sequence[str],
        existing_points: Sequence[np.ndarray],
        sep_tol: float,
        min_start_goal_dist: float,
        vlimit: float,
        direct_exclusion_tol: float = 0.0,
    ) -> Optional[MPQuery]:
        for _ in range(cls.MAX_PAIR_ATTEMPTS):
            start = cls._sample_zone_point(
                env,
                rng,
                start_zones,
                existing_points,
                sep_tol,
                direct_exclusion_tol=direct_exclusion_tol,
            )
            if start is None:
                continue
            goal = cls._sample_zone_point(
                env,
                rng,
                goal_zones,
                list(existing_points) + [start],
                sep_tol,
                direct_exclusion_tol=direct_exclusion_tol,
            )
            if goal is None:
                continue
            if np.linalg.norm(start - goal) < min_start_goal_dist:
                continue
            return cls._query(start, goal, vlimit)
        return None

    @classmethod
    def _sample_rect_query(
        cls,
        env,
        rng: np.random.RandomState,
        start_rects: Sequence[int],
        goal_rects: Sequence[int],
        existing_points: Sequence[np.ndarray],
        sep_tol: float,
        min_start_goal_dist: float,
        vlimit: float,
        direct_exclusion_tol: float = 0.0,
    ) -> Optional[MPQuery]:
        for _ in range(cls.MAX_PAIR_ATTEMPTS):
            start = cls._sample_rect_point(
                env,
                start_rects,
                rng,
                existing_points,
                sep_tol,
                direct_exclusion_tol=direct_exclusion_tol,
            )
            if start is None:
                continue
            goal = cls._sample_rect_point(
                env,
                goal_rects,
                rng,
                list(existing_points) + [start],
                sep_tol,
                direct_exclusion_tol=direct_exclusion_tol,
            )
            if goal is None:
                continue
            if np.linalg.norm(start - goal) < min_start_goal_dist:
                continue
            return cls._query(start, goal, vlimit)
        return None

    @classmethod
    def _sample_random_query(
        cls,
        env,
        rng: np.random.RandomState,
        existing_points: Sequence[np.ndarray],
        sep_tol: float,
        min_start_goal_dist: float,
        vlimit: float,
        direct_exclusion_tol: float = 0.0,
    ) -> Optional[MPQuery]:
        rect_indices = tuple(range(len(env.C_Space)))
        for _ in range(cls.MAX_PAIR_ATTEMPTS):
            if direct_exclusion_tol > 0.0:
                start = cls._sample_carved_rect_point(
                    env=env,
                    rect_indices=rect_indices,
                    rng=rng,
                    existing_points=existing_points,
                    exclusion_tol=direct_exclusion_tol,
                )
            else:
                start = cls._sample_point_with_predicate(
                    env=env,
                    rng=rng,
                    predicate=lambda _point: True,
                    existing_points=existing_points,
                    sep_tol=sep_tol,
                )
            if start is None:
                continue
            if direct_exclusion_tol > 0.0:
                goal = cls._sample_carved_rect_point(
                    env=env,
                    rect_indices=rect_indices,
                    rng=rng,
                    existing_points=list(existing_points) + [start],
                    exclusion_tol=direct_exclusion_tol,
                )
            else:
                goal = cls._sample_point_with_predicate(
                    env=env,
                    rng=rng,
                    predicate=lambda _point: True,
                    existing_points=list(existing_points) + [start],
                    sep_tol=sep_tol,
                )
            if goal is None:
                continue
            if np.linalg.norm(start - goal) < min_start_goal_dist:
                continue
            return cls._query(start, goal, vlimit)
        return None

    @classmethod
    def _sample_random_queries(
        cls,
        env,
        num_agents: int,
        rng: np.random.RandomState,
        sep_tol: float,
        min_start_goal_dist: float,
        vlimit: float,
        direct_exclusion_tol: float = 0.0,
    ) -> Optional[List[MPQuery]]:
        existing_points: List[np.ndarray] = []
        queries: List[MPQuery] = []
        for _agent_idx in range(num_agents):
            query = cls._sample_random_query(
                env=env,
                rng=rng,
                existing_points=existing_points,
                sep_tol=sep_tol,
                min_start_goal_dist=min_start_goal_dist,
                vlimit=vlimit,
                direct_exclusion_tol=direct_exclusion_tol,
            )
            if query is None:
                return None
            existing_points.extend([query.start, query.goal])
            queries.append(query)
        return queries

    @classmethod
    def _solve_independent_queries(
        cls,
        instance,
        queries: Sequence[MPQuery],
        reference_budget: float,
    ) -> Tuple[Optional[List[object]], Optional[Tuple[float, float, float]]]:
        sols = []
        total_cost = 0.0
        total_runtime = 0.0
        makespan = 0.0
        for query in queries:
            sol, entry = MRMPExperiment.exact_reference_solve(instance, query, budget=reference_budget)
            if sol is None or not entry.is_success:
                return None, None
            sols.append(sol)
            total_cost += float(entry.cost)
            total_runtime += float(entry.runtime)
            makespan = max(makespan, float(entry.cost))
        return sols, (total_cost, makespan, total_runtime)

    @classmethod
    def _validate_queries_quick(
        cls,
        instance,
        queries: Sequence[MPQuery],
    ) -> bool:
        for query in queries:
            is_valid, _ = instance.stgcs.validate_query(query)
            if not is_valid:
                return False
        return True

    @staticmethod
    def _proxy_metrics(queries: Sequence[MPQuery]) -> Tuple[float, float]:
        durations = [
            float(np.linalg.norm(query.goal - query.start) / max(float(query.vlimit), 1e-9))
            for query in queries
        ]
        if not durations:
            return 0.0, 0.0
        return float(sum(durations)), float(max(durations))

    @staticmethod
    def _pairwise_min_distance(points: Sequence[np.ndarray]) -> float:
        if len(points) < 2:
            return float("inf")
        min_dist = float("inf")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                min_dist = min(min_dist, float(np.linalg.norm(points[i] - points[j])))
        return min_dist

    @classmethod
    def _satisfies_endpoint_separation(
        cls,
        env,
        family: str,
        queries: Sequence[MPQuery],
    ) -> bool:
        if family not in cls.ENDPOINT_SEPARATION_FAMILIES:
            return True
        min_endpoint_separation = cls._endpoint_separation_tol(env, family)
        endpoints = [query.start for query in queries] + [query.goal for query in queries]
        return cls._pairwise_min_distance(endpoints) >= min_endpoint_separation

    @classmethod
    def _template_conflict_stats(cls) -> ConflictStats:
        return ConflictStats(
            num_conflicting_pairs=0,
            num_conflicting_agents=0,
            largest_component=0,
            density=0.0,
        )

    @classmethod
    def _conflict_stats(
        cls,
        instance,
        sols: Sequence[object],
    ) -> ConflictStats:
        conflict_checker = PriorityBasedSearch(instance.stgcs, None, instance.env.robot_radius)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(sols)))
        for i in range(len(sols)):
            for j in range(i + 1, len(sols)):
                if conflict_checker.collision_checking(sols[i].points, sols[j].points):
                    graph.add_edge(i, j)

        num_pairs = graph.number_of_edges()
        conflicting_nodes = [node for node in graph.nodes if graph.degree[node] > 0]
        conflicting_agents = len(conflicting_nodes)
        if conflicting_nodes:
            conflict_graph = graph.subgraph(conflicting_nodes)
            largest_component = max((len(component) for component in nx.connected_components(conflict_graph)), default=0)
        else:
            largest_component = 0
        density = 0.0
        if len(sols) >= 2:
            density = (2.0 * num_pairs) / (len(sols) * (len(sols) - 1))
        return ConflictStats(
            num_conflicting_pairs=num_pairs,
            num_conflicting_agents=conflicting_agents,
            largest_component=largest_component,
            density=density,
        )

    @classmethod
    def max_conflict_density(cls, domain_key: str) -> Optional[float]:
        return cls.MAX_CONFLICT_DENSITY_BY_DOMAIN.get(str(domain_key))

    @classmethod
    def conflict_density_allowed(cls, domain_key: str, density: float) -> bool:
        max_density = cls.max_conflict_density(domain_key)
        if max_density is None:
            return True
        return float(density) <= float(max_density)

    @classmethod
    def conflict_density_selection_enabled(cls, domain_key: str) -> bool:
        return False

    @classmethod
    def conflict_density_selection_pool_count(
        cls,
        domain_key: str,
        target_count: int,
        available_count: int,
    ) -> int:
        del domain_key
        return min(int(target_count), int(available_count))

    @classmethod
    def select_records_for_domain(
        cls,
        domain_key: str,
        records: Sequence[MRMPBenchmarkRecord],
        target_count: int,
    ) -> List[MRMPBenchmarkRecord]:
        del domain_key
        return list(records)[: int(target_count)]

    @classmethod
    def synthesize_record(
        cls,
        base_instance,
        base_record: BaseBenchmarkRecord,
        family: str,
        tier: str,
        rng: np.random.RandomState,
        instance_id: str,
        difficulty_mode: str = DIFFICULTY_MODE_TEMPLATE,
        min_conflicting_pairs: int = 0,
        on_status: StatusCallback | None = None,
    ) -> MRMPBenchmarkRecord:
        difficulty_mode = cls.validate_difficulty_mode(difficulty_mode)
        min_conflicting_pairs = cls.validate_min_conflicting_pairs(min_conflicting_pairs, difficulty_mode)
        domain_key = cls.record_domain_key(base_instance)
        num_agents = cls._num_agents(base_instance, tier, {})
        reference_budget = cls.independent_reference_budget(domain_key)
        for attempt_idx in range(cls.MAX_RECORD_ATTEMPTS):
            cls._emit_status(
                on_status,
                "synthesize_attempt",
                family=family,
                tier=tier,
                spatial_seed=base_instance.seed,
                attempt=attempt_idx + 1,
                max_attempts=cls.MAX_RECORD_ATTEMPTS,
            )
            queries = cls.sample_queries(base_instance, family, tier, rng)
            if queries is None or len(queries) != num_agents:
                cls._emit_status(
                    on_status,
                    "reject",
                    family=family,
                    tier=tier,
                    spatial_seed=base_instance.seed,
                    reason="sample",
                )
                continue
            if not cls._satisfies_endpoint_separation(
                base_instance.env,
                family,
                queries,
            ):
                cls._emit_status(
                    on_status,
                    "reject",
                    family=family,
                    tier=tier,
                    spatial_seed=base_instance.seed,
                    reason="endpoint_separation",
                )
                continue

            if difficulty_mode == cls.DIFFICULTY_MODE_TEMPLATE:
                if not cls._validate_queries_quick(base_instance, queries):
                    cls._emit_status(
                        on_status,
                        "reject",
                        family=family,
                        tier=tier,
                        spatial_seed=base_instance.seed,
                        reason="validate",
                    )
                    continue
                total_cost, makespan = cls._proxy_metrics(queries)
                total_runtime = 0.0
                conflict_stats = cls._template_conflict_stats()
            else:
                cls._emit_status(
                    on_status,
                    "independent_plans",
                    family=family,
                    tier=tier,
                    spatial_seed=base_instance.seed,
                )
                sols, metrics = cls._solve_independent_queries(
                    base_instance,
                    queries,
                    reference_budget=reference_budget,
                )
                if sols is None or metrics is None:
                    cls._emit_status(
                        on_status,
                        "reject",
                        family=family,
                        tier=tier,
                        spatial_seed=base_instance.seed,
                        reason="solve",
                    )
                    continue

                cls._emit_status(
                    on_status,
                    "conflict_screen",
                    family=family,
                    tier=tier,
                    spatial_seed=base_instance.seed,
                )
                conflict_stats = cls._conflict_stats(base_instance, sols)
                if conflict_stats.num_conflicting_pairs < min_conflicting_pairs:
                    cls._emit_status(
                        on_status,
                        "reject",
                        family=family,
                        tier=tier,
                        spatial_seed=base_instance.seed,
                        reason="conflict_count",
                        num_conflicting_pairs=conflict_stats.num_conflicting_pairs,
                        min_conflicting_pairs=min_conflicting_pairs,
                    )
                    continue
                max_conflict_density = cls.max_conflict_density(domain_key)
                if max_conflict_density is not None and conflict_stats.density > max_conflict_density:
                    cls._emit_status(
                        on_status,
                        "reject",
                        family=family,
                        tier=tier,
                        spatial_seed=base_instance.seed,
                        reason="conflict_density",
                        conflict_density=conflict_stats.density,
                        max_conflict_density=max_conflict_density,
                    )
                    continue

                total_cost, makespan, total_runtime = metrics
            return MRMPBenchmarkRecord(
                instance_id=instance_id,
                base_instance_id=base_record.instance_id,
                queries=[QuerySpec.from_query(query) for query in queries],
                traffic_family=family,
                traffic_tier=tier,
                num_agents=num_agents,
                independent_cost_sum=total_cost,
                independent_makespan=makespan,
                independent_runtime=total_runtime,
                num_conflicting_pairs=conflict_stats.num_conflicting_pairs,
                num_conflicting_agents=conflict_stats.num_conflicting_agents,
                conflict_largest_component=conflict_stats.largest_component,
                conflict_density=conflict_stats.density,
                stgcs_num_vertices=base_instance.stgcs.G.number_of_nodes(),
                stgcs_num_edges=base_instance.stgcs.G.number_of_edges(),
            )
        raise RuntimeError(
            f"Failed to synthesize MRMP record for seed={base_instance.seed}, family={family}, tier={tier}."
        )

    @staticmethod
    def record_domain_key(base_instance) -> str:
        if base_instance.__class__.__name__.startswith("Maze"):
            return "maze"
        return f"grid{base_instance.env.dim}d"

    @classmethod
    def sample_queries(
        cls,
        base_instance,
        family: str,
        tier: str,
        rng: np.random.RandomState,
    ) -> Optional[MRMPQuery]:
        raise NotImplementedError


class GridMRMPManifestBuilder(MRMPManifestBuilder):
    SIDE_BAND_RATIO = 0.25
    MIN_GRID_AGENTS = 2
    FAMILIES_BY_SPACE_DIM = {
        2: ("random", "switch-spread", "merge-flow"),
        3: ("random", "switch-spread", "merge-flow"),
    }
    DEFAULT_SIZES_BY_SPACE_DIM = {
        2: (1, 2, 3, 4, 5),
        3: (1, 2, 3),
    }
    DEFAULT_COUNT_PER_SIZE = 100

    @classmethod
    def _reference_grid_size(cls, space_dim: int) -> int:
        return int(max(cls.DEFAULT_SIZES_BY_SPACE_DIM[space_dim]))

    @classmethod
    def size_targets(
        cls,
        sizes: Sequence[int],
        counts_per_size: int | None = None,
        count_total: int | None = None,
    ) -> Dict[int, int]:
        if count_total is not None:
            if count_total < 0:
                raise ValueError(f"Grid MRMP total count must be non-negative, got {count_total}")
            base = count_total // len(sizes)
            remainder = count_total % len(sizes)
            targets = {int(size): base for size in sizes}
            for size in sizes[:remainder]:
                targets[int(size)] += 1
            return targets
        if counts_per_size is None:
            counts_per_size = cls.DEFAULT_COUNT_PER_SIZE
        if counts_per_size < 0:
            raise ValueError(f"Grid MRMP count per size must be non-negative, got {counts_per_size}")
        return {int(size): int(counts_per_size) for size in sizes}

    @classmethod
    def adaptive_tier_agent_count(cls, space_dim: int, tier: str, size: int) -> int:
        if size < 1:
            raise ValueError(f"Grid size must be positive, got {size}")
        base_count = cls.tier_agent_count(f"grid{space_dim}d", tier)
        reference_size = cls._reference_grid_size(space_dim)
        scaled_count = int(round(base_count * (float(size) / float(reference_size))))
        return max(cls.MIN_GRID_AGENTS, scaled_count)

    @classmethod
    def _grid_size(cls, base_instance, env_params: Dict[str, int] | None = None) -> int:
        if env_params is not None and "size" in env_params:
            return int(env_params["size"])
        if hasattr(base_instance, "gridN"):
            return int(base_instance.gridN)
        raise ValueError("Grid MRMP builder requires a grid size to determine the number of agents.")

    @classmethod
    def _num_agents(
        cls,
        base_instance,
        tier: str,
        env_params: Dict[str, int],
    ) -> int:
        return cls.adaptive_tier_agent_count(
            space_dim=int(base_instance.env.dim),
            tier=tier,
            size=cls._grid_size(base_instance, env_params),
        )

    @classmethod
    def _sample_valid_random_queries(
        cls,
        base_instance,
        rng: np.random.RandomState,
        num_agents: int,
        sep_tol: float,
        min_start_goal_dist: float,
        direct_exclusion_tol: float,
    ) -> Optional[List[MPQuery]]:
        env = base_instance.env
        existing_points: List[np.ndarray] = []
        queries: List[MPQuery] = []
        for _agent_idx in range(num_agents):
            query = None
            for _ in range(cls.MAX_QUERY_ATTEMPTS):
                candidate = cls._sample_random_query(
                    env=env,
                    rng=rng,
                    existing_points=existing_points,
                    sep_tol=sep_tol,
                    min_start_goal_dist=min_start_goal_dist,
                    vlimit=base_instance.stgcs.vlimit,
                    direct_exclusion_tol=direct_exclusion_tol,
                )
                if candidate is None:
                    continue
                if cls._validate_queries_quick(base_instance, [candidate]):
                    query = candidate
                    break
            if query is None:
                return None
            existing_points.extend([query.start, query.goal])
            queries.append(query)
        return queries

    @classmethod
    def _sample_valid_pattern_queries(
        cls,
        base_instance,
        patterns: Sequence[Tuple[Tuple[str, ...], Tuple[str, ...]]],
        rng: np.random.RandomState,
        num_agents: int,
        sep_tol: float,
        min_start_goal_dist: float,
        direct_exclusion_tol: float,
    ) -> Optional[List[MPQuery]]:
        env = base_instance.env
        existing_points: List[np.ndarray] = []
        queries: List[MPQuery] = []
        for agent_idx in range(num_agents):
            start_zones, goal_zones = patterns[agent_idx % len(patterns)]
            query = None
            for _ in range(cls.MAX_QUERY_ATTEMPTS):
                candidate = cls._sample_zone_query(
                    env=env,
                    rng=rng,
                    start_zones=start_zones,
                    goal_zones=goal_zones,
                    existing_points=existing_points,
                    sep_tol=sep_tol,
                    min_start_goal_dist=min_start_goal_dist,
                    vlimit=base_instance.stgcs.vlimit,
                    direct_exclusion_tol=direct_exclusion_tol,
                )
                if candidate is None:
                    continue
                if cls._validate_queries_quick(base_instance, [candidate]):
                    query = candidate
                    break
            if query is None:
                return None
            existing_points.extend([query.start, query.goal])
            queries.append(query)
        return queries

    @classmethod
    def _patterns_for_family(cls, space_dim: int, family: str) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        if family == "switch-spread":
            return [
                (("left",), ("right",)),
                (("right",), ("left",)),
            ]
        if space_dim == 2:
            return [
                (("left",), ("right",)),
                (("bottom",), ("right",)),
            ]
        return [
            (("left",), ("right",)),
            (("bottom",), ("right",)),
            (("front",), ("right",)),
        ]

    @classmethod
    def sample_queries(
        cls,
        base_instance,
        family: str,
        tier: str,
        rng: np.random.RandomState,
    ) -> Optional[MRMPQuery]:
        env = base_instance.env
        num_agents = cls._num_agents(
            base_instance=base_instance,
            tier=tier,
            env_params={"size": cls._grid_size(base_instance)},
        )
        bbox_diag = float(np.linalg.norm(env.ub - env.lb))
        endpoint_sep_tol = cls._endpoint_separation_tol(env, family)
        if family == "random":
            sep_tol = max(1.15 * cls._point_separation(env), endpoint_sep_tol)
            queries = cls._sample_valid_random_queries(
                base_instance=base_instance,
                rng=rng,
                num_agents=num_agents,
                sep_tol=sep_tol,
                min_start_goal_dist=0.28 * bbox_diag,
                direct_exclusion_tol=sep_tol if endpoint_sep_tol > 0.0 else 0.0,
            )
            return None if queries is None else MRMPQuery(queries)
        patterns = cls._patterns_for_family(env.dim, family)
        sep_tol = max(cls._point_separation(env), endpoint_sep_tol)
        if family == "switch-spread":
            sep_tol *= cls.SWITCH_SPREAD_SEPARATION_MULTIPLIER
        min_start_goal_dist = 0.35 * bbox_diag if family == "switch-spread" else 0.32 * bbox_diag
        queries = cls._sample_valid_pattern_queries(
            base_instance=base_instance,
            patterns=patterns,
            rng=rng,
            num_agents=num_agents,
            sep_tol=sep_tol,
            min_start_goal_dist=min_start_goal_dist,
            direct_exclusion_tol=sep_tol if endpoint_sep_tol > 0.0 else 0.0,
        )
        return None if queries is None else MRMPQuery(queries)

    @classmethod
    def _instance_id(cls, base_instance_id: str, tier: str, family: str) -> str:
        return f"mrmp-{base_instance_id}-{tier}-{family}"

    @classmethod
    def build_manifest(
        cls,
        base_records: Sequence[BaseBenchmarkRecord],
        space_dim: int,
        counts_per_size: int = DEFAULT_COUNT_PER_SIZE,
        count_total: int | None = None,
        sizes: Sequence[int] | None = None,
        difficulty_mode: str = MRMPManifestBuilder.DIFFICULTY_MODE_TEMPLATE,
        min_conflicting_pairs: int = 0,
        on_record: RecordCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> List[MRMPBenchmarkRecord]:
        difficulty_mode = cls.validate_difficulty_mode(difficulty_mode)
        min_conflicting_pairs = cls.validate_min_conflicting_pairs(min_conflicting_pairs, difficulty_mode)
        families = cls.FAMILIES_BY_SPACE_DIM[space_dim]
        sizes = cls.DEFAULT_SIZES_BY_SPACE_DIM[space_dim] if sizes is None else tuple(sizes)
        size_targets = cls.size_targets(sizes=sizes, counts_per_size=counts_per_size, count_total=count_total)
        records: List[MRMPBenchmarkRecord] = []
        candidates = [
            record
            for record in base_records
            if record.domain == "grid"
            and record.space_dim == space_dim
            and record.env_params.get("size") in sizes
        ]

        for size in sizes:
            size_target = int(size_targets[int(size)])
            quotas = cls._pair_targets(size_target, families)
            size_records = [
                record
                for record in candidates
                if int(record.env_params.get("size", -1)) == size
            ]
            ordinal = 0
            size_built = 0
            for base_record in size_records:
                if size_built >= size_target:
                    break
                base = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
                record = None
                tried_pairs: set[Tuple[str, str]] = set()
                for pair_group in (
                    cls._candidate_pairs(quotas, families, ordinal),
                    cls._ordered_candidate_pairs(families, ordinal),
                ):
                    if record is not None:
                        break
                    for tier, family in pair_group:
                        if (tier, family) in tried_pairs:
                            continue
                        tried_pairs.add((tier, family))
                        cls._emit_status(
                            on_status,
                            "candidate",
                            family=family,
                            tier=tier,
                            spatial_seed=base_record.spatial_seed,
                            failed_attempts=0,
                            remaining=sum(quotas.values()),
                            size=size,
                        )
                        try:
                            record = cls.synthesize_record(
                                base_instance=base,
                                base_record=base_record,
                                family=family,
                                tier=tier,
                                rng=np.random.RandomState(
                                    base_record.spatial_seed
                                    + 31 * ordinal
                                    + 101 * cls.TIERS.index(tier)
                                    + 17 * families.index(family)
                                ),
                                instance_id=cls._instance_id(base_record.instance_id, tier, family),
                                difficulty_mode=difficulty_mode,
                                min_conflicting_pairs=min_conflicting_pairs,
                                on_status=on_status,
                            )
                        except RuntimeError:
                            continue
                        if quotas[(tier, family)] > 0:
                            quotas[(tier, family)] -= 1
                        records.append(record)
                        size_built += 1
                        ordinal += 1
                        if on_record is not None:
                            on_record(record, records)
                        break
                if record is None:
                    continue
            if size_built < size_target:
                raise RuntimeError(
                    f"Not enough usable MRMP grid{space_dim}d base instances for size={size}. Requested {size_target}, built {size_built}."
                )

        return records

class MazeMRMPManifestBuilder(MRMPManifestBuilder):
    CENTER_BAND_RATIO = 0.5
    FAMILIES = ("random", "switch-spread", "merge-flow")
    DEFAULT_COUNT_TOTAL = 300

    @classmethod
    def _patterns_for_family(cls, family: str) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        return {
            "switch-spread": [
                (("left", "mid_y"), ("right", "mid_y")),
                (("right", "mid_y"), ("left", "mid_y")),
            ],
            "merge-flow": [
                (("left",), ("right",)),
                (("bottom",), ("right",)),
            ],
        }[family]

    @classmethod
    def sample_queries(
        cls,
        base_instance,
        family: str,
        tier: str,
        rng: np.random.RandomState,
    ) -> Optional[MRMPQuery]:
        env = base_instance.env
        num_agents = cls.tier_agent_count("maze", tier)
        bbox_diag = float(np.linalg.norm(env.ub - env.lb))
        endpoint_sep_tol = cls._endpoint_separation_tol(env, family)
        if family == "random":
            sep_tol = max(1.10 * cls._point_separation(env), endpoint_sep_tol)
            queries = cls._sample_random_queries(
                env=env,
                num_agents=num_agents,
                rng=rng,
                sep_tol=sep_tol,
                min_start_goal_dist=0.30 * bbox_diag,
                vlimit=base_instance.stgcs.vlimit,
                direct_exclusion_tol=sep_tol if endpoint_sep_tol > 0.0 else 0.0,
            )
            return None if queries is None else MRMPQuery(queries)
        patterns = cls._patterns_for_family(family)
        sep_tol = max(cls._point_separation(env), endpoint_sep_tol)
        if family == "switch-spread":
            sep_tol *= cls.SWITCH_SPREAD_SEPARATION_MULTIPLIER
        existing_points: List[np.ndarray] = []
        queries: List[MPQuery] = []
        min_start_goal_dist = 0.36 * bbox_diag if family == "switch-spread" else 0.33 * bbox_diag
        for agent_idx in range(num_agents):
            start_zones, goal_zones = patterns[agent_idx % len(patterns)]
            query = cls._sample_zone_query(
                env=env,
                rng=rng,
                start_zones=start_zones,
                goal_zones=goal_zones,
                existing_points=existing_points,
                sep_tol=sep_tol,
                min_start_goal_dist=min_start_goal_dist,
                vlimit=base_instance.stgcs.vlimit,
                direct_exclusion_tol=sep_tol if endpoint_sep_tol > 0.0 else 0.0,
            )
            if query is None:
                return None
            existing_points.extend([query.start, query.goal])
            queries.append(query)
        return MRMPQuery(queries)

    @classmethod
    def _instance_id(cls, base_instance_id: str, tier: str, family: str) -> str:
        return f"mrmp-{base_instance_id}-{tier}-{family}"

    @classmethod
    def build_manifest(
        cls,
        base_records: Sequence[BaseBenchmarkRecord],
        count_total: int = DEFAULT_COUNT_TOTAL,
        difficulty_mode: str = MRMPManifestBuilder.DIFFICULTY_MODE_TEMPLATE,
        min_conflicting_pairs: int = 0,
        on_record: RecordCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> List[MRMPBenchmarkRecord]:
        difficulty_mode = cls.validate_difficulty_mode(difficulty_mode)
        min_conflicting_pairs = cls.validate_min_conflicting_pairs(min_conflicting_pairs, difficulty_mode)
        records: List[MRMPBenchmarkRecord] = []
        quotas = cls._pair_targets(count_total, cls.FAMILIES)
        candidates = sorted(
            (record for record in base_records if record.domain == "maze"),
            key=lambda record: record.spatial_seed,
        )
        ordinal = 0
        for base_record in candidates:
            if len(records) >= count_total or sum(quotas.values()) <= 0:
                break
            base = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
            record = None
            tried_pairs: set[Tuple[str, str]] = set()
            for pair_group in (
                cls._candidate_pairs(quotas, cls.FAMILIES, ordinal),
                cls._ordered_candidate_pairs(cls.FAMILIES, ordinal),
            ):
                if record is not None:
                    break
                for tier, family in pair_group:
                    if (tier, family) in tried_pairs:
                        continue
                    tried_pairs.add((tier, family))
                    cls._emit_status(
                        on_status,
                        "candidate",
                        family=family,
                        tier=tier,
                        spatial_seed=base_record.spatial_seed,
                        failed_attempts=0,
                        remaining=sum(quotas.values()),
                        width=int(base_record.env_params["width"]),
                        height=int(base_record.env_params["height"]),
                    )
                    try:
                        record = cls.synthesize_record(
                            base_instance=base,
                            base_record=base_record,
                            family=family,
                            tier=tier,
                            rng=np.random.RandomState(
                                base_record.spatial_seed
                                + 41 * ordinal
                                + 19 * cls.FAMILIES.index(family)
                            ),
                            instance_id=cls._instance_id(base_record.instance_id, tier, family),
                            difficulty_mode=difficulty_mode,
                            min_conflicting_pairs=min_conflicting_pairs,
                            on_status=on_status,
                        )
                    except RuntimeError:
                        continue
                    if quotas[(tier, family)] > 0:
                        quotas[(tier, family)] -= 1
                    records.append(record)
                    ordinal += 1
                    if on_record is not None:
                        on_record(record, records)
                    break
            if record is None:
                continue
        if len(records) < count_total:
            raise RuntimeError(
                f"Not enough usable MRMP maze base instances. Requested {count_total}, built {len(records)}."
            )
        return records


class PBSExpansionMRMPManifestBuilder(GridMRMPManifestBuilder):
    FAMILY = "intersection"
    TIER = "10-robot"
    NUM_AGENTS = 10
    MIN_START_GOAL_DIST_RATIO = 0.35
    MIN_TRAJECTORY_INTERSECTING_PAIRS = 5

    @staticmethod
    def record_domain_key(base_instance) -> str:
        class_name = base_instance.__class__.__name__
        if class_name.startswith("Maze"):
            return "maze"
        if class_name.startswith("Iris2D"):
            return "iris-2d"
        return f"grid{base_instance.env.dim}d"

    @classmethod
    def _num_agents(
        cls,
        base_instance,
        tier: str,
        env_params: Dict[str, int],
    ) -> int:
        del base_instance, tier, env_params
        return cls.NUM_AGENTS

    @classmethod
    def sample_queries(
        cls,
        base_instance,
        family: str,
        tier: str,
        rng: np.random.RandomState,
    ) -> Optional[MRMPQuery]:
        if family != cls.FAMILY:
            raise ValueError(f"PBS expansion ablation only supports {cls.FAMILY!r} traffic.")
        if tier != cls.TIER:
            raise ValueError(f"PBS expansion ablation only supports {cls.TIER!r} traffic tier.")

        env = base_instance.env
        bbox_diag = float(np.linalg.norm(env.ub - env.lb))
        endpoint_sep_tol = cls._endpoint_separation_tol(env, family)
        sep_tol = max(cls._point_separation(env), endpoint_sep_tol)
        queries = cls._sample_valid_random_queries(
            base_instance=base_instance,
            rng=rng,
            num_agents=cls.NUM_AGENTS,
            sep_tol=sep_tol,
            min_start_goal_dist=cls.MIN_START_GOAL_DIST_RATIO * bbox_diag,
            direct_exclusion_tol=sep_tol,
        )
        return None if queries is None else MRMPQuery(queries)

    @classmethod
    def _solve_independent_queries(
        cls,
        instance,
        queries: Sequence[MPQuery],
        reference_budget: float,
    ) -> Tuple[Optional[List[object]], Optional[Tuple[float, float, float]]]:
        planner = MRMPExperiment._build_low_level_planner(
            instance,
            PBSExpansionAblation.LOW_LEVEL_SPEC,
            runtime_limit_secs=float(reference_budget),
        )
        sols = []
        total_cost = 0.0
        total_runtime = 0.0
        makespan = 0.0
        for query in queries:
            sol, runtime, status = planner.plan(instance.stgcs, query)
            if status == STPlanStatus.FAIL or sol is None:
                return None, None
            sols.append(sol)
            total_cost += float(sol.duration)
            total_runtime += float(runtime)
            makespan = max(makespan, float(sol.duration))
        return sols, (total_cost, makespan, total_runtime)

    @classmethod
    def _instance_id(cls, base_instance_id: str) -> str:
        return f"mrmp-{base_instance_id}-{cls.TIER}-{cls.FAMILY}"

    @classmethod
    def _prepare_low_level_heuristics(cls, instance, base_record: BaseBenchmarkRecord) -> None:
        if base_record.manifest_path is None:
            raise ValueError(
                f"Base record {base_record.instance_id!r} needs a manifest path to load Max heuristic caches."
            )
        BaseOfflineHeuristicStore.prepare_instance_for_search(
            instance,
            base_record.manifest_path,
            base_record,
            PBSExpansionAblation.required_heuristics(),
            online_td_timeout_secs=cls.independent_reference_budget(base_record.domain_key),
        )

    @classmethod
    def build_manifest(
        cls,
        base_records: Sequence[BaseBenchmarkRecord],
        on_record: RecordCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> List[MRMPBenchmarkRecord]:
        records: List[MRMPBenchmarkRecord] = []
        for ordinal, base_record in enumerate(base_records):
            cls._emit_status(
                on_status,
                "candidate",
                family=cls.FAMILY,
                tier=cls.TIER,
                spatial_seed=base_record.spatial_seed,
                remaining=len(base_records) - len(records),
                base_instance_id=base_record.instance_id,
            )
            base = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
            cls._prepare_low_level_heuristics(base, base_record)
            record = cls.synthesize_record(
                base_instance=base,
                base_record=base_record,
                family=cls.FAMILY,
                tier=cls.TIER,
                rng=np.random.RandomState(base_record.spatial_seed + 7919 * (ordinal + 1)),
                instance_id=cls._instance_id(base_record.instance_id),
                difficulty_mode=cls.DIFFICULTY_MODE_SOLVER,
                min_conflicting_pairs=cls.MIN_TRAJECTORY_INTERSECTING_PAIRS,
                on_status=on_status,
            )
            records.append(record)
            if on_record is not None:
                on_record(record, records)
        return records


class WindowedCoordinationMRMPManifestBuilder(PBSExpansionMRMPManifestBuilder):
    TIER = "20-robot"
    NUM_AGENTS = 20
    MIN_TRAJECTORY_INTERSECTING_PAIRS = 10
    MAX_CONFLICT_DENSITY_BY_DOMAIN = {
        "grid2d": 0.45,
        "maze": 0.45,
    }
    TARGET_CONFLICT_MEAN_BY_DOMAIN = {
        "grid2d": 0.25,
        "maze": 0.25,
    }
    TARGET_CONFLICT_MEDIAN_BY_DOMAIN = {
        "grid2d": 0.25,
        "maze": 0.25,
    }
    TARGET_CONFLICT_MAX_BY_DOMAIN = {
        "grid2d": 0.45,
        "maze": 0.45,
    }

    @classmethod
    def record_matches_benchmark(cls, record: MRMPBenchmarkRecord) -> bool:
        return (
            str(record.traffic_family) == cls.FAMILY
            and str(record.traffic_tier) == cls.TIER
            and int(record.num_agents) == int(cls.NUM_AGENTS)
            and cls.conflict_density_allowed(record.domain_key, record.conflict_density)
        )

    @classmethod
    def conflict_density_selection_enabled(cls, domain_key: str) -> bool:
        return str(domain_key) in cls.TARGET_CONFLICT_MEAN_BY_DOMAIN

    @classmethod
    def conflict_density_selection_pool_count(
        cls,
        domain_key: str,
        target_count: int,
        available_count: int,
    ) -> int:
        if not cls.conflict_density_selection_enabled(domain_key):
            return super().conflict_density_selection_pool_count(domain_key, target_count, available_count)
        return int(available_count)

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        if not values:
            return float("nan")
        sorted_values = sorted(float(value) for value in values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return sorted_values[mid]
        return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])

    @classmethod
    def _density_selection_score(
        cls,
        domain_key: str,
        records: Sequence[MRMPBenchmarkRecord],
    ) -> Tuple[float, float, float, float, Tuple[str, ...]]:
        densities = [float(record.conflict_density) for record in records]
        mean_density = float(sum(densities) / len(densities))
        median_density = cls._median(densities)
        max_density = max(densities)
        target_mean = float(cls.TARGET_CONFLICT_MEAN_BY_DOMAIN[str(domain_key)])
        target_median = float(cls.TARGET_CONFLICT_MEDIAN_BY_DOMAIN[str(domain_key)])
        target_max = float(cls.TARGET_CONFLICT_MAX_BY_DOMAIN[str(domain_key)])
        overflow = max(0.0, max_density - target_max)
        score = (
            abs(mean_density - target_mean)
            + abs(median_density - target_median)
            + 0.25 * abs(max_density - target_max)
            + 10.0 * overflow
        )
        return (
            score,
            abs(mean_density - target_mean),
            abs(median_density - target_median),
            abs(max_density - target_max),
            tuple(sorted(record.instance_id for record in records)),
        )

    @classmethod
    def _improve_density_selection(
        cls,
        domain_key: str,
        selected: Sequence[MRMPBenchmarkRecord],
        candidates: Sequence[MRMPBenchmarkRecord],
    ) -> List[MRMPBenchmarkRecord]:
        selected_records = list(selected)
        selected_ids = {record.instance_id for record in selected_records}
        unselected_records = [record for record in candidates if record.instance_id not in selected_ids]
        best_score = cls._density_selection_score(domain_key, selected_records)

        improved = True
        while improved:
            improved = False
            best_swap: Tuple[int, int, Tuple[float, float, float, float, Tuple[str, ...]]] | None = None
            for selected_idx, _selected_record in enumerate(selected_records):
                for unselected_idx, unselected_record in enumerate(unselected_records):
                    trial = list(selected_records)
                    trial[selected_idx] = unselected_record
                    trial_score = cls._density_selection_score(domain_key, trial)
                    if trial_score < best_score:
                        best_score = trial_score
                        best_swap = (selected_idx, unselected_idx, trial_score)
            if best_swap is not None:
                selected_idx, unselected_idx, _score = best_swap
                selected_records[selected_idx], unselected_records[unselected_idx] = (
                    unselected_records[unselected_idx],
                    selected_records[selected_idx],
                )
                improved = True
        return selected_records

    @classmethod
    def select_records_for_domain(
        cls,
        domain_key: str,
        records: Sequence[MRMPBenchmarkRecord],
        target_count: int,
    ) -> List[MRMPBenchmarkRecord]:
        domain_key = str(domain_key)
        if not cls.conflict_density_selection_enabled(domain_key):
            return super().select_records_for_domain(domain_key, records, target_count)
        target_count = int(target_count)
        candidates = sorted(
            records,
            key=lambda record: (float(record.conflict_density), record.instance_id),
        )
        if len(candidates) <= target_count:
            return candidates

        target_mean = float(cls.TARGET_CONFLICT_MEAN_BY_DOMAIN[domain_key])
        target_max = float(cls.TARGET_CONFLICT_MAX_BY_DOMAIN[domain_key])
        closest_to_mean = sorted(
            candidates,
            key=lambda record: (abs(float(record.conflict_density) - target_mean), record.instance_id),
        )[:target_count]
        tail_record = min(
            candidates,
            key=lambda record: (abs(float(record.conflict_density) - target_max), record.instance_id),
        )
        with_tail = [tail_record]
        with_tail.extend(
            record
            for record in closest_to_mean
            if record.instance_id != tail_record.instance_id
        )
        if len(with_tail) < target_count:
            with_tail.extend(
                record
                for record in candidates
                if record.instance_id not in {item.instance_id for item in with_tail}
            )
        initial_sets = (
            closest_to_mean,
            with_tail[:target_count],
        )
        best_records = min(
            (cls._improve_density_selection(domain_key, initial, candidates) for initial in initial_sets),
            key=lambda selected: cls._density_selection_score(domain_key, selected),
        )
        return sorted(best_records, key=lambda record: record.instance_id)


class MRMPPerformanceComparisonManifestBuilder(PBSExpansionMRMPManifestBuilder):
    TIER = "2,4,...,20-robot"
    NUM_AGENTS = MRMPPerformanceComparison.MAX_NUM_AGENTS
    FAMILY = MRMPPerformanceComparison.FAMILY
    MIN_TRAJECTORY_INTERSECTING_PAIRS = 0

    @classmethod
    def target_count(cls, replicates_per_robot_count: int) -> int:
        return len(MRMPPerformanceComparison.num_agent_values()) * int(replicates_per_robot_count)

    @classmethod
    def record_matches_benchmark(cls, record: MRMPBenchmarkRecord) -> bool:
        try:
            expected_tier = MRMPPerformanceComparison.traffic_tier(int(record.num_agents))
        except ValueError:
            return False
        return (
            str(record.traffic_family) == cls.FAMILY
            and str(record.traffic_tier) == expected_tier
        )

    @classmethod
    def _num_agents(
        cls,
        base_instance,
        tier: str,
        env_params: Dict[str, int],
    ) -> int:
        del base_instance, env_params
        return MRMPPerformanceComparison.num_agents_from_tier(tier)

    @classmethod
    def sample_queries(
        cls,
        base_instance,
        family: str,
        tier: str,
        rng: np.random.RandomState,
    ) -> Optional[MRMPQuery]:
        if family != cls.FAMILY:
            raise ValueError(f"MRMP performance comparison only supports {cls.FAMILY!r} traffic.")

        env = base_instance.env
        num_agents = cls._num_agents(base_instance, tier, {})
        bbox_diag = float(np.linalg.norm(env.ub - env.lb))
        endpoint_sep_tol = cls._endpoint_separation_tol(env, family)
        sep_tol = max(cls._point_separation(env), endpoint_sep_tol)
        queries = cls._sample_valid_random_queries(
            base_instance=base_instance,
            rng=rng,
            num_agents=num_agents,
            sep_tol=sep_tol,
            min_start_goal_dist=cls.MIN_START_GOAL_DIST_RATIO * bbox_diag,
            direct_exclusion_tol=sep_tol,
        )
        return None if queries is None else MRMPQuery(queries)

    @classmethod
    def _instance_id(cls, base_instance_id: str, num_agents: int) -> str:
        return f"mrmp-{base_instance_id}-{MRMPPerformanceComparison.traffic_tier(num_agents)}-{cls.FAMILY}"

    @staticmethod
    def _replicate_counts(
        records: Sequence[MRMPBenchmarkRecord],
    ) -> Dict[int, int]:
        counts = {num_agents: 0 for num_agents in MRMPPerformanceComparison.num_agent_values()}
        for record in records:
            num_agents = int(record.num_agents)
            if num_agents in counts:
                counts[num_agents] += 1
        return counts

    @classmethod
    def _current_existing_records(
        cls,
        existing_records: Sequence[MRMPBenchmarkRecord],
        replicates_per_robot_count: int,
    ) -> List[MRMPBenchmarkRecord]:
        counts = {num_agents: 0 for num_agents in MRMPPerformanceComparison.num_agent_values()}
        records: List[MRMPBenchmarkRecord] = []
        for record in existing_records:
            if not cls.record_matches_benchmark(record):
                continue
            num_agents = int(record.num_agents)
            if counts[num_agents] >= int(replicates_per_robot_count):
                continue
            records.append(record)
            counts[num_agents] += 1
        return records

    @classmethod
    def build_manifest(
        cls,
        base_records: Sequence[BaseBenchmarkRecord],
        replicates_per_robot_count: int = 1,
        existing_records: Sequence[MRMPBenchmarkRecord] | None = None,
        on_record: RecordCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> List[MRMPBenchmarkRecord]:
        replicates_per_robot_count = int(replicates_per_robot_count)
        if replicates_per_robot_count <= 0:
            raise ValueError("replicates_per_robot_count must be positive.")

        records = cls._current_existing_records(existing_records or (), replicates_per_robot_count)
        used_instance_ids = {record.instance_id for record in records}
        counts = cls._replicate_counts(records)

        for num_agents in MRMPPerformanceComparison.num_agent_values():
            tier = MRMPPerformanceComparison.traffic_tier(num_agents)
            for base_record in base_records:
                if counts[num_agents] >= replicates_per_robot_count:
                    break
                instance_id = cls._instance_id(base_record.instance_id, num_agents)
                if instance_id in used_instance_ids:
                    continue
                cls._emit_status(
                    on_status,
                    "candidate",
                    family=cls.FAMILY,
                    tier=tier,
                    num_agents=num_agents,
                    spatial_seed=base_record.spatial_seed,
                    remaining=replicates_per_robot_count - counts[num_agents],
                    base_instance_id=base_record.instance_id,
                )
                base = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
                cls._prepare_low_level_heuristics(base, base_record)
                try:
                    record = cls.synthesize_record(
                        base_instance=base,
                        base_record=base_record,
                        family=cls.FAMILY,
                        tier=tier,
                        rng=np.random.RandomState(
                            base_record.spatial_seed
                            + 7919 * int(num_agents)
                            + 104729 * (counts[num_agents] + 1)
                        ),
                        instance_id=instance_id,
                        difficulty_mode=cls.DIFFICULTY_MODE_SOLVER,
                        min_conflicting_pairs=cls.MIN_TRAJECTORY_INTERSECTING_PAIRS,
                        on_status=on_status,
                    )
                except RuntimeError as exc:
                    if not str(exc).startswith("Failed to synthesize MRMP record"):
                        raise
                    cls._emit_status(
                        on_status,
                        "candidate_failed",
                        base_instance_id=base_record.instance_id,
                        num_agents=num_agents,
                        reason=str(exc),
                    )
                    continue
                records.append(record)
                used_instance_ids.add(record.instance_id)
                counts[num_agents] += 1
                if on_record is not None:
                    on_record(record, records)

            if counts[num_agents] < replicates_per_robot_count:
                raise RuntimeError(
                    f"Unable to build {replicates_per_robot_count} {tier} MRMP performance-comparison records; "
                    f"built {counts[num_agents]}."
                )

        return records
