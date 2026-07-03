from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from pydrake.all import HPolyhedron, Iris, IrisOptions

from environment.env import Env
from environment.obstacle import StaticPolygon
from stgcs.geometry_utils import collinear, hpoly_to_vrep, make_hpolytope


@dataclass(frozen=True)
class Iris2DRegion:
    seed: np.ndarray
    hpoly: HPolyhedron
    vertices: np.ndarray


@dataclass(frozen=True)
class Iris2DProblem:
    domain: HPolyhedron
    domain_lb: np.ndarray
    domain_ub: np.ndarray
    obstacle_polygons: List[np.ndarray]
    original_obstacles: List[HPolyhedron]
    iris_obstacles: List[HPolyhedron]
    clearance: float
    options: IrisOptions

    @staticmethod
    def _as_2d_point(point: np.ndarray, name: str) -> np.ndarray:
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.shape != (2,):
            raise ValueError(f"{name} must be a 2D point, got shape {point.shape}.")
        return point

    @staticmethod
    def _ordered_vertices(vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=float)
        if len(vertices) <= 2 or collinear(vertices):
            return vertices
        hull = ConvexHull(vertices)
        return vertices[hull.vertices]

    @classmethod
    def _canonical_obstacle_polygon(cls, vertices: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(f"Obstacle vertices must have shape (n, 2), got {vertices.shape}.")

        vertices = np.unique(vertices, axis=0)
        if vertices.shape[0] < 3:
            raise ValueError("A 2D polygonal obstacle needs at least three unique vertices.")

        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]
        hull_hpoly = make_hpolytope(hull_vertices)
        slack = np.abs(hull_hpoly.A() @ vertices.T - hull_hpoly.b().reshape(-1, 1))
        vertices_on_boundary = np.any(slack <= tol, axis=0)
        if not np.all(vertices_on_boundary):
            raise ValueError(
                "IRIS expects convex obstacles; at least one obstacle vertex lies strictly inside "
                "the convex hull of the supplied polygon."
            )
        return hull_vertices

    @staticmethod
    def _normalize_halfspaces(hpoly: HPolyhedron, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        A = np.asarray(hpoly.A(), dtype=float)
        b = np.asarray(hpoly.b(), dtype=float).reshape(-1)
        norms = np.linalg.norm(A, axis=1)
        if np.any(norms <= tol):
            raise ValueError("Cannot normalize a halfspace with a near-zero normal.")
        return A / norms[:, None], b / norms

    @classmethod
    def inflate_obstacle(cls, obstacle: HPolyhedron, clearance: float) -> HPolyhedron:
        if clearance < 0.0:
            raise ValueError(f"clearance must be nonnegative, got {clearance}.")
        A, b = cls._normalize_halfspaces(obstacle)
        return HPolyhedron(A, b + clearance)

    @staticmethod
    def make_iris_options(
        *,
        iteration_limit: int,
        random_seed: int,
        require_seed_contained: bool = True,
    ) -> IrisOptions:
        options = IrisOptions()
        options.iteration_limit = int(iteration_limit)
        options.random_seed = int(random_seed)
        options.require_sample_point_is_contained = bool(require_seed_contained)
        return options

    @classmethod
    def from_polygon_obstacles(
        cls,
        obstacle_polygons: List[np.ndarray],
        domain_lb: np.ndarray,
        domain_ub: np.ndarray,
        *,
        clearance: float = 1e-4,
        iteration_limit: int = 50,
        random_seed: int = 0,
    ) -> "Iris2DProblem":
        domain_lb = cls._as_2d_point(domain_lb, "domain_lb")
        domain_ub = cls._as_2d_point(domain_ub, "domain_ub")
        if np.any(domain_lb >= domain_ub):
            raise ValueError(f"domain_lb must be strictly smaller than domain_ub, got {domain_lb}, {domain_ub}.")

        canonical_polygons = [
            cls._canonical_obstacle_polygon(vertices)
            for vertices in obstacle_polygons
        ]
        original_obstacles = [make_hpolytope(vertices) for vertices in canonical_polygons]
        iris_obstacles = [
            cls.inflate_obstacle(obstacle, clearance)
            for obstacle in original_obstacles
        ]
        options = cls.make_iris_options(
            iteration_limit=iteration_limit,
            random_seed=random_seed,
        )
        return cls(
            domain=HPolyhedron.MakeBox(domain_lb, domain_ub),
            domain_lb=domain_lb,
            domain_ub=domain_ub,
            obstacle_polygons=canonical_polygons,
            original_obstacles=original_obstacles,
            iris_obstacles=iris_obstacles,
            clearance=float(clearance),
            options=options,
        )

    @staticmethod
    def point_is_free(
        point: np.ndarray,
        domain: HPolyhedron,
        obstacles: List[HPolyhedron],
        *,
        tol: float = 1e-8,
    ) -> bool:
        if not domain.PointInSet(point, tol):
            return False
        return not any(obstacle.PointInSet(point, tol) for obstacle in obstacles)

    def grow_region(self, seed: np.ndarray, *, tol: float = 1e-8) -> Iris2DRegion:
        seed = self._as_2d_point(seed, "seed")
        if not self.point_is_free(seed, self.domain, self.iris_obstacles, tol=tol):
            raise ValueError("IRIS seed must be inside the domain and outside every clearance-inflated obstacle.")

        hpoly = Iris(self.iris_obstacles, seed, self.domain, self.options)
        if hpoly.IsEmpty():
            raise RuntimeError("IRIS returned an empty region.")
        if not hpoly.PointInSet(seed, tol):
            raise RuntimeError("IRIS returned a region that does not contain its seed.")
        if not hpoly.ContainedIn(self.domain, tol):
            raise RuntimeError("IRIS returned a region outside the domain.")
        if self.clearance > tol and any(hpoly.IntersectsWith(obstacle) for obstacle in self.original_obstacles):
            raise RuntimeError("IRIS region intersects an original obstacle; increase clearance or check the seed.")

        vertices = hpoly_to_vrep(hpoly)
        if vertices is None:
            raise RuntimeError("Could not convert the IRIS HPolyhedron to vertices.")
        return Iris2DRegion(seed=seed, hpoly=hpoly, vertices=self._ordered_vertices(vertices))

    def sample_free_seed(
        self,
        rng: np.random.Generator,
        existing_regions: List[Iris2DRegion],
        *,
        tol: float = 1e-8,
    ) -> Optional[np.ndarray]:
        sample = rng.uniform(self.domain_lb, self.domain_ub)
        if not self.point_is_free(sample, self.domain, self.iris_obstacles, tol=tol):
            return None
        if any(region.hpoly.PointInSet(sample, tol) for region in existing_regions):
            return None
        return sample

    @staticmethod
    def connectivity_edges(regions: List[Iris2DRegion]) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if regions[i].hpoly.IntersectsWith(regions[j].hpoly):
                    edges.append((i, j))
        return edges

    @classmethod
    def _as_seed_points(cls, seeds: np.ndarray | List[np.ndarray]) -> List[np.ndarray]:
        arr = np.asarray(seeds, dtype=float)
        if arr.shape == (2,):
            return [arr]
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
            return [arr[idx].copy() for idx in range(arr.shape[0])]
        raise ValueError(f"seeds must be a 2D point or an array/list of 2D points, got shape {arr.shape}.")

    @staticmethod
    def containing_region_index(
        regions: List[Iris2DRegion],
        point: np.ndarray,
        *,
        tol: float = 1e-8,
    ) -> Optional[int]:
        for idx, region in enumerate(regions):
            if region.hpoly.PointInSet(point, tol):
                return idx
        return None

    @staticmethod
    def region_indices_are_connected(
        num_regions: int,
        region_indices: List[int],
        edges: List[Tuple[int, int]],
    ) -> bool:
        required = set(region_indices)
        if len(required) <= 1:
            return True

        parent = list(range(num_regions))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(lhs: int, rhs: int) -> None:
            lhs_root, rhs_root = find(lhs), find(rhs)
            if lhs_root != rhs_root:
                parent[rhs_root] = lhs_root

        for lhs, rhs in edges:
            union(lhs, rhs)

        roots = {find(idx) for idx in required}
        return len(roots) == 1

    def grow_connected_regions(
        self,
        seeds: np.ndarray | List[np.ndarray],
        *,
        max_attempts: int = 5000,
        threshold: float = 0.8,
        min_coverage_samples: int = 100,
        random_seed: int = 0,
    ) -> tuple[List[Iris2DRegion], List[Tuple[int, int]]]:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {max_attempts}.")
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}.")
        if min_coverage_samples < 1:
            raise ValueError(f"min_coverage_samples must be positive, got {min_coverage_samples}.")

        rng = np.random.default_rng(random_seed)
        required_seeds = self._as_seed_points(seeds)
        regions: List[Iris2DRegion] = []
        seed_region_indices: List[int] = []
        for seed in required_seeds:
            region_idx = self.containing_region_index(regions, seed)
            if region_idx is None:
                regions.append(self.grow_region(seed))
                region_idx = len(regions) - 1
            seed_region_indices.append(region_idx)

        attempts = 0
        free_samples = 0
        covered_free_samples = 0

        while attempts < max_attempts:
            attempts += 1
            sample = rng.uniform(self.domain_lb, self.domain_ub)
            if not self.point_is_free(sample, self.domain, self.iris_obstacles):
                continue

            free_samples += 1
            if any(region.hpoly.PointInSet(sample) for region in regions):
                covered_free_samples += 1
                edges = self.connectivity_edges(regions)
                if (
                    free_samples >= min_coverage_samples
                    and covered_free_samples / free_samples >= threshold
                    and self.region_indices_are_connected(len(regions), seed_region_indices, edges)
                ):
                    break
                continue

            try:
                candidate = self.grow_region(sample)
            except (RuntimeError, ValueError):
                continue
            if any(candidate.hpoly.IntersectsWith(region.hpoly) for region in regions):
                regions.append(candidate)
                covered_free_samples += 1
                edges = self.connectivity_edges(regions)
                if (
                    free_samples >= min_coverage_samples
                    and covered_free_samples / free_samples >= threshold
                    and self.region_indices_are_connected(len(regions), seed_region_indices, edges)
                ):
                    break

        edges = self.connectivity_edges(regions)
        if not self.region_indices_are_connected(len(regions), seed_region_indices, edges):
            raise RuntimeError("Could not connect all required IRIS seed regions within max_attempts.")
        return regions, edges

class Iris2DEnvBuilder:
    DEFAULT_PARAMS: Dict[str, Any] = {
        "square_size": 10.0,
        "m": 9,
        "robot_radius": 0.25,
        "coverage_threshold": 0.7,
        "max_attempts": 5000,
        "coverage_samples": 500,
        "clearance": 0.25,
        "min_obstacle_radius": 0.20,
        "max_obstacle_radius": 0.60,
        "min_obstacle_vertices": 3,
        "max_obstacle_vertices": 6,
        "iteration_limit": 50,
    }

    @classmethod
    def normalize_env_params(cls, env_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(cls.DEFAULT_PARAMS)
        if env_params is not None:
            unknown = set(env_params) - set(cls.DEFAULT_PARAMS)
            if unknown:
                raise ValueError(f"Unsupported iris-2d env params: {sorted(unknown)}.")
            params.update(env_params)

        params["square_size"] = float(params["square_size"])
        params["m"] = int(params["m"])
        params["robot_radius"] = float(params["robot_radius"])
        params["coverage_threshold"] = float(params["coverage_threshold"])
        params["max_attempts"] = int(params["max_attempts"])
        params["coverage_samples"] = int(params["coverage_samples"])
        params["clearance"] = float(params["clearance"])
        params["min_obstacle_radius"] = float(params["min_obstacle_radius"])
        params["max_obstacle_radius"] = float(params["max_obstacle_radius"])
        params["min_obstacle_vertices"] = int(params["min_obstacle_vertices"])
        params["max_obstacle_vertices"] = int(params["max_obstacle_vertices"])
        params["iteration_limit"] = int(params["iteration_limit"])

        if params["square_size"] <= 0.0:
            raise ValueError("square_size must be positive.")
        if params["m"] < 0:
            raise ValueError("m must be nonnegative.")
        if params["robot_radius"] <= 0.0:
            raise ValueError("robot_radius must be positive.")
        if params["coverage_threshold"] <= 0.0 or params["coverage_threshold"] > 1.0:
            raise ValueError("coverage_threshold must be in (0, 1].")
        if params["max_attempts"] < 1:
            raise ValueError("max_attempts must be positive.")
        if params["coverage_samples"] < 1:
            raise ValueError("coverage_samples must be positive.")
        if params["clearance"] < 0.0:
            raise ValueError("clearance must be nonnegative.")
        if params["min_obstacle_radius"] <= 0.0:
            raise ValueError("min_obstacle_radius must be positive.")
        if params["max_obstacle_radius"] < params["min_obstacle_radius"]:
            raise ValueError("max_obstacle_radius must be at least min_obstacle_radius.")
        if params["min_obstacle_vertices"] < 3:
            raise ValueError("min_obstacle_vertices must be at least 3.")
        if params["max_obstacle_vertices"] < params["min_obstacle_vertices"]:
            raise ValueError("max_obstacle_vertices must be at least min_obstacle_vertices.")
        if params["iteration_limit"] < 1:
            raise ValueError("iteration_limit must be positive.")

        return params

    @staticmethod
    def random_convex_polygon(
        rng: np.random.Generator,
        center: np.ndarray,
        radius: float,
        num_vertices: int,
    ) -> np.ndarray:
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=num_vertices))
        radii = radius * rng.uniform(0.65, 1.0, size=num_vertices)
        points = center + np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None]
        hull = ConvexHull(points)
        return points[hull.vertices]

    @classmethod
    def sample_obstacle_polygons(
        cls,
        rng: np.random.Generator,
        domain_lb: np.ndarray,
        domain_ub: np.ndarray,
        params: Dict[str, Any],
    ) -> List[np.ndarray]:
        obstacles: List[np.ndarray] = []
        attempts = 0
        max_attempts = max(1000, 200 * int(params["m"]))
        while len(obstacles) < int(params["m"]) and attempts < max_attempts:
            attempts += 1
            radius = float(rng.uniform(params["min_obstacle_radius"], params["max_obstacle_radius"]))
            center = rng.uniform(domain_lb + radius, domain_ub - radius)
            num_vertices = int(rng.integers(
                int(params["min_obstacle_vertices"]),
                int(params["max_obstacle_vertices"]) + 1,
            ))
            vertices = cls.random_convex_polygon(rng, center, radius, num_vertices)
            if np.any(vertices < domain_lb) or np.any(vertices > domain_ub):
                continue
            obstacles.append(vertices)

        if len(obstacles) != int(params["m"]):
            raise RuntimeError(f"Could only place {len(obstacles)} polygonal obstacles out of {params['m']}.")
        return obstacles

    @staticmethod
    def sample_initial_seed(
        problem: Iris2DProblem,
        rng: np.random.Generator,
        max_attempts: int,
    ) -> np.ndarray:
        for _ in range(int(max_attempts)):
            seed = problem.sample_free_seed(rng, [])
            if seed is not None:
                return seed
        raise RuntimeError(f"Could not sample a free IRIS seed within {int(max_attempts)} attempts.")

    @staticmethod
    def estimate_coverage(
        problem: Iris2DProblem,
        regions: List[Iris2DRegion],
        rng: np.random.Generator,
        sample_count: int,
    ) -> float:
        free_samples = 0
        covered_free_samples = 0
        for _ in range(sample_count):
            sample = rng.uniform(problem.domain_lb, problem.domain_ub)
            if not problem.point_is_free(sample, problem.domain, problem.original_obstacles):
                continue
            free_samples += 1
            if any(region.hpoly.PointInSet(sample) for region in regions):
                covered_free_samples += 1
        if free_samples == 0:
            return 0.0
        return float(covered_free_samples) / float(free_samples)

    @classmethod
    def build_env(cls, seed: int, env_params: Optional[Dict[str, Any]] = None) -> Env:
        params = cls.normalize_env_params(env_params)
        rng = np.random.default_rng(int(seed))
        square_size = float(params["square_size"])
        domain_lb = np.array([0.0, 0.0])
        domain_ub = np.array([square_size, square_size])

        obstacle_polygons = cls.sample_obstacle_polygons(
            rng,
            domain_lb,
            domain_ub,
            params,
        )
        problem = Iris2DProblem.from_polygon_obstacles(
            obstacle_polygons,
            domain_lb,
            domain_ub,
            clearance=float(params["clearance"]),
            iteration_limit=int(params["iteration_limit"]),
            random_seed=int(seed),
        )
        seed_points = np.array([
            cls.sample_initial_seed(problem, rng, int(params["max_attempts"]))
        ])
        regions, edges = problem.grow_connected_regions(
            seeds=seed_points,
            max_attempts=int(params["max_attempts"]),
            threshold=float(params["coverage_threshold"]),
            random_seed=int(seed),
        )

        coverage = cls.estimate_coverage(
            problem,
            regions,
            np.random.default_rng(int(seed) + 1),
            int(params["coverage_samples"]),
        )
        if coverage < float(params["coverage_threshold"]):
            raise RuntimeError(
                f"IRIS-2D coverage estimate {coverage:.3f} is below "
                f"threshold {float(params['coverage_threshold']):.3f}."
            )

        env = Env(
            name=f"iris2d-m{int(params['m'])}-seed{int(seed):05d}",
            CSpace=[region.vertices for region in regions],
            robot_radius=float(params["robot_radius"]),
            OStatic=[StaticPolygon(vertices) for vertices in obstacle_polygons],
            ODynamic=[],
            edges=edges,
            domain_lb=domain_lb,
            domain_ub=domain_ub,
        )
        env.seed_points = seed_points
        env.coverage_estimate = coverage
        return env
