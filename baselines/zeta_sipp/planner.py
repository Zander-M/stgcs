from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydrake.all import HPolyhedron

from baselines.common import ShortestPathSolution
from environment.env import Env
from environment.obstacle import StaticPolygon, StaticSphere
from stgcs.interval import Interval
from stgcs.st_planner import MPQuery


TIME_EPS = 1e-9


class _ZetaUtils:
    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def merge_intervals(intervals: Iterable[Tuple[float, float]], t_max: float) -> List[Tuple[float, float]]:
        ordered: List[Tuple[float, float]] = []
        for start, end in intervals:
            lo = max(0.0, float(start))
            hi = min(float(end), float(t_max))
            if hi <= lo + TIME_EPS:
                continue
            ordered.append((lo, hi))
        ordered.sort(key=lambda item: (item[0], item[1]))

        merged: List[Tuple[float, float]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1] + TIME_EPS:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def flatten_intervals(intervals: Iterable[Tuple[float, float]]) -> List[float]:
        flat: List[float] = []
        for start, end in intervals:
            flat.extend([float(start), float(end)])
        return flat

    @staticmethod
    def complement_intervals(blocked: Sequence[Tuple[float, float]], t_max: float) -> List[Tuple[float, float]]:
        safe: List[Tuple[float, float]] = []
        cursor = 0.0
        for start, end in blocked:
            if start > cursor + TIME_EPS:
                safe.append((cursor, start))
            cursor = max(cursor, end)
            if cursor >= t_max - TIME_EPS:
                break
        if cursor < t_max - TIME_EPS:
            safe.append((cursor, t_max))
        elif abs(cursor - t_max) <= TIME_EPS:
            safe.append((t_max, t_max))
        return safe

    @staticmethod
    def intersect_intervals(a: Sequence[float], b: Sequence[float]) -> List[float]:
        lo = max(float(a[0]), float(b[0]))
        hi = min(float(a[1]), float(b[1]))
        if hi <= lo + TIME_EPS:
            return []
        return [lo, hi]

    @staticmethod
    def point_in_cspace(env: Env, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        return any(hpoly.PointInSet(point) for hpoly in env._CSpace_hpoly)

    @staticmethod
    def point_in_domain_box(env: Env, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        return bool(np.all(point >= env.lb - TIME_EPS) and np.all(point <= env.ub + TIME_EPS))

    @classmethod
    def point_in_discretized_space(cls, env: Env, point: np.ndarray) -> bool:
        if env.O_Static:
            return cls.point_in_domain_box(env, point)
        return cls.point_in_cspace(env, point)

    @staticmethod
    def cell_bounds(center: np.ndarray, cell_size: float) -> Tuple[np.ndarray, np.ndarray]:
        center = np.asarray(center, dtype=float)
        half_size = 0.5 * float(cell_size)
        return center - half_size, center + half_size

    @classmethod
    def cell_corners(cls, center: np.ndarray, cell_size: float) -> Tuple[np.ndarray, ...]:
        lower, upper = cls.cell_bounds(center, cell_size)
        return (
            np.array([lower[0], lower[1]], dtype=float),
            np.array([upper[0], lower[1]], dtype=float),
            np.array([upper[0], upper[1]], dtype=float),
            np.array([lower[0], upper[1]], dtype=float),
        )

    @staticmethod
    def cspace_cell_containment_required(env: Env) -> bool:
        if not env.O_Static:
            return True
        return str(env.name).lower().startswith("maze")

    @staticmethod
    def native_maze_grid_required(env: Env) -> bool:
        return str(env.name).lower().startswith("maze")

    @staticmethod
    def native_maze_grid_cell_size(env: Env) -> float:
        span = np.asarray(env.ub[:2], dtype=float) - np.asarray(env.lb[:2], dtype=float)
        if not np.allclose(span, np.round(span), atol=TIME_EPS):
            raise ValueError(f"Maze grid discretization requires integer domain span, got {span}.")
        return 1.0

    @classmethod
    def cell_in_domain_box(cls, env: Env, center: np.ndarray, cell_size: float) -> bool:
        lower, upper = cls.cell_bounds(center, cell_size)
        return bool(np.all(lower >= env.lb - TIME_EPS) and np.all(upper <= env.ub + TIME_EPS))

    @classmethod
    def cell_in_cspace(cls, env: Env, center: np.ndarray, cell_size: float) -> bool:
        corners = cls.cell_corners(center, cell_size)
        return any(
            all(hpoly.PointInSet(corner, TIME_EPS) for corner in corners)
            for hpoly in env._CSpace_hpoly
        )

    @classmethod
    def cell_in_discretized_space(cls, env: Env, center: np.ndarray, cell_size: float) -> bool:
        if cls.cspace_cell_containment_required(env):
            return cls.cell_in_cspace(env, center, cell_size)
        return cls.cell_in_domain_box(env, center, cell_size)

    @classmethod
    def native_maze_cell_is_free(cls, env: Env, center: np.ndarray) -> bool:
        return cls.point_in_cspace(env, center) and not cls.blocked_by_static(
            env,
            center,
            env.robot_radius,
        )

    @staticmethod
    def blocked_by_static(env: Env, point: np.ndarray, inflation: float) -> bool:
        return any(obstacle.is_colliding(point, inflation) for obstacle in env.O_Static)

    @staticmethod
    def segment_blocked_by_static(env: Env, start: np.ndarray, goal: np.ndarray, inflation: float) -> bool:
        return any(obstacle.is_colliding_lineseg(start, goal, inflation) for obstacle in env.O_Static)

    @classmethod
    def cell_blocked_by_static(
        cls,
        env: Env,
        center: np.ndarray,
        cell_size: float,
        inflation: float,
    ) -> bool:
        lower, upper = cls.cell_bounds(center, cell_size)
        inflated_box = HPolyhedron.MakeBox(lower - float(inflation), upper + float(inflation))
        for obstacle in env.O_Static:
            if isinstance(obstacle, StaticPolygon):
                if obstacle.hpoly.IntersectsWith(inflated_box):
                    return True
                continue
            if isinstance(obstacle, StaticSphere):
                closest = np.minimum(np.maximum(obstacle.pos, lower), upper)
                if np.linalg.norm(obstacle.pos - closest) <= obstacle.radius + float(inflation):
                    return True
                continue
            raise TypeError(f"Unsupported static obstacle type {type(obstacle)!r} for Zeta*-SIPP projection.")
        return False

    @classmethod
    def point_blocked_intervals(
        cls, env: Env, point: np.ndarray, inflation: float, t_max: float
    ) -> List[Tuple[float, float]]:
        blocked: List[Tuple[float, float]] = []
        for obstacle in env.O_Dynamic:
            for interval in obstacle.collision_intervals(point, inflation):
                blocked.append((interval.start, interval.end))
        return cls.merge_intervals(blocked, t_max)

    @staticmethod
    def line_seg_on_grid(x0: float, y0: float, x1: float, y1: float, distance: float) -> List[dict]:
        grids: List[dict] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x = math.floor(x0)
        y = math.floor(y0)

        dtdx = math.inf if dx <= TIME_EPS else 1.0 / dx
        dtdy = math.inf if dy <= TIME_EPS else 1.0 / dy

        d0 = 0.0
        n = 1

        if dx <= TIME_EPS:
            x_inc = 0
            t_next_horizontal = math.inf
        elif x1 > x0:
            x_inc = 1
            n += math.floor(x1) - x
            t_next_horizontal = (x + 1 - x0) * dtdx
        else:
            x_inc = -1
            n += x - math.floor(x1)
            t_next_horizontal = (x0 - x) * dtdx

        if dy <= TIME_EPS:
            y_inc = 0
            t_next_vertical = math.inf
        elif y1 > y0:
            y_inc = 1
            n += math.floor(y1) - y
            t_next_vertical = (y + 1 - y0) * dtdy
        else:
            y_inc = -1
            n += y - math.floor(y1)
            t_next_vertical = (y0 - y) * dtdy

        for _ in range(n):
            d1 = distance * min(t_next_horizontal, t_next_vertical, 1.0)
            grids.append({"x": x, "y": y, "dist": [d0, d1]})
            d0 = d1
            if t_next_horizontal > t_next_vertical:
                y += y_inc
                t_next_vertical += dtdy
            else:
                x += x_inc
                t_next_horizontal += dtdx
        return grids


class _KeyedHeap:
    def __init__(self, key_fn: Callable[[object], float]) -> None:
        self._key_fn = key_fn
        self._heap: List[Tuple[float, int, object]] = []
        self._counter = 0

    def push(self, item: object) -> None:
        heapq.heappush(self._heap, (float(self._key_fn(item)), self._counter, item))
        self._counter += 1

    def sort_element(self, item: object) -> None:
        self.push(item)

    def _discard_stale(self) -> None:
        while self._heap:
            key, _, item = self._heap[0]
            if abs(key - float(self._key_fn(item))) <= 1e-12:
                return
            heapq.heappop(self._heap)

    def peek_key(self) -> float:
        self._discard_stale()
        if not self._heap:
            return math.inf
        return float(self._heap[0][0])

    def shift(self) -> object:
        self._discard_stale()
        _, _, item = heapq.heappop(self._heap)
        return item

    def size(self) -> int:
        self._discard_stale()
        return len(self._heap)


@dataclass(eq=False)
class _Node:
    grid_point: np.ndarray
    world_point: np.ndarray
    safe_interval: Tuple[float, float]
    cell: "_Cell"
    f: float = math.inf
    g: float = math.inf
    g_low: float = math.inf
    h: float = 0.0
    parent: Optional["_Node"] = None
    closed: bool = False
    best_potential_parent: Optional["_Node"] = None
    potential_parents: List["_Node"] = field(default_factory=list)
    g_low_array: List[float] = field(default_factory=list)
    wait_time: float = 0.0


@dataclass(eq=False)
class _Cell:
    grid_point: np.ndarray
    world_point: np.ndarray
    weight: bool
    risk_interval: List[float]
    base_index: Optional[Tuple[int, int]]
    is_query_cell: bool = False
    h: float = 0.0
    fh: float = math.inf
    visited: bool = False
    closed: bool = False
    nodes: List[_Node] = field(default_factory=list)
    visible_cells: List["_Cell"] = field(default_factory=list)


class _ProjectedGrid:
    def __init__(self, env: Env, t_max: float, cell_size: float) -> None:
        if env.dim != 2:
            raise ValueError(f"Zeta*-SIPP projection currently supports only 2D environments, got dim={env.dim}")
        if cell_size <= 0.0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")

        self.env = env
        self.t_max = float(t_max)
        self.use_native_maze_grid = _ZetaUtils.native_maze_grid_required(env)
        self.cell_size = (
            _ZetaUtils.native_maze_grid_cell_size(env)
            if self.use_native_maze_grid
            else float(cell_size)
        )
        self.origin = np.asarray(env.lb[:2], dtype=float)
        span = np.asarray(env.ub[:2], dtype=float) - self.origin
        self.shape = tuple(int(np.ceil(float(axis) / self.cell_size)) for axis in span)
        self.half_diag = 0.5 * self.cell_size * math.sqrt(2.0)
        self.grids: List[List[_Cell]] = []
        self.cells: List[_Cell] = []
        self.checks = 0
        self.scanned_grids = 0
        self._build_cells()

    def _build_cells(self) -> None:
        for ix in range(self.shape[0]):
            row: List[_Cell] = []
            for iy in range(self.shape[1]):
                center_grid = np.array([ix + 0.5, iy + 0.5], dtype=float)
                center_world = self.grid_to_world(center_grid)
                if self.use_native_maze_grid:
                    weight = _ZetaUtils.native_maze_cell_is_free(self.env, center_world)
                    risk_inflation = self.env.robot_radius
                else:
                    weight = (
                        _ZetaUtils.cell_in_discretized_space(self.env, center_world, self.cell_size)
                        and not _ZetaUtils.cell_blocked_by_static(
                            self.env,
                            center_world,
                            self.cell_size,
                            self.env.robot_radius,
                        )
                    )
                    risk_inflation = self.env.robot_radius + self.half_diag
                risk_intervals = _ZetaUtils.flatten_intervals(
                    _ZetaUtils.point_blocked_intervals(
                        self.env,
                        center_world,
                        risk_inflation,
                        self.t_max,
                    )
                )
                cell = _Cell(
                    grid_point=center_grid,
                    world_point=center_world,
                    weight=weight,
                    risk_interval=risk_intervals,
                    base_index=(ix, iy),
                )
                if cell.weight:
                    for safe_interval in self.safe_intervals_for_point(center_world):
                        cell.nodes.append(_Node(center_grid.copy(), center_world.copy(), safe_interval, cell))
                row.append(cell)
                self.cells.append(cell)
            self.grids.append(row)

    def world_to_grid(self, point: np.ndarray) -> np.ndarray:
        grid = (np.asarray(point, dtype=float) - self.origin) / self.cell_size
        upper = np.asarray(self.shape, dtype=float) - 1e-9
        return np.clip(grid, 0.0, upper)

    def grid_to_world(self, point: np.ndarray) -> np.ndarray:
        return self.origin + self.cell_size * np.asarray(point, dtype=float)

    def safe_intervals_for_point(self, world_point: np.ndarray) -> List[Tuple[float, float]]:
        blocked = _ZetaUtils.point_blocked_intervals(self.env, world_point, self.env.robot_radius, self.t_max)
        safe = _ZetaUtils.complement_intervals(blocked, self.t_max)
        return [interval for interval in safe if interval[1] >= interval[0] - TIME_EPS]

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.shape[0] and 0 <= iy < self.shape[1]

    def get_cell(self, ix: int, iy: int) -> Optional[_Cell]:
        if not self.in_bounds(ix, iy):
            return None
        return self.grids[ix][iy]

    @staticmethod
    def _is_query_endpoint(endpoint: _Cell | _Node) -> bool:
        if isinstance(endpoint, _Node):
            return endpoint.cell.is_query_cell
        return endpoint.is_query_cell

    @staticmethod
    def _world_point(endpoint: _Cell | _Node) -> np.ndarray:
        if isinstance(endpoint, _Node):
            return endpoint.world_point
        return endpoint.world_point

    def _query_line_of_sight(self, a: _Cell | _Node, b: _Cell | _Node) -> bool:
        start = self._world_point(a)
        goal = self._world_point(b)
        if _ZetaUtils.segment_blocked_by_static(self.env, start, goal, self.env.robot_radius):
            return False

        if _ZetaUtils.cspace_cell_containment_required(self.env):
            return bool(self.env.is_segment_in_CSpace(start, goal))

        distance = float(np.linalg.norm(goal - start))
        sample_step = max(self.cell_size * 0.25, 1e-6)
        num_samples = max(2, int(math.ceil(distance / sample_step)) + 1)
        for alpha in np.linspace(0.0, 1.0, num_samples):
            point = start + alpha * (goal - start)
            if not _ZetaUtils.point_in_discretized_space(self.env, point):
                return False
            if _ZetaUtils.blocked_by_static(self.env, point, self.env.robot_radius):
                return False
        return True

    def line_of_sight_grid(self, a: _Cell | _Node, b: _Cell | _Node) -> bool:
        self.checks += 1
        if self._is_query_endpoint(a) or self._is_query_endpoint(b):
            return self._query_line_of_sight(a, b)

        if _ZetaUtils.segment_blocked_by_static(
            self.env,
            self._world_point(a),
            self._world_point(b),
            self.env.robot_radius,
        ):
            return False

        x0, y0 = float(a.grid_point[0]), float(a.grid_point[1])
        x1, y1 = float(b.grid_point[0]), float(b.grid_point[1])

        grid_count = 0

        def is_blocked(ix: int, iy: int) -> bool:
            nonlocal grid_count
            grid_count += 1
            cell = self.get_cell(ix, iy)
            return cell is None or not cell.weight

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x = math.floor(x0)
        y = math.floor(y0)

        if not float(x0).is_integer() and not float(y0).is_integer() and is_blocked(x, y):
            self.scanned_grids += grid_count
            return False
        if float(x0).is_integer() and float(y0).is_integer() and x1 > x0 and y1 > y0 and is_blocked(x, y):
            self.scanned_grids += grid_count
            return False

        n = 1
        if dx <= TIME_EPS:
            x_inc = 0
            error = math.inf
        elif x1 > x0:
            x_inc = 1
            n += math.floor(x1) - x
            error = (x + 1 - x0) * dy
        else:
            x_inc = -1
            n += x - math.floor(x1)
            error = (x0 - x) * dy

        if dy <= TIME_EPS:
            y_inc = 0
            error = -math.inf
        elif y1 > y0:
            y_inc = 1
            n += math.floor(y1) - y
            error -= (y + 1 - y0) * dx
        else:
            y_inc = -1
            n += y - math.floor(y1)
            error -= (y0 - y) * dx

        while n > 1:
            n -= 1
            if error > 0:
                y += y_inc
                error -= dx
            elif error < 0:
                x += x_inc
                error += dy
            else:
                if is_blocked(x + x_inc, y) and is_blocked(x, y + y_inc):
                    self.scanned_grids += grid_count
                    return False
                x += x_inc
                error += dy
                y += y_inc
                error -= dx
                n -= 1
            if is_blocked(x, y):
                self.scanned_grids += grid_count
                return False

        self.scanned_grids += grid_count
        return True


class _Shadowcast:
    def __init__(self, projection: _ProjectedGrid) -> None:
        self.projection = projection
        self.inverted = False
        self.max_cost = math.inf
        self.visible: List[_Cell] = []
        self.transforms = [
            {"xx": 1, "xy": 0, "yx": 0, "yy": 1},
            {"xx": 1, "xy": 0, "yx": 0, "yy": -1},
            {"xx": -1, "xy": 0, "yx": 0, "yy": 1},
            {"xx": -1, "xy": 0, "yx": 0, "yy": -1},
            {"xx": 0, "xy": 1, "yx": 1, "yy": 0},
            {"xx": 0, "xy": -1, "yx": 1, "yy": 0},
            {"xx": 0, "xy": 1, "yx": -1, "yy": 0},
            {"xx": 0, "xy": -1, "yx": -1, "yy": 0},
        ]

    def find(self, x: float, y: float) -> Optional[_Cell]:
        return self.projection.get_cell(math.floor(x), math.floor(y))

    def record(self, node: _Cell) -> None:
        self.visible.append(node)

    def in_range(self, node: _Cell) -> bool:
        if not self.inverted:
            return True
        return node.fh <= self.max_cost + TIME_EPS

    def blocked(self, node: Optional[_Cell]) -> bool:
        return node is None or (not node.weight) or (not self.in_range(node))

    def scan(self, origin: _Cell) -> List[_Cell]:
        self.visible = []
        if origin.is_query_cell:
            return []
        for transform in self.transforms:
            self.compute(origin, 1, 1.0, 0.0, transform)
        dedup: List[_Cell] = []
        seen: set[int] = set()
        for cell in self.visible:
            key = id(cell)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(cell)
        return dedup

    def compute(self, origin: _Cell, delta_x: int, top_slope: float, bottom_slope: float, transform: dict) -> None:
        if bottom_slope >= top_slope:
            return

        ymin = delta_x * bottom_slope
        ymax = delta_x * top_slope
        was_blocked = False
        y = math.floor(ymin)
        while y <= math.ceil(ymax):
            real_x = origin.grid_point[0] + transform["xx"] * delta_x + transform["xy"] * y
            real_y = origin.grid_point[1] + transform["yx"] * delta_x + transform["yy"] * y
            node = self.find(real_x, real_y)
            if not self.blocked(node):
                if ymin - TIME_EPS <= y <= ymax + TIME_EPS:
                    self.record(node)
                was_blocked = False
            else:
                if not was_blocked:
                    new_top_slope = (y - 0.5) / (delta_x + 0.5)
                    self.compute(origin, delta_x + 1, min(new_top_slope, top_slope), bottom_slope, transform)
                    was_blocked = True
                new_bottom_slope = (y + 0.5) / max(delta_x - 0.5, TIME_EPS)
                bottom_slope = max(bottom_slope, new_bottom_slope)
                if bottom_slope >= top_slope:
                    return
                ymin = delta_x * bottom_slope
            y += 1
        self.compute(origin, delta_x + 1, top_slope, bottom_slope, transform)


class _TOAASIPP:
    def __init__(
        self,
        projection: _ProjectedGrid,
        query: MPQuery,
        t_max: float,
        runtime_limit_secs: float,
        time_buffer: float,
    ) -> None:
        self.projection = projection
        self.query = query
        self.t_max = float(t_max)
        self.runtime_limit_secs = float(runtime_limit_secs)
        self.time_buffer = float(time_buffer)
        self.open_heap = _KeyedHeap(lambda node: node.f)
        self.bound_heap = _KeyedHeap(lambda cell: cell.fh)
        self.open_cells: List[_Cell] = []
        self.tree_cells: List[_Cell] = []
        self.steps = 0

    @staticmethod
    def _travel_time_world(a: np.ndarray, b: np.ndarray, vlimit: float) -> float:
        return float(np.max(np.abs(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))) / vlimit)

    def heuristic(self, a: _Cell | _Node, b: _Cell | _Node) -> float:
        return self._travel_time_world(a.world_point, b.world_point, self.query.vlimit)

    def visit_cell(self, cell: _Cell, end_cell: _Cell) -> None:
        if cell.visited:
            return
        if not cell.nodes:
            return
        if cell.h == 0.0 and cell is not end_cell:
            cell.h = self.heuristic(cell, end_cell)
        else:
            cell.h = self.heuristic(cell, end_cell)
        cell.visited = True
        self.tree_cells.append(cell)
        self.open_cells.append(cell)
        for node in cell.nodes:
            node.h = cell.h
            self.open_heap.push(node)

    def init_start(self, start_cell: _Cell, end_cell: _Cell) -> None:
        self.visit_cell(start_cell, end_cell)
        start_node = start_cell.nodes[0]
        start_node.g = float(self.query.t_start)
        start_node.g_low = float(self.query.t_start)
        start_node.f = start_node.g + start_node.h
        self.open_heap.sort_element(start_node)

    def init(self) -> Tuple[_Cell, _Cell]:
        start_grid = self.projection.world_to_grid(self.query.start)
        goal_grid = self.projection.world_to_grid(self.query.goal)

        start_intervals = [
            interval
            for interval in self.projection.safe_intervals_for_point(self.query.start)
            if interval[0] - TIME_EPS <= self.query.t_start <= interval[1] + TIME_EPS
        ]
        if not start_intervals:
            raise ValueError("Start query is not safe at t_start in projected Zeta*-SIPP.")

        goal_intervals = self.projection.safe_intervals_for_point(self.query.goal)
        if self.query.is_stay:
            goal_intervals = [interval for interval in goal_intervals if interval[1] >= self.t_max - TIME_EPS]
        if not goal_intervals:
            raise ValueError("Goal query is not feasible in projected Zeta*-SIPP under is_stay/t_max constraints.")

        start_cell = _Cell(
            grid_point=start_grid,
            world_point=np.asarray(self.query.start, dtype=float),
            weight=True,
            risk_interval=[],
            base_index=None,
            is_query_cell=True,
        )
        start_cell.nodes = [_Node(start_grid.copy(), np.asarray(self.query.start, dtype=float), start_intervals[0], start_cell)]

        goal_cell = _Cell(
            grid_point=goal_grid,
            world_point=np.asarray(self.query.goal, dtype=float),
            weight=True,
            risk_interval=[],
            base_index=None,
            is_query_cell=True,
        )
        goal_cell.nodes = [
            _Node(goal_grid.copy(), np.asarray(self.query.goal, dtype=float), interval, goal_cell)
            for interval in goal_intervals
        ]

        self.init_start(start_cell, goal_cell)
        if not np.allclose(self.query.start, self.query.goal):
            self.visit_cell(goal_cell, goal_cell)
        return start_cell, goal_cell

    def search(self) -> List[_Node]:
        start_ts = time.perf_counter()
        start_cell, end_cell = self.init()
        while self.open_heap.size() > 0 and self.open_heap.peek_key() < math.inf:
            if time.perf_counter() - start_ts > self.runtime_limit_secs:
                return []
            self.steps += 1
            current_node = self.find_next_closed_node()
            if current_node is not None:
                current_cell = current_node.cell
                if current_cell is end_cell:
                    return self.path_to(current_node)
                self.inverted_expansion(current_node, current_cell, end_cell)
            self.forward_expansion(start_cell, end_cell)
        return []

    def find_next_closed_node(self) -> Optional[_Node]:
        current_node = self.open_heap.shift()

        if current_node.best_potential_parent is not None:
            for idx, parent in enumerate(list(current_node.potential_parents)):
                if parent is current_node.best_potential_parent:
                    current_node.potential_parents.pop(idx)
                    current_node.g_low_array.pop(idx)
                    break

        if current_node.best_potential_parent is None:
            g_new = current_node.g
        else:
            g_new = self.transition(current_node.best_potential_parent, current_node) + current_node.best_potential_parent.g
        if g_new < current_node.g - TIME_EPS:
            current_node.g = g_new
            current_node.parent = current_node.best_potential_parent
            current_node.wait_time = max(0.0, g_new - current_node.g_low)

        current_node.g_low = current_node.g
        current_node.best_potential_parent = current_node.parent
        current_node.f = current_node.g_low + current_node.h

        if self.new_best_potential_parent_exists(current_node) or current_node.g >= math.inf:
            self.open_heap.push(current_node)
            return None

        best_open = self.open_heap.peek_key()
        best_bound = self.bound_heap.peek_key()
        if current_node.g + current_node.h <= best_open + TIME_EPS and current_node.g + current_node.h <= best_bound + TIME_EPS:
            current_node.closed = True
            self.update_close_cell(current_node.cell)
            return current_node

        self.open_heap.push(current_node)
        return None

    def update_close_cell(self, current_cell: _Cell) -> None:
        return

    def update_open_cell(self, current_cell: _Cell) -> None:
        if current_cell.nodes and all(node.closed for node in current_cell.nodes):
            if current_cell in self.open_cells:
                self.open_cells.remove(current_cell)

    def _visible_open_cells(self, current_cell: _Cell) -> List[_Cell]:
        visible: List[_Cell] = []
        for cell in self.open_cells:
            if cell is current_cell:
                continue
            if self.projection.line_of_sight_grid(current_cell, cell):
                visible.append(cell)
        return visible

    def inverted_expansion(self, current_node: _Node, current_cell: _Cell, end_cell: _Cell) -> None:
        self.update_open_cell(current_cell)
        if not current_cell.visible_cells:
            current_cell.visible_cells = self._visible_open_cells(current_cell)
        for cell in current_cell.visible_cells:
            if not cell.visited:
                self.visit_cell(cell, end_cell)
            for node in cell.nodes:
                if node.closed:
                    continue
                self.add_potential_parent(current_node, node)

    def forward_expansion(self, start_cell: _Cell, end_cell: _Cell) -> None:
        return

    def add_potential_parent(self, parent: _Node, node: _Node) -> None:
        g_low_new = parent.g + self.heuristic(parent, node)
        if g_low_new < node.g_low - TIME_EPS:
            node.g_low = g_low_new
            node.best_potential_parent = parent
            node.f = node.g_low + node.h
            self.open_heap.sort_element(node)
        node.potential_parents.append(parent)
        node.g_low_array.append(g_low_new)

    def new_best_potential_parent_exists(self, node: _Node) -> bool:
        node.g_low = node.g
        node.best_potential_parent = node.parent
        node.f = node.g_low + node.h
        exists = False
        for parent, g_low_new in zip(node.potential_parents, node.g_low_array):
            if g_low_new < node.g_low - TIME_EPS:
                node.g_low = g_low_new
                node.best_potential_parent = parent
                node.f = node.g_low + node.h
                exists = True
        return exists

    @staticmethod
    def path_to(node: _Node) -> List[_Node]:
        current = node
        path: List[_Node] = []
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def transition(self, parent: _Node, node: _Node) -> float:
        if parent is node:
            return 0.0
        return self.get_cost_sipp(parent, node)

    def get_cost_sipp(self, parent: _Node, node: _Node) -> float:
        travel_time = self._travel_time_world(parent.world_point, node.world_point, self.query.vlimit)
        if travel_time <= TIME_EPS:
            if parent.g <= node.safe_interval[1] + TIME_EPS:
                return max(0.0, node.safe_interval[0] - parent.g)
            return math.inf

        depart_time = max(parent.g, node.safe_interval[0] - travel_time)
        if depart_time > parent.safe_interval[1] + TIME_EPS:
            return math.inf

        grid_distance = float(np.linalg.norm(node.grid_point - parent.grid_point))
        trace_grids = _ZetaUtils.line_seg_on_grid(
            float(parent.grid_point[0]),
            float(parent.grid_point[1]),
            float(node.grid_point[0]),
            float(node.grid_point[1]),
            grid_distance,
        )

        intervals: List[List[float]] = []
        for grid in trace_grids:
            if grid_distance <= TIME_EPS:
                lam0 = lam1 = 0.0
            else:
                lam0 = _ZetaUtils.clamp(grid["dist"][0] / grid_distance, 0.0, 1.0)
                lam1 = _ZetaUtils.clamp(grid["dist"][1] / grid_distance, 0.0, 1.0)
            intervals.append([depart_time + lam0 * travel_time, depart_time + lam1 * travel_time])

        wait_time = max(0.0, depart_time - parent.g)
        for index in range(len(trace_grids)):
            wait_time = self.update_wait_time(trace_grids, intervals, index, wait_time, parent)
            if parent.g + wait_time > parent.safe_interval[1] + TIME_EPS:
                return math.inf

        arrival = parent.g + wait_time + travel_time
        if arrival < node.safe_interval[0] - TIME_EPS or arrival > node.safe_interval[1] + TIME_EPS:
            return math.inf
        if arrival > self.t_max + TIME_EPS:
            return math.inf
        return wait_time + travel_time

    def update_wait_time(
        self,
        trace_grids: Sequence[dict],
        intervals: List[List[float]],
        index: int,
        wait_time: float,
        parent: _Node,
    ) -> float:
        wait_time_old = wait_time
        grid = trace_grids[index]
        cell = self.projection.get_cell(int(grid["x"]), int(grid["y"]))
        if cell is None:
            return math.inf

        for offset in range(0, len(cell.risk_interval), 2):
            risk_interval = [cell.risk_interval[offset], cell.risk_interval[offset + 1]]
            conflict = _ZetaUtils.intersect_intervals(risk_interval, intervals[index])
            if not conflict:
                continue
            delta = risk_interval[1] - intervals[index][0] + self.time_buffer
            wait_time += delta
            for interval in intervals:
                interval[0] += delta
                interval[1] += delta

        if wait_time > wait_time_old + TIME_EPS:
            for prior in range(index):
                wait_time = self.update_wait_time(trace_grids, intervals, prior, wait_time, parent)
                if wait_time >= math.inf:
                    return math.inf

        if parent.g + wait_time > parent.safe_interval[1] + TIME_EPS:
            return math.inf
        return wait_time


class _ZetaSIPP(_TOAASIPP):
    def __init__(
        self,
        projection: _ProjectedGrid,
        query: MPQuery,
        t_max: float,
        runtime_limit_secs: float,
        time_buffer: float,
    ) -> None:
        super().__init__(projection, query, t_max, runtime_limit_secs, time_buffer)
        self.close_cells: List[_Cell] = []

    def init(self) -> Tuple[_Cell, _Cell]:
        start_cell, end_cell = super().init()
        self.init_bound(start_cell, end_cell)
        return start_cell, end_cell

    def init_bound(self, start_cell: _Cell, end_cell: _Cell) -> None:
        start_cell.fh = float(self.query.t_start) + start_cell.h
        for cell in self.projection.cells:
            if not cell.weight or cell is start_cell or not cell.nodes:
                continue
            self.bound_cell(cell, start_cell, end_cell)

    def bound_cell(self, cell: _Cell, start_cell: _Cell, end_cell: _Cell) -> None:
        cell.h = self.heuristic(cell, end_cell)
        cell.fh = float(self.query.t_start) + self.heuristic(start_cell, cell) + cell.h
        self.bound_heap.push(cell)

    def update_close_cell(self, current_cell: _Cell) -> None:
        if not current_cell.closed:
            current_cell.closed = True
            self.close_cells.append(current_cell)

    def forward_expansion(self, start_cell: _Cell, end_cell: _Cell) -> None:
        while self.bound_heap.size() > 0 and (
            self.open_heap.size() == 0 or self.bound_heap.peek_key() <= self.open_heap.peek_key() + TIME_EPS
        ):
            new_cell = self.bound_heap.shift()
            self.visit_cell(new_cell, end_cell)
            self.inverted_scan(new_cell)

    def inverted_scan(self, new_cell: _Cell) -> None:
        for cell in self.close_cells:
            if self.projection.line_of_sight_grid(cell, new_cell):
                if new_cell not in cell.visible_cells:
                    cell.visible_cells.append(new_cell)
                for parent in cell.nodes:
                    if not parent.closed:
                        continue
                    for node in new_cell.nodes:
                        self.add_potential_parent(parent, node)


class _ZetaStarSIPP(_ZetaSIPP):
    def __init__(
        self,
        projection: _ProjectedGrid,
        query: MPQuery,
        t_max: float,
        runtime_limit_secs: float,
        time_buffer: float,
    ) -> None:
        super().__init__(projection, query, t_max, runtime_limit_secs, time_buffer)
        self.shadowcast = _Shadowcast(projection)
        self.shadowcast.inverted = True
        self.buffer = math.sqrt(2.0) * projection.cell_size / query.vlimit

    def inverted_scan(self, scan_cell: _Cell) -> None:
        self.shadowcast.max_cost = scan_cell.fh + self.buffer
        visible_cells = self.shadowcast.scan(scan_cell)
        self.projection.scanned_grids += len(visible_cells)
        scan_cell.visible_cells = [cell for cell in visible_cells if cell.visited]
        for query_cells in (self.close_cells, self.open_cells):
            for cell in query_cells:
                if not cell.is_query_cell:
                    continue
                if cell is scan_cell or not self.projection.line_of_sight_grid(cell, scan_cell):
                    continue
                if cell not in scan_cell.visible_cells:
                    scan_cell.visible_cells.append(cell)
        for cell in scan_cell.visible_cells:
            if scan_cell not in cell.visible_cells:
                cell.visible_cells.append(scan_cell)
            if not cell.closed:
                continue
            for parent in cell.nodes:
                if not parent.closed:
                    continue
                for node in scan_cell.nodes:
                    self.add_potential_parent(parent, node)

    def inverted_expansion(self, current_node: _Node, current_cell: _Cell, end_cell: _Cell) -> None:
        self.update_open_cell(current_cell)
        if not current_cell.visible_cells:
            if current_cell.is_query_cell:
                current_cell.visible_cells = self._visible_open_cells(current_cell)
            else:
                current_cell.visible_cells = self.shadowcast.scan(current_cell)
                self.projection.scanned_grids += len(current_cell.visible_cells)
                current_cell.visible_cells = [cell for cell in current_cell.visible_cells if cell.visited]
                for cell in self.open_cells:
                    if not cell.is_query_cell or cell is current_cell:
                        continue
                    if self.projection.line_of_sight_grid(current_cell, cell) and cell not in current_cell.visible_cells:
                        current_cell.visible_cells.append(cell)

        for cell in current_cell.visible_cells:
            for node in cell.nodes:
                if node.closed:
                    continue
                self.add_potential_parent(current_node, node)


class ZetaStarSIPPPlanner:
    def __init__(
        self,
        env: Env,
        seed: int,
        vlimit: float,
        cell_size: Optional[float] = None,
        runtime_limit_secs: float = math.inf,
        time_buffer: float = 1e-3,
        use_fov: bool = True,
    ) -> None:
        self.env = env
        self.seed = seed
        self.vlimit = float(vlimit)
        self.cell_size = float(cell_size) if cell_size is not None else max(env.robot_radius * 2.0, 0.1)
        self.runtime_limit_secs = float(runtime_limit_secs)
        self.time_buffer = float(time_buffer)
        self.use_fov = bool(use_fov)
        self.last_projection: Optional[_ProjectedGrid] = None
        self.last_path: List[_Node] = []

    @staticmethod
    def componentwise_travel_time(start: np.ndarray, goal: np.ndarray, vlimit: float) -> float:
        return _TOAASIPP._travel_time_world(np.asarray(start, dtype=float), np.asarray(goal, dtype=float), vlimit)

    def solve(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        t_start: float,
        t_max: float,
        is_stay: bool = True,
    ) -> ShortestPathSolution:
        query = MPQuery(
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
            t_start=float(t_start),
            is_stay=bool(is_stay),
            vlimit=float(self.vlimit),
        )
        return self.plan(query, t_max)

    def plan(self, query: MPQuery, t_max: float) -> ShortestPathSolution:
        ts = time.perf_counter()
        projection = _ProjectedGrid(self.env.copy(), t_max=t_max, cell_size=self.cell_size)
        self.last_projection = projection
        if np.allclose(query.start, query.goal):
            feasible_intervals = [
                interval
                for interval in projection.safe_intervals_for_point(query.start)
                if interval[0] - TIME_EPS <= query.t_start <= interval[1] + TIME_EPS
                and (not query.is_stay or interval[1] >= t_max - TIME_EPS)
            ]
            if not feasible_intervals:
                return ShortestPathSolution(False, -1.0, time.perf_counter() - ts, [], [])
            zero_segment = np.hstack([query.start, query.t_start, query.goal, query.t_start])
            return ShortestPathSolution(
                is_success=True,
                cost=0.0,
                time=time.perf_counter() - ts,
                vertex_path=[],
                trajectory=[zero_segment],
                itvl=Interval(float(query.t_start), float(query.t_start)),
                dim=self.env.dim + 1,
            )
        planner_cls = _ZetaStarSIPP if self.use_fov else _ZetaSIPP
        planner = planner_cls(
            projection=projection,
            query=query,
            t_max=t_max,
            runtime_limit_secs=self.runtime_limit_secs,
            time_buffer=self.time_buffer,
        )

        try:
            path = planner.search()
        except ValueError:
            return ShortestPathSolution(False, -1.0, time.perf_counter() - ts, [], [])

        self.last_path = path
        if not path:
            return ShortestPathSolution(False, -1.0, time.perf_counter() - ts, [], [])
        return self._path_to_solution(path, query, time.perf_counter() - ts)

    def _path_to_solution(self, path: Sequence[_Node], query: MPQuery, runtime: float) -> ShortestPathSolution:
        if not path:
            return ShortestPathSolution(False, -1.0, runtime, [], [])

        trajectory: List[np.ndarray] = []
        current_time = float(query.t_start)
        for prev, curr in zip(path[:-1], path[1:]):
            travel_time = self.componentwise_travel_time(prev.world_point, curr.world_point, query.vlimit)
            depart_time = curr.g - travel_time
            if depart_time > current_time + TIME_EPS:
                trajectory.append(
                    np.hstack([prev.world_point, current_time, prev.world_point, depart_time])
                )
            if travel_time > TIME_EPS or not np.allclose(prev.world_point, curr.world_point):
                trajectory.append(
                    np.hstack([prev.world_point, depart_time, curr.world_point, curr.g])
                )
            current_time = curr.g

        if not trajectory:
            final_time = float(path[-1].g)
            trajectory.append(
                np.hstack([query.start, query.t_start, query.goal, final_time])
            )

        return ShortestPathSolution(
            is_success=True,
            cost=float(path[-1].g - query.t_start),
            time=runtime,
            vertex_path=[],
            trajectory=trajectory,
            itvl=Interval(float(query.t_start), float(path[-1].g)),
            dim=self.env.dim + 1,
        )
