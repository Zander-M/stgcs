from __future__ import annotations
from typing import Any, Dict, List, Optional, Deque, Tuple
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from itertools import product
from enum import IntEnum
import time

import numpy as np

from pydrake.all import (
    RandomGenerator, HPolyhedron, GraphOfConvexSets as GCS,
    ConvexSet, Point as DrakePoint, Binding, Constraint, VPolytope
)

from stgcs.graph import STGCS
from stgcs.gcs_solver import solve_convex_restriction
from stgcs.gcs_solver import EDGE_KEY, MPGCSInstance, make_Cartesian_power_hpoly
from stgcs.trajectory import STTrajectory
from stgcs.bfs.best_first_search import DominanceCheck, SearchNode
from stgcs.geometry_utils import HPolyhedronSampler, hpoly_to_vrep

from scipy.spatial import ConvexHull, QhullError


import logging
logger = logging.getLogger(__name__)



class AStarDominanceCheck(DominanceCheck):

    def __init__(self) -> None:
        self._g = defaultdict(lambda: np.inf)

    def reset(self) -> None:
        self._g = defaultdict(lambda: np.inf)

    @staticmethod
    def _arrival_cost(n: SearchNode) -> float:
        if n.g is not None:
            return n.g
        if n.sol is None:
            return np.inf
        return n.sol.duration

    def check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:
        if n.sol is None:
            return True

        g = self._arrival_cost(n)
        if g >= self._g[n.vertex_name]:
            return True

        self._g[n.vertex_name] = g
        return False

    def is_stale(self, n: SearchNode) -> bool:
        return self._arrival_cost(n) > self._g[n.vertex_name]

    @property
    def precomputation_time(self) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return "Astar"


class GlobalUpperBoundDominanceCheck(DominanceCheck):
    
    def __init__(self, ub:float, ub_comp_time:float, epsilon:float) -> None:
        self._ub = ub * epsilon * 1.05 # add small slack to avoid numerical issues
        self._ub_computation_time = ub_comp_time
    
    def reset(self, ub:float, ub_comp_time:float, epsilon:float) -> None:
        self._ub = ub * epsilon * 1.05
        self._ub_computation_time = ub_comp_time

    def check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:
        if n.sol is None:
            return True

        if n.f > self._ub > 0:
            logger.debug(f"Dominated as f > ub. Path: {n.vertex_path}, f={n.f}, ub={self._ub}")
            return True
        return False

    @property
    def precomputation_time(self) -> float:
        return self._ub_computation_time

    @property
    def name(self) -> str:
        return "GUB"


class PositionBasedDominanceCheck(DominanceCheck):
    
    def __init__(
        self, 
        seed: int = 0,
        num_samples_per_vertex: int = 1,
        deque_size_per_vertex: Optional[int] = None
    ) -> None:
        self.seed = seed
        self._deque: Dict[str, Deque[List[float]]] = defaultdict(lambda: deque(maxlen=deque_size_per_vertex))
        # self._debug: Dict[str, Deque[List[Tuple[List[str], List[float]]]]] = defaultdict(lambda: deque(maxlen=config.deque_size_per_vertex))
        self._vertex_set_samples: Dict[str, List[ConvexSet]] = defaultdict(list)
        self._max_len_per_vertex = deque_size_per_vertex
        self._num_samples_per_vertex = num_samples_per_vertex
        self._solver_options = None
        self._edge_cache: Optional[Dict[Tuple[str, str], Any]] = None
        self._convex_restriction_time = 0.0
        self._convex_restriction_calls = 0
      
    def reset(self) -> None:
        self._deque = defaultdict(lambda: deque(maxlen=self._max_len_per_vertex))
        self._vertex_set_samples = defaultdict(list)

    def set_runtime_context(self, solver_options: object, edge_cache: Dict[Tuple[str, str], Any]) -> None:
        self._solver_options = solver_options
        self._edge_cache = edge_cache
        self._convex_restriction_time = 0.0
        self._convex_restriction_calls = 0

    @staticmethod
    def _sampling_subspace(convex_set: HPolyhedron) -> Optional[np.ndarray]:
        return HPolyhedronSampler.sampling_subspace(convex_set)

    def _init_vertex_set_samples(self, vertex_name:str, convex_set:ConvexSet, tmax:float) -> None:
        if isinstance(convex_set, DrakePoint):
            self._vertex_set_samples[vertex_name] = [convex_set]
        elif isinstance(convex_set, HPolyhedron):
            rng = RandomGenerator(self.seed)
            last_st_sample = convex_set.MaybeGetFeasiblePoint()
            for _ in range(self._num_samples_per_vertex):
                try:
                    if last_st_sample is None:
                        raise RuntimeError(
                            "Expected a non-empty HPolyhedron when sampling position-based dominance sets."
                        )
                    st_sample = HPolyhedronSampler.uniform_sample(
                        convex_set,
                        rng,
                        last_st_sample,
                        tol=1e-6,
                        context=f"PositionBasedDominanceCheck[{vertex_name}]",
                    )
                except Exception:
                    st_sample = convex_set.MaybeGetFeasiblePoint()

                spatial_sample = st_sample[:-1]
                st_line_set = HPolyhedron.MakeBox(
                    lb = np.append(spatial_sample, 0.0),
                    ub = np.append(spatial_sample, tmax)
                ).Intersection(convex_set)
                self._vertex_set_samples[vertex_name].append(st_line_set)
                last_st_sample = st_sample
        else:
            raise NotImplementedError(f"Convex set type {type(convex_set)} not supported yet.")

    def check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:
        if n.sol is None:
            logger.debug(f"Dominated as sol is None. Path: {n.vertex_path}")
            return True

        if n.vertex_name not in self._vertex_set_samples:
            self._init_vertex_set_samples(
                n.vertex_name, 
                stgcs.get_vertex(n.vertex_name).st_hpoly,
                stgcs.tmax
            )
        
        frontier = self._deque[n.vertex_name]
        candidate_duration = float(n.sol.duration)
        for costs in frontier:
            if all(cost <= candidate_duration for cost in costs):
                logger.debug(
                    f"delta_pos[{n.vertex_name}]. Dominated by duration lower bound: path={n.vertex_path}"
                )
                return True

        if isinstance(stgcs.get_vertex(n.vertex_name).st_hpoly, DrakePoint):
            candidate_costs, all_inf = [n.sol.duration], False
        else:
            candidate_costs, all_inf = [], True
            for sample in self._vertex_set_samples[n.vertex_name]:
                ts = time.perf_counter()
                sol = restrict_final_position(
                    n.vertex_path,
                    sample,
                    stgcs,
                    gcs,
                    solver_options=self._solver_options,
                    edge_cache=self._edge_cache,
                )
                self._convex_restriction_time += time.perf_counter() - ts
                self._convex_restriction_calls += 1
                if sol is None:
                    candidate_costs.append(np.inf)
                else:
                    all_inf = False
                    candidate_costs.append(sol.duration)

        dominated = all_inf and len(frontier) > 0
        for costs in frontier:
            if all(c1 <= c2 for c1, c2 in zip(costs, candidate_costs)):
                dominated = True
                logger.debug(f"delta_pos[{n.vertex_name}]. Dominated: costs={candidate_costs}, path={n.vertex_path}")
                # logger.debug(f"delta_pos[{n.vertex_name}]={self._debug[n.vertex_name]}. Dominated: costs={candidate_costs}, path={n.vertex_path}")
                break
            
        if not dominated:
            frontier.append(candidate_costs)
            # self._debug[n.vertex_name].append((n.vertex_path.copy(), candidate_costs))

        return dominated
    
    @property
    def precomputation_time(self) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return "delta_pos"

    @property
    def convex_restriction_time(self) -> float:
        return self._convex_restriction_time

    @property
    def convex_restriction_calls(self) -> int:
        return self._convex_restriction_calls


def restrict_final_position(
    vertex_path:List[str],
    cvs:ConvexSet,
    stgcs:STGCS,
    gcs:GCS,
    solver_options: object = None,
    edge_cache: Optional[Dict[Tuple[str, str], Any]] = None,
) -> Optional[STTrajectory]:
    if solver_options is None:
        solver_options = MPGCSInstance.default_solver_options()
    v_last = gcs.GetVertexByName(vertex_path[-1])
    v_last_set: ConvexSet = v_last.set()
    if isinstance(v_last_set, DrakePoint):
        return solve_convex_restriction(gcs, vertex_path, options=solver_options, edge_cache=edge_cache)
    
    elif isinstance(v_last_set, HPolyhedron):
        final_vertex = gcs.AddVertex(make_Cartesian_power_hpoly(cvs, 2), "final_pt")
        try:
            e_last = gcs.AddEdge(v_last, final_vertex, name="final_edge")
            for cstr in stgcs.edge_constraints:
                e_last.AddConstraint(Binding[Constraint](cstr, np.append(e_last.xu(), e_last.xv())))

            E = []
            for tail, head in zip(vertex_path[:-1], vertex_path[1:]):
                edge_key = (tail, head)
                if edge_cache is not None:
                    edge = edge_cache.get(edge_key)
                    if edge is None:
                        edge = gcs.GetEdgeByName(EDGE_KEY(tail, head))
                        edge_cache[edge_key] = edge
                else:
                    edge = gcs.GetEdgeByName(EDGE_KEY(tail, head))
                E.append(edge)
            E.append(e_last)
            res =  gcs.SolveConvexRestriction(E, solver_options)
            sol = None
            if res.is_success():
                points = [res.GetSolution(e.xu()) for e in E] + [res.GetSolution(E[-1].xv())]
                sol = STTrajectory(vertex_path, points[:-1], stgcs.dimension)   
            return sol
        finally:
            gcs.RemoveVertex(final_vertex)

    else:
        raise NotImplementedError(f"Convex set type {type(v_last.set())} not supported yet.")


class ArrivalStateContainmentDominanceCheck(DominanceCheck):

    class GreedyOptions:
        OFF = 0
        KEEP_EARLIEST = 1

    def __init__(
        self, vmin:np.ndarray, vmax:np.ndarray, greedy_option:GreedyOptions = GreedyOptions.OFF
    ) -> None:
        # for each vertex, every Cone(p) does not contain any other Cone(q)
        self._P: Dict[str, List[tuple]] = defaultdict(list)
        self._vmin = vmin
        self._vmax = vmax
        self.greedy_option = greedy_option

    def reset(self) -> None:
        self._P = defaultdict(list)
    
    def check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:    
        if n.sol is None:
            logger.debug(f"Dominated as sol is None. Path: {n.vertex_path}")
            return True
        
        p = n.sol.points[-1][n.sol.dim:]
        p_key = tuple(p.tolist())

        is_dominated, dominated = False, []
        for q in self._P[n.vertex_name]:
            if earliest_cone_containment(q, p, self._vmin, self._vmax):
                # Cone(q) contains Cone(p), so no need to check further
                # since Cone(p) cannot contain any other Cone(r)
                is_dominated = True
                break
            elif earliest_cone_containment(p, q, self._vmin, self._vmax):
                # Cone(p) contains Cone(q), so q can be removed
                # and Cone(p) will never be contained by any other Cone(r)
                # however, we still need to check other q's for containment removal
                dominated.append(q)    
            
        assert not (is_dominated and dominated != []), "Logical error in dominance checking"
        for q in dominated:
            self._P[n.vertex_name].remove(q)

        if is_dominated:
            logger.debug(f"Dominated by VelCone {q}. Path: {n.vertex_path}, p={p}")
            return True
        elif self.greedy_option == ArrivalStateContainmentDominanceCheck.GreedyOptions.KEEP_EARLIEST \
             and len(self._P[n.vertex_name]) > 0:
            assert len(self._P[n.vertex_name]) == 1, "Logical error in keep_earliest"
            if p[-1] < self._P[n.vertex_name][0][-1]:
                logger.debug(f"Replace {self._P[n.vertex_name][0]} by earlier {p}")
                self._P[n.vertex_name] = [p_key]
            return True
        else:
            self._P[n.vertex_name].append(p_key)
            return False
        
    @property
    def precomputation_time(self) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return "delta_state"


def earliest_cone_containment(
    p:np.ndarray, q:np.ndarray, vmin:np.ndarray, vmax:np.ndarray
) -> bool:
    """ Return True if Cone(p) contains Cone(q) by checking a sufficient condition:
        (tp <= tq) and the projected cuboid of Cone(p) at tq contains point q
    """
    if p[-1] > q[-1]:
        return False
    
    dt = q[-1] - p[-1]
    lb = p[:-1] + vmin * dt
    ub = p[:-1] + vmax * dt
    return np.all(lb <= q[:-1]) and np.all(q[:-1] <= ub)


class SetContainmentDominanceCheck(DominanceCheck):

    class _Frontier:
        def __init__(self) -> None:
            self.paths: List[tuple] = []
            self.times: List[float] = []
            self.points: List[np.ndarray] = []

        def __len__(self) -> int:
            return len(self.paths)

        def __contains__(self, path_key: tuple) -> bool:
            return path_key in self.paths

        def __eq__(self, other: object) -> bool:
            if isinstance(other, list):
                return self.paths == other
            return super().__eq__(other)

        def __repr__(self) -> str:
            return repr(self.paths)

        def prefix_end(self, time_value: float) -> int:
            return bisect_right(self.times, float(time_value))

        def suffix_start(self, time_value: float) -> int:
            return bisect_left(self.times, float(time_value))

        def insert(self, path_key: tuple, point: np.ndarray) -> None:
            time_value = float(point[-1])
            insert_idx = bisect_right(self.times, time_value)
            self.paths.insert(insert_idx, path_key)
            self.times.insert(insert_idx, time_value)
            self.points.insert(insert_idx, np.asarray(point, dtype=float))

        def remove(self, dominated: set[tuple]) -> None:
            if not dominated:
                return
            retained = [
                (path, time_value, point)
                for path, time_value, point in zip(self.paths, self.times, self.points)
                if path not in dominated
            ]
            if retained:
                self.paths, self.times, self.points = [list(values) for values in zip(*retained)]
            else:
                self.paths, self.times, self.points = [], [], []

        def slice(self, start: int, end: Optional[int] = None) -> Tuple[List[tuple], np.ndarray]:
            paths = self.paths[start:end]
            points = self.points[start:end]
            if not points:
                return paths, np.empty((0, 0))
            return paths, np.asarray(points, dtype=float)

    class Option(IntEnum):
        VERTEX_ONLY = 0
        EDGE_ONLY = 1
        BOTH = 2

    def __init__(
        self, vmin:np.ndarray, vmax:np.ndarray, tmax:float, option:Option = Option.VERTEX_ONLY,
    ) -> None:
        # for each vertex, every Cone(p) does not contain any other Cone(q)
        self._option = option
        self._paths_on_verts: Dict[str, SetContainmentDominanceCheck._Frontier] = defaultdict(self._Frontier)
        self._paths_on_edges: Dict[Tuple[str, str], SetContainmentDominanceCheck._Frontier] = defaultdict(self._Frontier)
        self._arrival_pt_cache: Dict[tuple, np.ndarray] = {}
        self._intersection_cache: Dict[Tuple[str, str], HPolyhedron|DrakePoint] = {}
        self._reachable_set_cache: Dict[tuple, Optional[HPolyhedron]] = {}
        self._reachable_vertex_cache: Dict[tuple, Optional[np.ndarray]] = {}
        self._vmin = vmin
        self._vmax = vmax
        self._tmax = tmax

    def reset(self) -> None:
        self._paths_on_verts = defaultdict(self._Frontier)
        self._paths_on_edges = defaultdict(self._Frontier)
        self._arrival_pt_cache = {}
        self._intersection_cache = {}
        self._reachable_set_cache = {}
        self._reachable_vertex_cache = {}
    
    def check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:    
        if self._option == SetContainmentDominanceCheck.Option.EDGE_ONLY:
            return self._edge_based_check(n, stgcs, gcs)
        elif self._option == SetContainmentDominanceCheck.Option.VERTEX_ONLY:
            return self._vertex_based_check(n, stgcs, gcs)
        elif self._option == SetContainmentDominanceCheck.Option.BOTH:
            return self._edge_based_check(n, stgcs, gcs) or self._vertex_based_check(n, stgcs, gcs)

    def _vertex_based_check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:    
        if n.sol is None:
            logger.debug(f"Dominated as sol is None. Path: {n.vertex_path}")
            return True
        
        path_key = n.vertex_path_tuple
        p = n.sol.points[-1][n.sol.dim:]
        p_istc = self.get_intersection(stgcs, path_key[-2], n.vertex_name)
        self._arrival_pt_cache[path_key] = p

        frontier = self._paths_on_verts[n.vertex_name]
        dominated_by_end = frontier.prefix_end(p[-1])
        prefix_paths, prefix_points = frontier.slice(0, dominated_by_end)
        candidate_mask = self._cone_contains_point_mask(prefix_points, p)
        for path, q in zip(self._masked_paths(prefix_paths, candidate_mask), prefix_points[candidate_mask]):
            if self._exact_cone_containment(q, p, p_istc, path_key):
                # existing Cone(q) dominates p_istc, break
                self._clear_path_cache(path_key)
                logger.debug(f"Dominated by ExactConeContainment {q}. Path: {n.vertex_path}, p={p}")
                return True

        dominated = set()
        dominates_start = frontier.suffix_start(p[-1])
        suffix_paths, suffix_points = frontier.slice(dominates_start)
        dominated_mask = self._point_cone_contains_mask(p, suffix_points)
        for path, q in zip(self._masked_paths(suffix_paths, dominated_mask), suffix_points[dominated_mask]):
            q_istc = self.get_intersection(stgcs, path[-2], path[-1])
            if self._exact_cone_containment(p, q, q_istc, path):
                dominated.add(path)

        self._remove_dominated_paths(frontier, dominated)
        frontier.insert(path_key, p)
        return False

    def _edge_based_check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:
        if n.sol is None:
            logger.debug(f"Dominated as sol is None. Path: {n.vertex_path}")
            return True
        
        path_key = n.vertex_path_tuple
        if len(path_key) < 2:
            return False # first vertex, no dominance
        
        arrival_edge = (path_key[-2], n.vertex_name)
        p = n.sol.points[-1][n.sol.dim:]
        self._arrival_pt_cache[path_key] = p

        edge_istc = self.get_intersection(stgcs, *arrival_edge)
        
        frontier = self._paths_on_edges[arrival_edge]
        dominated_by_end = frontier.prefix_end(p[-1])
        prefix_paths, prefix_points = frontier.slice(0, dominated_by_end)
        candidate_mask = self._cone_contains_point_mask(prefix_points, p)
        for path, q in zip(self._masked_paths(prefix_paths, candidate_mask), prefix_points[candidate_mask]):
            if self._exact_cone_containment(q, p, edge_istc, path_key):
                # existing Cone(q) dominates p_istc, break
                self._clear_path_cache(path_key)
                logger.debug(f"Dominated by tri-ExactConeContainment {q}. Path: {n.vertex_path}, p={p}")
                return True

        dominated = set()
        dominates_start = frontier.suffix_start(p[-1])
        suffix_paths, suffix_points = frontier.slice(dominates_start)
        dominated_mask = self._point_cone_contains_mask(p, suffix_points)
        for path, q in zip(self._masked_paths(suffix_paths, dominated_mask), suffix_points[dominated_mask]):
            if self._exact_cone_containment(p, q, edge_istc, path):
                dominated.add(path)

        self._remove_dominated_paths(frontier, dominated)
        frontier.insert(path_key, p)
        return False

    @staticmethod
    def _masked_paths(paths: List[tuple], mask: np.ndarray) -> List[tuple]:
        if len(paths) == 0:
            return []
        return [path for path, keep in zip(paths, mask.tolist()) if keep]

    def _remove_dominated_paths(self, frontier: SetContainmentDominanceCheck._Frontier, dominated: set[tuple]) -> None:
        if not dominated:
            return
        for path in dominated:
            self._clear_path_cache(path)
        frontier.remove(dominated)

    def _clear_path_cache(self, path_key: tuple) -> None:
        self._arrival_pt_cache.pop(path_key, None)
        self._reachable_set_cache.pop(path_key, None)
        self._reachable_vertex_cache.pop(path_key, None)

    def _cone_contains_point_mask(self, starts: np.ndarray, target: np.ndarray) -> np.ndarray:
        if starts.size == 0:
            return np.zeros((0,), dtype=bool)
        dt = target[-1] - starts[:, -1]
        mask = dt >= 0
        if not np.any(mask):
            return mask
        lb = starts[:, :-1] + dt.reshape(-1, 1) * self._vmin
        ub = starts[:, :-1] + dt.reshape(-1, 1) * self._vmax
        return mask & np.all(lb <= target[:-1], axis=1) & np.all(target[:-1] <= ub, axis=1)

    def _point_cone_contains_mask(self, start: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if targets.size == 0:
            return np.zeros((0,), dtype=bool)
        dt = targets[:, -1] - start[-1]
        mask = dt >= 0
        if not np.any(mask):
            return mask
        lb = start[:-1] + dt.reshape(-1, 1) * self._vmin
        ub = start[:-1] + dt.reshape(-1, 1) * self._vmax
        return mask & np.all(lb <= targets[:, :-1], axis=1) & np.all(targets[:, :-1] <= ub, axis=1)

    @staticmethod
    def _reachable_set(q_hpoly: HPolyhedron, tmin: float) -> HPolyhedron:
        space_dim = q_hpoly.A().shape[1] - 1
        return q_hpoly.Intersection(HPolyhedron(
            A = np.array([0] * space_dim + [-1.0]).reshape(1, -1),
            b = np.array([-tmin]),
        ))

    @staticmethod
    def _all_earliest_cone_containment(
        p: np.ndarray, qs: np.ndarray, vmin: np.ndarray, vmax: np.ndarray
    ) -> bool:
        if qs.size == 0:
            return True
        if qs.ndim == 1:
            qs = qs.reshape(1, -1)
        dt = qs[:, -1] - p[-1]
        if np.any(dt < 0):
            return False
        lb = p[:-1] + dt.reshape(-1, 1) * vmin
        ub = p[:-1] + dt.reshape(-1, 1) * vmax
        return bool(np.all(lb <= qs[:, :-1]) and np.all(qs[:, :-1] <= ub))

    def _reachable_data(
        self, path_key: tuple, q: np.ndarray, q_hpoly: HPolyhedron|DrakePoint
    ) -> Tuple[Optional[HPolyhedron], Optional[np.ndarray], bool]:
        if not isinstance(q_hpoly, HPolyhedron):
            return None, None, False

        if path_key not in self._reachable_set_cache:
            reachable_set = self._reachable_set(q_hpoly, q[-1])
            if reachable_set.IsEmpty():
                self._reachable_set_cache[path_key] = None
                self._reachable_vertex_cache[path_key] = None
            else:
                self._reachable_set_cache[path_key] = reachable_set
                self._reachable_vertex_cache[path_key] = hpoly_to_vrep(reachable_set)

        return self._reachable_set_cache[path_key], self._reachable_vertex_cache[path_key], True

    def _exact_cone_containment(
        self, p: np.ndarray, q: np.ndarray, q_hpoly: HPolyhedron|DrakePoint, path_key: tuple,
    ) -> bool:
        if isinstance(q_hpoly, HPolyhedron) and not earliest_cone_containment(p, q, self._vmin, self._vmax):
            return False

        reachable_set, reachable_vertices, is_precomputed = self._reachable_data(path_key, q, q_hpoly)
        return exact_cone_containment(
            p,
            q,
            q_hpoly,
            self._vmin,
            self._vmax,
            reachable_set=reachable_set,
            reachable_vertices=reachable_vertices,
            is_precomputed=is_precomputed,
        )

    def get_intersection(self, stgcs:STGCS, u:str, v:str) -> HPolyhedron|DrakePoint:
        if (u, v) not in self._intersection_cache:
            u_hpoly = stgcs.get_vertex(u).st_hpoly
            v_hpoly = stgcs.get_vertex(v).st_hpoly
            if isinstance(u_hpoly, DrakePoint):
                self._intersection_cache[(u, v)] = u_hpoly
            elif isinstance(v_hpoly, DrakePoint):
                self._intersection_cache[(u, v)] = v_hpoly
            else:
                itsc = u_hpoly.Intersection(v_hpoly)
                self._intersection_cache[(u, v)] = itsc

        return self._intersection_cache[(u, v)]
    
    @property
    def precomputation_time(self) -> float:
        return 0.0

    @property
    def name(self) -> str:
        if self._option == SetContainmentDominanceCheck.Option.VERTEX_ONLY:
            return "delta_set"
        elif self._option == SetContainmentDominanceCheck.Option.EDGE_ONLY:
            return "delta_set(e)"
        else:
            return "delta_set(b)"


def exact_cone_containment(
    p:np.ndarray,
    q:np.ndarray,
    q_hpoly:HPolyhedron|DrakePoint,
    vmin:np.ndarray,
    vmax:np.ndarray,
    reachable_set: Optional[HPolyhedron] = None,
    reachable_vertices: Optional[np.ndarray] = None,
    is_precomputed: bool = False,
) -> bool:
    """ return True if Cone(p) contains all Cone(p) for all reachable corner pts q in q_hpoly """
    if isinstance(q_hpoly, DrakePoint):
        return earliest_cone_containment(p, q_hpoly.x(), vmin, vmax)

    if not earliest_cone_containment(p, q, vmin, vmax):
        return False
    
    if is_precomputed:
        enlarged_reachable_set = reachable_set
    else:
        enlarged_reachable_set = SetContainmentDominanceCheck._reachable_set(q_hpoly, q[-1])

    if enlarged_reachable_set is None:
        return True

    if enlarged_reachable_set.IsEmpty():
        # no reachable points most likely due to numerical issues
        # fall back to Cone containment check at point q
        return True

    if enlarged_reachable_set.PointInSet(p):
        # p is in reachable set, so Cone(p) cannot contain all Cone(q)
        return False

    verts = reachable_vertices if is_precomputed else hpoly_to_vrep(enlarged_reachable_set)
    if verts is None:
        # V-rep failed, fall back to point check
        return True

    return SetContainmentDominanceCheck._all_earliest_cone_containment(p, np.asarray(verts), vmin, vmax)


def get_reachable_bounding_pts(
    hpoly:HPolyhedron, vmin:np.ndarray, vmax:np.ndarray, tmin:float, tmax:float
) -> np.ndarray:
    space_dim = hpoly.A().shape[1] - 1
    reachable_set = hpoly.Intersection(HPolyhedron(
        A = np.array([0] * space_dim + [-1.0]).reshape(1, -1),
        b = np.array([-tmin]),
    ))

    V = hpoly_to_vrep(reachable_set).T
    
    A = np.empty((0, space_dim + 1))
    b = np.empty((0,))
    for v in product(*zip(vmin, vmax)):
        v = np.array(v)
        A = np.block([
            [A],
            [np.eye(space_dim), -v.reshape(-1, 1)],
            [np.zeros((1, space_dim)), np.zeros((1, 1))]
        ])
        b = np.hstack([b, np.hstack([v, 1]) * tmax])
    
    pts = (A @ V + b.reshape(-1, 1)).reshape(2 ** space_dim, space_dim + 1, -1)
    pts = pts.transpose(0, 2, 1).reshape(-1, space_dim + 1)
    hull = ConvexHull(pts[:, :-1])
    return hull.points[hull.vertices]
