from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from itertools import product

import os, time, pickle, heapq
import numpy as np
from tqdm import tqdm

from stgcs.graph import STGCS
from stgcs.geometry_utils import hpoly_to_vrep, find_min_travel_time_state_to_spatial_point
from stgcs.gcs_solver import (
    EDGE_KEY, GCS_SOURCE_NAME, GCS_TARGET_NAME, GCS, MPGCSInstance, make_Cartesian_power_hpoly
)
from stgcs.gcs_solver import solve_convex_restriction
from stgcs.bfs.best_first_search import Heuristic, SearchNode
from stgcs.bfs.dominance_check import SetContainmentDominanceCheck
from stgcs.trajectory import STTrajectory

from pydrake.all import Binding, Constraint, Cost, HPolyhedron, VPolytope, Point as DrakePoint

import logging
logger = logging.getLogger(__name__)


class MotionOnlyHeuristic(Heuristic):

    """ Minimum travel-time lower bound from an arrival set to the target point """

    def __init__(self, stgcs:STGCS) -> None:
        ts = time.perf_counter()
        self._precomputation_time = time.perf_counter() - ts

    @staticmethod
    def travel_time_lower_bound(start: np.ndarray, goal: np.ndarray, vlimit: float) -> float:
        start = np.asarray(start, dtype=float).reshape(-1)
        goal = np.asarray(goal, dtype=float).reshape(-1)
        return float(np.max(np.abs(start - goal) / float(vlimit)))

    @staticmethod
    def spatial_points(convex_set: DrakePoint | object, dimension: int) -> List[np.ndarray]:
        if isinstance(convex_set, DrakePoint):
            return [convex_set.x()[:dimension]]

        vertices = hpoly_to_vrep(convex_set)
        spatial_points = np.unique([v[:dimension] for v in vertices], axis=0)
        return [point for point in spatial_points]

    @staticmethod
    def intersect_arrival_sets(
        lhs: HPolyhedron | DrakePoint | None,
        rhs: HPolyhedron | DrakePoint | None,
    ) -> HPolyhedron | DrakePoint | None:
        if lhs is None or rhs is None:
            return None

        if isinstance(lhs, DrakePoint) and isinstance(rhs, DrakePoint):
            return lhs if np.allclose(lhs.x(), rhs.x()) else None

        if isinstance(lhs, DrakePoint):
            return lhs if rhs.PointInSet(lhs.x()) else None

        if isinstance(rhs, DrakePoint):
            return rhs if lhs.PointInSet(rhs.x()) else None

        intersection = lhs.Intersection(rhs)
        if intersection.IsEmpty():
            return None
        return intersection

    @classmethod
    def motion_only_domain(
        cls,
        v_name: str,
        stgcs: STGCS,
        node: Optional[SearchNode],
        x: Optional[np.ndarray] = None,
    ) -> HPolyhedron | DrakePoint:
        domain, _ = cls.motion_only_domain_with_cache_policy(v_name, stgcs, node, x)
        return domain

    @classmethod
    def motion_only_domain_with_cache_policy(
        cls,
        v_name: str,
        stgcs: STGCS,
        node: Optional[SearchNode],
        x: Optional[np.ndarray] = None,
    ) -> Tuple[HPolyhedron | DrakePoint, bool]:
        vertex_set = stgcs.get_vertex(v_name).st_hpoly
        if node is None or node.parent is None:
            return vertex_set, True

        if node.vertex_name != v_name:
            raise ValueError(
                f"Motion-only heuristic node/vertex mismatch: node is at {node.vertex_name!r}, "
                f"but the heuristic was queried for {v_name!r}."
            )

        predecessor_name = node.parent.vertex_name
        predecessor_set = stgcs.get_vertex(predecessor_name).st_hpoly
        interface = cls.intersect_arrival_sets(predecessor_set, vertex_set)
        if interface is not None:
            return interface, True

        if x is None:
            raise ValueError(
                f"Motion-only heuristic expected a non-empty arrival interface for edge "
                f"({predecessor_name!r}, {v_name!r})."
            )

        state = np.asarray(x, dtype=float).reshape(-1)
        expected_dim = stgcs.dimension + 1
        if state.shape != (expected_dim,):
            raise ValueError(
                f"Motion-only heuristic fallback state has shape {state.shape}, "
                f"expected ({expected_dim},) for edge ({predecessor_name!r}, {v_name!r})."
            )
        return DrakePoint(state), False

    def get(self, stgcs:STGCS, gcs:GCS, source:str=GCS_SOURCE_NAME, target:str=GCS_TARGET_NAME) -> Callable[[str, np.ndarray, STGCS, Optional[SearchNode]], float]:
        target_vertex = stgcs.get_vertex(target)
        goal_spatial_point = np.array([itvl.start for itvl in target_vertex.space_itvls], dtype=float)
        if not np.allclose(goal_spatial_point, [itvl.end for itvl in target_vertex.space_itvls]):
            raise ValueError("Motion-only heuristic expects the query target to be spatially fixed.")
        cache: Dict[Tuple[Optional[str], str], float] = {}
        
        def _h_func(v_name:str, x:np.ndarray, stgcs:STGCS, node:Optional[SearchNode]=None) -> float:
            if v_name == target or 'target' in v_name:
                return 0.0

            predecessor_name = None if node is None or node.parent is None else node.parent.vertex_name
            cache_key = (predecessor_name, v_name)
            if cache_key in cache:
                return cache[cache_key]

            domain, cacheable = self.motion_only_domain_with_cache_policy(v_name, stgcs, node, x)
            min_time, _ = find_min_travel_time_state_to_spatial_point(
                domain,
                goal_spatial_point,
                stgcs.vlimit,
            )
            if cacheable:
                cache[cache_key] = min_time
            return min_time

        return _h_func

    @property
    def name(self) -> str:
        return "h_mot"

    @property
    def computation_time(self) -> float:
        return self._precomputation_time


class ZeroHeuristic(Heuristic):

    def __init__(self, stgcs: Optional[STGCS] = None) -> None:
        del stgcs
        self._precomputation_time = 0.0

    def get(
        self,
        stgcs: STGCS,
        gcs: GCS,
        source: str = GCS_SOURCE_NAME,
        target: str = GCS_TARGET_NAME,
    ) -> Callable[[str, np.ndarray, STGCS, Optional[SearchNode]], float]:
        del stgcs, gcs, source, target
        return lambda v_name, x, stgcs, node=None: 0.0

    @property
    def name(self) -> str:
        return "h_zero"

    @property
    def computation_time(self) -> float:
        return self._precomputation_time


class MaxHeuristic(Heuristic):
    _INSTANCE_HEURISTIC_ATTRS = (
        "motion_only_heuristic",
        "triplet_relaxation_heuristic",
        "interface_to_set_cost_table_heuristic",
    )

    def __init__(self, heuristics: Sequence[Heuristic]) -> None:
        self._heuristics = tuple(heur for heur in heuristics if heur is not None)
        if len(self._heuristics) == 0:
            raise ValueError("MaxHeuristic requires at least one available heuristic.")
        self._precomputation_time = sum(float(heur.computation_time) for heur in self._heuristics)

    @classmethod
    def from_instance(cls, instance: object) -> MaxHeuristic:
        return cls(cls.available_heuristics(instance))

    @classmethod
    def available_heuristics(cls, instance: object) -> List[Heuristic]:
        heuristics: List[Heuristic] = []
        for attr in cls._INSTANCE_HEURISTIC_ATTRS:
            heur = getattr(instance, attr, None)
            if heur is not None:
                heuristics.append(heur)
        return heuristics

    def get(
        self,
        stgcs: STGCS,
        gcs: GCS,
        source: str = GCS_SOURCE_NAME,
        target: str = GCS_TARGET_NAME,
    ) -> Callable[[str, np.ndarray, STGCS, Optional[SearchNode]], float]:
        h_funcs = [heur.get(stgcs, gcs, source, target) for heur in self._heuristics]

        def _h_func(v_name: str, x: np.ndarray, stgcs: STGCS, node: Optional[SearchNode] = None) -> float:
            return max(float(h_func(v_name, x, stgcs, node)) for h_func in h_funcs)

        return _h_func

    @property
    def name(self) -> str:
        return "h_max"

    @property
    def computation_time(self) -> float:
        return self._precomputation_time


class TripletRelaxationHeuristic(Heuristic):

    @staticmethod
    def _update_root_set2set_dist(
        root_set2set_dist: Dict[Tuple[str, str], float],
        stgcs: STGCS,
        source_name: str,
        target_name: str,
        cost: float,
    ) -> None:
        for source_root in stgcs.get_vertex(source_name).root_name:
            for target_root in stgcs.get_vertex(target_name).root_name:
                key = (source_root, target_root)
                root_set2set_dist[key] = min(root_set2set_dist.get(key, float("inf")), cost)

    @staticmethod
    def _update_root_triplet_lb_cost(
        root_triplet_lb_cost: Dict[Tuple[str, str, str], float],
        stgcs: STGCS,
        source_name: str,
        middle_name: str,
        target_name: str,
        cost: float,
    ) -> None:
        for source_root in stgcs.get_vertex(source_name).root_name:
            for middle_root in stgcs.get_vertex(middle_name).root_name:
                for target_root in stgcs.get_vertex(target_name).root_name:
                    key = (source_root, middle_root, target_root)
                    root_triplet_lb_cost[key] = min(root_triplet_lb_cost.get(key, float("inf")), cost)

    def __init__(self, stgcs:STGCS, gcs:GCS, use_update:bool=False) -> None:
        backbone = self.build_backbone(stgcs, gcs)
        self._initialize_from_backbone(stgcs, backbone, use_update=use_update, show_progress=True)

    @classmethod
    def build_backbone(cls, stgcs: STGCS, gcs: GCS) -> Dict[str, object]:
        ts = time.perf_counter()
        triplet_lb_cost: Dict[Tuple[str, str, str], float] = {}
        root_triplet_lb_cost: Dict[Tuple[str, str, str], float] = {}
        for u, v in stgcs.G.edges:
            for w in stgcs.G.successors(v):
                if u != w:  # inherently no 1-hop cycle (u -> v -> u)
                    cost = cls._calc_lower_bound_cost(u, v, w, gcs)
                    triplet_lb_cost[(u, v, w)] = cost
                    cls._update_root_triplet_lb_cost(root_triplet_lb_cost, stgcs, u, v, w, cost)
        return {
            "triplet_lb_cost": triplet_lb_cost,
            "root_triplet_lb_cost": root_triplet_lb_cost,
            "computation_time": time.perf_counter() - ts,
        }

    @classmethod
    def from_backbone(
        cls,
        stgcs: STGCS,
        backbone: Dict[str, object],
        use_update: bool = False,
        show_progress: bool = True,
    ) -> TripletRelaxationHeuristic:
        heur = cls.__new__(cls)
        heur._initialize_from_backbone(stgcs, backbone, use_update=use_update, show_progress=show_progress)
        return heur

    @classmethod
    def build_pair(
        cls,
        stgcs: STGCS,
        gcs: GCS,
        show_progress: bool = True,
    ) -> Tuple[TripletRelaxationHeuristic, TripletRelaxationHeuristic]:
        backbone = cls.build_backbone(stgcs, gcs)
        return (
            cls.from_backbone(stgcs, backbone, use_update=False, show_progress=show_progress),
            cls.from_backbone(stgcs, backbone, use_update=True, show_progress=False),
        )

    def _initialize_from_backbone(
        self,
        stgcs: STGCS,
        backbone: Dict[str, object],
        use_update: bool,
        show_progress: bool,
    ) -> None:
        self._triplet_lb_cost = dict(backbone["triplet_lb_cost"])
        self._root_triplet_lb_cost = dict(backbone.get("root_triplet_lb_cost", {}))
        self.use_update = use_update
        self._backbone_precomputation_time = float(backbone["computation_time"])

        ts = time.perf_counter()
        if use_update:
            self._triplet_added: List[Tuple[str,str,str]] = []
        else:
            self._set2set_dist: Dict[Tuple[str, str], float] = {}
            self._root_set2set_dist = dict(backbone.get("root_set2set_dist", {}))
            if not self._root_set2set_dist:
                for target in tqdm(
                    stgcs.G.nodes,
                    desc="Computing set-to-set distances over h_tri",
                    disable=not show_progress,
                ):
                    g = self._backward_search(target, stgcs)
                    for u_name, cost in g.items():
                        self._set2set_dist[(u_name, target)] = cost
                        self._update_root_set2set_dist(
                            self._root_set2set_dist,
                            stgcs,
                            u_name,
                            target,
                            cost,
                        )

        self._precomputation_time = self._backbone_precomputation_time + time.perf_counter() - ts

    def get(self, stgcs:STGCS, gcs:GCS, source:str=GCS_SOURCE_NAME, target:str=GCS_TARGET_NAME) -> Callable[[str, np.ndarray, STGCS, Optional[SearchNode]], float]:
        if not self.use_update:
            def _h_func(v_name:str, x:np.ndarray, stgcs:STGCS, node:Optional[SearchNode]=None) -> float:
                v_root_names = stgcs.get_vertex(v_name).root_name
                t_root_names = stgcs.get_vertex(target).root_name
                return min([self._root_set2set_dist.get((v_root, t_root), float("inf"))
                            for v_root in v_root_names for t_root in t_root_names])
            
            return _h_func
        
        # reset updated triplet costs and do backward search again
        self._reset_updated(stgcs, gcs, target)
        g = self._backward_search(target, stgcs)
        return lambda v_name, x, stgcs, node=None: g[v_name]

    @staticmethod
    def _calc_lower_bound_cost(u:str, v:str, w:str, gcs:GCS) -> float:
        sol = solve_convex_restriction(gcs, vertex_path=[u, v, w])
        if sol is None:
            return float("inf")
        
        return sol.duration

    def _triplet_lower_bound_cost(self, pred: str, curr: str, parent: str, stgcs: STGCS) -> float:
        exact_key = (pred, curr, parent)
        if exact_key in self._triplet_lb_cost:
            return self._triplet_lb_cost[exact_key]

        best = float("inf")
        pred_roots = stgcs.get_vertex(pred).root_name
        curr_roots = stgcs.get_vertex(curr).root_name
        parent_roots = stgcs.get_vertex(parent).root_name
        for pred_root in pred_roots:
            for curr_root in curr_roots:
                for parent_root in parent_roots:
                    root_key = (pred_root, curr_root, parent_root)
                    best = min(best, self._root_triplet_lb_cost.get(root_key, float("inf")))
                    if pred_root == curr_root or curr_root == parent_root:
                        best = min(best, 0.0)
        return best

    def _backward_search(self, target:str, stgcs:STGCS) -> Dict[str, float]:      
        g: Dict[str, float] = defaultdict(lambda: float("inf"))
        expanded, g[target] = set(), 0
        Q = [(0, target, None)]

        while len(Q) > 0:
            _, curr, parent = heapq.heappop(Q)
            expanded.add(curr)
            for pred in stgcs.G.predecessors(curr):
                if pred in expanded:
                    continue
                if parent is None:
                    cost = 0.0
                else:
                    cost = self._triplet_lower_bound_cost(pred, curr, parent, stgcs)

                tentative_g = g[curr] + cost
                if g[pred] > tentative_g:
                    g[pred] = tentative_g
                    heapq.heappush(Q, (g[pred], pred, curr))
                    # print(f"Updated h_tri backward search: g[{pred}] = {g[pred]} via ({pred}, {curr}, {parent}) with cost {cost}")
        
        return g

    def _reset_updated(self, stgcs:STGCS, gcs:GCS, target:str) -> None:
        # Clear previous triplets that were added
        for triplet in self._triplet_added:
            if triplet in self._triplet_lb_cost:
                del self._triplet_lb_cost[triplet]
        
        # Reset the triplet_added list for new computation
        self._triplet_added = []
                
        # add triplet (source, v, w)
        for v in stgcs.G.successors(GCS_SOURCE_NAME):
            for w in stgcs.G.successors(v):
                # self._triplet_lb_cost[(GCS_SOURCE_NAME, v, w)] = 0.0
                self._triplet_added.append((GCS_SOURCE_NAME, v, w))
            
        # add triplet (residing_v, sub_target, target) & (u, residing_v, dummay_target)
        for sub_target in stgcs.G.predecessors(GCS_TARGET_NAME):
            for residing_v in stgcs.G.predecessors(sub_target):
                self._triplet_added.append((residing_v, sub_target, GCS_TARGET_NAME))
                for u in stgcs.G.predecessors(residing_v):
                    self._triplet_added.append((u, residing_v, sub_target))
        
        # calculate the lower bound cost for each triplet added
        for triplet in self._triplet_added:
            if triplet not in self._triplet_lb_cost:
                cost = self._calc_lower_bound_cost(*triplet, gcs)
                if np.isfinite(cost):
                    self._triplet_lb_cost[triplet] = cost

    def save(self, filename:str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def backbone_payload(self) -> Dict[str, object]:
        triplet_lb_cost = dict(self._triplet_lb_cost)
        if self.use_update:
            for triplet in self._triplet_added:
                triplet_lb_cost.pop(triplet, None)
        payload = {
            "triplet_lb_cost": triplet_lb_cost,
            "root_triplet_lb_cost": dict(self._root_triplet_lb_cost),
            "computation_time": self._backbone_precomputation_time,
        }
        if not self.use_update and hasattr(self, "_root_set2set_dist"):
            payload["root_set2set_dist"] = dict(self._root_set2set_dist)
        return payload

    def save_backbone(self, filename:str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.backbone_payload(), f)

    @staticmethod
    def load(filename:str) -> TripletRelaxationHeuristic:
        with open(filename, 'rb') as f:
            heuristic: TripletRelaxationHeuristic = pickle.load(f)
        return heuristic

    @staticmethod
    def load_backbone(filename:str) -> Dict[str, object]:
        with open(filename, 'rb') as f:
            backbone: Dict[str, object] = pickle.load(f)
        return backbone

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        if "_backbone_precomputation_time" not in self.__dict__:
            self._backbone_precomputation_time = float(self._precomputation_time)

    @property
    def computation_time(self) -> float:
        return self._precomputation_time

    @property
    def backbone_computation_time(self) -> float:
        return self._backbone_precomputation_time
    
    @property
    def name(self) -> str:
        return "h_tri" if self.use_update else "h_tri(static)"


class InterfaceToSetCostTableHeuristic(Heuristic):
    """Compute the interface-to-set cost table using set-containment dominance checks."""

    def __init__(
        self, stgcs:STGCS, gcs:GCS, vlimit:float, 
        heur:Heuristic, timeout_secs:float
    ) -> None:
        self._ts = time.perf_counter()
        self.vlimit = vlimit
        self.heur = heur
        self._timeout_secs = timeout_secs
        self._successful = True
        # setting default heuristic as 0.0, in case timeout happens
        self._dist: Dict[Tuple[str, str], float] = {}
        self._incoming_dist: Dict[Tuple[str, str, str], float] = {}
        self._pair_target_predecessor: Dict[Tuple[str, str], Optional[str]] = {}
        
        for source_name in tqdm(stgcs.G.nodes, desc="Computing h_tab"):
            g, predecessors = self._backward_search_dijkstra(source_name, stgcs, gcs)
            if time.perf_counter() - self._ts >= self._timeout_secs:
                logger.warning("Terminating h_tab precomputation due to timeout.")
                self._successful = False
                break
            else:
                for target_name, dist in g.items():
                    self._cache_pair_result(
                        source_name,
                        target_name,
                        dist,
                        predecessors.get(target_name),
                    )
                for predecessor_name in stgcs.G.predecessors(source_name):
                    interface_g = self._backward_search_dijkstra_from_interface(
                        predecessor_name,
                        source_name,
                        stgcs,
                        gcs,
                    )
                    for target_name, dist in interface_g.items():
                        self._cache_incoming_result(
                            predecessor_name,
                            source_name,
                            target_name,
                            dist,
                            stgcs,
                        )
                    if time.perf_counter() - self._ts >= self._timeout_secs:
                        logger.warning("Terminating h_tab precomputation due to timeout.")
                        self._successful = False
                        break
                if not self._successful:
                    break
        
        self._precomputation_time = self.heur.computation_time + time.perf_counter() - self._ts

    def _cache_pair_result(
        self,
        source_name: str,
        target_name: str,
        cost: float,
        predecessor: Optional[str],
    ) -> None:
        self._dist[(source_name, target_name)] = cost
        self._pair_target_predecessor[(source_name, target_name)] = predecessor

    def _cache_incoming_result(
        self,
        predecessor_name: str,
        source_name: str,
        target_name: str,
        cost: float,
        stgcs: STGCS,
    ) -> None:
        for predecessor_root in stgcs.get_vertex(predecessor_name).root_name:
            for source_root in stgcs.get_vertex(source_name).root_name:
                for target_root in stgcs.get_vertex(target_name).root_name:
                    key = (predecessor_root, source_root, target_root)
                    self._incoming_dist[key] = min(self._incoming_dist.get(key, float("inf")), cost)

    @classmethod
    def _incoming_interface(
        cls,
        predecessor_name: str,
        source_name: str,
        stgcs: STGCS,
    ) -> HPolyhedron | DrakePoint | None:
        return MotionOnlyHeuristic.intersect_arrival_sets(
            stgcs.get_vertex(predecessor_name).st_hpoly,
            stgcs.get_vertex(source_name).st_hpoly,
        )

    @classmethod
    def _add_edge_constraints_and_costs(
        cls,
        edge: object,
        stgcs: STGCS,
    ) -> None:
        edge_vars = np.append(edge.u().x(), edge.v().x())
        for cost in stgcs.edge_costs:
            edge.AddCost(Binding[Cost](cost, edge_vars))
        for constraint in stgcs.edge_constraints:
            edge.AddConstraint(Binding[Constraint](constraint, edge_vars))

    @classmethod
    def _solve_convex_restriction_from_interface(
        cls,
        predecessor_name: str,
        source_name: str,
        vertex_path: List[str],
        stgcs: STGCS,
        gcs: GCS,
    ) -> Optional[STTrajectory]:
        if len(vertex_path) == 0 or vertex_path[0] != source_name:
            raise ValueError(
                f"Interface-constrained h_tab path must start at {source_name!r}; "
                f"received {vertex_path!r}."
            )

        interface = cls._incoming_interface(predecessor_name, source_name, stgcs)
        if interface is None:
            return None

        restricted_path = vertex_path[:-1] if vertex_path[-1] == GCS_TARGET_NAME else vertex_path
        if len(restricted_path) <= 1:
            return None

        temp_source_name = f"h_tab-interface-source:{predecessor_name}->{source_name}"
        temp_source_vertex = gcs.AddVertex(
            make_Cartesian_power_hpoly(interface, 2),
            temp_source_name,
        )
        try:
            source_edge = gcs.AddEdge(
                temp_source_vertex,
                gcs.GetVertexByName(source_name),
                EDGE_KEY(temp_source_name, source_name),
            )
            cls._add_edge_constraints_and_costs(source_edge, stgcs)
            original_edges = [
                gcs.GetEdgeByName(EDGE_KEY(tail, head))
                for tail, head in zip(restricted_path[:-1], restricted_path[1:])
            ]
            result = gcs.SolveConvexRestriction(
                [source_edge] + original_edges,
                MPGCSInstance.default_solver_options(),
            )
            if not result.is_success():
                return None

            points = [result.GetSolution(edge.xu()) for edge in original_edges]
            return STTrajectory(restricted_path[:-1], points, stgcs.dimension)
        finally:
            gcs.RemoveVertex(temp_source_vertex)

    def _backward_search_dijkstra(
        self,
        source: str,
        stgcs: STGCS,
        gcs: GCS,
    ) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """Perform forward one-to-all uniform-cost search from ``source``."""

        OPEN = [SearchNode.from_source(source)]
        V = set(stgcs.G.nodes)
        g: Dict[str, float] = {}
        predecessors: Dict[str, Optional[str]] = {}
        dc = SetContainmentDominanceCheck(
            -self.vlimit*np.ones(stgcs.dimension),
             self.vlimit*np.ones(stgcs.dimension),
             tmax=stgcs.tmax,
        )
        
        while len(OPEN) > 0:
            n: SearchNode = heapq.heappop(OPEN)
            if n.vertex_name not in g:
                V.discard(n.vertex_name)
                g[n.vertex_name] = n.f  # f = g since h is always 0
                predecessors[n.vertex_name] = None if n.parent is None else n.parent.vertex_name
                if len(V) == 0:
                    break

            for successor in stgcs.G.successors(n.vertex_name):
                if n.has_visited(successor):
                    continue

                if time.perf_counter() - self._ts >= self._timeout_secs:
                    logger.warning("Timeout in InterfaceToSetCostTableHeuristic backward search.")
                    return g, predecessors

                n_next = SearchNode.from_parent(child_vertex_name=successor, parent=n)
                n_next.sol = solve_convex_restriction(gcs, n_next.vertex_path)
                if n_next.sol is None:
                    n_next.f = float("inf")
                else:
                    n_next.f = n_next.sol.duration
                    is_dominated = dc.check(n_next, stgcs, gcs)
                    if not is_dominated:
                        heapq.heappush(OPEN, n_next)

        return g, predecessors

    def _backward_search_dijkstra_from_interface(
        self,
        predecessor: str,
        source: str,
        stgcs: STGCS,
        gcs: GCS,
    ) -> Dict[str, float]:
        """One-to-all h_tab search starting from the actual ``predecessor -> source`` interface."""

        if self._incoming_interface(predecessor, source, stgcs) is None:
            return {}

        OPEN = [SearchNode.from_source(source)]
        V = set(stgcs.G.nodes)
        g: Dict[str, float] = {}
        dc = SetContainmentDominanceCheck(
            -self.vlimit*np.ones(stgcs.dimension),
             self.vlimit*np.ones(stgcs.dimension),
             tmax=stgcs.tmax,
        )

        while len(OPEN) > 0:
            n: SearchNode = heapq.heappop(OPEN)
            if n.vertex_name not in g:
                V.discard(n.vertex_name)
                g[n.vertex_name] = n.f
                if len(V) == 0:
                    break

            for successor in stgcs.G.successors(n.vertex_name):
                if n.has_visited(successor):
                    continue

                if time.perf_counter() - self._ts >= self._timeout_secs:
                    logger.warning("Timeout in InterfaceToSetCostTableHeuristic interface backward search.")
                    return g

                n_next = SearchNode.from_parent(child_vertex_name=successor, parent=n)
                n_next.sol = self._solve_convex_restriction_from_interface(
                    predecessor,
                    source,
                    n_next.vertex_path,
                    stgcs,
                    gcs,
                )
                if n_next.sol is None:
                    n_next.f = float("inf")
                else:
                    n_next.f = n_next.sol.duration
                    is_dominated = dc.check(n_next, stgcs, gcs)
                    if not is_dominated:
                        heapq.heappush(OPEN, n_next)

        return g

    @staticmethod
    def _fixed_goal_spatial_point(target_vertex: object) -> np.ndarray:
        goal_spatial_point = np.array([itvl.start for itvl in target_vertex.space_itvls], dtype=float)
        if not np.allclose(goal_spatial_point, [itvl.end for itvl in target_vertex.space_itvls]):
            raise ValueError("h_tab expects the query target to be spatially fixed.")
        return goal_spatial_point

    def _goal_correction_by_root(
        self,
        stgcs: STGCS,
        target_vertex: object,
        goal_spatial_point: np.ndarray,
    ) -> Dict[str, float]:
        correction_by_root: Dict[str, float] = {}
        goal_parents = list(getattr(target_vertex, "target_parent_list", []) or [])

        if not goal_parents:
            for root_name in getattr(target_vertex, "root_name", []):
                correction_by_root[root_name] = 0.0
            return correction_by_root

        for goal_parent in goal_parents:
            goal_parent_name = getattr(goal_parent, "name", None)
            goal_parent_roots = list(getattr(goal_parent, "root_name", []))
            if not goal_parent_roots:
                continue

            if goal_parent_name is None:
                for root_name in goal_parent_roots:
                    correction_by_root[root_name] = min(correction_by_root.get(root_name, float("inf")), 0.0)
                continue

            predecessor_names = list(stgcs.G.predecessors(goal_parent_name))
            if not predecessor_names:
                for root_name in goal_parent_roots:
                    correction_by_root[root_name] = min(correction_by_root.get(root_name, float("inf")), 0.0)
                continue

            best_correction = float("inf")
            goal_parent_hpoly = getattr(goal_parent, "st_hpoly", None)
            if goal_parent_hpoly is None:
                continue

            for predecessor_name in predecessor_names:
                predecessor_vertex = stgcs.get_vertex(predecessor_name)
                predecessor_hpoly = getattr(predecessor_vertex, "st_hpoly", None)
                if predecessor_hpoly is None:
                    continue

                intersection = MotionOnlyHeuristic.intersect_arrival_sets(goal_parent_hpoly, predecessor_hpoly)
                if intersection is None:
                    continue
                correction, _ = find_min_travel_time_state_to_spatial_point(
                    intersection,
                    goal_spatial_point,
                    self.vlimit,
                )
                best_correction = min(best_correction, correction)

            for root_name in goal_parent_roots:
                correction_by_root[root_name] = min(
                    correction_by_root.get(root_name, float("inf")),
                    best_correction,
                )

        return correction_by_root

    def get(self, stgcs:STGCS, gcs:GCS, source:str=GCS_SOURCE_NAME, target:str=GCS_TARGET_NAME) -> Callable[[str, np.ndarray, STGCS, Optional[SearchNode]], float]:
        del gcs, source
        target_vertex = stgcs.get_vertex(target)
        goal_spatial_point = self._fixed_goal_spatial_point(target_vertex)
        goal_correction_by_root = self._goal_correction_by_root(stgcs, target_vertex, goal_spatial_point)
        
        def _state_to_goal_cost(x: np.ndarray) -> Optional[float]:
            state = np.asarray(x, dtype=float).reshape(-1)
            space_dim = goal_spatial_point.shape[0]
            if state.shape[0] < space_dim:
                return None
            return MotionOnlyHeuristic.travel_time_lower_bound(
                state[:space_dim],
                goal_spatial_point,
                self.vlimit,
            )

        def _vertex_bound(
            v_root_names: List[str],
            t_root_names: List[str],
            state_to_goal_cost: Optional[float],
        ) -> float:
            min_dist = float("inf")
            for v_root, t_root in product(v_root_names, t_root_names):
                set_to_set_cost = self._dist.get((v_root, t_root), float("inf"))
                if not np.isfinite(set_to_set_cost):
                    continue
                pair_cost = set_to_set_cost + goal_correction_by_root.get(t_root, 0.0)
                if v_root == t_root and state_to_goal_cost is not None:
                    pair_cost = min(pair_cost, state_to_goal_cost)
                min_dist = min(min_dist, pair_cost)
            return min_dist

        def _incoming_bound(
            predecessor_root_names: List[str],
            v_root_names: List[str],
            t_root_names: List[str],
        ) -> float:
            min_dist = float("inf")
            for predecessor_root, v_root, t_root in product(
                predecessor_root_names,
                v_root_names,
                t_root_names,
            ):
                set_to_set_cost = self._incoming_dist.get(
                    (predecessor_root, v_root, t_root),
                    float("inf"),
                )
                if not np.isfinite(set_to_set_cost):
                    continue
                min_dist = min(
                    min_dist,
                    set_to_set_cost + goal_correction_by_root.get(t_root, 0.0),
                )
            return min_dist

        def _h_func(v_name:str, x:np.ndarray, stgcs:STGCS, node:Optional[SearchNode]=None) -> float:
            if v_name == target:
                return 0.0
            if node is not None and node.vertex_name != v_name:
                raise ValueError(
                    f"h_tab node/vertex mismatch: node is at {node.vertex_name!r}, "
                    f"but the heuristic was queried for {v_name!r}."
                )
            v_root_names = stgcs.get_vertex(v_name).root_name
            t_root_names = target_vertex.root_name
            vertex_bound = _vertex_bound(v_root_names, t_root_names, _state_to_goal_cost(x))

            if (
                node is None
                or node.parent is None
                or node.parent.vertex_name == GCS_SOURCE_NAME
            ):
                return vertex_bound

            predecessor_root_names = stgcs.get_vertex(node.parent.vertex_name).root_name
            incoming_bound = _incoming_bound(predecessor_root_names, v_root_names, t_root_names)
            if not np.isfinite(incoming_bound):
                return vertex_bound

            return max(vertex_bound, incoming_bound)
        
        return _h_func

    @property
    def name(self) -> str:
        return "h_tab"
    
    @property
    def computation_time(self) -> float:
        return self._precomputation_time

    def save(self, filename:str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        if "_pair_target_predecessor" not in self.__dict__:
            self._pair_target_predecessor = {}
        if "_incoming_dist" not in self.__dict__:
            self._incoming_dist = {}
    
    @staticmethod
    def load(filename:str) -> InterfaceToSetCostTableHeuristic:
        with open(filename, 'rb') as f:
            heuristic: InterfaceToSetCostTableHeuristic = pickle.load(f)
        return heuristic
