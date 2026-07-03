from __future__ import annotations
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple
from dataclasses import dataclass

import time
import heapq
import itertools
import numpy as np

from stgcs.graph import STGCS
from stgcs.gcs_solver import GCS_SOURCE_NAME, GCS_TARGET_NAME, EDGE_KEY, GCS, MPGCSInstance
from stgcs.gcs_solver import solve_convex_restriction
from stgcs.geometry_utils import HPolyhedronSampler
from stgcs.trajectory import STTrajectory

from pydrake.all import HPolyhedron, Point as DrakePoint, RandomGenerator


import logging
logger = logging.getLogger(__name__)


class TieBreak(Enum):
    FIFO = 1
    LIFO = 2


class DominationChecker(ABC):

    @abstractmethod
    def reset(self) -> None:
        """ Reset the internal state of the domination checker. """
        raise NotImplementedError("reset method not implemented.")

    @abstractmethod
    def check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:
        """ Returns True if the node is dominated, False otherwise. 
            Only add the node if it is not dominated. """
        raise NotImplementedError("check method not implemented.")
    
    @property
    def name(self) -> str:
        """ Returns the name of the domination checker. """
        raise NotImplementedError("name property not implemented.")

    @property
    def precomputation_time(self) -> float:
        """ Returns the computation time of the domination checker. """
        raise NotImplementedError("precomputation_time property not implemented.")

    def is_stale(self, n: SearchNode) -> bool:
        """Returns True if the node has been superseded by a better node."""
        return False

    def set_runtime_context(self, solver_options: object, edge_cache: Dict[Tuple[str, str], Any]) -> None:
        """Attach per-search reusable objects used by expensive checks."""
        del solver_options, edge_cache

    @property
    def convex_restriction_time(self) -> float:
        return 0.0

    @property
    def convex_restriction_calls(self) -> int:
        return 0


class Heuristic(ABC):
    """ Abstract base class for heuristic funciton, defined on root STVertex """

    @abstractmethod
    def get(self, stgcs:STGCS, gcs:GCS, source:str, target:str) -> Callable[[str, np.ndarray, STGCS, Optional[SearchNode]], float]:
        raise NotImplementedError("get method not implemented.")
    
    @property
    def name(self) -> str:
        raise NotImplementedError("name property not implemented.")

    @property
    def computation_time(self) -> float:
        raise NotImplementedError("computation_time property not implemented.")


class SearchNode:
    """A node in the search tree."""

    def __init__(
        self,
        f: Optional[float],
        vertex_name: str,
        edge_path: Optional[Sequence[str]] = None,
        vertex_path: Optional[Sequence[str]] = None,
        parent: Optional[SearchNode] = None,
        sol: Optional[STTrajectory] = None,
        successor_name: Optional[str] = None,
        g: Optional[float] = None,
        h: Optional[float] = None,
        trace_id: Optional[int] = None,
    ) -> None:
        self.f = f
        self.vertex_name = vertex_name
        self.parent = parent
        self.sol = sol
        self.successor_name = successor_name
        self.g = g
        self.h = h
        self.trace_id = trace_id

        if vertex_path is not None:
            self._vertex_path_tuple: Optional[Tuple[str, ...]] = tuple(vertex_path)
        elif parent is None:
            self._vertex_path_tuple = (vertex_name,)
        else:
            self._vertex_path_tuple = None

        if edge_path is not None:
            self._edge_path_tuple: Optional[Tuple[str, ...]] = tuple(edge_path)
        elif parent is None:
            self._edge_path_tuple = ()
        else:
            self._edge_path_tuple = None

        if vertex_path is not None:
            self._visited_vertices: FrozenSet[str] = frozenset(vertex_path)
        elif parent is None:
            self._visited_vertices = frozenset({vertex_name})
        else:
            self._visited_vertices = parent._visited_vertices | {vertex_name}

    def __lt__(self, other: SearchNode):
        return self.f < other.f

    @property
    def vertex_path_tuple(self) -> Tuple[str, ...]:
        if self._vertex_path_tuple is None:
            if self.parent is None:
                self._vertex_path_tuple = (self.vertex_name,)
            else:
                self._vertex_path_tuple = self.parent.vertex_path_tuple + (self.vertex_name,)
        return self._vertex_path_tuple

    @property
    def edge_path_tuple(self) -> Tuple[str, ...]:
        if self._edge_path_tuple is None:
            if self.parent is None:
                self._edge_path_tuple = ()
            else:
                self._edge_path_tuple = self.parent.edge_path_tuple + (
                    EDGE_KEY(self.parent.vertex_name, self.vertex_name),
                )
        return self._edge_path_tuple

    @property
    def vertex_path(self) -> List[str]:
        return list(self.vertex_path_tuple)

    @vertex_path.setter
    def vertex_path(self, path: Sequence[str]) -> None:
        self._vertex_path_tuple = tuple(path)
        self._visited_vertices = frozenset(path)

    @property
    def edge_path(self) -> List[str]:
        return list(self.edge_path_tuple)

    @edge_path.setter
    def edge_path(self, path: Sequence[str]) -> None:
        self._edge_path_tuple = tuple(path)

    def has_visited(self, vertex_name: str) -> bool:
        return vertex_name in self._visited_vertices

    @classmethod
    def from_source(
        cls, source_vertex_name: str, 
        successor_name:Optional[str]=None
    ) -> SearchNode:
        return cls(
            f=0,
            vertex_name=source_vertex_name,
            edge_path=[],
            vertex_path=[source_vertex_name],
            successor_name=successor_name,
            g=0.0,
            h=0.0,
        )

    @classmethod
    def from_parent(
        cls, child_vertex_name: str, parent: SearchNode, 
        successor_name:Optional[str]=None
    ) -> SearchNode:
        return cls(
            f=None,
            vertex_name=child_vertex_name,
            parent=parent,
            successor_name=successor_name,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sol"] = None  # Do not serialize `sol`
        return state

    def __setstate__(self, state):
        vertex_path = state.pop("vertex_path", None)
        edge_path = state.pop("edge_path", None)
        self.__dict__.update(state)
        if "_vertex_path_tuple" not in self.__dict__:
            if vertex_path is not None:
                self._vertex_path_tuple = tuple(vertex_path)
            elif self.parent is None:
                self._vertex_path_tuple = (self.vertex_name,)
            else:
                self._vertex_path_tuple = None
        if "_edge_path_tuple" not in self.__dict__:
            if edge_path is not None:
                self._edge_path_tuple = tuple(edge_path)
            elif self.parent is None:
                self._edge_path_tuple = ()
            else:
                self._edge_path_tuple = None
        if "_visited_vertices" not in self.__dict__:
            self._visited_vertices = frozenset(self.vertex_path_tuple)

    def __repr__(self):
        return (
            f"SearchNode(f={self.f:.3f}, vertex={self.vertex_name}, "
            f"successor={self.successor_name}, "
            f"path={self.vertex_path})"
        )


@dataclass
class SearchTraceNode:
    trace_id: int
    parent_id: Optional[int]
    vertex_name: str
    vertex_path: List[str]
    successor_name: Optional[str]
    f: Optional[float] = None
    g: Optional[float] = None
    h: Optional[float] = None
    end_state: Optional[List[float]] = None
    trajectory_points: Optional[List[List[float]]] = None
    status: str = "open"
    generated_order: Optional[int] = None
    expanded_order: Optional[int] = None

    @classmethod
    def from_search_node(cls, trace_id: int, generated_order: int, node: SearchNode) -> SearchTraceNode:
        return cls(
            trace_id=trace_id,
            parent_id=node.parent.trace_id if node.parent is not None else None,
            vertex_name=node.vertex_name,
            vertex_path=list(node.vertex_path_tuple),
            successor_name=node.successor_name,
            f=node.f,
            g=node.g,
            h=node.h,
            end_state=None if node.sol is None else node.sol.xT.tolist(),
            trajectory_points=cls._trajectory_points(node),
            generated_order=generated_order,
        )

    def update_from_search_node(self, node: SearchNode, status: Optional[str] = None) -> None:
        self.f = node.f
        self.g = node.g
        self.h = node.h
        self.end_state = None if node.sol is None else node.sol.xT.tolist()
        self.trajectory_points = self._trajectory_points(node)
        if status is not None:
            self.status = status

    @staticmethod
    def _trajectory_points(node: SearchNode) -> Optional[List[List[float]]]:
        if node.sol is None:
            return None
        return [point.tolist() for point in node.sol.points]

    @staticmethod
    def _json_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    def to_dict(self) -> Dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "vertex_name": self.vertex_name,
            "vertex_path": self.vertex_path,
            "successor_name": self.successor_name,
            "f": self._json_float(self.f),
            "g": self._json_float(self.g),
            "h": self._json_float(self.h),
            "end_state": self.end_state,
            "trajectory_points": self.trajectory_points,
            "status": self.status,
            "generated_order": self.generated_order,
            "expanded_order": self.expanded_order,
        }


class SearchTrace:

    def __init__(self) -> None:
        self._next_trace_id = 0
        self._generated_order = 0
        self._expanded_order = 0
        self._nodes: Dict[int, SearchTraceNode] = {}
        self.root_id: Optional[int] = None
        self.goal_id: Optional[int] = None
        self.termination_reason: Optional[str] = None

    def register_node(self, node: SearchNode) -> None:
        trace_id = self._next_trace_id
        self._next_trace_id += 1
        node.trace_id = trace_id
        trace_node = SearchTraceNode.from_search_node(trace_id, self._generated_order, node)
        self._generated_order += 1
        self._nodes[trace_id] = trace_node
        if self.root_id is None:
            self.root_id = trace_id

    def update_node(self, node: SearchNode, status: Optional[str] = None) -> None:
        if node.trace_id is None:
            return
        self._nodes[node.trace_id].update_from_search_node(node, status=status)

    def mark_expanded(self, node: SearchNode) -> None:
        if node.trace_id is None:
            return
        trace_node = self._nodes[node.trace_id]
        trace_node.update_from_search_node(node, status="expanded")
        trace_node.expanded_order = self._expanded_order
        self._expanded_order += 1

    def mark_goal(self, node: SearchNode) -> None:
        if node.trace_id is None:
            return
        self.goal_id = node.trace_id
        self._nodes[node.trace_id].update_from_search_node(node, status="goal")

    def finalize(self, reason: str) -> None:
        self.termination_reason = reason

    def to_dict(self) -> Dict[str, object]:
        children: Dict[int, List[int]] = {trace_id: [] for trace_id in self._nodes}
        for trace_node in self._nodes.values():
            if trace_node.parent_id is not None:
                children.setdefault(trace_node.parent_id, []).append(trace_node.trace_id)
        for trace_id in children:
            children[trace_id].sort(key=lambda child_id: self._nodes[child_id].generated_order or 0)

        nodes = []
        for trace_node in sorted(self._nodes.values(), key=lambda item: item.generated_order or 0):
            payload = trace_node.to_dict()
            payload["children_ids"] = children.get(trace_node.trace_id, [])
            nodes.append(payload)

        return {
            "root_id": self.root_id,
            "goal_id": self.goal_id,
            "termination_reason": self.termination_reason,
            "nodes": nodes,
        }
    

ConvexRestrictionCache = Dict[Tuple[str, ...], Optional[STTrajectory]]
_CACHE_MISS = object()


class SearchAlgorithm:

    """ Search algorithm for Spatiotemporal Planning on ST-GCS 
    
    - Example Usage:
        - For GCS*-like (1-sampling + Shortcut heuristic):
            gcs_star = SearchAlgorithm(
                        heuristics=HeurShortCut(),
                        domination_checker=Sampling_DC()
                    )
        - For IxG-like:
            gcs = stgcs.get_gcs_instance().gcs
            ixg = SearchAlgorithm(
                    heuristics=HeurLowerBoundGraph(stgcs, gcs),
                    domination_checker=AStar_DC()
                )

            ixg_star = SearchAlgorithm(
                heuristics = lbg_no_update,
                domination_checker = GlobalUpperBound_DC(
                    ub=cost, ub_comp_time=astar.runtime, epsilon=eps)
            )
    """

    def __init__(
        self,
        heuristics: Heuristic,
        heuristic_inflation_factor: float = 1.0,
        domination_checker: Optional[List[DominationChecker]] = None,
        tiebreak: TieBreak = TieBreak.FIFO,
        record_trace: bool = False,
        convex_restriction_cache: Optional[ConvexRestrictionCache] = None,
    ):
        self._ts = time.perf_counter()
        self._domination_check_time = 0.0
        self._convex_restriction_time = 0.0
        self._convex_restriction_calls = 0
        self._timeout_seconds = float('inf')
        self._OPEN = []
        self._heur: Heuristic = heuristics
        self._epsilon = heuristic_inflation_factor
        self._dc: List[DominationChecker] = [] if domination_checker is None else domination_checker
        self._convex_restriction_cache: ConvexRestrictionCache = (
            {} if convex_restriction_cache is None else convex_restriction_cache
        )
        self._restriction_solver_options = MPGCSInstance.default_solver_options()
        self._edge_cache: Dict[Tuple[str, str], Any] = {}

        self._tiebreak = tiebreak
        self._record_trace = record_trace
        self.trace: Optional[SearchTrace] = None
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)

    def set_convex_restriction_cache(self, cache: ConvexRestrictionCache) -> None:
        self._convex_restriction_cache = cache

    def reset_domination_checkers(self) -> None:
        for checker in self._dc:
            checker.reset()

    @staticmethod
    def _convex_restriction_cache_key(vertex_path: Sequence[str]) -> Tuple[str, ...]:
        path_key = vertex_path if isinstance(vertex_path, tuple) else tuple(vertex_path)
        if path_key[-1] == GCS_TARGET_NAME:
            return path_key[:-1]
        return path_key

    def _solve_convex_restriction(self, gcs: GCS, vertex_path: Sequence[str]) -> Optional[STTrajectory]:
        cache_key = self._convex_restriction_cache_key(vertex_path)
        cached = self._convex_restriction_cache.get(cache_key, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached

        ts = time.perf_counter()
        sol = solve_convex_restriction(
            gcs,
            list(cache_key),
            options=self._restriction_solver_options,
            edge_cache=self._edge_cache,
        )
        self._convex_restriction_time += time.perf_counter() - ts
        self._convex_restriction_calls += 1
        self._convex_restriction_cache[cache_key] = sol
        return sol

    def _push_open(self, node: SearchNode) -> None:
        if node.f is None:
            raise ValueError(f"Cannot push search node without an f-value: {node.vertex_name}")
        heapq.heappush(self._OPEN, (node.f, next(self._counter), node))

    def _pop_open(self) -> SearchNode:
        return heapq.heappop(self._OPEN)[2]

    def run(
        self, stgcs:STGCS, gcs:GCS, 
        source:str=GCS_SOURCE_NAME, target:str=GCS_TARGET_NAME,
        timeout_seconds:float=float('inf')
    ) -> Optional[STTrajectory]:
    
        self._ts = time.perf_counter()
        self._timeout_seconds = timeout_seconds
        self._planning_time = time.perf_counter() - self._ts
        self.n_expanded = 0
        self.n_generated = 0
        self._domination_check_time = 0.0
        self._convex_restriction_time = 0.0
        self._convex_restriction_calls = 0
        self._edge_cache = {}
        for checker in self._dc:
            checker.set_runtime_context(self._restriction_solver_options, self._edge_cache)
        self.trace = SearchTrace() if self._record_trace else None

        if source == target:
            logger.debug("Source is the same as target.")
            vertex_set = stgcs.get_vertex(source).st_hpoly
            if isinstance(vertex_set, DrakePoint):
                pt = vertex_set.x()
            elif isinstance(vertex_set, HPolyhedron):
                pt = HPolyhedronSampler.uniform_sample(
                    vertex_set,
                    RandomGenerator(),
                    context=f"SearchAlgorithm.run[{source}]",
                )
            else:
                raise NotImplementedError(f"Unsupported source set type {type(vertex_set)}")
            root = SearchNode.from_source(source)
            root.sol = STTrajectory([source], [np.tile(pt, 2)], stgcs.dimension)
            root.g = 0.0
            root.h = 0.0
            if self.trace is not None:
                self.trace.register_node(root)
                self.trace.mark_goal(root)
                self.trace.finalize("goal")
            return root.sol

        h_func = self._heur.get(stgcs, gcs, source, target)
        root = SearchNode.from_source(source)
        if self.trace is not None:
            self.trace.register_node(root)
        self._OPEN = []
        self._push_open(root)

        while len(self._OPEN) > 0:
            n: SearchNode = self._pop_open()
            self._planning_time = time.perf_counter() - self._ts
            if self._planning_time >= self._timeout_seconds:
                if self.trace is not None:
                    self.trace.finalize("timeout")
                break
            # Skip OPEN entries superseded by a cheaper same-vertex node.
            if any(checker.is_stale(n) for checker in self._dc):
                if self.trace is not None:
                    self.trace.update_node(n, status="dominated")
                continue
            if self.trace is not None:
                self.trace.mark_expanded(n)
            sol = self.goal_condition(n, gcs, target)

            self._planning_time = time.perf_counter() - self._ts
            if self._planning_time >= self._timeout_seconds:
                if self.trace is not None:
                    self.trace.finalize("timeout")
                break

            if sol is not None:
                n.sol = sol
                n.g = sol.duration
                if self.trace is not None:
                    self.trace.mark_goal(n)
                    self.trace.finalize("goal")
                return sol
            
            self.expand_node(n, h_func, stgcs, gcs)

        if self.trace is not None and self.trace.termination_reason is None:
            self.trace.finalize("exhausted")
    
    def goal_condition(self, n:SearchNode, gcs:GCS, target:str) -> Optional[STTrajectory]:
        if n.vertex_name == target:
            if n.sol is None:
                return self._solve_convex_restriction(gcs, n.vertex_path_tuple)
            return n.sol

    def expand_node(
        self,
        n:SearchNode,
        h:Callable[[str, np.ndarray, STGCS, Optional[SearchNode]], float],
        stgcs:STGCS,
        gcs:GCS,
    ) -> None:
        # print(f"Expanding node: {n}")
        for successor in stgcs.G.successors(n.vertex_name):
            if time.perf_counter() - self._ts >= self._timeout_seconds:
                break

            if n.has_visited(successor):
                logger.debug(f"Cycle detected: {n.vertex_path} -> {successor}")
                continue

            n_next = SearchNode.from_parent(child_vertex_name=successor, parent=n)
            if self.trace is not None:
                self.trace.register_node(n_next)
            n_next.sol = self._solve_convex_restriction(gcs, n_next.vertex_path_tuple)
            if n_next.sol is None:
                n_next.f = float("inf")   # infeasible path
                n_next.g = float("inf")
                if self.trace is not None:
                    self.trace.update_node(n_next, status="infeasible")
            else:
                n_next.g = n_next.sol.duration
                n_next.h = h(n_next.vertex_name, n_next.sol.xT, stgcs, n_next)
                n_next.f = n_next.g + self._epsilon * n_next.h
                is_dominated = self.domination_check(n_next, stgcs, gcs)
                if not is_dominated or "target" in successor:
                    self._push_open(n_next)
                    self.n_generated += 1
                    if self.trace is not None:
                        self.trace.update_node(n_next, status="open")
                elif self.trace is not None:
                    self.trace.update_node(n_next, status="dominated")
        
        self.n_expanded += 1

    def domination_check(self, n:SearchNode, stgcs:STGCS, gcs:GCS) -> bool:
        ts = time.perf_counter()
        ret = False
        for checker in self._dc:
            if checker.check(n, stgcs, gcs):
                logger.debug(f"Dominated by {checker.__class__.__name__}. Path: {n.vertex_path}")                 
                ret = True
                break

        self._domination_check_time += time.perf_counter() - ts
        return ret

    @property
    def runtime(self) -> float:
        return self._planning_time + sum(dc.precomputation_time for dc in self._dc)

    @property
    def dc_time(self) -> float:
        return self._domination_check_time

    @property
    def convex_restriction_time(self) -> float:
        return self._convex_restriction_time

    @property
    def convex_restriction_calls(self) -> int:
        return self._convex_restriction_calls

    @property
    def domination_convex_restriction_time(self) -> float:
        return float(sum(checker.convex_restriction_time for checker in self._dc))

    @property
    def domination_convex_restriction_calls(self) -> int:
        return int(sum(checker.convex_restriction_calls for checker in self._dc))
