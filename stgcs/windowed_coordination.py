from __future__ import annotations
from itertools import combinations
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence

import numpy as np
import networkx as nx

from stgcs.pbs import (
    DEFAULT_CHILD_EXPANSION_MODE,
    ChildExpansionMode,
    LowLevelPlanCacheKey,
    PriorityBasedSearch,
    PBSNode,
)
from stgcs.graph import STGCS
from stgcs.trajectory import STTrajectory
from stgcs.st_planner import MPQuery, STPlanner, STPlanStatus


class WindowedPBS(PriorityBasedSearch):
    ENABLE_LOW_LEVEL_REPLAN_REDUCTION: bool = True
    
    def __init__(
        self, stgcs:STGCS, st_planner:STPlanner, robot_radius:float,
        t_horizon: float,
        reached: List[bool],
        child_expansion_mode: ChildExpansionMode = DEFAULT_CHILD_EXPANSION_MODE,
    ) -> None:
        super().__init__(stgcs, st_planner, robot_radius, child_expansion_mode=child_expansion_mode)
        self.t_horizon = t_horizon
        self._reached_prior: List[bool] = list(reached)
        self._window_chunk_cache: dict[Tuple[int, float], STTrajectory] = {}

    def partial_orders(self, i:int, j:int) -> List[Tuple[int, int]]:
        """Return preferred child order for reached-agent priority deferral.

        For LAZY mode this controls stack append order directly (LIFO stack).
        For ranked child-expansion modes, WindowedPBS also applies the same
        preference via `_child_expansion_priority`.
        """
        if self._reached_prior[i] and not self._reached_prior[j]:
            return [(i, j), (j, i)]
        elif not self._reached_prior[i] and self._reached_prior[j]:
            return [(j, i), (i, j)]
        else:
            return super().partial_orders(i, j)

    def _child_expansion_priority(
        self,
        parent: PBSNode,
        child: PBSNode,
        conflict_i: int,
        conflict_j: int,
        ordered_pair: Tuple[int, int],
    ) -> int:
        """Prefer the child where a goal-reached robot yields to an unreached one."""
        i, j = ordered_pair
        if self._reached_prior[i] and not self._reached_prior[j]:
            return 1
        if not self._reached_prior[i] and self._reached_prior[j]:
            return 0
        return 0

    @staticmethod
    def _window_chunk_cache_key(node: PBSNode, agent_idx:int, tf:float) -> Tuple[int, float]:
        return (node.solution_cache_serial(agent_idx), float(tf))

    def _get_window_chunk(
        self,
        traj: STTrajectory,
        tf: Optional[float]=None,
        cache_key: Optional[Tuple[int, float]]=None,
    ) -> STTrajectory:
        """Return trajectory restricted to [0, tf], padded with waiting if needed."""
        tf = self.t_horizon if tf is None else tf
        if cache_key is not None:
            chunk = self._window_chunk_cache.get(cache_key)
            if chunk is not None:
                return chunk

        chunk = traj.get_chunk(tf=tf)
        if chunk.xT[-1] < tf:
            # Keep occupying the last reached position until the window end.
            chunk.vertex_path.append(chunk.vertex_path[-1])
            chunk.points.append(np.hstack([chunk.xT, chunk.xT[:-1], tf]))
        if cache_key is not None:
            self._window_chunk_cache[cache_key] = chunk
        return chunk

    def _get_node_window_chunk(
        self,
        node: PBSNode,
        agent_idx:int,
        tf: Optional[float]=None,
    ) -> STTrajectory:
        resolved_tf = self.t_horizon if tf is None else tf
        return self._get_window_chunk(
            node.sols[agent_idx],
            tf=resolved_tf,
            cache_key=self._window_chunk_cache_key(node, agent_idx, resolved_tf),
        )

    def _reservation_stay_modes(self, agent_idx:int, queries:List[MPQuery]) -> Tuple[bool, bool]:
        del agent_idx, queries
        return False, False

    def _reservation_trajectory_points(
        self,
        node:PBSNode,
        agent_idx:int,
        queries:List[MPQuery],
        reserve_tf: float,
    ) -> List[np.ndarray]:
        del queries
        # Reserve one tiny step beyond the window end to prevent boundary leakage at tf.
        return self._get_node_window_chunk(node, agent_idx, tf=reserve_tf).points

    def _conflicts_with_predecessors(
        self,
        node: PBSNode,
        agent_idx:int,
        high_priority_agents: Sequence[int],
        tf: float,
    ) -> bool:
        if not high_priority_agents:
            return False
        agent_indices = tuple(high_priority_agents) + (agent_idx,)
        points_by_agent = {
            idx: self._get_node_window_chunk(node, idx, tf=tf).points
            for idx in agent_indices
        }
        first_conflict, _ = self._scan_node_pair_conflicts(
            node,
            agent_indices=agent_indices,
            pair_order=((k, agent_idx) for k in high_priority_agents),
            points_by_agent=points_by_agent,
            goal_stay_by_agent={idx: False for idx in agent_indices},
            cc_to_t0=False,
            stop_at_first=True,
        )
        return first_conflict is not None

    def update_node(self, node:PBSNode, idx:int, queries:List[MPQuery]) -> bool:
        replan_list = set([idx] + [j for j in range(len(node.sols)) if nx.has_path(node.prio_graph, idx, j)])
        
        for j in nx.topological_sort(node.prio_graph.subgraph(replan_list)):
            high_priority_agents = self._high_priority_agents(node, j)
            reserve_tf = self.t_horizon + self._stgcs_base.dt
            if self.ENABLE_LOW_LEVEL_REPLAN_REDUCTION:
                replan = self._conflicts_with_predecessors(
                    node,
                    j,
                    high_priority_agents,
                    tf=reserve_tf,
                )
            else:
                replan = j == idx
                if not replan:
                    replan = self._conflicts_with_predecessors(
                        node,
                        j,
                        high_priority_agents,
                        tf=self.t_horizon,
                    )

            if replan:
                safe_radius = 2 * self.robot_radius
                reservation_key = self._reservation_cache_key(
                    node,
                    high_priority_agents,
                    queries,
                    reserve_tf,
                    safe_radius,
                )
                plan_cache_key: Optional[LowLevelPlanCacheKey] = None
                if self.ENABLE_LOW_LEVEL_REPLAN_REDUCTION:
                    plan_cache_key = self._low_level_plan_cache_key(j, queries[j], reservation_key)
                    cached_plan = self._low_level_plan_cache.get(plan_cache_key)
                    if cached_plan is not None:
                        cached_sol, _, solution_cache_serial = cached_plan
                        node.set_solution(j, cached_sol, cache_serial=solution_cache_serial)
                        self.print(f"\t\u2713 Reuses cached plan for {j}.")
                        continue
                    if plan_cache_key in self._structural_failure_cache:
                        self.print(f"\t\u2713 Reuses cached structural failure for {j}.")
                        return False

                stgcs_reserved, ecd_calls, ecd_runtime = self._reserved_stgcs_for_node(
                    node,
                    high_priority_agents,
                    queries,
                    reserve_tf=reserve_tf,
                    reservation_key=reservation_key,
                )
                self._profiler["ecd"] += np.array([ecd_calls, ecd_runtime])

                if self.ENABLE_LOW_LEVEL_REPLAN_REDUCTION:
                    valid_query, _ = stgcs_reserved.validate_query(queries[j])
                    if not valid_query:
                        assert plan_cache_key is not None
                        self._structural_failure_cache.add(plan_cache_key)
                        self.print(f"\t\u2713 Fails to update plan for {j}: structural query infeasibility.")
                        return False

                sol, runtime, status = self.st_planner.plan(stgcs_reserved, queries[j])
                self._record_low_level_profile(runtime)
                
                if status == STPlanStatus.FAIL or sol is None:
                    self.print(f"\t\u2713 Fails to update plan for {j}: {stgcs_reserved.graph_size_string}")
                    return False
                node.set_solution(j, sol)
                if self.ENABLE_LOW_LEVEL_REPLAN_REDUCTION:
                    assert plan_cache_key is not None
                    self._low_level_plan_cache[plan_cache_key] = (
                        sol,
                        status,
                        node.solution_cache_serial(j),
                    )
                node._stgcs_num_edges.append(stgcs_reserved.G.number_of_edges())
                self.print(f"\t\u2713 Succeeds to update plan for {j}: {stgcs_reserved.graph_size_string}")
        return True

    def find_first_conflict(self, num_agents:int, node:PBSNode) -> Tuple[Optional[int], Optional[int]]:
        agent_indices = tuple(range(num_agents))
        points_by_agent = {
            idx: self._get_node_window_chunk(node, idx).points
            for idx in agent_indices
        }
        first_conflict, _ = self._scan_node_pair_conflicts(
            node,
            agent_indices=agent_indices,
            pair_order=combinations(agent_indices, 2),
            points_by_agent=points_by_agent,
            goal_stay_by_agent={idx: False for idx in agent_indices},
            cc_to_t0=False,
            stop_at_first=True,
        )
        if first_conflict is None:
            return None, None
        return first_conflict

    def number_of_conflicts(
        self,
        num_agents:int,
        node:PBSNode,
        max_count: Optional[int]=None,
    ) -> int:
        agent_indices = tuple(range(num_agents))
        points_by_agent = {
            idx: self._get_node_window_chunk(node, idx).points
            for idx in agent_indices
        }
        _, num_conflicts = self._scan_node_pair_conflicts(
            node,
            agent_indices=agent_indices,
            pair_order=combinations(agent_indices, 2),
            points_by_agent=points_by_agent,
            goal_stay_by_agent={idx: False for idx in agent_indices},
            cc_to_t0=False,
            stop_at_first=False,
            max_conflicts=max_count,
        )
        return num_conflicts


class WindowedPP(WindowedPBS):
    """Fixed-order prioritized planning restricted to a finite time window."""

    def __init__(
        self,
        stgcs: STGCS,
        st_planner: STPlanner,
        robot_radius: float,
        t_horizon: float,
        reached: List[bool],
        priority_order: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            stgcs,
            st_planner,
            robot_radius,
            t_horizon=t_horizon,
            reached=reached,
        )
        self.priority_order = None if priority_order is None else tuple(int(idx) for idx in priority_order)

    @staticmethod
    def _resolved_priority_order(
        priority_order: Optional[Sequence[int]],
        num_agents: int,
    ) -> Tuple[int, ...]:
        if priority_order is None:
            return tuple(range(num_agents))
        order = tuple(int(idx) for idx in priority_order)
        if len(order) != num_agents or sorted(order) != list(range(num_agents)):
            raise ValueError(
                "priority_order must be a permutation of all agent indices "
                f"0..{num_agents - 1}, got {order}."
            )
        return order

    @staticmethod
    def _priority_graph(num_agents: int, priority_order: Sequence[int]) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_agents))
        for higher_pos, higher in enumerate(priority_order):
            for lower in priority_order[higher_pos + 1 :]:
                graph.add_edge(int(higher), int(lower))
        return graph

    @staticmethod
    def _empty_solution_node(num_agents: int, stgcs: STGCS, priority_order: Sequence[int]) -> PBSNode:
        node = PBSNode(WindowedPP._priority_graph(num_agents, priority_order))
        for _ in range(num_agents):
            node.append_solution(STTrajectory([], [], stgcs.dimension))
        return node

    def _reset_run_state(self, queries: Sequence[MPQuery]) -> None:
        self.last_solution_prio_graph = None
        self.last_solution_order = tuple()
        self._queries = tuple(queries)
        self.num_popped_nodes = 0
        self.num_generated_children = 0
        self.num_update_calls = 0
        self._collision_geometry_cache.clear()
        self._reserved_stgcs_cache.clear()
        self._low_level_plan_cache.clear()
        self._structural_failure_cache.clear()
        self._ecd_pair_cache.clear()
        self._collision_pair_cache.clear()
        self._window_chunk_cache.clear()

    def _plan_agent(
        self,
        node: PBSNode,
        agent_idx: int,
        high_priority_agents: Sequence[int],
        queries: List[MPQuery],
    ) -> bool:
        reserve_tf = self.t_horizon + self._stgcs_base.dt
        safe_radius = 2 * self.robot_radius
        reservation_key = self._reservation_cache_key(
            node,
            high_priority_agents,
            queries,
            reserve_tf,
            safe_radius,
        )
        stgcs_reserved, ecd_calls, ecd_runtime = self._reserved_stgcs_for_node(
            node,
            high_priority_agents,
            queries,
            reserve_tf=reserve_tf,
            reservation_key=reservation_key,
        )
        self._profiler["ecd"] += np.array([ecd_calls, ecd_runtime])

        sol, runtime, status = self.st_planner.plan(stgcs_reserved, queries[agent_idx])
        self._record_low_level_profile(runtime)
        if status == STPlanStatus.FAIL or sol is None:
            self.print(f"\tFailed to plan agent {agent_idx}: {stgcs_reserved.graph_size_string}")
            return False

        node.set_solution(agent_idx, sol)
        node._stgcs_num_edges.append(stgcs_reserved.G.number_of_edges())
        self.print(f"\tPlanned agent {agent_idx}: {stgcs_reserved.graph_size_string}")
        return True

    def run(
        self,
        queries: List[MPQuery],
        timeout_secs: float,
        verbose: bool = True,
    ) -> Tuple[List[STTrajectory], float, bool]:
        self._verbose = verbose
        start_time = time.perf_counter()
        self._reset_run_state(queries)

        num_agents = len(queries)
        priority_order = self._resolved_priority_order(self.priority_order, num_agents)
        node = self._empty_solution_node(num_agents, self._stgcs_base, priority_order)
        planned_agents: List[int] = []

        self.print(f"\n-> Windowed-PP: initial STGCS: {self._stgcs_base.graph_size_string}")
        for agent_idx in priority_order:
            if time.perf_counter() - start_time > timeout_secs:
                self.print("\n-> Windowed-PP: Timeout.")
                return node.sols, time.perf_counter() - start_time, False
            if not self._plan_agent(node, agent_idx, planned_agents, queries):
                return node.sols, time.perf_counter() - start_time, False
            planned_agents.append(agent_idx)

        ci, cj = self.find_first_conflict(num_agents, node)
        if ci is not None:
            self.print(f"\n-> Windowed-PP: Found conflict between {ci} and {cj}.")
            return node.sols, time.perf_counter() - start_time, False

        self.last_solution_prio_graph = node.prio_graph.copy()
        self.last_solution_order = priority_order
        print(f"\nWindowed-PP: Successfully found a valid set of plans: {priority_order}")
        return node.sols, time.perf_counter() - start_time, True


def adjust_window(
    stgcs: STGCS,
    state: WindowCoordinationState,
    window_span: float,
    tol: float = 1e-9,
) -> bool:

    right_slack = max(0.0, stgcs.tmax - state.right_window)
    grow_right = min(window_span, right_slack)
    new_right = state.right_window + grow_right

    remaining_growth = 2 * window_span - grow_right
    left_slack = max(0.0, state.left_window - state.min_window_start)
    rollback_left = min(remaining_growth, left_slack)
    new_left = state.left_window - rollback_left

    did_rollback = rollback_left > tol
    adjusted = (new_right - state.right_window > tol) or did_rollback
    if not adjusted:
        return False

    state.left_window, state.right_window = new_left, new_right
    if did_rollback:
        for i in range(len(state.queries)):
            if state.solutions[i].size > 0:
                state.solutions[i] = state.solutions[i].get_chunk(tf=state.left_window)
                state.queries[i].start = state.solutions[i].xT[:-1]

    for i in range(len(state.queries)):
        state.queries[i].t_start = 0.0
        state.reached[i] = same_position(state.queries[i].start, state.queries[i].goal)

    # Any horizon change invalidates priority-order oscillation history.
    state.priority_order_history.clear()
    return True


def forward_simulation(
    queries: List[MPQuery],
    sol: Optional[List[STTrajectory]],
    pbs_success: bool,
    t_offset: float,
    t_horizon: float,
) -> Tuple[List[STTrajectory], List[bool]]:
    if (not pbs_success) or sol is None:
        return [], [False for _ in range(len(queries))]

    reached = [False for _ in range(len(queries))]
    partial_sols = []

    for i in range(len(queries)):         
        xt = sol[i].lerp(t_horizon)
        reached[i] = same_position(xt[:-1], queries[i].goal)
        subtraj = sol[i].get_chunk(tf=t_horizon)
        
        # if a robot reached goal earlier, then add waiting action there
        if sol[i].xT[-1] < t_horizon and reached[i]:
            subtraj.vertex_path.append(sol[i].vertex_path[-1])
            subtraj.points.append( np.hstack([subtraj.xT, queries[i].goal, t_horizon]) )

        # pad the subtraj to ensure time continuity
        for j in range(subtraj.size):
            subtraj.xA(j)[-1] += t_offset
            subtraj.xB(j)[-1] += t_offset

        partial_sols.append(subtraj)

    return partial_sols, reached


def stall_detection(
    reached: List[bool],
    priority_graph: Optional[nx.DiGraph],
    priority_order_history: List[PriorityOrderSignature],
    agent_positions: Optional[Sequence[np.ndarray]] = None,
    max_priority_cycle_period: int = 6,
) -> bool:
    """
    Detect periodic priority-order patterns.
    """
    unreached_inds = [i for i, r in enumerate(reached) if not r]
    if len(unreached_inds) == 0:
        assert all(reached)
        return False

    if priority_graph is None or len(unreached_inds) < 2:
        priority_order_history.clear()
        return False

    unreached_tuple = tuple(unreached_inds)
    order_signature = tuple(
        nx.lexicographical_topological_sort(
            priority_graph.subgraph(unreached_inds), key=lambda idx: idx
        )
    )
    position_signature = None
    if agent_positions is not None:
        position_signature = tuple(
            tuple(float(value) for value in np.round(np.asarray(agent_positions[idx], dtype=float).ravel(), 6))
            for idx in unreached_inds
        )
    priority_order_history.append((unreached_tuple, order_signature, position_signature))

    # check for repeated priority orders with the same unreached group, 
    # which indicates cyclic priority changes without progress
    max_period = min(max_priority_cycle_period, len(priority_order_history) // 2)
    for period in range(1, max_period + 1):
        if (
            priority_order_history[-period:]
            == priority_order_history[-2 * period : -period]
        ):
            return True

    return False


def same_position(pa: np.ndarray, pb: np.ndarray, tol: float = 1e-6) -> bool:
    return np.allclose(pa, pb, atol=tol)


PriorityOrderSignature = Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    Optional[Tuple[Tuple[float, ...], ...]],
]


@dataclass
class WindowCoordinationState:
    queries: List[MPQuery]
    solutions: List[STTrajectory]
    committed_solutions: List[STTrajectory]
    priority_order_history: List[PriorityOrderSignature]
    reached: List[bool]
    left_window: float
    right_window: float
    min_window_start: float

    @staticmethod
    def solution_makespan(solutions: Sequence[STTrajectory]) -> float:
        return max((sol.duration for sol in solutions if sol.size > 0), default=0.0)

    def commit_current_solutions(self) -> None:
        if self.solution_makespan(self.solutions) >= self.solution_makespan(self.committed_solutions):
            self.committed_solutions = [sol.copy() for sol in self.solutions]

    def best_partial_solutions(self) -> List[STTrajectory]:
        if self.solution_makespan(self.committed_solutions) >= self.solution_makespan(self.solutions):
            return self.committed_solutions
        return self.solutions


@dataclass
class WindowedCoordinationReturn:
    success: bool = False
    solutions: List[STTrajectory] = field(default_factory=list)
    mp: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    gcs: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    gub: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    search: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    cr: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    dc: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    dc_cr: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    ecd: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    cc: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    wpbs: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    wpp: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    runtime: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    pbs_popped_nodes: int = 0
    pbs_generated_children: int = 0
    pbs_update_calls: int = 0

    def __repr__(self):
        status = "success" if self.success else "partial-or-failed"
        return "WindowedCoordinationReturn:\n" + \
                f"\tStatus: {status}\n" + \
                f"\t# trajectories: {len(self.solutions)}\n" + \
                f"\tRuntime: {self.runtime[0]:.4f} secs\n" + \
                f"\tWPBS: # calls={self.wpbs[0]:.0f}, cum-time={self.wpbs[1]:.4f} secs\n" + \
                f"\tWPP: # calls={self.wpp[0]:.0f}, cum-time={self.wpp[1]:.4f} secs\n" + \
                f"\tWPBS nodes: popped={self.pbs_popped_nodes}, children={self.pbs_generated_children}, updates={self.pbs_update_calls}\n" + \
                f"\tMP: # calls={self.mp[0]:.0f}, cum-time={self.mp[1]:.4f} secs\n" + \
                f"\tGCS build: # calls={self.gcs[0]:.0f}, cum-time={self.gcs[1]:.4f} secs\n" + \
                f"\tGUB: # calls={self.gub[0]:.0f}, cum-time={self.gub[1]:.4f} secs\n" + \
                f"\tSearch: # calls={self.search[0]:.0f}, cum-time={self.search[1]:.4f} secs\n" + \
                f"\tConvex restriction: # calls={self.cr[0]:.0f}, cum-time={self.cr[1]:.4f} secs\n" + \
                f"\tDominance check: # calls={self.dc[0]:.0f}, cum-time={self.dc[1]:.4f} secs\n" + \
                f"\tDominance CR: # calls={self.dc_cr[0]:.0f}, cum-time={self.dc_cr[1]:.4f} secs\n" + \
                f"\tECD: # calls={self.ecd[0]:.0f}, cum-time={self.ecd[1]:.4f} secs\n" + \
                f"\tCC: # calls={self.cc[0]:.0f}, cum-time={self.cc[1]:.4f} secs\n"
