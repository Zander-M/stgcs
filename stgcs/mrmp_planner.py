from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

import numpy as np

from stgcs.graph import STGCS
from stgcs.pbs import DEFAULT_CHILD_EXPANSION_MODE, ChildExpansionMode, PriorityBasedSearch
from stgcs.st_planner import MPQuery, STPlanStatus, STPlanner
from stgcs.trajectory import STTrajectory
from stgcs.windowed_coordination import (
    WindowCoordinationState,
    WindowedCoordinationReturn,
    WindowedPBS,
    WindowedPP,
    adjust_window,
    forward_simulation,
    stall_detection,
)


@dataclass
class MRMPQuery(Sequence[MPQuery]):
    queries: List[MPQuery]

    @classmethod
    def from_queries(cls, queries: MRMPQuery | Sequence[MPQuery]) -> MRMPQuery:
        if isinstance(queries, cls):
            return queries
        return cls(list(queries))

    def copy(self) -> MRMPQuery:
        return MRMPQuery(deepcopy(self.queries))

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[MPQuery]:
        return iter(self.queries)

    def __getitem__(self, index: int | slice) -> MPQuery | List[MPQuery]:
        return self.queries[index]

    def __repr__(self) -> str:
        return f"MRMPQuery(num_agents={len(self.queries)})"


def pp(
    stgcs: STGCS,
    st_planner: STPlanner,
    queries: MRMPQuery | Sequence[MPQuery],
    robot_radius: float,
    timeout_secs: float = np.inf,
    priority_order: Optional[Sequence[int]] = None,
    pp_verbose: bool = False,
) -> Tuple[List[STTrajectory], WindowedCoordinationReturn]:
    start_time = time.perf_counter()
    queries = MRMPQuery.from_queries(queries).copy().queries
    num_agents = len(queries)
    priority_order = WindowedPP._resolved_priority_order(priority_order, num_agents)
    runner = PriorityBasedSearch(stgcs, st_planner, robot_radius)
    runner._verbose = pp_verbose
    runner.last_solution_prio_graph = None
    runner.last_solution_order = tuple()
    runner._queries = tuple(queries)
    node = WindowedPP._empty_solution_node(num_agents, stgcs, priority_order)
    result = WindowedCoordinationReturn()
    planned_agents: List[int] = []

    def finish(success: bool) -> Tuple[List[STTrajectory], WindowedCoordinationReturn]:
        runtime = time.perf_counter() - start_time
        result.success = bool(success)
        result.runtime = np.array([runtime])
        result.wpp = np.array([1.0, runtime])
        result.mp = runner._profiler["mp"].copy()
        result.gcs = runner._profiler["gcs"].copy()
        result.gub = runner._profiler["gub"].copy()
        result.search = runner._profiler["search"].copy()
        result.cr = runner._profiler["cr"].copy()
        result.dc = runner._profiler["dc"].copy()
        result.dc_cr = runner._profiler["dc_cr"].copy()
        result.ecd = runner._profiler["ecd"].copy()
        result.cc = runner._profiler["cc"].copy()
        result.pbs_popped_nodes = int(runner.num_popped_nodes)
        result.pbs_generated_children = int(runner.num_generated_children)
        result.pbs_update_calls = int(runner.num_update_calls)
        result.solutions = [sol.copy() for sol in node.sols]
        return node.sols, result

    runner.print(f"\n-> PP: initial STGCS: {runner._stgcs_base.graph_size_string}")
    for agent_idx in priority_order:
        if time.perf_counter() - start_time > timeout_secs:
            runner.print("\n-> PP: Timeout.")
            return finish(False)

        reserve_tf = runner.tmax
        safe_radius = 2 * runner.robot_radius
        reservation_key = runner._reservation_cache_key(
            node,
            planned_agents,
            queries,
            reserve_tf,
            safe_radius,
        )
        stgcs_reserved, ecd_calls, ecd_runtime = runner._reserved_stgcs_for_node(
            node,
            planned_agents,
            queries,
            reserve_tf=reserve_tf,
            reservation_key=reservation_key,
        )
        runner._profiler["ecd"] += np.array([ecd_calls, ecd_runtime])

        sol, runtime, status = runner.st_planner.plan(stgcs_reserved, queries[agent_idx])
        runner._record_low_level_profile(runtime)
        if status == STPlanStatus.FAIL or sol is None:
            runner.print(f"\tFailed to plan agent {agent_idx}: {stgcs_reserved.graph_size_string}")
            return finish(False)

        node.set_solution(agent_idx, sol)
        node._stgcs_num_edges.append(stgcs_reserved.G.number_of_edges())
        runner.print(f"\tPlanned agent {agent_idx}: {stgcs_reserved.graph_size_string}")
        planned_agents.append(agent_idx)

    ci, cj = runner.find_first_conflict(num_agents, node)
    if ci is not None:
        runner.print(f"\n-> PP: Found conflict between {ci} and {cj}.")
        return finish(False)

    runner.last_solution_prio_graph = node.prio_graph.copy()
    runner.last_solution_order = priority_order
    print(f"\nPP: Successfully found a valid set of plans: {priority_order}")
    return finish(True)


def windowed_pbs(
    stgcs: STGCS,
    st_planner: STPlanner,
    queries: MRMPQuery | Sequence[MPQuery],
    robot_radius: float,
    window_span: float=None,
    execution_horizon_factor: float = 1.0,
    timeout_secs: float=np.inf,
    dynamic_window_adjustment: bool = True,
    child_expansion_mode: ChildExpansionMode = DEFAULT_CHILD_EXPANSION_MODE,
    wpbs_verbose: bool = False,
) -> Tuple[List[STTrajectory], WindowedCoordinationReturn]:

    ts = time.perf_counter()
    queries = MRMPQuery.from_queries(queries).copy().queries
    N = len(queries)
    execution_horizon_factor = float(execution_horizon_factor)
    if (
        not np.isfinite(execution_horizon_factor)
        or execution_horizon_factor < 0.0
        or execution_horizon_factor > 1.0
    ):
        raise ValueError("execution_horizon_factor must be finite and in [0, 1].")
    query_t_starts = np.array([q.t_start for q in queries], dtype=float)
    window_span = 5 * robot_radius / stgcs.vlimit if window_span is None else window_span
    min_window_start = float(np.min(query_t_starts))
    if not np.allclose(query_t_starts, min_window_start, atol=1e-9):
        raise ValueError(
            "windowed_pbs currently requires synchronized starts "
            "(all queries must share the same t_start)."
        )
    for q in queries:
        q.t_start = 0.0

    state = WindowCoordinationState(
        queries=queries,
        solutions=[STTrajectory([], [], stgcs.dimension) for _ in range(N)],
        committed_solutions=[STTrajectory([], [], stgcs.dimension) for _ in range(N)],
        priority_order_history=[],
        reached=[False for _ in range(N)],
        left_window=min_window_start,
        right_window=min(stgcs.tmax, min_window_start + window_span),
        min_window_start=min_window_start,
    )
    result = WindowedCoordinationReturn()

    while time.perf_counter() - ts < timeout_secs:
        print(f"-> Windowed-PBS: window = [{state.left_window:.2f} -> {state.right_window:.2f}].")
        dist_to_goal_entries = []
        for i, is_reached in enumerate(state.reached):
            if is_reached:
                dist_to_goal_entries.append("✓")
            else:
                dist_to_goal = np.linalg.norm(state.queries[i].start - state.queries[i].goal)
                dist_to_goal_entries.append(f"{dist_to_goal:.2f}")
        compact_entry_separator = ",\t"
        for start in range(0, len(dist_to_goal_entries), 10):
            chunk = dist_to_goal_entries[start:start + 10]
            print(f"\t[{compact_entry_separator.join(chunk)}]")

        window_queries = deepcopy(state.queries)
        planning_horizon = state.right_window - state.left_window
        terminal_window = state.right_window >= stgcs.tmax - 1e-9
        # Once the window is clamped at tmax, there is no future lookahead to preserve.
        # Execute the whole terminal window instead of taking an asymptotic beta tail.
        execution_horizon = (
            planning_horizon
            if terminal_window and execution_horizon_factor > 0.0
            else execution_horizon_factor * planning_horizon
        )
        if execution_horizon <= 1e-9 and not all(state.reached):
            print("\t\u2717 Windowed-PBS cannot advance with a zero execution horizon.")
            break
        for i, query in enumerate(window_queries):
            t_earliest_arrival = query.t_start + np.max(np.abs(query.goal - query.start) / query.vlimit)
            # Any agent that can finish within this window will be padded as occupying its goal
            # through the window end, so the low-level query must enforce stay semantics too.
            query.is_stay = bool(state.reached[i] or t_earliest_arrival <= planning_horizon + 1e-9)

        wpbs = WindowedPBS(
            stgcs, st_planner, robot_radius,
            t_horizon = planning_horizon,
            reached = state.reached,
            child_expansion_mode=child_expansion_mode,
        )
        pbs_sols, runtime, pbs_success = wpbs.run(
            window_queries,
            timeout_secs - (time.perf_counter() - ts),
            verbose=wpbs_verbose,
        )

        result.wpbs += np.array([1, runtime])
        result.mp += wpbs._profiler["mp"]
        result.gcs += wpbs._profiler["gcs"]
        result.gub += wpbs._profiler["gub"]
        result.search += wpbs._profiler["search"]
        result.cr += wpbs._profiler["cr"]
        result.dc += wpbs._profiler["dc"]
        result.dc_cr += wpbs._profiler["dc_cr"]
        result.ecd += wpbs._profiler["ecd"]
        result.cc += wpbs._profiler["cc"]
        result.pbs_popped_nodes += wpbs.num_popped_nodes
        result.pbs_generated_children += wpbs.num_generated_children
        result.pbs_update_calls += wpbs.num_update_calls

        partial_sols, state.reached = forward_simulation(
            state.queries,
            pbs_sols,
            pbs_success=pbs_success,
            t_offset=state.left_window,
            t_horizon=execution_horizon,
        )
        if len(partial_sols) == 0 or stall_detection(
                state.reached,
                priority_graph=wpbs.last_solution_prio_graph,
                priority_order_history=state.priority_order_history,
                agent_positions=[partial_sol.xT[:-1] for partial_sol in partial_sols],
            ):
            print("\t\u2717 Windowed-PBS failed to find partial solution for current window or stall detected. ")
            if not dynamic_window_adjustment:
                print("\t\u2717 Failed to adjust window since dynamic adjustment is disabled.")
                break
            elif not adjust_window(stgcs, state, window_span):
                print("\t\u2717 Failed to adjust window since it is already at the maximum size. Returning failure.")
                break
            print("\t\u2713 Adjusted window for next iteration: "
                  f"window = [{state.left_window:.2f} -> {state.right_window:.2f}].")
        else:
            # proceed
            for i, partial_sol in enumerate(partial_sols):
                state.solutions[i].append_chunk(partial_sol)
                state.queries[i].t_start = 0.0
                state.queries[i].start = state.solutions[i].xT[:-1]
            state.commit_current_solutions()

            # return if all agents reached goals
            if all(state.reached):
                print("-> All agents reached goals!")
                result.runtime = np.array([time.perf_counter() - ts])
                result.success = True
                result.solutions = [sol.copy() for sol in state.solutions]
                return state.solutions, result

            state.left_window = state.left_window + execution_horizon
            state.right_window = min(stgcs.tmax, state.left_window + window_span)
            if not (state.right_window > state.left_window + 1e-9):
                break

            print(
                f"\t\u2713 Windowed-PBS succeeded. Advancing by execution horizon "
                f"{execution_horizon:.2f} (planning={planning_horizon:.2f}, beta={execution_horizon_factor:g})."
            )

    print("\t\u2713 Windowed-PBS failed to find solution.")
    result.runtime = np.array([time.perf_counter() - ts])
    result.success = False
    failure_solutions = state.best_partial_solutions()
    result.solutions = [sol.copy() for sol in failure_solutions]
    return failure_solutions, result


def windowed_pp(
    stgcs: STGCS,
    st_planner: STPlanner,
    queries: MRMPQuery | Sequence[MPQuery],
    robot_radius: float,
    window_span: float = None,
    execution_horizon_factor: float = 1.0,
    timeout_secs: float = np.inf,
    dynamic_window_adjustment: bool = True,
    priority_order: Optional[Sequence[int]] = None,
    wpp_verbose: bool = False,
) -> Tuple[List[STTrajectory], WindowedCoordinationReturn]:
    start_time = time.perf_counter()
    queries = MRMPQuery.from_queries(queries).copy().queries
    num_agents = len(queries)
    priority_order = WindowedPP._resolved_priority_order(priority_order, num_agents)
    execution_horizon_factor = float(execution_horizon_factor)
    if (
        not np.isfinite(execution_horizon_factor)
        or execution_horizon_factor < 0.0
        or execution_horizon_factor > 1.0
    ):
        raise ValueError("execution_horizon_factor must be finite and in [0, 1].")

    query_t_starts = np.array([query.t_start for query in queries], dtype=float)
    window_span = 5 * robot_radius / stgcs.vlimit if window_span is None else window_span
    min_window_start = float(np.min(query_t_starts))
    if not np.allclose(query_t_starts, min_window_start, atol=1e-9):
        raise ValueError(
            "windowed_pp currently requires synchronized starts "
            "(all queries must share the same t_start)."
        )
    for query in queries:
        query.t_start = 0.0

    state = WindowCoordinationState(
        queries=queries,
        solutions=[STTrajectory([], [], stgcs.dimension) for _ in range(num_agents)],
        committed_solutions=[STTrajectory([], [], stgcs.dimension) for _ in range(num_agents)],
        priority_order_history=[],
        reached=[False for _ in range(num_agents)],
        left_window=min_window_start,
        right_window=min(stgcs.tmax, min_window_start + window_span),
        min_window_start=min_window_start,
    )
    result = WindowedCoordinationReturn()

    while time.perf_counter() - start_time < timeout_secs:
        print(f"-> Windowed-PP: window = [{state.left_window:.2f} -> {state.right_window:.2f}].")
        dist_to_goal_entries = []
        for agent_idx, is_reached in enumerate(state.reached):
            if is_reached:
                dist_to_goal_entries.append("reached")
            else:
                dist_to_goal = np.linalg.norm(state.queries[agent_idx].start - state.queries[agent_idx].goal)
                dist_to_goal_entries.append(f"{dist_to_goal:.2f}")
        compact_entry_separator = ",\t"
        for start in range(0, len(dist_to_goal_entries), 10):
            chunk = dist_to_goal_entries[start : start + 10]
            print(f"\t[{compact_entry_separator.join(chunk)}]")

        window_queries = deepcopy(state.queries)
        planning_horizon = state.right_window - state.left_window
        terminal_window = state.right_window >= stgcs.tmax - 1e-9
        execution_horizon = (
            planning_horizon
            if terminal_window and execution_horizon_factor > 0.0
            else execution_horizon_factor * planning_horizon
        )
        if execution_horizon <= 1e-9 and not all(state.reached):
            print("\tWindowed-PP cannot advance with a zero execution horizon.")
            break
        for agent_idx, query in enumerate(window_queries):
            t_earliest_arrival = query.t_start + np.max(np.abs(query.goal - query.start) / query.vlimit)
            query.is_stay = bool(state.reached[agent_idx] or t_earliest_arrival <= planning_horizon + 1e-9)

        wpp = WindowedPP(
            stgcs,
            st_planner,
            robot_radius,
            t_horizon=planning_horizon,
            reached=state.reached,
            priority_order=priority_order,
        )
        pp_sols, runtime, pp_success = wpp.run(
            window_queries,
            timeout_secs - (time.perf_counter() - start_time),
            verbose=wpp_verbose,
        )

        result.wpp += np.array([1, runtime])
        result.mp += wpp._profiler["mp"]
        result.gcs += wpp._profiler["gcs"]
        result.gub += wpp._profiler["gub"]
        result.search += wpp._profiler["search"]
        result.cr += wpp._profiler["cr"]
        result.dc += wpp._profiler["dc"]
        result.dc_cr += wpp._profiler["dc_cr"]
        result.ecd += wpp._profiler["ecd"]
        result.cc += wpp._profiler["cc"]

        partial_sols, state.reached = forward_simulation(
            state.queries,
            pp_sols,
            pbs_success=pp_success,
            t_offset=state.left_window,
            t_horizon=execution_horizon,
        )
        if len(partial_sols) == 0 or stall_detection(
            state.reached,
            priority_graph=wpp.last_solution_prio_graph,
            priority_order_history=state.priority_order_history,
            agent_positions=[partial_sol.xT[:-1] for partial_sol in partial_sols],
        ):
            print("\tWindowed-PP failed to find partial solution for current window or stall detected.")
            if not dynamic_window_adjustment:
                print("\tFailed to adjust window since dynamic adjustment is disabled.")
                break
            if not adjust_window(stgcs, state, window_span):
                print("\tFailed to adjust window since it is already at the maximum size. Returning failure.")
                break
            print(
                "\tAdjusted window for next iteration: "
                f"window = [{state.left_window:.2f} -> {state.right_window:.2f}]."
            )
        else:
            for agent_idx, partial_sol in enumerate(partial_sols):
                state.solutions[agent_idx].append_chunk(partial_sol)
                state.queries[agent_idx].t_start = 0.0
                state.queries[agent_idx].start = state.solutions[agent_idx].xT[:-1]
            state.commit_current_solutions()

            if all(state.reached):
                print("-> All agents reached goals!")
                result.runtime = np.array([time.perf_counter() - start_time])
                result.success = True
                result.solutions = [sol.copy() for sol in state.solutions]
                return state.solutions, result

            state.left_window = state.left_window + execution_horizon
            state.right_window = min(stgcs.tmax, state.left_window + window_span)
            if not (state.right_window > state.left_window + 1e-9):
                break

            print(
                f"\tWindowed-PP succeeded. Advancing by execution horizon "
                f"{execution_horizon:.2f} (planning={planning_horizon:.2f}, beta={execution_horizon_factor:g})."
            )

    print("\tWindowed-PP failed to find solution.")
    result.runtime = np.array([time.perf_counter() - start_time])
    result.success = False
    failure_solutions = state.best_partial_solutions()
    result.solutions = [sol.copy() for sol in failure_solutions]
    return failure_solutions, result
