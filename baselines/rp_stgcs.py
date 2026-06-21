from __future__ import annotations
from typing import List, Dict, Tuple

import time
import numpy as np

from environment.env import Env
from mrmp.stgcs import STGCS, BASE_MAX_ROUNDED_PATHS, BASE_MAX_ROUNDING_TRIALS
from mrmp.ecd import reserve as ecd_reserve
from mrmp.region_reservation.bvc import halfplane_reserve as bvc_reserve
from mrmp.region_reservation.cvt import cvt_reserve
from mrmp.utils import timeit, make_hpolytope
from mrmp.graph import ShortestPathSolution


def randomized_prioritized_planning(
    env:Env, tmax:float, vlimit:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    seed:int, max_ordering_trials:int, timeout_secs:float, scaler_multiplier:float=1.0,
    reservation_method:str='ecd',
) -> Tuple[List[ShortestPathSolution], int]:

    """ prioritized planner with randomized ordering """
    rng = np.random.RandomState(seed)
    num_agents = len(starts)
    ordering = [int(i) for i in range(num_agents)]

    visited = set()
    ts = time.perf_counter()
    while len(visited) < max_ordering_trials:
        rng.shuffle(ordering)
        ordering_tuple = tuple(ordering)
        if ordering_tuple in visited:
            continue

        print(f"-> PP: using total ordering: {ordering}. time elapsed={time.perf_counter() - ts}")
        visited.add(ordering_tuple)

        stgcs = STGCS.from_env(env, t0=0.0, tmax=tmax, vlimit=vlimit)

        sol, num_edges_stgcs = prioritized_planning(
            stgcs, ordering, env.robot_radius, starts, goals, t0s,
            timeout_secs = timeout_secs - (time.perf_counter() - ts),
            scaler_multiplier = scaler_multiplier,
            reservation_method = reservation_method,
        )

        if sol != []:
            return [sol[_idx] for _idx in range(num_agents)], num_edges_stgcs

        if time.perf_counter() - ts > timeout_secs:
            break

    return [], -1


def sequential_planning(
    env:Env, tmax:float, vlimit:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    timeout_secs:float, scaler_multiplier:float=1.0,
    reservation_method:str='ecd',
) -> Tuple[List[ShortestPathSolution], int]:

    ts = time.perf_counter()
    stgcs = STGCS.from_env(env, t0=0.0, tmax=tmax, vlimit=vlimit)

    ordering = [int(i) for i in range(len(starts))]
    sol, num_edges_stgcs = prioritized_planning(
        stgcs, ordering, env.robot_radius, starts, goals, t0s,
        timeout_secs = timeout_secs - (time.perf_counter() - ts),
        scaler_multiplier = scaler_multiplier,
        reservation_method = reservation_method,
    )

    if sol != []:
        return [sol[_idx] for _idx in range(len(starts))], num_edges_stgcs

    return [], -1


def prioritized_planning(
    stgcs:STGCS, ordering:List[int], robot_radius:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    timeout_secs:float, scaler_multiplier:float,
    reservation_method:str='ecd',
) -> Tuple[Dict[int, ShortestPathSolution], int]:
    """
    Plan agents in priority order using the given reservation method.

    reservation_method: one of 'ecd', 'bvc', 'cvt'.
      - 'ecd': Exact Convex Decomposition (original method; splits vertices).
      - 'bvc': Buffered Voronoi Cell halfspaces (no topology changes; faster).
      - 'cvt': Centroidal Voronoi Tessellation (joint partition; uses all trajectories).
    """
    ts = time.perf_counter()
    num_agents = len(starts)
    solution = {}
    stgcs_i = stgcs
    for idx, i in enumerate(ordering):
        print(f"\tplanning for agent {i} [{reservation_method}], STGCS: |V|={stgcs.G.n_vertices}, |E|={stgcs.G.n_edges}")
        start, goal, t0 = starts[i], goals[i], t0s[i]

        if reservation_method == 'cvt':
            # CVT builds a per-agent graph from all planned trajectories + j's linear estimate.
            planned_trajs = [solution[ordering[k]].trajectory for k in range(idx)]
            stgcs_i = cvt_reserve(stgcs, planned_trajs, robot_radius, start, goal, stgcs.tmax)
        else:
            stgcs_i = stgcs  # ECD/BVC accumulate into shared stgcs

        sol = stgcs_i.solve(start, goal, t0,
                            relaxation=True,
                            max_rounded_paths = int(BASE_MAX_ROUNDED_PATHS * scaler_multiplier),
                            max_rounding_trials = int(BASE_MAX_ROUNDING_TRIALS * scaler_multiplier))
        if not sol.is_success:
            print(f"\t PP failed to find a solution for {i}")
            break
        if time.perf_counter() - ts > timeout_secs:
            print(f"\t PP timed out for {i}")
            break

        solution[i] = sol

        if len(solution) == num_agents:
            break

        if reservation_method == 'ecd':
            stgcs = ecd_reserve(stgcs, sol.trajectory, 2 * robot_radius)
        elif reservation_method == 'bvc':
            stgcs = bvc_reserve(stgcs, sol.trajectory, robot_radius)
        # cvt: stgcs stays unchanged; reservation is computed fresh per agent

    if len(solution) == num_agents:
        return solution, stgcs_i.G.n_edges

    return [], -1

