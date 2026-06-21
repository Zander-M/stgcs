"""
Benchmark comparing ECD, BVC, CVT, and Lazy-BVC reservation methods.

Sweeps over multiple environments and agent counts. All available instances
per configuration are tested. A per-trial subprocess timeout safely kills
pathological cases (e.g. ECD on complex2d with many agents) without crashing
Drake's C extensions.

Run from the project root:
    conda activate stgcs
    PYTHONPATH=. python tests/test_reservation.py
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from environment.problems import MRMP
import __main__
__main__.MRMP = MRMP


# ── configuration ──────────────────────────────────────────────────────────────

DATA_DIR = "/Users/zdrrrm/Desktop/Projects/stgcs/data"

PROBLEM_SETS = {
    "simple2d":  f"{DATA_DIR}/simple2d.ps",
    "empty2d":   f"{DATA_DIR}/empty2d.ps",
    "complex2d": f"{DATA_DIR}/complex2d.ps",
}

AGENT_COUNTS = [2, 4, 6, 8]

TIMEOUT_SECS = 60  # per method × trial


# ── result containers ──────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    success: bool = False
    timed_out: bool = False
    wall_time: float = 0.0
    total_cost: float = 0.0
    peak_edges: int = 0
    collision_free: bool = False


@dataclass
class MethodStats:
    name: str
    results: List[TrialResult] = field(default_factory=list)

    def add(self, r: TrialResult):
        self.results.append(r)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_success(self) -> int:
        return sum(r.success for r in self.results)

    @property
    def success_rate(self) -> str:
        return f"{self.n_success}/{self.n}"

    @property
    def avg_time(self) -> str:
        vals = [r.wall_time for r in self.results if r.success]
        return f"{np.mean(vals):.2f}s" if vals else "  —  "

    @property
    def avg_cost(self) -> str:
        vals = [r.total_cost for r in self.results if r.success]
        return f"{np.mean(vals):.2f}" if vals else "  —  "

    @property
    def avg_edges(self) -> str:
        vals = [r.peak_edges for r in self.results if r.success]
        return f"{np.mean(vals):.0f}" if vals else "  —  "

    @property
    def safe_rate(self) -> str:
        succeeded = [r for r in self.results if r.success]
        if not succeeded:
            return "  —  "
        s = sum(r.collision_free for r in succeeded)
        return f"{s}/{len(succeeded)}"

    @property
    def timeout_count(self) -> int:
        return sum(r.timed_out for r in self.results)


# ── subprocess worker (must be top-level for pickling) ─────────────────────────

def _trial_worker(
    env_name: str,
    tmax: float,
    vlimit: float,
    robot_radius: float,
    ordering: List[int],
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    t0s: List[float],
    method_name: str,
) -> dict:
    """
    Runs one planner × trial in an isolated process.
    Returns a plain dict so nothing Drake-internal needs to cross the process boundary.
    """
    import sys
    sys.path.insert(0, "/Users/zdrrrm/Desktop/Projects/stgcs")

    from environment.examples import SIMPLE2D, EMPTY2D, COMPLEX2D
    from environment.problems import MRMP as _MRMP
    import __main__ as _main
    _main.MRMP = _MRMP

    from mrmp.stgcs import STGCS, BASE_MAX_ROUNDED_PATHS, BASE_MAX_ROUNDING_TRIALS
    from mrmp.ecd import reserve as ecd_reserve
    from mrmp.region_reservation.bvc import halfplane_reserve as bvc_reserve
    from mrmp.region_reservation.cvt import cvt_reserve
    from mrmp.region_reservation.lazy_bvc import lazy_bvc_reserve
    from mrmp.pbs import collision_checking

    _ENV_MAP = {"simple2d": SIMPLE2D, "empty2d": EMPTY2D, "complex2d": COMPLEX2D}
    env = _ENV_MAP[env_name]
    stgcs_base = STGCS.from_env(env, t0=0.0, tmax=tmax, vlimit=vlimit)
    n_agents = len(ordering)

    # ── inner planners ────────────────────────────────────────────────────────

    def _solve(st, i):
        return st.solve(starts[i], goals[i], t0s[i],
                        relaxation=True,
                        max_rounded_paths=BASE_MAX_ROUNDED_PATHS,
                        max_rounding_trials=BASE_MAX_ROUNDING_TRIALS)

    def _ecd():
        stgcs = stgcs_base.copy()
        peak = stgcs.G.n_edges
        sol_map = {}
        for i in ordering:
            sol = _solve(stgcs, i)
            if not sol.is_success:
                return None, peak
            sol_map[i] = sol
            if len(sol_map) < n_agents:
                stgcs = ecd_reserve(stgcs, sol.trajectory, 2 * robot_radius)
                peak = max(peak, stgcs.G.n_edges)
        return sol_map, peak

    def _halfplane():
        stgcs = stgcs_base.copy()
        peak = stgcs.G.n_edges
        sol_map = {}
        for i in ordering:
            sol = _solve(stgcs, i)
            if not sol.is_success:
                return None, peak
            sol_map[i] = sol
            if len(sol_map) < n_agents:
                stgcs = bvc_reserve(stgcs, sol.trajectory, robot_radius)
                peak = max(peak, stgcs.G.n_edges)
        return sol_map, peak

    def _cvt():
        peak = stgcs_base.G.n_edges
        sol_map = {}
        for idx, i in enumerate(ordering):
            planned = [sol_map[ordering[k]].trajectory for k in range(idx)]
            st_i = cvt_reserve(stgcs_base, planned, robot_radius,
                               starts[i], goals[i], stgcs_base.tmax)
            peak = max(peak, st_i.G.n_edges)
            sol = _solve(st_i, i)
            if not sol.is_success:
                return None, peak
            sol_map[i] = sol
        return sol_map, peak

    def _lazy_bvc():
        peak = stgcs_base.G.n_edges
        sol_map = {}
        trajectories = []
        for k, i in enumerate(ordering):
            if not trajectories:
                sol = _solve(stgcs_base.copy(), i)
                if not sol.is_success:
                    return None, peak
                sol_map[i] = sol
                trajectories.append(sol.trajectory)
                continue
            sol = None
            for max_hp in [1, 2, 3, 4]:
                st_i = stgcs_base.copy()
                for traj in trajectories:
                    st_i = lazy_bvc_reserve(st_i, traj, robot_radius,
                                            max_halfplanes=max_hp)
                peak = max(peak, st_i.G.n_edges)
                sol = _solve(st_i, i)
                if sol.is_success:
                    break
            if not sol.is_success:
                return None, peak
            sol_map[i] = sol
            trajectories.append(sol.trajectory)
        return sol_map, peak

    _planners = {"ecd": _ecd, "halfplane": _halfplane, "cvt": _cvt, "lazy_bvc": _lazy_bvc}

    try:
        solution, peak_edges = _planners[method_name]()
    except Exception as e:
        return {"success": False, "peak_edges": 0, "error": str(e)}

    if solution is None or len(solution) < n_agents:
        return {"success": False, "peak_edges": peak_edges}

    total_cost = sum(solution[i].cost for i in range(n_agents))

    def _check_safe():
        agents = sorted(solution.keys())
        for ii in range(len(agents)):
            for jj in range(ii + 1, len(agents)):
                if collision_checking(solution[agents[ii]].trajectory,
                                      solution[agents[jj]].trajectory,
                                      robot_radius, 0.0, tmax):
                    return False
        return True

    return {
        "success": True,
        "peak_edges": peak_edges,
        "total_cost": total_cost,
        "collision_free": _check_safe(),
    }


# ── printing ───────────────────────────────────────────────────────────────────

COL = "{:<10}  {:>8}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}"

def _header():
    print(COL.format("Method", "Success", "Avg time", "Avg cost", "Avg edges", "Safe", "Timeout"))
    print("-" * 76)

def _row(s: MethodStats):
    print(COL.format(s.name, s.success_rate, s.avg_time, s.avg_cost,
                     s.avg_edges, s.safe_rate, str(s.timeout_count)))


# ── benchmark ──────────────────────────────────────────────────────────────────

def run_benchmark(quick: bool = False):
    problem_sets = (
        {"simple2d": PROBLEM_SETS["simple2d"]} if quick else PROBLEM_SETS
    )
    agent_counts = [2, 4] if quick else AGENT_COUNTS
    max_trials   = 3      if quick else None
    timeout      = 30     if quick else TIMEOUT_SECS

    grand: Dict[str, MethodStats] = {n: MethodStats(name=n) for n in _PLANNER_NAMES}

    for ps_name, ps_path in problem_sets.items():
        print(f"\nLoading {ps_name} ...")
        with open(ps_path, "rb") as f:
            problem_set = pickle.load(f)

        for n_agents in agent_counts:
            problems: List[MRMP] = problem_set.get(n_agents, [])
            if not problems:
                continue
            if max_trials is not None:
                problems = problems[:max_trials]

            print(f"\n{'='*76}")
            print(f"  {ps_name}  |  n_agents={n_agents}  |  {len(problems)} trials"
                  f"  |  timeout={timeout}s/trial")
            print(f"{'='*76}")

            cfg: Dict[str, MethodStats] = {n: MethodStats(name=n) for n in _PLANNER_NAMES}

            for trial_idx, mrmp in enumerate(problems):
                env   = mrmp.env
                tmax  = mrmp.tmax
                starts, goals, t0s = mrmp.starts, mrmp.goals, mrmp.T0s
                ordering = list(range(n_agents))
                print(f"\n  Trial {trial_idx + 1}/{len(problems)}")

                for method_name in _PLANNER_NAMES:
                    ts = time.perf_counter()
                    timed_out = False
                    ret = None

                    with ProcessPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(
                            _trial_worker,
                            ps_name, tmax, mrmp.vlimit, env.robot_radius,
                            ordering, starts, goals, list(t0s), method_name,
                        )
                        try:
                            ret = future.result(timeout=timeout)
                        except FuturesTimeout:
                            timed_out = True
                            future.cancel()
                        except Exception as e:
                            print(f"    [{method_name}] WORKER ERROR: {e}")

                    elapsed = time.perf_counter() - ts
                    result = TrialResult(timed_out=timed_out)

                    if ret and ret.get("success"):
                        result.success = True
                        result.wall_time = elapsed
                        result.total_cost = ret["total_cost"]
                        result.peak_edges = ret["peak_edges"]
                        result.collision_free = ret["collision_free"]
                        status = (f"SUCCESS  t={elapsed:.2f}s  cost={result.total_cost:.1f}"
                                  f"  edges={result.peak_edges}  safe={result.collision_free}")
                    elif timed_out:
                        status = f"TIMEOUT  (>{TIMEOUT_SECS}s)"
                    else:
                        err = (ret or {}).get("error", "")
                        status = f"FAILED   t={elapsed:.2f}s" + (f"  [{err}]" if err else "")

                    print(f"    [{method_name:<8}] {status}")
                    cfg[method_name].add(result)
                    grand[method_name].add(result)

            print(f"\n  --- {ps_name} / n_agents={n_agents} ---")
            _header()
            for n, s in cfg.items():
                _row(s)

    print(f"\n{'='*76}")
    print(f"{'GRAND SUMMARY':^76}")
    print(f"{'='*76}")
    _header()
    for n, s in grand.items():
        _row(s)


_PLANNER_NAMES = ["ecd", "halfplane", "cvt", "lazy_bvc"]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="simple2d only, 2+4 agents, 3 trials, 30s timeout")
    args = ap.parse_args()
    run_benchmark(quick=args.quick)
