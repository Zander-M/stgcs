from __future__ import annotations
from typing import List, Optional

import networkx as nx
import numpy as np
from pydrake.all import HPolyhedron

from mrmp.stgcs import STGCS
from mrmp.interval import Interval, AABB
from mrmp.utils import squash_multi_points
from mrmp.ecd import update_edge, time_cropping_mid


def _spatial_generator(traj: List[np.ndarray], v_itvl: Interval, sdim: int) -> Optional[np.ndarray]:
    """Spatial centroid of a trajectory within a vertex's time window."""
    pts = []
    for seg in traj:
        xp, xq = seg[:sdim + 1], seg[-(sdim + 1):]
        seg_itvl = Interval(xp[-1], xq[-1])
        t_lo = max(v_itvl.start, seg_itvl.start)
        t_hi = min(v_itvl.end, seg_itvl.end)
        if t_lo >= t_hi:
            continue
        dt = xq[-1] - xp[-1]
        if abs(dt) < 1e-10:
            pts.extend([xp[:sdim].copy(), xp[:sdim].copy()])
        else:
            k_lo = (t_lo - xp[-1]) / dt
            k_hi = (t_hi - xp[-1]) / dt
            pts.append(xp[:sdim] + k_lo * (xq[:sdim] - xp[:sdim]))
            pts.append(xp[:sdim] + k_hi * (xq[:sdim] - xp[:sdim]))
    return np.mean(pts, axis=0) if pts else None


def cvt_reserve(
    stgcs: STGCS,
    trajectories: List[List[np.ndarray]],
    safe_radius: float,
    current_start: np.ndarray,
    current_goal: np.ndarray,
    tmax: float,
) -> STGCS:
    """
    Reserve space for the current agent (j) using a Centroidal Voronoi Tessellation.

    For each contested GCS vertex, the vertex is split into temporal slices and the
    "during" slice is replaced by agent j's Voronoi cell — the portion of the vertex
    closer to j's estimated position than to any higher-priority agent's position.
    A clearance gap of 2*safe_radius is enforced between adjacent Voronoi cells.

    Unlike ECD/BVC (which accumulate reservations into a shared stgcs), each call to
    cvt_reserve produces a fresh per-agent stgcs.  Call it once per agent to plan
    within, passing all previously planned trajectories.

    Args:
        stgcs:          Base STGCS (not modified in place).
        trajectories:   Planned trajectories of all higher-priority agents.
        safe_radius:    Robot radius; the gap between cells is 2*safe_radius.
        current_start:  Spatial start of the current agent j (shape: (sdim,)).
        current_goal:   Spatial goal of the current agent j (shape: (sdim,)).
        tmax:           Planning horizon for the linear-interpolation generator.

    Returns:
        A new STGCS with contested vertices replaced by j's Voronoi cells.
    """
    new = stgcs.copy()

    adj_graph = nx.Graph()
    for u, V in stgcs.G._adjacency_list.items():
        adj_graph.add_edges_from([(u, v) for v in V])

    v_hpolys = {
        v_name: squash_multi_points(v.convex_set.set, dim=stgcs.dim)
        for v_name, v in stgcs.G.vertices.items()
    }

    split_map = {v_name: {v_name} for v_name in stgcs.G.vertex_names}
    sdim = stgcs.dim - 1

    for v_name, v in stgcs.G.vertices.items():
        v_hpoly = v_hpolys[v_name]

        # Identify which planned trajectories contest this vertex
        overlapping_trajs = []
        t_union_lo, t_union_hi = np.inf, -np.inf
        for traj in trajectories:
            for seg in traj:
                xp, xq = seg[:stgcs.dim], seg[-stgcs.dim:]
                seg_itvl = Interval(xp[-1], xq[-1])
                seg_spat_bounds = [
                    Interval(min(xp[d], xq[d]) - safe_radius, max(xp[d], xq[d]) + safe_radius)
                    for d in range(sdim)
                ]
                if AABB(seg_spat_bounds + [seg_itvl], v.space_bounds + [v.itvl]):
                    t_lo = max(v.itvl.start, seg_itvl.start)
                    t_hi = min(v.itvl.end, seg_itvl.end)
                    if t_lo < t_hi:
                        t_union_lo = min(t_union_lo, t_lo)
                        t_union_hi = max(t_union_hi, t_hi)
                        overlapping_trajs.append(traj)
                        break

        if not overlapping_trajs:
            continue

        # Generator for agent j: linear interpolation at vertex time midpoint
        t_mid = (v.itvl.start + v.itvl.end) / 2.0
        k = np.clip(t_mid / tmax, 0.0, 1.0) if tmax > 1e-10 else 0.0
        g_j = current_start + k * (current_goal - current_start)

        # Build CVT halfspace constraints: j's cell is closer to g_j than to any g_k
        A_rows, b_vals = [], []
        for traj_k in overlapping_trajs:
            g_k = _spatial_generator(traj_k, v.itvl, sdim)
            if g_k is None:
                continue
            diff = g_k - g_j
            norm = np.linalg.norm(diff)
            if norm < 1e-8:
                continue
            n_kj = diff / norm  # unit vector from g_j toward g_k
            b_mid = np.dot(n_kj, (g_j + g_k) / 2.0)
            # j's Voronoi cell (with clearance r): n_kj^T x_spatial <= b_mid - safe_radius
            A_rows.append(np.append(n_kj, 0.0))
            b_vals.append(b_mid - safe_radius)

        if not A_rows:
            continue

        hs = HPolyhedron(np.array(A_rows), np.array(b_vals))
        old_v = new.remove_vertex(v_name)
        new_sub_names = []

        # Before overlap
        if t_union_lo > v.itvl.start + 1e-6:
            bot = time_cropping_mid(v_hpoly, v.itvl.start, t_union_lo)
            if bot is not None and not bot.IsEmpty():
                sub = new.try_add_vertex(bot, Interval(v.itvl.start, t_union_lo))
                if sub is not None:
                    new_sub_names.append(sub.name)

        # During overlap: restrict to j's Voronoi cell
        mid_base = time_cropping_mid(v_hpoly, t_union_lo, t_union_hi)
        if mid_base is not None and not mid_base.IsEmpty():
            mid_tight = mid_base.Intersection(hs)
            if not mid_tight.IsEmpty():
                sub = new.try_add_vertex(mid_tight, Interval(t_union_lo, t_union_hi))
                if sub is not None:
                    new_sub_names.append(sub.name)

        # After overlap
        if t_union_hi < v.itvl.end - 1e-6:
            top = time_cropping_mid(v_hpoly, t_union_hi, v.itvl.end)
            if top is not None and not top.IsEmpty():
                sub = new.try_add_vertex(top, Interval(t_union_hi, v.itvl.end))
                if sub is not None:
                    new_sub_names.append(sub.name)

        split_map[v_name] = set(new_sub_names)

    new = update_edge(new, adj_graph, split_map)
    return new
