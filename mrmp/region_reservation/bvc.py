from __future__ import annotations
from typing import List

import networkx as nx
import numpy as np
from pydrake.all import HPolyhedron

from mrmp.stgcs import STGCS
from mrmp.interval import Interval, AABB
from mrmp.utils import squash_multi_points
from mrmp.ecd import (
    ECDPair, update_edge, slice as ecd_slice,
    time_cropping_bot_top,
    parallelepiped_side_halfspace_1d, parallelepiped_side_halfspace_2d,
)


def halfplane_reserve(stgcs: STGCS, trajectory: List[np.ndarray], safe_radius: float) -> STGCS:
    """
    Halfplane reservation using ECD parallelepiped face halfspaces.

    For each GCS vertex overlapping a trajectory segment:

    - STAYING segments (agent stationary at start/goal extended to t0/tmax):
      ECD-style full split — all parallelepiped face halfspaces are used,
      keeping every safe half (left + right in 1D, four sides in 2D).

    - MOTION segments (agent actually moving): among the parallelepiped face
      halfspaces, select the ONE face where the vertex centroid p_v is on the
      safe side.  This is the halfplane where "before → during" connectivity
      is preserved: the before sub-vertex (full spatial extent, t ≤ t_lo)
      and the during_safe sub-vertex (halfplane, t ∈ [t_lo, t_hi]) share a
      common feasible point at t = t_lo, because p_v satisfies the halfplane.
      Only the safe half is kept; the occupied half is discarded.

    Vertex splitting delegates to ECD's slice() for correct before/during/after
    temporal structure.  Sub-vertex count vs. full ECD (2D spatial):
      - staying: same as ECD (2 during sub-vertices in 1D, 4 in 2D)
      - motion:  1 during sub-vertex (vs. 2 or 4 in full ECD) — leaner graph
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
    dim = stgcs.dim
    halfspace_func = (
        parallelepiped_side_halfspace_1d if dim == 2
        else parallelepiped_side_halfspace_2d
    )

    # Extend trajectory with staying segments at t0/tmax, matching the
    # staying-segment convention used by collision_checking in pbs.py.
    traj = list(trajectory)
    if traj[0][dim - 1] != stgcs.t0:
        traj.insert(0, np.hstack([traj[0][:sdim], [stgcs.t0], traj[0][:dim]]))
    if traj[-1][-1] != stgcs.tmax:
        traj.append(np.hstack([traj[-1][-dim:], traj[-1][-dim:-1], [stgcs.tmax]]))

    for v_name, v in stgcs.G.vertices.items():
        v_hpoly = v_hpolys[v_name]
        p_v = np.array([(b.start + b.end) / 2.0 for b in v.space_bounds])

        ecd_pairs_for_v: List[ECDPair] = []

        for seg in traj:
            xp, xq = seg[:dim], seg[-dim:]
            seg_itvl = Interval(xp[-1], xq[-1])
            seg_spat_bounds = [
                Interval(min(xp[d], xq[d]) - 2 * safe_radius,
                         max(xp[d], xq[d]) + 2 * safe_radius)
                for d in range(sdim)
            ]
            if not AABB(seg_spat_bounds + [seg_itvl], v.space_bounds + [v.itvl]):
                continue

            t_lo = max(v.itvl.start, seg_itvl.start)
            t_hi = min(v.itvl.end, seg_itvl.end)
            if t_lo >= t_hi:
                continue

            is_staying = np.linalg.norm(xp[:sdim] - xq[:sdim]) < 1e-6
            face_hs = halfspace_func(xp, xq, 2 * safe_radius)
            bot_hs, top_hs = time_cropping_bot_top(dim, xp[-1], xq[-1])
            bounds = [
                Interval(min(xp[d], xq[d]) - 2 * safe_radius,
                         max(xp[d], xq[d]) + 2 * safe_radius)
                for d in range(sdim)
            ]
            bounds.append(Interval(xp[-1], xq[-1]))

            if is_staying:
                # All face halfspaces → ECD-style two-sided split.
                ecd_pairs_for_v.append(ECDPair(bot_hs, top_hs, face_hs, bounds))
            else:
                # Select the ONE face halfspace where the vertex centroid lies
                # on the safe side, so that before→during connectivity holds.
                t_v_mid = (t_lo + t_hi) / 2.0
                p_v_full = np.concatenate([p_v, [t_v_mid]])
                best_margin = -np.inf
                best_hs = face_hs[0]
                for hs in face_hs:
                    margin = float(hs.b()[0] - (hs.A() @ p_v_full)[0])
                    if margin > best_margin:
                        best_margin = margin
                        best_hs = hs
                ecd_pairs_for_v.append(ECDPair(bot_hs, top_hs, [best_hs], bounds))

        if not ecd_pairs_for_v:
            continue

        new.remove_vertex(v_name)
        new_sub_names = []
        for res in ecd_slice(v_hpoly, ecd_pairs_for_v, v.itvl.start, v.itvl.end):
            sub = new.try_add_vertex(*res)
            if sub is not None:
                new_sub_names.append(sub.name)
        split_map[v_name] = set(new_sub_names)

    new = update_edge(new, adj_graph, split_map)
    return new


# Alias kept for callers using the old name
bvc_reserve = halfplane_reserve
