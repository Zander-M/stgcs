from __future__ import annotations
from typing import List

import networkx as nx
import numpy as np
from pydrake.all import HPolyhedron

from mrmp.stgcs import STGCS
from mrmp.interval import Interval, AABB
from mrmp.utils import squash_multi_points
from mrmp.ecd import (
    update_edge, time_cropping_mid,
    parallelepiped_side_halfspace_1d, parallelepiped_side_halfspace_2d,
)


def _ecd_halfplanes(xp: np.ndarray, xq: np.ndarray, safe_radius: float, dim: int) -> List[HPolyhedron]:
    if dim == 2:
        return parallelepiped_side_halfspace_1d(xp, xq, safe_radius)
    elif dim == 3:
        return parallelepiped_side_halfspace_2d(xp, xq, safe_radius)
    else:
        raise ValueError(f"Unsupported dimension {dim}")


def lazy_bvc_reserve(
    stgcs: STGCS,
    trajectory: List[np.ndarray],
    safe_radius: float,
    max_halfplanes: int = 1,
) -> STGCS:
    """
    Reserve space using lazy halfplane expansion over ECD parallelepiped faces.

    For each GCS vertex whose space-time window overlaps the trajectory, the vertex is
    temporally split into before/during/after slices. For the "during" slice, up to
    max_halfplanes ECD parallelepiped halfplanes are added as separate sub-vertices
    (disjunctive options for the planning agent). Halfplanes are ordered so the one
    most aligned with the BVC centroid direction is tried first.

    max_halfplanes controls the trade-off:
      1  — cheapest; one safe sub-vertex per conflict vertex (BVC-like)
      2+ — more options; GCS can route through any of the sub-vertices
      4  — equivalent to ECD for single-segment conflicts (2D spatial)

    Args:
        stgcs:          The base STGCS (not modified in place).
        trajectory:     Planned trajectory as List[np.ndarray] of shape (2*dim,).
        safe_radius:    Robot radius; internally uses 2*safe_radius for ECD halfplane width.
        max_halfplanes: Max sub-vertices to create per "during" temporal slice.

    Returns:
        A new STGCS with lazy BVC constraints applied.
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
        p_v_spat = np.array([(b.start + b.end) / 2.0 for b in v.space_bounds])

        # Collect candidate halfplanes from all overlapping segments, scored by BVC alignment
        scored_halfplanes: List[tuple[float, HPolyhedron]] = []
        t_union_lo, t_union_hi = np.inf, -np.inf

        for seg in trajectory:
            xp, xq = seg[:stgcs.dim], seg[-stgcs.dim:]
            seg_itvl = Interval(xp[-1], xq[-1])

            seg_spat_bounds = [
                Interval(min(xp[d], xq[d]) - safe_radius, max(xp[d], xq[d]) + safe_radius)
                for d in range(sdim)
            ]
            if not AABB(seg_spat_bounds + [seg_itvl], v.space_bounds + [v.itvl]):
                continue

            t_lo = max(v.itvl.start, seg_itvl.start)
            t_hi = min(v.itvl.end, seg_itvl.end)
            if t_lo >= t_hi:
                continue

            t_union_lo = min(t_union_lo, t_lo)
            t_union_hi = max(t_union_hi, t_hi)

            # Space-time centroid of v at this overlap window, used for scoring
            t_mid = 0.5 * (t_lo + t_hi)
            p_v_st = np.append(p_v_spat, t_mid)

            # ECD halfplanes for this segment (full clearance = 2 * safe_radius)
            halfplanes = _ecd_halfplanes(xp, xq, 2.0 * safe_radius, stgcs.dim)

            # Score: margin of v's centroid inside each safe halfspace (A·x ≤ b).
            # Higher margin = v's centroid is more safely on this side = better BVC alignment.
            for hs in halfplanes:
                margin = float(hs.b()[0]) - float(hs.A()[0] @ p_v_st)
                scored_halfplanes.append((margin, hs))

        if not scored_halfplanes:
            continue

        # Sort by alignment score descending; pick the top max_halfplanes
        scored_halfplanes.sort(key=lambda x: x[0], reverse=True)
        selected = [hs for _, hs in scored_halfplanes[:max_halfplanes]]

        new.remove_vertex(v_name)
        new_sub_names = []

        # Before overlap
        if t_union_lo > v.itvl.start + 1e-6:
            bot = time_cropping_mid(v_hpoly, v.itvl.start, t_union_lo)
            if bot is not None and not bot.IsEmpty():
                sub = new.try_add_vertex(bot, Interval(v.itvl.start, t_union_lo))
                if sub is not None:
                    new_sub_names.append(sub.name)

        # During overlap: one sub-vertex per selected halfplane (disjunctive options)
        mid_base = time_cropping_mid(v_hpoly, t_union_lo, t_union_hi)
        if mid_base is not None and not mid_base.IsEmpty():
            for hs in selected:
                sub_set = mid_base.Intersection(hs)
                if not sub_set.IsEmpty():
                    sub = new.try_add_vertex(sub_set, Interval(t_union_lo, t_union_hi))
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
