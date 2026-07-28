from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from itertools import combinations


import numpy as np
import networkx as nx

from pydrake.all import (
    Binding, HPolyhedron, Constraint, Cost,
    Point as DrakePoint,
    GraphOfConvexSets as GCS,
    LinearConstraint, LinearEqualityConstraint, LorentzConeConstraint,
    L2NormCost
)
from stgcs.gcs_solver import (
    MPGCSInstance, GCS_SOURCE_NAME, GCS_TARGET_NAME, EDGE_KEY, make_Cartesian_power_hpoly
)

from stgcs.interval import Interval, AABB
from stgcs.geometry_utils import (
    remove_hpoly_redundancies, get_hpoly_bounds, make_hpolytope
)


import logging
logger = logging.getLogger(__name__)


class STVertex:

    def __init__(
        self,
        name: str,
        st_hpoly: HPolyhedron,
        time_itvl: Interval,
        space_itvls: List[Interval],
        parent: Optional[STVertex] = None,
        target_parent_list: List[STVertex]=[]
    ) -> None:
        self.name: str = name
        self.st_hpoly: HPolyhedron = st_hpoly
        self.time_itvl: Interval = time_itvl
        self.space_itvls: List[Interval] = space_itvls
        self.parent: Optional[STVertex] = parent
        self.depth: int = self.parent.depth + 1 if self.parent is not None else 0
        self.target_parent_list: List[STVertex] = target_parent_list
        assert self.parent is None or len(self.target_parent_list) == 0, \
            "Either parent or target_parent_list should be set, not both."

    @property
    def root(self) -> STVertex|List[STVertex]:
        if self.target_parent_list: # special case for target vertex
            return [tp.root for tp in self.target_parent_list]
        
        if self.parent is None:
            return self
        else:
            return self.parent.root
        
    @property
    def root_name(self) -> List[str]:
        if self.target_parent_list: # special case for target vertex
            return [tp.root.name for tp in self.target_parent_list]
        
        if self.parent is None:
            return [self.name]
        else:
            return [self.parent.root.name]


class STGCS:
    _reusable_gcs_cache: Dict[Tuple[int, Tuple[Tuple[str, int], ...], Tuple[Tuple[str, str], ...]], GCS] = {}

    def __init__(
        self, spatial_sets:List[np.ndarray],
        t0:float=0, tmax:float=1e2, vlimit:float=1.0, dt:float=1e-6,
        order:int=2,
    ) -> None:

        self.spatial_sets = spatial_sets
        self.t0, self.tmax, self.vlimit, self.dt = t0, tmax, vlimit, dt
        # order = joint control points per GCS vertex. 2 = today's linear segment
        # (default, unchanged behavior); >2 = cubic (or general-order) Bezier
        # segment per vertex (AGENT.md).
        self.order = order

        self._spatial_hpolys = [make_hpolytope(r) for r in spatial_sets]
        self.dimension = self._spatial_hpolys[0].ambient_dimension()

        self.source: Optional[STVertex] = None
        self.source_comp: Optional[np.ndarray] = None
        self.target: Optional[STVertex] = None

        self._init_constraints_costs()

        self.G = nx.DiGraph()
        self._st_vertices: Dict[str, STVertex] = {}
        self._dummy_targets: Dict[str, STVertex] = {}
        self._next_vertex_index: int = 0
        self._reusable_gcs_signature = None
        self._reusable_gcs: Optional[GCS] = None

    def __getstate__(self) -> Dict[str, object]:
        state = dict(self.__dict__)
        state["vertex_costs"] = []
        state["vertex_constraints"] = []
        state["edge_costs"] = []
        state["edge_constraints"] = []
        state["_reusable_gcs_signature"] = None
        state["_reusable_gcs"] = None
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        self._init_constraints_costs()
        self._reusable_gcs_signature = None
        self._reusable_gcs = None

    def copy(self) -> STGCS:
        ret = STGCS(
            spatial_sets = self.spatial_sets,
            t0 = self.t0, tmax = self.tmax,
            vlimit = self.vlimit, dt = self.dt,
            order = self.order,
        )
        for v_name in self.G.nodes:
            ret.G.add_node(v_name, **self.G.nodes[v_name])
        for u_name, v_name in self.G.edges:
            ret.G.add_edge(u_name, v_name, **self.G.edges[u_name, v_name])
        ret._st_vertices = {name: vertex for name, vertex in self._st_vertices.items()}
        ret.source = self.source
        ret.source_comp = self.source_comp
        ret.target = self.target
        ret._dummy_targets = {name: vertex for name, vertex in self._dummy_targets.items()}
        ret._next_vertex_index = self._next_vertex_index
        return ret

    def get_vertex(self, name:str) -> Optional[STVertex]:
        if name == GCS_SOURCE_NAME:
            return self.source
        elif name == GCS_TARGET_NAME:
            return self.target
        elif name in self._dummy_targets:
            return self._dummy_targets[name]
    
        return self._st_vertices[name]

    def add_vertex(
        self, cvx_set:HPolyhedron, time_bound:Interval, space_bounds:List[Interval]=None,
        size_filter_tol:float=1e-6, name:str=None, parent:STVertex=None,
        remove_redundancies: bool = True,
    ) -> Optional[STVertex]:
        if time_bound.duration <= size_filter_tol:
            return
        
        if cvx_set.ambient_dimension() == self.dimension + 1:
            st_hpoly = remove_hpoly_redundancies(cvx_set) if remove_redundancies else cvx_set
        else:
            raise "cvx_set must be a time-extruded set"

        if space_bounds is None:
            space_bounds = []
            dims = [int(_i) for _i in range(self.dimension)]
            for lb, ub in zip(*get_hpoly_bounds(st_hpoly, dim=dims)):
                if ub - lb <= size_filter_tol:
                    return
                space_bounds.append(Interval(lb, ub))
        
        if name is None:
            name = f"v{self._next_vertex_index}"
            self._next_vertex_index += 1
        
        vertex = STVertex(name, st_hpoly, time_bound, space_bounds, parent)
        
        self.G.add_node(name)
        assert name not in self._st_vertices, f"Vertex {name} already exists!"
        self._st_vertices[name] = vertex

        return vertex

    def add_edges_bidir(self, u_name:str, v_name:str) -> None:
        if u_name == v_name or u_name not in self.G or v_name not in self.G:
            return
        
        u, v = self.get_vertex(u_name), self.get_vertex(v_name)

        if AABB(u.space_itvls, v.space_itvls) and \
           u.st_hpoly.IntersectsWith(v.st_hpoly):
            if np.allclose(u.time_itvl.end, v.time_itvl.start) and u.time_itvl.start <= v.time_itvl.end:
                self._add_edge(u_name, v_name)
            elif np.allclose(v.time_itvl.end, u.time_itvl.start) and v.time_itvl.start <= u.time_itvl.end:
                self._add_edge(v_name, u_name)
            else:
                self._add_edge(u_name, v_name)
                self._add_edge(v_name, u_name)
    
    def remove_vertex_from_graph(self, name:str) -> STVertex:
        """ remove a vertex from the graph and return the removed vertex.
            Note that the vertex is not deleted from the _st_vertices dictionary."""
        vertex = self.get_vertex(name)
        if name in self.G.nodes:
            self.G.remove_node(name)
        return vertex

    def validate_query(self, mp_query:object) -> Tuple[bool, Optional[str]]:
        self._clean_source_targets()
        try:
            found_src = self._init_source_vertex(mp_query.start, mp_query.t_start)
            if not found_src:
                start_state = np.hstack([mp_query.start, mp_query.t_start]).tolist()
                return False, (
                    f"Start state {start_state} is not contained in any ST set according to the real STGCS source check."
                )

            t_earliest_arrival = mp_query.t_start + np.max(abs(mp_query.goal - mp_query.start) / self.vlimit)
            if t_earliest_arrival > self.tmax + 1e-9:
                return False, (
                    f"Goal {mp_query.goal.tolist()} is unreachable by tmax={self.tmax:.4f}. "
                    f"The real STGCS target check requires arrival no earlier than {t_earliest_arrival:.4f}."
                )

            found_tar = self._init_target_vertex(
                mp_query.goal,
                t_earliest_arrival,
                mp_query.is_stay,
                emit_warnings=False,
            )
            if not found_tar:
                return False, self._describe_invalid_target(mp_query.goal, t_earliest_arrival, mp_query.is_stay)

            return True, None
        finally:
            self._clean_source_targets()

    def get_gcs_instance(
        self,
        mp_query:Optional[object]=None,
        reuse_base: bool=False,
    ) -> Optional[MPGCSInstance]:
        """ constructs and returns a drake GCS object, optinally with GCS vertices of source and target """
        
        # 0) unset existing source and target first
        self._clean_source_targets()

        # 1) find source and target vertices
        if mp_query is not None:
            found_src = self._init_source_vertex(mp_query.start, mp_query.t_start)
            t_earliest_arrival = mp_query.t_start + np.max(abs(mp_query.goal - mp_query.start) / self.vlimit)
            found_tar = self._init_target_vertex(mp_query.goal, t_earliest_arrival, mp_query.is_stay)
            if not found_src or not found_tar:
                logger.warning("Failed to initialize source or target vertex")
                self._clean_source_targets()
                return None

        if reuse_base and mp_query is not None:
            return self._get_reusable_gcs_instance()
        
        # 2) instantiate GCS object
        _gcs = GCS()

        # source vertex
        if self.source is not None:
            src_gcs_vert = _gcs.AddVertex(make_Cartesian_power_hpoly(self.source.st_hpoly, self.order), GCS_SOURCE_NAME)
            src_gcs_vert.AddConstraint(Binding[Constraint](
                LinearEqualityConstraint(np.eye((self.dimension + 1) * self.order), np.tile(self.source_comp, self.order)),
                src_gcs_vert.x()))
        else:
            src_gcs_vert = None
        
        # target & dummy-target vertices
        if self.target is not None:
            tar_gcs_vert = _gcs.AddVertex(make_Cartesian_power_hpoly(self.target.st_hpoly, self.order), GCS_TARGET_NAME)
            assert len(self._dummy_targets) > 0, "No dummy targets found for the target vertex"
            for v_name in self._dummy_targets.keys():
                dummy_target_gcs_vertex = _gcs.AddVertex(make_Cartesian_power_hpoly(self.target.st_hpoly, self.order), v_name)
        else:
            tar_gcs_vert = None

        # all other vertices
        for v_name in self.G.nodes:
            vertex = self.get_vertex(v_name)
            gcs_vert = _gcs.AddVertex(make_Cartesian_power_hpoly(vertex.st_hpoly, self.order), v_name)
            self._add_gcs_vertex_costs_constraints(gcs_vert)
        
        # all other edges
        for tail_name, head_name in self.G.edges:
            tail_gcs_vert = _gcs.GetVertexByName(tail_name)
            head_gcs_vert = _gcs.GetVertexByName(head_name)
            # init a GCS.Edge and add costs and constraints
            e = _gcs.AddEdge(tail_gcs_vert, head_gcs_vert, EDGE_KEY(tail_name, head_name))
            self._add_gcs_edge_costs_constraints(e, tail_gcs_vert, head_gcs_vert)
        
        return MPGCSInstance(_gcs, src_gcs_vert, tar_gcs_vert)

    def _add_gcs_vertex_costs_constraints(self, gcs_vert:GCS.Vertex) -> None:
        for cost in self.vertex_costs:
            gcs_vert.AddCost(Binding[Cost](cost, gcs_vert.x()))
        for cstr in self.vertex_constraints:
            gcs_vert.AddConstraint(Binding[Constraint](cstr, gcs_vert.x()))

    def _add_gcs_edge_costs_constraints(
        self,
        edge:GCS.Edge,
        tail_gcs_vert:GCS.Vertex,
        head_gcs_vert:GCS.Vertex,
    ) -> None:
        for cost in self.edge_costs:
            edge.AddCost(Binding[Cost](cost, np.append(tail_gcs_vert.x(), head_gcs_vert.x())))

        # Edges touching a source/target/dummy-target vertex get C0-only continuity:
        # those vertices have no well-defined internal velocity, so requiring a C1
        # tangent-match there would spuriously force a zero velocity at the
        # start/end of the path. Edges between two real per-region vertices get the
        # full C0+C1 constraint list. For order == 2 the two lists are identical
        # (C0 only, as before), so this is a no-op behavior change there.
        boundary_names = {GCS_SOURCE_NAME, GCS_TARGET_NAME, *self._dummy_targets.keys()}
        is_boundary = tail_gcs_vert.name() in boundary_names or head_gcs_vert.name() in boundary_names
        constraints = self.edge_constraints_boundary if is_boundary else self.edge_constraints
        for cstr in constraints:
            edge.AddConstraint(Binding[Constraint](cstr, np.append(tail_gcs_vert.x(), head_gcs_vert.x())))

    def _base_gcs_signature(self) -> Tuple[int, Tuple[Tuple[str, int], ...], Tuple[Tuple[str, str], ...]]:
        planning_names = tuple(self._planning_vertex_names())
        planning_name_set = set(planning_names)
        planning_vertices = tuple(
            (v_name, id(self.get_vertex(v_name).st_hpoly))
            for v_name in planning_names
        )
        planning_edges = tuple(
            (tail_name, head_name)
            for tail_name, head_name in self.G.edges
            if tail_name in planning_name_set and head_name in planning_name_set
        )
        # `order` must be part of the signature: `_reusable_gcs_cache` is a class
        # attribute shared across every STGCS instance regardless of order, so two
        # instances with different `order` but the same vertex/edge signature would
        # otherwise silently retrieve each other's cached GCS, built with the wrong
        # CartesianPower.
        return self.order, planning_vertices, planning_edges

    def _ensure_reusable_base_gcs(self) -> GCS:
        signature = self._base_gcs_signature()
        if self._reusable_gcs is not None and self._reusable_gcs_signature == signature:
            return self._reusable_gcs
        cached_gcs = self._reusable_gcs_cache.get(signature)
        if cached_gcs is not None:
            self._reusable_gcs_signature = signature
            self._reusable_gcs = cached_gcs
            return cached_gcs

        _gcs = GCS()
        _, planning_vertices, planning_edges = signature
        planning_names = tuple(v_name for v_name, _ in planning_vertices)
        for v_name in planning_names:
            vertex = self.get_vertex(v_name)
            gcs_vert = _gcs.AddVertex(make_Cartesian_power_hpoly(vertex.st_hpoly, self.order), v_name)
            self._add_gcs_vertex_costs_constraints(gcs_vert)

        for tail_name, head_name in planning_edges:
            tail_gcs_vert = _gcs.GetVertexByName(tail_name)
            head_gcs_vert = _gcs.GetVertexByName(head_name)
            edge = _gcs.AddEdge(tail_gcs_vert, head_gcs_vert, EDGE_KEY(tail_name, head_name))
            self._add_gcs_edge_costs_constraints(edge, tail_gcs_vert, head_gcs_vert)

        self._reusable_gcs_signature = signature
        self._reusable_gcs = _gcs
        self._reusable_gcs_cache[signature] = _gcs
        return _gcs

    def _get_reusable_gcs_instance(self) -> MPGCSInstance:
        _gcs = self._ensure_reusable_base_gcs()
        query_vertices: List[GCS.Vertex] = []

        src_gcs_vert = _gcs.AddVertex(make_Cartesian_power_hpoly(self.source.st_hpoly, self.order), GCS_SOURCE_NAME)
        src_gcs_vert.AddConstraint(Binding[Constraint](
            LinearEqualityConstraint(np.eye((self.dimension + 1) * self.order), np.tile(self.source_comp, self.order)),
            src_gcs_vert.x()))
        query_vertices.append(src_gcs_vert)

        tar_gcs_vert = _gcs.AddVertex(make_Cartesian_power_hpoly(self.target.st_hpoly, self.order), GCS_TARGET_NAME)
        query_vertices.append(tar_gcs_vert)
        for v_name in self._dummy_targets.keys():
            query_vertices.append(
                _gcs.AddVertex(make_Cartesian_power_hpoly(self.target.st_hpoly, self.order), v_name)
            )

        planning_name_set = set(self._planning_vertex_names())
        query_name_set = {GCS_SOURCE_NAME, GCS_TARGET_NAME, *self._dummy_targets.keys()}
        for tail_name, head_name in self.G.edges:
            if tail_name in planning_name_set and head_name in planning_name_set:
                continue
            if tail_name not in query_name_set and head_name not in query_name_set:
                continue
            tail_gcs_vert = _gcs.GetVertexByName(tail_name)
            head_gcs_vert = _gcs.GetVertexByName(head_name)
            edge = _gcs.AddEdge(tail_gcs_vert, head_gcs_vert, EDGE_KEY(tail_name, head_name))
            self._add_gcs_edge_costs_constraints(edge, tail_gcs_vert, head_gcs_vert)

        def cleanup() -> None:
            for gcs_vertex in query_vertices:
                _gcs.RemoveVertex(gcs_vertex)

        return MPGCSInstance(_gcs, src_gcs_vert, tar_gcs_vert, cleanup=cleanup)

    def clear_query_vertices(self) -> None:
        self._clean_source_targets()

    def make_leaves_roots(self) -> None:
        roots = list(self.leaf_vertices)
        self._st_vertices = {vertex.name: vertex for vertex in roots}

    def _add_edge(self, tail_name:str, head_name:str) -> None:
        self.G.add_edge(tail_name, head_name)

    def _init_constraints_costs(self) -> None:
        self.vertex_costs, self.vertex_constraints = [], []
        self.edge_costs, self.edge_constraints, self.edge_constraints_boundary = [], [], []

        d, order, block = self.dimension, self.order, self.dimension + 1

        if order == 2:
            # --- unchanged: today's linear segment, L-infinity velocity box ---
            A_vmax = np.hstack([
                 np.eye(d),  self.vlimit * np.ones((d, 1)),
                -np.eye(d), -self.vlimit * np.ones((d, 1))])
            A_vmin = np.hstack([
                -np.eye(d),  self.vlimit * np.ones((d, 1)),
                 np.eye(d), -self.vlimit * np.ones((d, 1))])
            A_dt = np.array([0] * d + [1] + [0] * d + [-1])
            A = np.vstack([A_vmax, A_vmin, A_dt])
            b = np.hstack([np.zeros(d), np.zeros(d), -self.dt])

            self.vertex_constraints.append(LinearConstraint(A, -np.inf * np.ones_like(b), b))
            self.vertex_costs.append(L2NormCost(A_dt.reshape(1, -1), np.zeros(1)))

            # define the continuity constraint for each edge (C0 only)
            A_cont = np.block([
                np.zeros((block, block)), np.eye(block), -np.eye(block), np.zeros((block, block))
            ])
            b_cont = np.zeros(block)
            self.edge_constraints.append(LinearEqualityConstraint(A_cont, b_cont))
            self.edge_constraints_boundary = list(self.edge_constraints)
            return

        # --- order > 2: cubic (or general-order) Bezier segment per GCS vertex ---
        # Joint control points Q_0..Q_{order-1} = (P_i, T_i); flat layout per vertex
        # is [P_0, T_0, P_1, T_1, ..., P_{order-1}, T_{order-1}], each block of size
        # `block`.
        n = order * block

        def p_slice(i: int) -> slice:
            offset = i * block
            return slice(offset, offset + d)

        def t_index(i: int) -> int:
            return i * block + d

        # Monotone time (F4/F5): each consecutive pair needs >= dt (implies
        # monotonicity, and is strictly stronger than only an overall
        # T_{order-1}-T_0 >= dt). A per-span floor rules out the solver collapsing
        # an interior control-point spacing to ~0 while satisfying only the overall
        # bound -- interior time *allocation* is still free (uncosted), just
        # bounded away from zero.
        A_mono = np.zeros((order - 1, n))
        for i in range(order - 1):
            A_mono[i, t_index(i)] = 1.0
            A_mono[i, t_index(i + 1)] = -1.0
        b_mono = np.full(order - 1, -self.dt)
        self.vertex_constraints.append(LinearConstraint(A_mono, -np.inf * np.ones(order - 1), b_mono))

        # cost is total segment duration only (T_{order-1} - T_0); interior time
        # allocation between control points is unchanged in *kind* from order == 2,
        # just at different indices.
        A_dt_total = np.zeros(n)
        A_dt_total[t_index(0)] = -1.0
        A_dt_total[t_index(order - 1)] = 1.0
        self.vertex_costs.append(L2NormCost(A_dt_total.reshape(1, -1), np.zeros(1)))

        # Velocity (F8, isotropic L2 form): per-span second-order-cone constraint
        # ||P_{i+1}-P_i|| <= vlimit*(T_{i+1}-T_i), one per consecutive control-point
        # pair. Deliberately an L2 ball here, not this file's order == 2 L-infinity
        # box above: F8's certificate (the Bezier derivative controls are scaled
        # consecutive differences of the degree-3 controls) is inherently
        # isotropic, so an L-infinity box would under-certify the curve's true
        # worst-case speed between the directions it happens to check.
        for i in range(order - 1):
            A_soc = np.zeros((d + 1, n))
            A_soc[0, t_index(i + 1)] = self.vlimit
            A_soc[0, t_index(i)] = -self.vlimit
            A_soc[1:, p_slice(i + 1)] = np.eye(d)
            A_soc[1:, p_slice(i)] = -np.eye(d)
            self.vertex_constraints.append(LorentzConeConstraint(A_soc, np.zeros(d + 1)))

        # Edge continuity (F3): C0 always (tail's last control point == head's
        # first); C1 additionally, only between two real per-region vertices (see
        # _add_gcs_edge_costs_constraints for why boundary edges stay C0-only).
        A_c0 = np.zeros((block, 2 * n))
        A_c0[:, (order - 1) * block: order * block] = np.eye(block)
        A_c0[:, n: n + block] = -np.eye(block)
        self.edge_constraints_boundary.append(LinearEqualityConstraint(A_c0, np.zeros(block)))
        self.edge_constraints.append(LinearEqualityConstraint(A_c0, np.zeros(block)))

        A_c1 = np.zeros((block, 2 * n))
        A_c1[:, (order - 1) * block: order * block] = np.eye(block)         # + Q_tail[-1]
        A_c1[:, (order - 2) * block: (order - 1) * block] = -np.eye(block)  # - Q_tail[-2]
        A_c1[:, n: n + block] = np.eye(block)                               # + Q_head[0]
        A_c1[:, n + block: n + 2 * block] = -np.eye(block)                  # - Q_head[1]
        self.edge_constraints.append(LinearEqualityConstraint(A_c1, np.zeros(block)))

    def _init_source_vertex(self, source:np.ndarray, t_start:float) -> bool:
        # assume source is contained in only one convex set of a vertex; 
        # otherwise, it uses the first one found
        source_found = False
        self.source_comp = np.hstack([source, t_start])
        for v_name in self.G.nodes:
            v = self.get_vertex(v_name)
            if v.st_hpoly.PointInSet(self.source_comp):
                self.source = STVertex(GCS_SOURCE_NAME, DrakePoint(self.source_comp), 
                                  time_itvl = Interval(t_start, t_start), 
                                  space_itvls = [Interval(p, p) for p in source],
                                  parent=v)
                self.G.add_node(GCS_SOURCE_NAME)
                self._add_edge(GCS_SOURCE_NAME, v_name)
                source_found = True
                break
        
        return source_found

    def _planning_vertex_names(self) -> List[str]:
        """Return real ST vertices, excluding temporary query vertices."""
        return [
            v_name
            for v_name in self.G.nodes
            if v_name not in {GCS_SOURCE_NAME, GCS_TARGET_NAME}
            and v_name not in self._dummy_targets
        ]

    def _describe_invalid_target(self, goal:np.ndarray, t_earliest_arrival:float, is_stay:bool) -> str:
        goal = np.asarray(goal, dtype=float).reshape(-1)
        if t_earliest_arrival > self.tmax + 1e-9:
            return (
                f"Goal {goal.tolist()} is unreachable by tmax={self.tmax:.4f}. "
                f"The real STGCS target check requires arrival no earlier than {t_earliest_arrival:.4f}."
            )

        target = HPolyhedron.MakeBox(lb=np.hstack([goal, t_earliest_arrival]), ub=np.hstack([goal, self.tmax]))
        anytime_target = HPolyhedron.MakeBox(lb=np.hstack([goal, self.t0]), ub=np.hstack([goal, self.tmax]))
        target_tmax = np.hstack([goal, self.tmax])
        candidate_names: List[str] = []
        anytime_names: List[str] = []
        intersections: Dict[str, HPolyhedron] = {}
        v_target_tmax = None

        for v_name in self._planning_vertex_names():
            v_hpoly = self.get_vertex(v_name).st_hpoly
            itsc = None
            anytime_itsc = None
            if isinstance(v_hpoly, HPolyhedron):
                itsc = target.Intersection(v_hpoly)
                if itsc.IsEmpty():
                    itsc = None
                anytime_itsc = anytime_target.Intersection(v_hpoly)
                if anytime_itsc.IsEmpty():
                    anytime_itsc = None
            elif isinstance(v_hpoly, DrakePoint):
                v_point = v_hpoly.x()
                if target.PointInSet(v_point):
                    itsc = HPolyhedron.MakeBox(lb=v_point, ub=v_point)
                if anytime_target.PointInSet(v_point):
                    anytime_itsc = HPolyhedron.MakeBox(lb=v_point, ub=v_point)

            if anytime_itsc is not None:
                anytime_names.append(v_name)
            if itsc is not None:
                candidate_names.append(v_name)
                intersections[v_name] = itsc
                if v_hpoly.PointInSet(target_tmax):
                    v_target_tmax = v_name

        if len(candidate_names) == 0:
            if len(anytime_names) > 0:
                return (
                    f"Goal {goal.tolist()} intersects ST sets only before the earliest reachable time "
                    f"{t_earliest_arrival:.4f}. The real STGCS target check uses goal over "
                    f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
                )
            return (
                f"Goal {goal.tolist()} is not contained in any ST set over the valid arrival window "
                f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
            )

        if not is_stay:
            return (
                f"Goal {goal.tolist()} failed the real STGCS target initialization over "
                f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
            )

        if v_target_tmax is None:
            return (
                f"Stay target is invalid: goal {goal.tolist()} is not contained in any ST set at tmax={self.tmax:.4f}."
            )

        T = nx.Graph()
        T.add_nodes_from(candidate_names)
        for name_1, name_2 in combinations(candidate_names, 2):
            if intersections[name_1].IntersectsWith(intersections[name_2]):
                T.add_edge(name_1, name_2)

        valid = [v_name for v_name in candidate_names if nx.has_path(T, v_name, v_target_tmax)]
        if len(valid) == 0:
            return (
                f"Stay target is invalid: goal {goal.tolist()} is not continuously supported by intersecting ST sets "
                f"through tmax={self.tmax:.4f}."
            )

        return (
            f"Goal {goal.tolist()} failed the real STGCS target initialization over "
            f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
        )

    def _init_target_vertex(
        self,
        goal:np.ndarray,
        t_earliest_arrival:float,
        is_stay:bool,
        emit_warnings: bool = True,
    ) -> bool:
        if t_earliest_arrival > self.tmax + 1e-9:
            if emit_warnings:
                logger.warning("Earliest target arrival exceeds tmax")
            return False
        candidates: List[Tuple[str, str]] = []
        target = HPolyhedron.MakeBox(lb=np.hstack([goal, t_earliest_arrival]), ub=np.hstack([goal, self.tmax]))
        target_tmax = np.hstack([goal, self.tmax])
        v_target_tmax = None    # record the vertex containing the goal position at tmax
        intersections: Dict[str, HPolyhedron] = {}
        for v_name in self._planning_vertex_names():
            v_hpoly = self.get_vertex(v_name).st_hpoly
            if isinstance(v_hpoly, HPolyhedron):
                itsc = target.Intersection(v_hpoly)
                if itsc.IsEmpty():
                    itsc = None
            elif isinstance(v_hpoly, DrakePoint):
                v_point = v_hpoly.x()
                itsc = HPolyhedron.MakeBox(lb=v_point, ub=v_point)
                if not target.PointInSet(v_point):
                    itsc = None
            else:
                itsc = None
            
            if itsc is not None:
                v_tar_name = f"sub-{GCS_TARGET_NAME}-{len(candidates)}"
                candidates.append((v_tar_name, v_name))   # (dummy_target, residing_vertex)
                intersections[v_tar_name] = itsc
                if v_hpoly.PointInSet(target_tmax):
                    v_target_tmax = v_tar_name
        
        if len(candidates) == 0:
            if emit_warnings:
                logger.warning("No target vertices found")
            return False
        
        if not is_stay:
            valid = candidates
        elif v_target_tmax is None:
            if emit_warnings:
                logger.warning("No target vertex found at tmax; cannot create a stay target")
            return False
        else:
            # filter candidates to only those that can reach the target at tmax
            T = nx.Graph()
            T.add_nodes_from([name for name, _ in candidates])
            for (name_1, _), (name_2, _) in combinations(candidates, 2):
                if intersections[name_1].IntersectsWith(intersections[name_2]):
                    T.add_edge(name_1, name_2)

            valid = [(vtar, v_residing) for vtar, v_residing in candidates if nx.has_path(T, vtar, v_target_tmax)]
                    
        if len(valid) == 0:
            if emit_warnings:
                logger.warning("No valid target vertices")
            return False

        self.target = STVertex(GCS_TARGET_NAME, target,
                                time_itvl = Interval(t_earliest_arrival, self.tmax), 
                                space_itvls = [Interval(p, p) for p in goal],
                                target_parent_list=[self.get_vertex(_v[1]) for _v in valid])
        self.G.add_node(GCS_TARGET_NAME)

        # add edges: (residing-vertex -> dummy-target) and (dummy-target -> hypertarget)
        # dummy-target vertices are used to connect the residing vertices to the hypertarget and constrain the target
        # this allows for multiple paths to the hypertarget, which can be useful for relaxation
        for v_dummy_tar, v_residing in valid:
            self._dummy_targets[v_dummy_tar] = STVertex(v_dummy_tar, target, 
                                    time_itvl = Interval(t_earliest_arrival, self.tmax), 
                                    space_itvls = [Interval(p, p) for p in goal],
                                    parent=self.get_vertex(v_residing))
            self.G.add_node(v_dummy_tar)
            self._add_edge(v_residing, v_dummy_tar)
            self._add_edge(v_dummy_tar, GCS_TARGET_NAME)
        
        return True

    def _clean_source_targets(self) -> None:
        if not (self.source is None and self.target is None and len(self._dummy_targets) == 0):
            logger.debug("Source and target vertices already set; Unset them first")
            self.remove_vertex_from_graph(GCS_SOURCE_NAME)
            self.remove_vertex_from_graph(GCS_TARGET_NAME)
            for v_name in self._dummy_targets.keys():
                self.remove_vertex_from_graph(v_name)
                
            self._dummy_targets.clear()
            self.source, self.target, self.source_comp = None, None, None

    @property
    def graph_size_string(self) -> str:
        return f"|V|={self.G.number_of_nodes()} |E|={self.G.number_of_edges()}"

    @property
    def leaf_vertices(self) -> List[STVertex]:
        return [self.get_vertex(v_name) for v_name in self.G.nodes]
