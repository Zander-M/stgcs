from __future__ import annotations

# Code modified from https://github.com/RobotLocomotion/gcs-science-robotics

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import logging
import networkx as nx
import numpy as np

from pydrake.all import (
    CommonSolverOption,
    GraphOfConvexSets as GCS,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    MosekSolver,
    MosekSolverDetails,
    Point as DrakePoint,
    SolverOptions,
)

from stgcs.trajectory import STTrajectory


logger = logging.getLogger(__name__)


GCS_SOURCE_NAME = "source"
GCS_TARGET_NAME = "target"
EDGE_KEY = lambda tail, head: f"({tail}, {head})"


class MPGCSInstance:

    def __init__(
        self,
        gcs: GCS,
        source: GCS.Vertex,
        target: GCS.Vertex,
        cleanup: Optional[Callable[[], None]] = None,
    ) -> None:
        self.gcs = gcs
        self.source = source
        self.target = target
        self._cleanup = cleanup
        self._is_cleaned = False

    def cleanup(self) -> None:
        if self._cleanup is not None and not self._is_cleaned:
            self._cleanup()
            self._is_cleaned = True

    @staticmethod
    def default_solver_options() -> GraphOfConvexSetsOptions:
        options = GraphOfConvexSetsOptions()
        options.solver_options = SolverOptions()
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_LOG", 2)
        return options


def make_Cartesian_power_hpoly(cvs:HPolyhedron|DrakePoint, power:int) -> HPolyhedron|DrakePoint:
    """Create the Cartesian power of a given HPolyhedron."""
    if isinstance(cvs, DrakePoint):
        return DrakePoint(np.tile(cvs.x(), power))
    elif isinstance(cvs, HPolyhedron):
        return cvs.CartesianPower(power)
    else:
        raise TypeError("Input must be either HPolyhedron or Point.")


# Helper functions used be various rounding strategies
def depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn):
    visited_vertices = [source]
    path_vertices = [source]
    path_edges = []
    while path_vertices[-1] != target:
        candidate_edges = getCandidateEdgesFn(path_vertices[-1], visited_vertices)
        if len(candidate_edges) == 0:
            if len(path_vertices) > 0 and len(path_edges) > 0:
                path_vertices.pop()
                path_edges.pop()
        else:
            next_edge, next_vertex = edgeSelectorFn(candidate_edges)
            visited_vertices.append(next_vertex)
            path_vertices.append(next_vertex)
            path_edges.append(next_edge)
    return path_edges

def incomingEdges(gcs):
    incoming_edges = {v.id(): [] for v in gcs.Vertices()}
    for e in gcs.Edges():
        incoming_edges[e.v().id()].append(e)
    return incoming_edges

def outgoingEdges(gcs):
    outgoing_edges = {u.id(): [] for u in gcs.Vertices()}
    for e in gcs.Edges():
        outgoing_edges[e.u().id()].append(e)
    return outgoing_edges

def extractEdgeFlows(gcs, result):
    return {e.id(): result.GetSolution(e.phi()) for e in gcs.Edges()}

def greedyEdgeSelector(candidate_edges, flows):
    candidate_flows = [flows[e.id()] for e in candidate_edges]
    return candidate_edges[np.argmax(candidate_flows)]

def randomEdgeSelector(candidate_edges, flows):
    candidate_flows = np.array([flows[e.id()] for e in candidate_edges])
    probabilities = candidate_flows/sum(candidate_flows)
    return np.random.choice(candidate_edges, p=probabilities)

# Rounding Strategies
def greedyForwardPathSearch(gcs, result, source, target, flow_tol=1e-5, **kwargs):

    outgoing_edges = outgoingEdges(gcs)
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        keepEdge = lambda e: e.v() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in outgoing_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        e = greedyEdgeSelector(candidate_edges, flows)
        return e, e.v()

    return [depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn)]

def runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_paths=10, max_trials=1000):
    paths = []
    trials = 0
    while len(paths) < max_paths and trials < max_trials:
        trials += 1
        try:
            path = depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn)
            if path not in paths:
                paths.append(path)
        except IndexError as e:
            pass
    return paths

def randomForwardPathSearch(gcs, result, source, target, max_paths=10, max_trials=100, seed=None, flow_tol=1e-5, **kwargs):

    if seed is not None:
        np.random.seed(seed)

    outgoing_edges = outgoingEdges(gcs)
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        keepEdge = lambda e: e.v() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in outgoing_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e.v()

    return runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_paths, max_trials)

def greedyBackwardPathSearch(gcs, result, source, target, flow_tol=1e-5, **kwargs):

    incoming_edges = incomingEdges(gcs)
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        keepEdge = lambda e: e.u() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in incoming_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        e = greedyEdgeSelector(candidate_edges, flows)
        return e, e.u()

    return [depthFirst(target, source, getCandidateEdgesFn, edgeSelectorFn)[::-1]]

def randomBackwardPathSearch(gcs, result, source, target, max_paths=10, max_trials=100, seed=None, flow_tol=1e-5, **kwargs):

    if seed is not None:
        np.random.seed(seed)

    incoming_edges = incomingEdges(gcs)
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        keepEdge = lambda e: e.u() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in incoming_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e.u()

    paths = [pathsource, ]

    return [path[::-1] for path in runTrials(target, source, getCandidateEdgesFn, edgeSelectorFn, max_paths, max_trials)]

def MipPathExtraction(gcs, result, source, target, **kwargs):
    return greedyForwardPathSearch(gcs, result, source, target)

def averageVertexPositionGcs(gcs, result, source, target, flow_min=1e-3, **kwargs):

    G = nx.DiGraph()
    G.add_nodes_from(gcs.Vertices())

    vertex_data = {}
    for v in gcs.Vertices():
        vertex_data[v.id()] = np.zeros(v.set().ambient_dimension() + 1)

    for e in gcs.Edges():
        vertex_data[e.u().id()][:-1] += e.GetSolutionPhiXu(result)
        vertex_data[e.u().id()][-1] += result.GetSolution(e.phi())
        if e.v() == target:
            vertex_data[target.id()][:-1] += e.GetSolutionPhiXv(result)
            vertex_data[target.id()][-1] += result.GetSolution(e.phi())

    for v in gcs.Vertices():
        if vertex_data[v.id()][-1] > flow_min:
            vertex_data[v.id()] = vertex_data[v.id()][:-1] / vertex_data[v.id()][-1]
        else:
            vertex_data[v.id()] = v.set().ChebyshevCenter()

    for e in gcs.Edges():
        G.add_edge(e.u(), e.v())
        e_cost = 0
        for cost in e.GetCosts():
            if len(cost.variables()) == e.u().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(vertex_data[e.u().id()])
            elif len(cost.variables()) == e.u().set().ambient_dimension() + e.v().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(np.append(vertex_data[e.u().id()], vertex_data[e.v().id()]))
            else:
                raise Exception("Unclear what variables are used in this cost.")
        G.edges[e.u(), e.v()]['l'] = np.squeeze(e_cost)
        if G.edges[e.u(), e.v()]['l'] < 0:
            raise RuntimeError(f"Averaged length of edge {e} is negative. Consider increasing flow_min.")

    path_vertices = nx.dijkstra_path(G, source, target, 'l')

    path_edges = []
    for u, v in zip(path_vertices[:-1], path_vertices[1:]):
        for e in gcs.Edges():
            if e.u() == u and e.v() == v:
                path_edges.append(e)
                break

    return [path_edges]


def solve(
    gcs_instance:MPGCSInstance, options:Optional[GraphOfConvexSetsOptions]=None,
    rounding:bool=True, max_rounded_paths:int=1000, max_runtime:float=float('inf')
) -> Optional[STTrajectory]:
    
    if options is None:
        options = MPGCSInstance.default_solver_options()
    
    options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_OPTIMIZER_MAX_TIME", float(max_runtime))
    options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", float(max_runtime))
    
    if rounding:
        options.convex_relaxation = True
        options.max_rounded_paths = max_rounded_paths
        options.max_rounding_trials = max_rounded_paths
    else:
        options.convex_relaxation = False

    result = gcs_instance.gcs.SolveShortestPath(gcs_instance.source, gcs_instance.target, options)
    best_path = MipPathExtraction(gcs_instance.gcs, result, gcs_instance.source, gcs_instance.target)[0]
    best_result = result
    
    solver_details = best_result.get_solver_details()
    if not isinstance(solver_details, MosekSolverDetails):
        raise TypeError("Expected MosekSolverDetails from the solver details")

    if not (solver_details.rescode == 0 and (solver_details.solution_status == 1 or solver_details.solution_status == 9)):
        logger.warning("Failed to find the optimal solution")
        return
    else:
        edge_path = best_path[:-2]
        points = [best_result.GetSolution(e.xv()) for e in edge_path]
        space_dim = int(points[0].shape[0] // 2) - 1
        vertex_path = [e.v().name() for e in edge_path]
        traj = STTrajectory(vertex_path, points, space_dim)
    
        for i in range(traj.size - 1):
            p_t2 = traj.points[i][-1]
            q_t1 = traj.points[i+1][space_dim]
            if not np.allclose(p_t2, q_t1):
                logger.warning(f"Invalid Solution: Time discontinuity.")
                return

        return traj


def solve_convex_restriction(
    gcs:GCS,
    vertex_path:List[str],
    options:Optional[GraphOfConvexSetsOptions]=None,
    edge_cache:Optional[Dict[Tuple[str, str], Any]]=None,
) -> Optional[STTrajectory]:
    if options is None:
        options = MPGCSInstance.default_solver_options()

    if vertex_path[-1] == GCS_TARGET_NAME:
        vertex_path = vertex_path[:-1]

    E = []
    for tail, head in zip(vertex_path[:-1], vertex_path[1:]):
        edge_key = (tail, head)
        if edge_cache is not None:
            edge = edge_cache.get(edge_key)
            if edge is None:
                edge = gcs.GetEdgeByName(EDGE_KEY(tail, head))
                edge_cache[edge_key] = edge
        else:
            edge = gcs.GetEdgeByName(EDGE_KEY(tail, head))
        E.append(edge)
    res = gcs.SolveConvexRestriction(E, options)
    if res.is_success():
        points = [res.GetSolution(e.xu()) for e in E]
        space_dim = int(points[0].shape[0] // 2) - 1
        return STTrajectory(vertex_path[:-1], points, space_dim)
