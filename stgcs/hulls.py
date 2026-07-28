from __future__ import annotations

import numpy as np
from pydrake.all import HPolyhedron

from stgcs.geometry_utils import make_hpolytope


def minkowski_sum_vertices(control_points: np.ndarray, footprint_vertices: np.ndarray) -> np.ndarray:
    """ V-rep Minkowski sum: conv(A) + conv(B) == conv(A + B), i.e. every pairwise
        sum of a control point and a footprint vertex. `footprint_vertices` has zero
        time extent -- a footprint occupies a spatial region, not a time interval
        (AGENT.md SS3/SS4) -- so its last column must be all zeros. """
    control_points = np.asarray(control_points, dtype=float)
    footprint_vertices = np.asarray(footprint_vertices, dtype=float)
    if not np.allclose(footprint_vertices[:, -1], 0.0):
        raise ValueError("footprint_vertices must have zero time extent (last column == 0)")
    summed = control_points[:, None, :] + footprint_vertices[None, :, :]
    return summed.reshape(-1, control_points.shape[-1])


def inflate_hull(control_points: np.ndarray, footprint_vertices: np.ndarray) -> HPolyhedron:
    """ Minkowski-inflate a Bezier segment's control-point hull by a footprint
        (AGENT.md F1, F7), and reduce the resulting H-representation.
        `ConvexHull` triangulates every planar facet into simplices, roughly
        doubling the row count with true duplicates -- do not skip
        `.ReduceInequalities()`. """
    raw_vertices = minkowski_sum_vertices(control_points, footprint_vertices)
    return make_hpolytope(raw_vertices).ReduceInequalities()
