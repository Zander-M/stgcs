from __future__ import annotations
from typing import Tuple

import numpy as np


def de_casteljau_eval(control_points: np.ndarray, s: float) -> np.ndarray:
    """ Evaluate a degree-(order-1) Bezier curve at parameter s in [0, 1] via
        De Casteljau's algorithm. `control_points` is (order, dim); generic over
        `dim` -- used identically for pure-time, pure-space, or joint (space, time)
        control points (AGENT.md F1). """
    pts = np.asarray(control_points, dtype=float).copy()
    n = pts.shape[0]
    for k in range(1, n):
        pts[: n - k] = (1 - s) * pts[: n - k] + s * pts[1 : n - k + 1]
    return pts[0]


def de_casteljau_split(control_points: np.ndarray, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Split a Bezier segment at parameter s into two segments of the same order,
        via De Casteljau's corner-cutting construction (AGENT.md F5: corner-cutting
        preserves monotonicity of the time controls). Returns (left, right) control
        points, each (order, dim); left[-1] == right[0] == curve(s). """
    pts = np.asarray(control_points, dtype=float)
    n = pts.shape[0]
    triangle = [pts.copy()]
    cur = pts.copy()
    for _ in range(1, n):
        cur = (1 - s) * cur[:-1] + s * cur[1:]
        triangle.append(cur)
    left = np.array([tri[0] for tri in triangle])
    right = np.array([tri[-1] for tri in triangle[::-1]])
    return left, right


def invert_time(t_controls: np.ndarray, t_star: float, tol: float = 1e-9, max_iter: int = 100) -> float:
    """ Invert a segment's (monotone, AGENT.md F4/F5) time-control polynomial t(s)
        to find s0 in [0, 1] such that t(s0) == t_star, via bisection. Monotonicity
        of `t_controls` guarantees a unique root whenever
        t_controls[0] <= t_star <= t_controls[-1]. """
    t_controls = np.asarray(t_controls, dtype=float).reshape(-1)
    t0, t1 = t_controls[0], t_controls[-1]
    if t_star <= t0:
        return 0.0
    if t_star >= t1:
        return 1.0

    lo, hi = 0.0, 1.0
    t_col = t_controls.reshape(-1, 1)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t_mid = de_casteljau_eval(t_col, mid)[0]
        if abs(t_mid - t_star) < tol:
            return mid
        if t_mid < t_star:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
