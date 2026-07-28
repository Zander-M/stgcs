from __future__ import annotations
from typing import List
from copy import deepcopy

import numpy as np

from stgcs.bezier import de_casteljau_eval, invert_time


class STSplineTrajectory:
    """ order>2 counterpart to `STTrajectory`: one Bezier segment (order joint
        space-time control points) per GCS vertex traversed, per AGENT.md SS3 --
        a GCS vertex already *is* one Bezier segment (F3), so there is no separate
        "spline" object spanning multiple regions.

        Kept API-compatible with `STTrajectory` on the surface every caller actually
        uses (`.xA`/`.xB`/`.x0`/`.xT`/`.duration`/`.size`/`.dim`/`.lerp`/
        `.find_segment_index`), so order-agnostic callers (the search loop, dominance
        checks that only look at entry/exit states) work unchanged. `.points[idx]` is
        NOT interchangeable with `STTrajectory.points[idx]` -- here it is an
        `(order, dim)` control-point matrix per vertex, not a flat 2-point vector;
        callers must go through the accessors, not index `.points` directly. """

    def __init__(self, vertex_path: List[str], points: List[np.ndarray], dim: int, order: int):
        assert len(vertex_path) == len(points), "|vertex_path| != |points|"
        self.vertex_path = vertex_path
        self.dim = dim + 1  # include time dimension, same convention as STTrajectory
        self.order = order
        self.points = [
            np.asarray(p, dtype=float).reshape(order, self.dim) for p in points
        ]

    def copy(self) -> STSplineTrajectory:
        return STSplineTrajectory(
            deepcopy(self.vertex_path), [p.copy() for p in self.points], self.dim - 1, self.order
        )

    @property
    def size(self) -> int:
        return len(self.points)

    def control_points(self, idx: int) -> np.ndarray:
        return self.points[idx]

    @property
    def x0(self) -> np.ndarray:
        return self.xA(0)

    @property
    def xT(self) -> np.ndarray:
        return self.xB(self.size - 1)

    @property
    def duration(self) -> float:
        return self.xT[-1] - self.x0[-1]

    def xA(self, idx: int) -> np.ndarray:
        return self.points[idx][0]

    def xB(self, idx: int) -> np.ndarray:
        return self.points[idx][-1]

    def find_segment_index(self, t: float) -> int:
        if t <= self.x0[-1]:
            return 0

        if t >= self.xT[-1]:
            return self.size - 1

        for idx in range(self.size):
            ta, tb = self.xA(idx)[-1], self.xB(idx)[-1]
            if round(t - ta, 6) >= 0 and round(t - tb, 6) <= 0:
                return idx

        raise AssertionError(f"t: {t} not found in any segment of this trajectory")

    def lerp(self, t: float) -> np.ndarray:
        """ Evaluate the curve at time t via De Casteljau on the inverted time
            parameter (AGENT.md F5: t(s) is invertible per segment since time
            controls are monotone). """
        if t <= self.x0[-1]:
            return self.xA(0)

        if t >= self.xT[-1]:
            return self.xB(self.size - 1)

        idx = self.find_segment_index(t)
        cps = self.points[idx]
        s0 = invert_time(cps[:, -1], t)
        return de_casteljau_eval(cps, s0)
