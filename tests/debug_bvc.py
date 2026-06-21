"""Debug BVC vs ECD on 2-agent simple2d problems."""
import pickle
import sys
import numpy as np
from environment.problems import MRMP  # register class before unpickling
import __main__
__main__.MRMP = MRMP  # patch so pickle finds __main__.MRMP
from mrmp.stgcs import STGCS
from mrmp.ecd import reserve as ecd_reserve
from mrmp.region_reservation.bvc import bvc_reserve
from mrmp.region_reservation.cvt import cvt_reserve

with open('/Users/zdrrrm/Desktop/Projects/stgcs/data/simple2d.ps', 'rb') as f:
    ps = pickle.load(f)

problems = ps[2][:4]
for ti, mrmp in enumerate(problems):
    env = mrmp.env
    starts, goals, t0s = mrmp.starts, mrmp.goals, mrmp.T0s
    print(f'\nTrial {ti}: r={env.robot_radius:.3f}')
    print(f'  Agent 0: start={starts[0].round(2)}, goal={goals[0].round(2)}, t0={t0s[0]:.2f}')
    print(f'  Agent 1: start={starts[1].round(2)}, goal={goals[1].round(2)}, t0={t0s[1]:.2f}')

    stgcs = STGCS.from_env(env, t0=0, tmax=mrmp.tmax, vlimit=mrmp.vlimit)
    print(f'  Base STGCS: |V|={stgcs.G.n_vertices}, |E|={stgcs.G.n_edges}')

    sol0 = stgcs.solve(starts[0], goals[0], t0s[0], relaxation=True)
    print(f'  Agent 0: success={sol0.is_success}, cost={sol0.cost:.2f}')
    if not sol0.is_success:
        continue

    stgcs_bvc = bvc_reserve(stgcs.copy(), sol0.trajectory, env.robot_radius)
    stgcs_ecd = ecd_reserve(stgcs.copy(), sol0.trajectory, 2 * env.robot_radius)
    stgcs_cvt = cvt_reserve(stgcs.copy(), [sol0.trajectory], env.robot_radius,
                             starts[1], goals[1], mrmp.tmax)

    print(f'  After BVC: |V|={stgcs_bvc.G.n_vertices}, |E|={stgcs_bvc.G.n_edges}')
    print(f'  After ECD: |V|={stgcs_ecd.G.n_vertices}, |E|={stgcs_ecd.G.n_edges}')
    print(f'  After CVT: |V|={stgcs_cvt.G.n_vertices}, |E|={stgcs_cvt.G.n_edges}')

    sol1_ecd = stgcs_ecd.solve(starts[1], goals[1], t0s[1], relaxation=True)
    sol1_bvc = stgcs_bvc.solve(starts[1], goals[1], t0s[1], relaxation=True)
    sol1_cvt = stgcs_cvt.solve(starts[1], goals[1], t0s[1], relaxation=True)

    print(f'  Agent 1 ECD: success={sol1_ecd.is_success}, cost={sol1_ecd.cost:.2f}')
    print(f'  Agent 1 BVC: success={sol1_bvc.is_success}, cost={sol1_bvc.cost:.2f}')
    print(f'  Agent 1 CVT: success={sol1_cvt.is_success}, cost={sol1_cvt.cost:.2f}')
