from pathlib import Path
from functools import partial, wraps
from itertools import accumulate
from operator import add

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pickle as pkl
import torch

from neurodiffeq import diff
from neurodiffeq.conditions import IVP, NoCondition
from neurodiffeq.solvers import Solver1D
from neurodiffeq.generators import Generator1D
from neurodiffeq.utils import set_seed

from util import OperatorI
import visualization_helper


DOMAIN = np.linspace(0, 1, 1000)


def get_operator(lmd1, lmd2, ceil=True):
    op1 = OperatorI(lmd1, ceil=ceil).bind_domain(DOMAIN)
    op2 = OperatorI(lmd2, ceil=ceil).bind_domain(DOMAIN)
    return lambda t: op2(op1(t))


def eqs(u1, u2, u3, t):
    return [
        diff(u1, t, order=2) + 3 * diff(u1, t) + 2 * u1 - (2*t**2 + 8*t + 7),
        diff(u2, t, order=2) + u2 - (t**2 + t + 3),
        diff(u3, t, order=2) - diff(u3, t) - (1 - 2*t),
    ]

if __name__ == "__main__":
    set_seed(0)
    visualization_helper.setup()

    conditions = [IVP(0, 1, 1) for _ in range(3)]
    solver = Solver1D(eqs, conditions, t_min=0, t_max=1)
    solver.fit(1000)
    u1, u2, u3 = solver.get_solution(best=False)(DOMAIN, to_numpy=True)
    v = DOMAIN ** 2 + DOMAIN + 1
    err1, err2, err3 = u1 - v, u2 - v, u3 - v

    res1, res2, res3 = solver.get_residuals(DOMAIN, best=False, to_numpy=True)
    bound1 = get_operator(-1, -2)(abs(res1))
    bound2 = get_operator(0, 0)(abs(res2))
    bound3 = get_operator(1, 0)(abs(res3))
    assert (abs(err1) <= bound1).all() and (abs(err2) <= bound2).all() and (abs(err3) <= bound3).all()

    fast_bound1 = (1 / 2) * abs(res1).max() * np.ones_like(DOMAIN)
    fast_bound2 = (1 / 2) * abs(res2).max() * (DOMAIN ** 2)

    assert (get_operator(-1, -2, ceil=False)(abs(res1)) <= fast_bound1).mean()
    assert (get_operator(0, 0, ceil=False)(abs(res2)) <= fast_bound2).mean()

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), dpi=70)
    titles = [r"$v'' + 3 v' + 2v = f(t)$", r"$v'' + v = g(t)$", r"$v'' - v' = h(t)$"]
    fast_bounds = [fast_bound1, fast_bound2, None]

    for err, bound, ax, title, fbound in zip([err1, err2, err3], [bound1, bound2, bound3], axes, titles, fast_bounds):
        if fbound is not None:
            ax.plot(DOMAIN, fbound, color='orange', linestyle='--', label=r'$\mathcal{B}_\mathrm{loose}$')
        ax.plot(DOMAIN, bound, 'r:', label=r'$\mathcal{B}_\mathrm{tight}$')
        ax.plot(DOMAIN, abs(err), label=r'$|\eta|$')
        ax.legend(prop=dict(size=12), loc='upper left')
        ax.set_xticks([0, 0.5, 1], ['0', '$t$', '1'])
        ax.set_title(title, fontdict=dict(fontsize=14))
        ax.tick_params(axis='y', which='major', labelsize=10)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)

    fig.savefig(visualization_helper.get_folder() / '2nd-order.pdf')