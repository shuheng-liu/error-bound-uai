from pathlib import Path
from functools import partial, wraps
from itertools import accumulate
from operator import add

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import ticker
import pickle as pkl
import torch

from neurodiffeq import diff
from neurodiffeq.conditions import IVP
from neurodiffeq.solvers import Solver1D
from neurodiffeq.generators import Generator1D
from neurodiffeq.utils import set_seed

from util import OperatorIStable
import visualization_helper


def EQ_LINEAR(u, t): 
    return diff(u, t, order=2) + 3 * diff(u, t) + 2 * u


TMAX = 2
TRAIN_GENERATOR = Generator1D(int(32 * TMAX), 0, TMAX, method='uniform')
VALID_GENERATOR = Generator1D(int(32 * TMAX), 0, TMAX, method='equally-spaced')
MAX_EPOCHS = 1000
_DOMAIN_POINTS = 100000
_PLOT_POINTS = 1000
DOMAIN = np.linspace(0, TMAX, _DOMAIN_POINTS)

# index for plotting
IDX = np.arange(0, _DOMAIN_POINTS, max(1, _DOMAIN_POINTS // _PLOT_POINTS), dtype=int)

_I1 = OperatorIStable(-1).bind_domain(DOMAIN)
_I2 = OperatorIStable(-2).bind_domain(DOMAIN)
I = lambda phis: _I2(_I1(phis))


def get_solver0():
    solver = Solver1D(
        lambda u, t: [EQ_LINEAR(u, t) - torch.cos(t)],
        [IVP(0, 1, 1)],
        train_generator=TRAIN_GENERATOR,
        valid_generator=VALID_GENERATOR,
        n_batches_valid=0,
    )

    solver.fit(max_epochs=MAX_EPOCHS)
    return solver


def get_solver1(u0fn):
    solver = Solver1D(
        lambda u, t: [EQ_LINEAR(u, t) + u0fn(t) ** 3],
        [IVP(0, 0, 0)],
        train_generator=TRAIN_GENERATOR,
        valid_generator=VALID_GENERATOR,
        n_batches_valid=0,
    )

    solver.fit(max_epochs=MAX_EPOCHS)
    return solver


def get_solver2(u0fn, u1fn):
    solver = Solver1D(
        lambda u, t: [EQ_LINEAR(u, t) + 3 * u0fn(t) ** 2 * u1fn(t)],
        [IVP(0, 0, 0)],
        train_generator=TRAIN_GENERATOR,
        valid_generator=VALID_GENERATOR,
        n_batches_valid=0,
    )

    solver.fit(max_epochs=MAX_EPOCHS)
    return solver


def get_solver3(u0fn, u1fn, u2fn):
    def eq(u, t):
        u0, u1, u2 = u0fn(t), u1fn(t), u2fn(t)
        return [
            EQ_LINEAR(u, t)
            + 3 * u0 ** 2 * u2
            + 3 * u0 * u1 ** 2
        ]

    solver = Solver1D(
        eq, [IVP(0, 0, 0)],
        train_generator=TRAIN_GENERATOR,
        valid_generator=VALID_GENERATOR,
        n_batches_valid=0,
    )

    solver.fit(max_epochs=MAX_EPOCHS)
    return solver


def get_solver4(u0fn, u1fn, u2fn, u3fn):
    def eq(u, t):
        u0, u1, u2, u3 = u0fn(t), u1fn(t), u2fn(t), u3fn(t)
        return [
            EQ_LINEAR(u, t)
            + 3 * u0 ** 2 * u3
            + 6 * u0 * u1 * u2
            + u1 ** 3
        ]

    solver = Solver1D(
        eq, [IVP(0, 0, 0)],
        train_generator=TRAIN_GENERATOR,
        valid_generator=VALID_GENERATOR,
        n_batches_valid=0,
    )

    solver.fit(max_epochs=MAX_EPOCHS)
    return solver


def get_solver5(u0fn, u1fn, u2fn, u3fn, u4fn):
    def eq(u, t):
        u0, u1, u2, u3, u4 = u0fn(t), u1fn(t), u2fn(t), u3fn(t), u4fn(t)
        return [
            EQ_LINEAR(u, t)
            + 3 * u0 ** 2 * u4
            + 6 * u0 * u1 * u3
            + 3 * u0 * u2 ** 2
            + 3 * u1 ** 2 * u2
        ]

    solver = Solver1D(
        eq, [IVP(0, 0, 0)],
        train_generator=TRAIN_GENERATOR,
        valid_generator=VALID_GENERATOR,
        n_batches_valid=0,
    )

    solver.fit(max_epochs=MAX_EPOCHS)
    return solver


def get_solver6(u0fn, u1fn, u2fn, u3fn, u4fn, u5fn):
    def eq(u, t):
        u0, u1, u2, u3, u4, u5 = u0fn(t), u1fn(t), u2fn(t), u3fn(t), u4fn(t), u5fn(t)
        return [
            EQ_LINEAR(u, t)
            + 3 * u0 ** 2 * u5
            + 6 * u0 * u1 * u4
            + 6 * u0 * u2 * u3
            + 3 * u1 ** 2 * u3
            + 3 * u1 * u2 ** 2
        ]

    solver = Solver1D(
        eq, [IVP(0, 0, 0)],
        train_generator=TRAIN_GENERATOR,
        valid_generator=VALID_GENERATOR,
        n_batches_valid=0,
    )

    solver.fit(max_epochs=MAX_EPOCHS)
    return solver


get_solver_fns = [
    get_solver0,
    get_solver1,
    get_solver2,
    get_solver3,
    get_solver4,
    get_solver5,
    get_solver6,
]

def _nonlin3(Bi, ui):
    return Bi**3 + 3 * Bi * ui**2 + 3 * Bi**2 * np.abs(ui)

def _nonlin21(Bi, ui, Bj, uj):
    x = ui**2 * Bj + Bi * (np.abs(uj) + Bj) * (np.abs(ui) + 2 * Bi)
    return x * 3

def _nonlin111(Bi, ui, Bj, uj, Bk, uk):
    x = (np.abs(ui) + Bi) * (np.abs(uj) + Bj) * (np.abs(uk) + Bk) - np.abs(ui*uj*uk)
    return x * 6

def get_bounds(_us, _rs):
    _us = [_u(DOMAIN, to_numpy=True) for _u in _us]
    _Bs = []

    res = lambda i: I(abs(_rs[i]))
    nl3 = lambda i: I(_nonlin3(_Bs[i], _us[i]))
    nl21 = lambda i, j: I(_nonlin21(_Bs[i], _us[i], _Bs[j], _us[j]))
    nl111 = lambda i, j, k: I(_nonlin111(_Bs[i], _us[i], _Bs[j], _us[j], _Bs[k], _us[k]))

    _Bs.append(res(0))
    _Bs.append(res(1) + nl3(0))
    _Bs.append(res(2) + nl21(0, 1))
    _Bs.append(res(3) + nl21(0, 2) + nl21(1, 0))
    _Bs.append(res(4) + nl21(0, 3) + nl111(0, 1, 2) + nl3(1))
    _Bs.append(res(5) + nl21(0, 4) + nl111(0, 1, 3) + nl21(2, 0) + nl21(1, 2))
    _Bs.append(res(6) + nl21(0, 5) + nl111(0, 1, 4) + nl111(0, 2, 3) + nl21(1, 3) + nl21(2, 1))

    return _Bs

def get_u_v(eps, us):
    v2 = solve_ivp(
        fun=lambda t, y: np.vstack([y[1], - 2 * y[0] - 3 * y[1] - eps * y[0]**3 + np.cos(t)]),
        t_span=(0.0, TMAX),
        y0=(1.0, 1.0), 
        method='RK45',
        t_eval=DOMAIN, 
        dense_output=True,
        vectorized=True,
        rtol=1e-4,
        atol=1e-6,
    ).sol

    v = lambda t: v2(t)[0]

    def u(t, to_numpy=True, **kwargs):
        return sum(ui(t, to_numpy=to_numpy, **kwargs) * eps ** i for i, ui in enumerate(us))
    
    return u, v

if __name__ == "__main__":
    set_seed(0)
    visualization_helper.setup()

    try:
        with open('duffing-residuals.pkl', 'rb') as f:
            rs = pkl.load(f)
        with open('duffing-bounds.pkl', 'rb') as f:
            bounds = pkl.load(f)
        with open('duffing-solutions.pkl', 'rb') as f:
            us = pkl.load(f)
        print("Cache Data Loaded.")
    except FileNotFoundError:
        print("Failed to load cached data. Retraining started.")
        us, rs, solvers = [], [], []
        for i, get_solver_fn in enumerate(get_solver_fns):
            set_seed(0)
            _solver = get_solver_fn(*us)
            _u = _solver.get_solution(best=False)
            _r = _solver.get_residuals(DOMAIN, best=False, to_numpy=True)

            solvers.append(_solver)
            us.append(_u)
            rs.append(_r)
            print(f"max residual is of eq[{i}] is", abs(_r).max())
        bounds = get_bounds(us, rs)

        with open('duffing-residuals.pkl', 'wb') as f:
            pkl.dump(rs, f)
        with open('duffing-bounds.pkl', 'wb') as f:
            pkl.dump(bounds, f)
        with open('duffing-solutions.pkl', 'wb') as f:
            pkl.dump(us, f)
    

    epsilons = np.linspace(-0.9, 0.9, 8)
    fig, axes = plt.subplots(3, 3, figsize=(6, 3.5), dpi=70)
    axes = axes.flatten()
    assert len(epsilons) == len(axes) - 1

    for ax, epsilon in zip(axes, epsilons):
        for i in range(len(us)):
            u, _ = get_u_v(epsilon, us[: i + 1])
            ax.plot(DOMAIN[IDX], u(DOMAIN[IDX]), ':', label=rf'deg 0$\sim${i}' if i > 0 else 'deg 0')
        _, v = get_u_v(epsilon, us)
        ax.plot(DOMAIN[IDX], v(DOMAIN[IDX]), label='RKF45')
        ax.text(0.05, 0.1, rf'$\varepsilon={epsilon:.3f}$', fontdict=dict(fontsize=16), transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='x', which='major', pad=0)
        ax.tick_params(axis='y', which='major', pad=2)
        ax.grid(visible=False)
        ax.set_xticks([0, TMAX / 2, TMAX], labels=['0', '$t$', f'{TMAX}'])
    axes[-2].legend(prop=dict(size=12), ncol=2, loc=(1.0, -0.1), borderaxespad=0, handletextpad=0.0, columnspacing=0.1)
    axes[-1].remove()
    plt.subplots_adjust(wspace=0.3, hspace=0.25, left=0.075, bottom=0.05, right=1.00, top=.99)
    fig.savefig(visualization_helper.get_folder() / 'duffing-solution.pdf', bbox_inches=0)

    fig, axes = plt.subplots(3, 3, figsize=(6, 3.5), dpi=70)
    axes = axes.flatten()
    assert len(epsilons) == len(axes) - 1
    for ax, epsilon in zip(axes, epsilons):
        accumulated_bounds = accumulate(bound * abs(epsilon) ** i for i, bound in enumerate(bounds))
        for i, ab in enumerate(accumulated_bounds):
            ax.plot(DOMAIN[IDX], ab[IDX], ':', label=rf'Up to $\mathcal{{B}}_{i}$')
        u, v = get_u_v(epsilon, us)
        ax.plot(DOMAIN[IDX], np.abs(u(DOMAIN[IDX]) - v(DOMAIN[IDX])), label='abs err')
        ax.text(0.05, 0.75, rf'$\varepsilon={epsilon:.3f}$', fontdict=dict(fontsize=16), transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='x', which='major', pad=0)
        ax.tick_params(axis='y', which='major', pad=2)
        ax.grid(visible=False)
        ax.set_xticks([0, TMAX / 2, TMAX], labels=['0', '$t$', f'{TMAX}'])
    axes[-2].legend(prop=dict(size=12), ncol=2, loc=(1.0, -0.1), borderaxespad=0, handletextpad=0.0, columnspacing=0.1)
    axes[-1].remove()
    plt.subplots_adjust(wspace=0.3, hspace=0.25, left=0.075, bottom=0.05, right=1.00, top=.99)
    fig.savefig(visualization_helper.get_folder() / 'duffing-error.pdf', bbox_inches=0)