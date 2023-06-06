from pathlib import Path
from functools import partial
from string import ascii_uppercase

import numpy as np
import matplotlib.pyplot as plt
import torch

from neurodiffeq import diff
from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.solvers import Solver2D
from neurodiffeq.utils import set_seed

from util import OperatorIStable
import visualization_helper

X0s = [-1, -0.5, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, -0.5, -1, -1, -1, -1]
Y0s = [1, 1, 1, 1, 1, 0.5, 0, -0.5, -1, -1, -1, -1, -1, -0.5, 0, 0.5]
POINT_NAMES = ascii_uppercase[:len(X0s)]

I = OperatorIStable(-1, ceil=True)


def plot_characteristics():
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    x, y = np.meshgrid(x, y)
    u = -x - y
    v = x - y

    _theta = np.linspace(0, 2 * np.pi, 40).reshape(-1, 1)
    start_points = np.hstack([np.cos(_theta), np.sin(_theta)])

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=70)

    ax.streamplot(
        x, y, u, v,
        color='0.7',
        arrowsize=0.0,
        start_points=start_points,
        broken_streamlines=False,
    )
    ax.set_aspect('equal')
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], label=['-1', '-0.5', '0', '0.5', '1'])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1], label=['-1', '-0.5', '0', '0.5', '1'])
    ax.tick_params(axis='both', which='major', labelsize=16)

    for i, (x0, y0) in enumerate(zip(X0s, Y0s)):
        x, y, _ = get_curve(x0, y0)
        ax.plot(x, y, color='red', linestyle='--')
        ax.text(x0 + 0.05, y0 + 0.05, f'${POINT_NAMES[i]}$', fontdict=dict(fontsize=14))

    ax.plot(X0s, Y0s, 'o', color='black')

    ax.set_xlabel('$x$', fontdict=dict(fontsize=16))
    ax.set_ylabel('$y$', fontdict=dict(fontsize=16), rotation=0)
    fig.savefig(visualization_helper.get_folder() / 'characteristics.pdf', bbox_inches='tight')


def get_curve(x0, y0, s_max=10, steps=1000):
    R0 = np.sqrt(x0 ** 2 + y0 ** 2)
    theta0 = np.arctan2(y0, x0)
    s = np.linspace(0, s_max, steps)
    x = R0 * np.exp(-s) * np.cos(s + theta0)
    y = R0 * np.exp(-s) * np.sin(s + theta0)
    return x, y, s


if __name__ == "__main__":
    set_seed(0)
    visualization_helper.setup()

    plot_characteristics()

    pde = lambda u, x, y: [
        (-x - y) * diff(u, x) + (x - y) * diff(u, y) + u - (3 * x - 2 * y)
    ]
    conditions = [DirichletBVP2D(
        x_min=-1, x_min_val=lambda y: 3 * y - 2,
        x_max=+1, x_max_val=lambda y: 3 * y + 2,
        y_min=-1, y_min_val=lambda x: 2 * x - 3,
        y_max=+1, y_max_val=lambda x: 2 * x + 3,
    )]

    solver = Solver2D(
        pde, conditions,
        xy_min=(-1, -1),
        xy_max=(1, 1),
        n_batches_valid=1,
    )

    solver.fit(1000)

    _x = np.linspace(-1, 1, 100)
    _y = np.linspace(-1, 1, 100)
    DOMAIN = np.meshgrid(_x, _y)

    u = partial(solver.get_solution(best=False), to_numpy=True)
    v = lambda x, y: 2 * x + 3 * y

    fig, axes = plt.subplots(6, 3, figsize=(6, 8), dpi=70)
    axes = axes.flatten()
    for x0, y0, ax, point_name in zip(X0s, Y0s, axes, POINT_NAMES):
        x_s, y_s, s = get_curve(x0, y0)
        err_s = u(x_s, y_s) - v(x_s, y_s)
        residual_s = solver.get_residuals(x_s, y_s, best=False, to_numpy=True)
        bound_s = I(abs(residual_s), s)

        assert (abs(err_s) <= bound_s * 1.001).all()

        ax.plot(s, abs(err_s), label=r'Abs. Err')
        ax.plot(s, bound_s, 'r:', label=r'Bound')
        ax.text(0.25, 0.75, f'Starting at ${point_name}$', fontdict=dict(fontsize=12), transform=ax.transAxes)
        ax.set_xticks([0, 5, 10], ['0', '$s$', '10'])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='x', which='major', pad=0)
        ax.tick_params(axis='y', which='major', pad=2)

    axes[-3].legend(loc=(1.05, 0.25), prop=dict(size=16))
    axes[-2].remove()
    axes[-1].remove()
    plt.subplots_adjust(wspace=0.15, hspace=0.4, left=0.05, bottom=0.05, right=0.99, top=.95)
    fig.savefig(visualization_helper.get_folder() / 'pde-error-bound.pdf', bbox_inches=0)
