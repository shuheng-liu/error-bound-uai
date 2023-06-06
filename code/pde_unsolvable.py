from pathlib import Path
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

from neurodiffeq import diff
from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.solvers import Solver2D
from neurodiffeq.utils import set_seed
from neurodiffeq.callbacks import ActionCallback

import visualization_helper

def plot_characteristics():
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    x, y = np.meshgrid(x, y)
    u = x**2 + y ** 2 + 1
    v = x**2 - y ** 2 - 2

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=70)

    ax.streamplot(
        x, y, u, v,
        density=0.4,
        color='0.7',
        arrowsize=0.0,
        broken_streamlines=False,
    )
    ax.set_aspect('equal')
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], label=['-1', '-0.5', '0', '0.5', '1'])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1], label=['-1', '-0.5', '0', '0.5', '1'])
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xlabel('$x$', fontdict=dict(fontsize=16))
    ax.set_ylabel('$y$', fontdict=dict(fontsize=16), rotation=0)

    fig.savefig(visualization_helper.get_folder() / 'unsolvable-characteristics.pdf', bbox_inches='tight')


class ComputeMetrics(ActionCallback):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.max_residual_over_c = []
        self.max_abs_err = []
        _x = np.linspace(-1, 1, 500)
        _y = np.linspace(-1, 1, 500)
        self.x, self.y = np.meshgrid(_x, _y)
        self.x, self.y = self.x.flatten(), self.y.flatten()
        print('__init__ ', self.x.shape, self.y.shape)
    
    def __call__(self, solver: Solver2D):
        res = solver.get_residuals(self.x, self.y, to_numpy=True, best=False)
        c = 3 - 2 * self.x
        self.max_residual_over_c.append(abs(res / c).max())
        u = solver.get_solution(best=False)(self.x, self.y, to_numpy=True)
        v = np.zeros_like(u) + 2
        self.max_abs_err.append(abs(u - v).max())

        assert self.max_abs_err[-1] <= self.max_residual_over_c[-1]


if __name__ == "__main__":
    set_seed(0)
    visualization_helper.setup()

    plot_characteristics()

    try:
        with open('pde-max-residual-over-c.pkl', 'rb') as f:
            max_residual_over_c = pkl.load(f)
        with open('pde-max-abs-err.pkl', 'rb') as f:
            max_abs_err =  pkl.load(f)
        print('Loaded cached data.')
    except FileNotFoundError:
        print('Failed to loaded cached data. Retraining started.')
        pde = lambda u, x, y: [ 
            (x**2 + y**2+1) * diff(u, x)
            + (x**2 - y**2 + 2) *diff(u, y) 
            + (3 - 2*x) * u
            - (6 - 4*x)
        ]

        conditions = [DirichletBVP2D(
            x_min=-1, x_min_val=lambda y: 2,
            x_max=+1, x_max_val=lambda y: 2,
            y_min=-1, y_min_val=lambda x: 2,
            y_max=+1, y_max_val=lambda x: 2,
        )]

        solver = Solver2D(
            pde, conditions,
            xy_min=(-1, -1), 
            xy_max=(1, 1),
            n_batches_valid=0,
        )

        metrics_cb = ComputeMetrics()

        solver.fit(1000, callbacks=[metrics_cb])
        max_abs_err = metrics_cb.max_abs_err
        max_residual_over_c = metrics_cb.max_residual_over_c

        with open('pde-max-residual-over-c.pkl', 'wb') as f:
            pkl.dump(max_residual_over_c, f)
        with open('pde-max-abs-err.pkl', 'wb') as f:
            pkl.dump(max_abs_err, f)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 1.5), dpi=70)
    epochs = 1 + np.arange(len(max_residual_over_c))
    ax.plot(epochs, max_residual_over_c, 'r:', label=r'$\mathcal B(x, y) \equiv B$')
    ax.plot(epochs, max_abs_err, color='blue', label=r'$\displaystyle \max_{(x, y)\in\Omega}|\eta(x, y)|$')
    ax.set_xlabel('Number of Training Epochs', fontdict=dict(fontsize=14))
    ax.set_yscale('log')
    ax.legend(prop=dict(size=14))
    ax.tick_params(axis='both', which='major', labelsize=16)

    fig.savefig(visualization_helper.get_folder() / 'pde-constant-bound.pdf', bbox_inches='tight')
