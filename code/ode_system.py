from pathlib import Path

import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import torch

from neurodiffeq import diff
from neurodiffeq.conditions import IVP, NoCondition
from neurodiffeq.solvers import Solver1D
from neurodiffeq.utils import set_seed

from util import OperatorIStable
import visualization_helper



def stackup(fn):
    def inner(t, to_numpy=True):
        t = t.reshape(-1, 1)
        out = fn(t, to_numpy=to_numpy)
        if isinstance(out[0], torch.Tensor):
            return torch.cat(out, dim=1)
        else:
            return np.concatenate(out, axis=1)
    return inner


if __name__ == "__main__":
    set_seed(0)

    visualization_helper.setup()

    J = np.array([
        [4, 1, 0, 0, 0, 0],  # Jordan Block 1: shape 3x3, eigenvlaue=4
        [0, 4, 1, 0, 0, 0],
        [0, 0, 4, 0, 0, 0],
        [0, 0, 0, 3, 1, 0],  # Jordan Block 2: shape 2x2: eigenvalue=3
        [0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 2],  # Jordan Block 3: shape 1x1, eigenvalue=2
    ])

    P = ortho_group.rvs(dim=6)
    # P = np.random.rand(6, 6)
    A = P @ J @ np.linalg.pinv(P)
    A_tensor = torch.tensor(A)
    P_tensor = torch.tensor(P)

    def v_fn(t, to_numpy=True):
        if isinstance(t, np.ndarray):
            t = torch.tensor(t, requires_grad=True)
        t = t.reshape(-1, 1)
        out = torch.cat([
            torch.sin(t),
            torch.log(t + 1),
            t + 1,
            t ** 2,
            torch.exp(t),
            torch.cos(t),
        ], dim=1) @ P_tensor.T

        return out.detach().cpu().numpy() if to_numpy else out

    def eq(*args):
        *us, t = args
        dudt = torch.cat([diff(u, t) for u in us], dim=1)
        Au = torch.cat(us, dim=1) @ A_tensor.T
        force = torch.cat([
            torch.cos(t) + 4 * torch.sin(t) + torch.log(1+t),
            1/(1+t) + 4 * torch.log(1+t) + (t+1),
            4 * t + 5,
            2 * t + 3 * t**2 + torch.exp(t),
            4 * torch.exp(t),
            2 * torch.cos(t) - torch.sin(t),
        ], dim=1) @ P_tensor.T

        assert len(Au.shape) == 2 and Au.shape[1] == 6
        assert dudt.shape == Au.shape == force.shape

        return [dudt + Au - force]

    v0s = (np.array([[0, 0, 1, 0, 1, 1]]) @ P.T).flatten().tolist()

    solver = Solver1D(
        ode_system=eq,
        conditions=[IVP(0, v0) for v0 in v0s],
        t_min=0,
        t_max=1,
        n_batches_valid=1,
    )

    solver.fit(1000)
    u_fn = stackup(solver.get_solution(best=False))
    DOMAIN = np.linspace(0, 1, 10000)
    us = u_fn(DOMAIN, to_numpy=True)
    vs = v_fn(DOMAIN, to_numpy=True)
    errs = us - vs

    residual = solver.get_residuals(
        DOMAIN, to_numpy=True, best=False, no_reshape=True)

    def get_operator_matrix_output(psi):
        psi1, psi2, psi3, psi4, psi5, psi6 = psi.T
        I41 = OperatorIStable(-4, integral_multiplicity=1).bind_domain(DOMAIN)
        I42 = OperatorIStable(-4, integral_multiplicity=2).bind_domain(DOMAIN)
        I43 = OperatorIStable(-4, integral_multiplicity=3).bind_domain(DOMAIN)
        I31 = OperatorIStable(-3, integral_multiplicity=1).bind_domain(DOMAIN)
        I32 = OperatorIStable(-3, integral_multiplicity=2).bind_domain(DOMAIN)
        I21 = OperatorIStable(-2, integral_multiplicity=1).bind_domain(DOMAIN)
        return np.stack([
            I41(psi1) + I42(psi2) + I43(psi3),
            I41(psi2) + I43(psi3),
            I41(psi3),
            I31(psi4) + I32(psi5),
            I31(psi5),
            I21(psi6),
        ], axis=0).T  # shape: n x 6

    # componenet-wise bound
    I_r_abs = get_operator_matrix_output(abs(residual) @ abs(np.linalg.pinv(P)).T)
    B_vector = I_r_abs @ abs(P).T
    assert (abs(errs) <= B_vector).all()

    # norm bound
    r_norm = np.linalg.norm(residual, ord=2, axis=1, keepdims=True)
    r_norm_expanded = r_norm * np.ones_like(residual)
    I = get_operator_matrix_output(r_norm_expanded)
    I_norm = np.linalg.norm(I, ord=2, axis=1)
    cond_P = np.linalg.cond(P, p=2)
    B_scalar = cond_P * I_norm
    assert (np.linalg.norm(errs, axis=1) <= B_scalar).all()

    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=70, figsize=(6, 4), height_ratios=[5, 2])
    colors = ['blue', 'red', 'orange', 'magenta', '0.3', 'green']
    for i in range(6):
        ax1.plot(DOMAIN, abs(B_vector)[:, i], ':', color=colors[i], zorder=100+i, label=rf'$\mathbf{{\mathcal{{B}}}}_{i+1}$')
        ax1.plot(DOMAIN, abs(errs)[:, i], color=colors[i], label=rf'$|\eta_{i+1}|$')
    ax1.set_ylabel(r'Component Abs. Err.', fontdict=dict(fontsize=16))
    ax1.legend(ncol=6, prop=dict(size=14), borderaxespad=0, handletextpad=0.2, columnspacing=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    ax2.plot(DOMAIN, B_scalar, ':', color='black', label=rf'$\mathcal{{B}}$')
    ax2.plot(DOMAIN, np.linalg.norm(errs, axis=1), color='black',label=r'$\|\eta\|$')
    ax2.set_xlabel('Temporal Domain: $t \in I = [0, 1]$', fontdict=dict(fontsize=16))
    ax2.set_ylabel(r'Err. Norm', fontdict=dict(fontsize=16))
    ax2.legend(ncol=2, prop=dict(size=14), loc='upper left', handletextpad=0.2, columnspacing=0.4)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    fig.savefig(visualization_helper.get_folder() / 'system-bound.pdf', bbox_inches='tight')