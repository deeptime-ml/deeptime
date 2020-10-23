r"""
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
"""

import numpy as np
import scipy.sparse

from . import objective_sparse

from .linsolve import mydot
from .linsolve_sparse import factor_aug as factor
from .linsolve_sparse import solve_factorized_aug as solve_factorized

__all__ = ['solve_mle_rev', ]

"""Algorithm parameters"""
GAMMA_MIN = 0.0001
GAMMA_MAX = 0.01
GAMMA_BAR = 0.49
KAPPA = 0.01
TAU = 0.5
RHO = min(0.2, min((0.5 * GAMMA_BAR) ** (1.0 / TAU), 1.0 - KAPPA))
SIGMA = 0.1
BETA = 100000


def mynorm(x):
    return np.linalg.norm(x)


def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


def primal_dual_solve(func, x0, Dfunc, A, b, G, h, args=(), tol=1e-10,
                      maxiter=100, full_output=False):
    """Wrap calls to function and Jacobian"""
    fcalls, func = wrap_function(func, args)
    Dfcalls, Dfunc = wrap_function(Dfunc, args)

    M, N = G.shape
    P, N = A.shape

    """Total number of inequality constraints"""
    m = M

    def gap(z):
        r"""Gap-function"""
        l = z[N + P:N + P + M]
        s = z[N + P + M:]
        return mydot(l, s) / m

    def centrality(z):
        r"""Centrality function"""
        l = z[N + P:N + P + M]
        s = z[N + P + M:]
        return np.min(l * s)

    def KKT(z, sigma=0.0):
        r"""KKT system (possible perturbed)."""

        """Primal variable"""
        x = z[0:N]

        """Multiplier for equality constraints"""
        nu = z[N:N + P]

        """Multiplier for inequality constraints"""
        l = z[N + P:N + P + M]

        """Slacks"""
        s = z[N + P + M:]

        """Evaluate objective function"""
        F = func(x)

        """Dual infeasibility"""
        rdual = F + mydot(A.transpose(), nu) + mydot(G.transpose(), l)

        """Primal infeasibilities"""
        rprim1 = mydot(A, x) - b
        rprim2 = mydot(G, x) - h + s

        """Complementary slackness (perturbed)"""
        mu = gap(z)
        rslack = l * s - sigma * mu

        return np.hstack((rdual, rprim1, rprim2, rslack))

    def step_fast(z, KKTval, LU, G, A, mu, beta, gamma, alpha0):
        r"""Affine scaling step."""
        dz = solve_factorized(z, KKTval, LU, G, A)

        """Reduce step length until slacks s and multipliers l are positive"""
        alpha = 1.0 * alpha_0
        k = 0
        for k in range(10):
            z_new = z + alpha * dz
            if np.all(z_new[N + P:] > 0.0):
                break
            alpha *= 0.5
            k += 1
        if k == 10 - 1:
            raise RuntimeError("Maximum steplength reduction reached")

        """Reduce step length until iterates lie in correct neighborhood"""
        for k in range(10):
            z_new = z + alpha * dz
            KKTval_new = KKT(z_new)
            dual = mynorm(KKTval_new[0:N])
            prim = mynorm(KKTval_new[N:N + P + M])
            mu_new = gap(z_new)
            cent_new = centrality(z_new)

            if (dual <= beta * mu_new and prim <= beta * mu_new and
                    cent_new >= gamma * mu_new):
                break
            alpha *= 0.5
            # alpha *= 0.95
        if k == 10 - 1:
            raise RuntimeError("Maximum steplength reduction reached")
        return z_new, mu_new

    def step_safe(z, KKTval, LU, G, A, mu, beta, gamma):
        r"""Centering step."""
        dz = solve_factorized(z, KKTval, LU, G, A)

        """Reduce step length until slacks s and multipliers l are positive"""
        alpha = 1.0
        k = 0
        for k in range(10):
            z_new = z + alpha * dz
            if np.all(z_new[N + P:] > 0.0):
                break
            alpha *= 0.5
            k += 1
        if k == 10 - 1:
            raise RuntimeError("Maximum steplength reduction (pos.) reached")

        """Reduce step length until iterates lie in correct neighborhood
        and mu fulfills Armijo condition"""
        k = 0
        for k in range(10):
            z_new = z + alpha * dz
            KKTval_new = KKT(z_new, sigma=SIGMA)
            dual = mynorm(KKTval_new[0:N])
            prim = mynorm(KKTval_new[N:N + P + M])
            mu_new = gap(z_new)
            cent_new = centrality(z_new)

            if (dual <= beta * mu_new and prim <= beta * mu_new and
                    cent_new >= gamma * mu_new and
                    mu_new <= (1.0 - KAPPA * alpha * (1.0 - SIGMA)) * mu):
                break
            alpha *= 0.5
        if k == 10 - 1:
            raise RuntimeError("Maximum steplength reduction reached")
        return z_new, mu_new

    """INITIALIZATION"""

    """Initial Slacks for inequality constraints"""
    s0 = -1.0 * (mydot(G, x0) - h)

    """Initial multipliers for inequality constraints"""
    l0 = 1.0 * np.ones(M)

    """Initial multipliers for equality constraints"""
    nu0 = np.zeros(P)

    """Initial point"""
    z0 = np.hstack((x0, nu0, l0, s0))

    """Initial KKT-values"""
    KKTval0 = KKT(z0, sigma=0.0)
    mu0 = gap(z0)
    dual0 = mynorm(KKTval0[0:N])
    prim0 = mynorm(KKTval0[N:N + P + M])

    """Initial neighborhood"""
    beta = BETA * np.sqrt(dual0 ** 2 + prim0 ** 2) / mu0
    gamma = 1.0 * GAMMA_MAX

    """Number of fast steps"""
    t = 0

    """Number of iterations"""
    n = 0

    """Dummy variable for step type"""
    step_type = " "

    """MAIN LOOP"""
    z = z0
    x = z0[0:N]
    KKTval = KKTval0
    dual = dual0
    prim = prim0
    mu = mu0
    Dfunc_val = Dfunc(x)
    LU = factor(z, Dfunc_val, G, A)

    if full_output:
        info = {'z': []}
        info['z'].append(z)

    for n in range(maxiter):
        """Attempt fast step"""
        beta_new = (1.0 + GAMMA_BAR ** (t + 1)) * beta
        gamma_new = GAMMA_MIN + GAMMA_BAR ** (t + 1) * (GAMMA_MAX - GAMMA_MIN)
        alpha_0 = 1.0 - np.sqrt(mu) / GAMMA_BAR ** t

        if alpha_0 > 0.0:
            z_new, mu_new = step_fast(z, KKTval, LU, G, A, mu,
                                      beta_new, gamma_new, alpha_0)
            if mu_new < RHO * mu:
                """Fast successful"""
                z = z_new
                mu = mu_new
                beta = beta_new
                gamma = gamma_new
                t += 1
                step_type = "f"
            else:
                """Perturbed right-had side"""
                KKTval_pert = 1.0 * KKTval
                KKTval_pert[N + P + M:] -= SIGMA * mu
                z, mu = step_safe(z, KKTval_pert, LU, G, A, mu, beta, gamma)
                step_type = "s"
        else:
            """Perturbed right-hand side"""
            KKTval_pert = 1.0 * KKTval
            KKTval_pert[N + P + M:] -= SIGMA * mu
            z, mu = step_safe(z, KKTval_pert, LU, G, A, mu, beta, gamma)
            step_type = "s"

        """Compute new iterates"""
        KKTval = KKT(z, sigma=0.0)
        dual = mynorm(KKTval[0:N])
        prim = mynorm(KKTval[N:N + P + M])
        x = z[0:N]
        Dfunc_val = Dfunc(x)
        LU = factor(z, Dfunc_val, G, A)
        if full_output:
            info['z'].append(z)
        if mu < tol and dual < tol and prim < tol:
            break
    if n == maxiter - 1:
        raise RuntimeError("Maximum number of iterations reached")

    if full_output:
        return z[0:N], info
    else:
        return z[0:N]


def solve_mle_rev(C, tol=1e-10, maxiter=100, full_output=False,
                  return_statdist=True):
    """Number of states"""
    M = C.shape[0]

    """Initial guess for primal-point"""
    z0 = np.zeros(2 * M)
    z0[0:M] = 1.0

    """Inequality constraints"""
    # G = np.zeros((M, 2*M))
    # G[np.arange(M), np.arange(M)] = -1.0
    G = -1.0 * scipy.sparse.eye(M, n=2 * M, k=0)
    h = np.zeros(M)

    """Equality constraints"""
    A = np.zeros((1, 2 * M))
    A[0, M] = 1.0
    b = np.array([0.0])

    """Scaling"""
    c0 = C.max()
    C = C / c0

    """Symmetric part"""
    Cs = C + C.T

    """Column sum"""
    c = C.sum(axis=0)

    if scipy.sparse.issparse(C):
        Cs = Cs.tocsr()
        c = c.A1
        A = scipy.sparse.csr_matrix(A)
        F = objective_sparse.F
        DF = objective_sparse.DFsym
        convert_solution = objective_sparse.convert_solution
    else:
        raise ValueError("dense not supported")

    """PDIP iteration"""
    res = primal_dual_solve(F, z0, DF, A, b, G, h,
                            args=(Cs, c),
                            maxiter=maxiter, tol=tol,
                            full_output=full_output)
    if full_output:
        z, info = res
    else:
        z = res
    pi, P = convert_solution(z, Cs)

    result = [P]
    if return_statdist:
        result.append(pi)
    if full_output:
        result.append(info)

    return tuple(result) if len(result) > 1 else result[0]
