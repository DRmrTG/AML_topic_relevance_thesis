# lra_nnls_hals.py
import numpy as np


def lra_nnls_hals_updt(Y, Z, U, V, alphaparam=0.5, delta=0.01):
    """
    NNLS solver via accelerated HALS block-coordinate descent.

    Solves: min_{V >= 0} ||YZ - UV||_F^2

    Adapted from:
    Gillis, N., & Glineur, F. (2012). Accelerated multiplicative updates
    and hierarchical ALS algorithms for nonnegative matrix factorization.
    Neural Computation, 24(4), 1085-1105.

    Parameters
    ----------
    Y          : (m x p) matrix
    Z          : (p x n) matrix  — target is M = Y @ Z
    U          : (m x r) matrix  — fixed factor
    V          : (r x n) matrix  — factor to update (initialisation)
    alphaparam : iteration control parameter (default: 0.5)
    delta      : stopping threshold (default: 0.01)

    Returns
    -------
    V    : updated (r x n) nonnegative matrix
    UtU  : U.T @ U  (r x r)
    UtM  : U.T @ M  (r x n)
    """
    V = V.copy().astype(float)

    KX = np.sum(Y > 0) + np.sum(Z > 0)
    n = V.shape[1]
    m, r = U.shape
    maxiter = int(np.floor(1 + alphaparam * (KX + m * r) / (n * r + n)))

    # precomputations
    UtU = U.T @ U                              # (r x r)
    UtM = (U.T @ Y) @ Z                        # (r x n)

    # coordinate descent
    eps0 = 0.0
    eps  = 1.0
    cnt  = 0

    while eps >= delta**2 * eps0 and cnt < maxiter:
        nodelta = 0.0
        for k in range(r):
            deltaV = np.maximum(
                (UtM[k, :] - UtU[k, :] @ V) / UtU[k, k],
                -V[k, :]
            )
            V[k, :] += deltaV
            nodelta += deltaV @ deltaV

            # safety: prevent zero rows
            if np.all(V[k, :] == 0):
                V[k, :] = 1e-16 * np.max(V)

        if cnt == 0:
            eps0 = nodelta
        eps = nodelta
        cnt += 1

    return V, UtU, UtM