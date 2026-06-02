# nnsvdlrc.py
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import issparse
from lra_nnls_hals import lra_nnls_hals_updt


def nnsvdlrc(X, r, delta=0.05, maxiter=20):
    """
    SVD-based initialization for NMF using Nonnegative SVD with Low-Rank Correction.

    Atif M. Syed, Sameer Qazi, Nicolas Gillis (2019).
    Improved SVD-based Initialization for Nonnegative Matrix Factorization
    using Low-Rank Correction. Pattern Recognition Letters, 122, 53-59.

    Parameters
    ----------
    X       : sparse or dense nonnegative (m x n) matrix
    r       : factorization rank
    delta   : stopping parameter for low-rank correction (default: 0.05)
    maxiter : maximum iterations for low-rank correction (default: 20)

    Returns
    -------
    W : (m x r) nonnegative matrix
    H : (r x n) nonnegative matrix
    Y : (m x p) left truncated SVD factor  s.t. Y @ Z ≈ X
    Z : (p x n) right truncated SVD factor
    e : relative error ||X - WH||_F / ||X||_F throughout correction
    """
    p = int(np.floor(r / 2 + 1))

    # truncated SVD — svds returns singular values in ascending order, so reverse
    u, s, vt = svds(X, k=p)
    idx = np.argsort(s)[::-1]
    u, s, vt = u[:, idx], s[idx], vt[idx, :]

    sqrt_s = np.sqrt(s)
    Y = u * sqrt_s                             # (m x p)
    Z = (vt.T * sqrt_s).T                      # (p x n)

    m = X.shape[0]
    W = np.zeros((m, r))
    H = np.zeros((r, X.shape[1]))

    # best rank-one approximation
    W[:, 0] = np.abs(Y[:, 0])
    H[0, :] = np.abs(Z[0, :])

    # next (r-1) rank-one factors
    i = 1
    j = 1
    while i < r:
        if i % 2 == 1:
            W[:, i] = np.maximum(Y[:, j], 0)
            H[i, :] = np.maximum(Z[j, :], 0)
        else:
            W[:, i] = np.maximum(-Y[:, j], 0)
            H[i, :] = np.maximum(-Z[j, :], 0)
            j += 1
        i += 1

    # scale (W, H)
    WtY  = W.T @ Y                             # (r x p)
    WtYZ = WtY @ Z                             # (r x n)
    WtW  = W.T @ W                             # (r x r)
    HHt  = H @ H.T                             # (r x r)
    scaling = np.sum(WtYZ * H) / np.sum(WtW * HHt)
    sqrt_sc = np.sqrt(scaling)
    W    = W * sqrt_sc
    H    = H * sqrt_sc
    WtYZ = WtYZ * sqrt_sc
    WtW  = WtW * scaling
    HHt  = HHt * scaling

    # relative error
    ZZt = Z @ Z.T                              # (p x p)
    YtY = Y.T @ Y                              # (p x p)
    nX  = np.sqrt(np.sum(YtY * ZZt))
    e   = [np.sqrt(max(nX**2 - 2 * np.sum(WtYZ * H) + np.sum(WtW * HHt), 0)) / nX]

    # low-rank correction via accelerated HALS
    k = 0
    while (k == 0 or e[k - 1] - e[k] > delta * e[0]) and k < maxiter:
        W, _, _ = lra_nnls_hals_updt(Z.T, Y.T, H.T, W.T)
        W = W.T
        H, WtW, WtYZ = lra_nnls_hals_updt(Y, Z, W, H)
        err = np.sqrt(max(nX**2 - 2 * np.sum(WtYZ * H) + np.sum(WtW * (H @ H.T)), 0)) / nX
        e.append(err)
        k += 1

    return W, H, Y, Z, np.array(e)