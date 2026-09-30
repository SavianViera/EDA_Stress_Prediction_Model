# ---------------------------------------------------------------------------
# Filter matrix construction
# ---------------------------------------------------------------------------

import numpy as np
from scipy.linalg import toeplitz

def make_H(n, fs, tau1, tau2, mode):
    """
    Parameters
    ----------
    n : int
        Signal length, needs to be bigger than fs*40.
    tau1, tau2 : float
        tau1 > tau2 > 0.
    mode : str
        One of 'top', 'tril', 'middle'.

    Returns
    -------
    H : ndarray, shape (n, n)
    """
    # Using model from Compressed Sensing paper, construct H
    # updated 12/18/2025
    # Xuemei Chen (converted to Python)

    T = 40  # duration in seconds
    scale = 1
    TT = T / scale
    #fs = 4  # signal sampled at every 1/fs second
    f_len = int(TT * fs)

    u = np.linspace(0, TT, f_len)
    uu = u * scale
    f = 2 * (np.exp(-uu / tau1) - np.exp(-uu / tau2))  # 160 coordinates

    ncols_main_H = n + f_len - 1
    h = np.zeros(n + f_len - 1)
    h[0:f_len] = f

    # Construct the Toeplitz matrix
    hflip = np.flip(h)          # fliplr(h')
    r = np.roll(hflip, 1)       # circshift(hflip, 1)
    H_temp = toeplitz(h, r)     # toeplitz(h, r)

    main_H = H_temp[:, 0:n]     # n+f_len-1 by n

    if mode == 'top':
        H = main_H[0:n, :]
    elif mode == 'tril':
        H = main_H[0:n, :]
        H = np.tril(H)
    elif mode == 'middle':
        center_idx = int(np.floor(ncols_main_H / 2))  # 1-based reference
        half_n = int(np.floor(n / 2))
        start_row_1idx = center_idx - half_n
        end_row_1idx = n + center_idx - half_n - 1
        H = main_H[start_row_1idx - 1:end_row_1idx, :]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return H

"""
ADMM solver for:

    min_{X,L}  lambda * ||X||_1 + ||L||_*
    s.t.       H X + L = M,   X >= 0,   L >= 0

Outer loop:  standard 2-block ADMM on f(X) = lambda*||X||_1 + I_{X>=0}(X)
             and g(L) = ||L||_* + I_{L>=0}(L), coupled by H X + L = M.

X-subproblem:  min_{X>=0} lambda*<1,X> + (rho/2)||H X - c||_F^2
               solved in closed form if H = I, otherwise by FISTA
               (projected accelerated gradient) since the objective on
               X>=0 is just a nonnegative-least-squares + linear term.

L-subproblem:  min_{L>=0} ||L||_* + (rho/2)||L - b||_F^2
               has no closed form (SVT does not respect L>=0), so it is
               solved by a small *inner* ADMM that alternates
               singular-value thresholding and projection onto L>=0.
"""


# --------------------------------------------------------------------------
# Basic proximal building blocks
# --------------------------------------------------------------------------

def svt(Y, tau):
    """Singular value thresholding: prox of tau*||.||_* at Y."""
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    r = np.count_nonzero(s_thresh)
    return (U[:, :r] * s_thresh[:r]) @ Vt[:r, :]


def project_nonneg(X):
    return np.maximum(X, 0.0)


# --------------------------------------------------------------------------
# L-subproblem:  argmin_{L>=0} ||L||_* + (rho/2)||L-b||_F^2
# solved by a short inner ADMM (SVT block <-> nonneg-projection block)
# --------------------------------------------------------------------------

def prox_nuclear_nonneg(b, tau, inner_iters=50, inner_rho=1.0, tol=1e-8):
    """
    Solve  argmin_{L>=0} tau*||L||_* + 0.5*||L-b||_F^2
    via inner ADMM on variables (L1, L2) with L1 = L2:
        L1 = argmin tau*||L1||_* + 0.5*||L1-b||_F^2 + (inner_rho/2)||L1-L2+U||_F^2
        L2 = argmin_{L2>=0}                              (inner_rho/2)||L1-L2+U||_F^2
        U  = U + L1 - L2

    The L1-step itself has a closed form because it is the prox of a
    *sum* of two quadratics plus a nuclear norm: combine the two
    quadratic terms first, then apply SVT.
    """
    L1 = b.copy()
    L2 = project_nonneg(b.copy())
    U = np.zeros_like(b)

    # Combine  0.5*||L1-b||^2 + (inner_rho/2)||L1-L2+U||^2
    # = ( (1+inner_rho)/2 ) * || L1 - w ||^2 + const,  w = (b + inner_rho*(L2-U)) / (1+inner_rho)
    # so the SVT threshold becomes tau/(1+inner_rho).
    denom = 1.0 + inner_rho
    svt_tau = tau / denom

    for _ in range(inner_iters):
        w = (b + inner_rho * (L2 - U)) / denom
        L1_new = svt(w, svt_tau)

        L2_new = project_nonneg(L1_new + U)

        U = U + (L1_new - L2_new)

        primal_res = np.linalg.norm(L1_new - L2_new)
        change = np.linalg.norm(L1_new - L1) + np.linalg.norm(L2_new - L2)
        L1, L2 = L1_new, L2_new

        if primal_res < tol and change < tol:
            break

    # Return the feasible (nonnegative) copy
    return L2


# --------------------------------------------------------------------------
# X-subproblem:  argmin_{X>=0} lambda*<1,X> + (rho/2)||H X - c||_F^2
# --------------------------------------------------------------------------

def prox_l1_nonneg_linear(H, c, lam, rho, X0=None,
                           max_iters=200, tol=1e-8, HtH=None, Ht=None):
    """
    Solve  argmin_{X>=0} lam*sum(X) + (rho/2)||H X - c||_F^2

    Closed form when H = I:   X = max(c - lam/rho, 0)   (one-sided soft-threshold)
    Otherwise: FISTA (accelerated projected gradient).
        grad(X) = rho * H^T (H X - c) + lam * ones_like(X)
        step    = 1 / (rho * L_H),  L_H = largest eigenvalue of H^T H
    """
    m, p = H.shape
    n = c.shape[1]

    # Fast path: H is identity (or effectively square-orthonormal-diagonal-1)
    if H.shape[0] == H.shape[1] and np.allclose(H, np.eye(m)):
        return project_nonneg(c - lam / rho)

    if Ht is None:
        Ht = H.T
    if HtH is None:
        HtH = Ht @ H

    # Lipschitz constant of the smooth part's gradient
    L_H = np.linalg.eigvalsh(HtH).max()
    step = 1.0 / (rho * L_H + 1e-12)

    X = np.zeros((p, n)) if X0 is None else X0.copy()
    Y = X.copy()
    t = 1.0
    ones = np.ones((p, n))

    for _ in range(max_iters):
        grad = rho * (Ht @ (H @ Y - c)) + lam * ones
        X_new = project_nonneg(Y - step * grad)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        Y = X_new + ((t - 1) / t_new) * (X_new - X)

        if np.linalg.norm(X_new - X) < tol * (1 + np.linalg.norm(X)):
            X = X_new
            break

        X, t = X_new, t_new

    return X

def prox_nuclear_nonneg_pg(b, tau, iters=100, step=0.5, accelerate=True,
                            tol=1e-8):
    """
    Approximate solve of  argmin_{L>=0} tau*||L||_* + 0.5*||L-b||_F^2
    via projected proximal-gradient (optionally FISTA-accelerated).
 
    This is CHEAPER than prox_nuclear_nonneg (the inner-ADMM version):
    it is a single loop with one SVD per iteration and no dual variable
    or inner penalty parameter to tune. It is also only APPROXIMATE:
    the update
 
        L_{t+1} = project_nonneg( SVT_{step*tau}( Y_t - step*(Y_t - b) ) )
 
    treats the quadratic term 0.5||L-b||^2 as the smooth part (gradient
    Y-b, Lipschitz constant 1) and the nuclear norm as the nonsmooth
    part (prox = SVT); nonnegativity is then enforced by projecting the
    result at every iteration rather than through an exact joint prox.
    A gradient step of step=1 collapses to the single-shot "SVT then
    clip" heuristic (which ignores L>=0 during the SVT step); step<1
    lets the running iterate carry information about the nonnegativity
    constraint into the next shrinkage step, improving the approximation
    at the cost of needing more iterations.
 
    Parameters
    ----------
    b : (m, n) array          target matrix (e.g. M - H X - U)
    tau : float               nuclear-norm threshold (= 1/rho in the outer ADMM)
    iters : int               number of proximal-gradient iterations
    step : float in (0, 1]    step size; step=1 is the fastest but most
                               biased (no interaction between the two
                               nonsmooth terms across iterations)
    accelerate : bool         use FISTA-style momentum
    tol : float               relative change stopping tolerance
 
    Returns
    -------
    L : (m, n) array, entrywise nonnegative
    """
    L = project_nonneg(b.copy())
    Y = L.copy()
    t = 1.0
 
    for _ in range(iters):
        grad = Y - b                      # gradient of 0.5||Y-b||^2
        Z = Y - step * grad
        Z = svt(Z, step * tau)
        L_new = project_nonneg(Z)
 
        if accelerate:
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            Y = L_new + ((t - 1) / t_new) * (L_new - L)
            t = t_new
        else:
            Y = L_new
 
        if np.linalg.norm(L_new - L) < tol * (1 + np.linalg.norm(L)):
            L = L_new
            break
        L = L_new
 
    return L
 
# --------------------------------------------------------------------------
# Outer ADMM
# --------------------------------------------------------------------------

def admm_nn_separation(H, M, lam, rho=1.0, max_iters=500,
                        tol_pri=1e-6, tol_dual=1e-6,
                        x_inner_iters=200, l_inner_iters=50,
                        l_method="admm", l_pg_step=0.5,
                        verbose=False):
    """
    Solve   min_{X,L}  lam*||X||_1 + ||L||_*
            s.t.        H X + L = M,  X >= 0,  L >= 0
 
    Parameters
    ----------
    H : (m, p) array
    M : (m, n) array
    lam : float, sparsity weight
    rho : float, ADMM penalty parameter
    l_method : {"admm", "pg"}
        How to solve the L-subproblem each outer iteration.
        "admm" (default) - exact inner ADMM (prox_nuclear_nonneg):
                            more accurate per outer iteration, more
                            expensive (nested loop with dual variable).
        "pg"              - projected proximal-gradient
                            (prox_nuclear_nonneg_pg): cheaper single
                            loop, approximate; typically needs more
                            outer ADMM iterations to reach the same
                            accuracy, but each call is lighter-weight.
    l_pg_step : float
        Step size used when l_method="pg" (ignored otherwise).
 
    Returns
    -------
    X, L : the estimated sparse and low-rank nonnegative matrices
    history : dict of residual norms per iteration
    """
    m, p = H.shape
    _, n = M.shape
 
    X = np.zeros((p, n))
    L = project_nonneg(M.copy())
    U = np.zeros((m, n))  # scaled dual variable
 
    Ht = H.T
    HtH = Ht @ H
 
    history = {"r_norm": [], "s_norm": []}
    L_prev = L.copy()
 
    for k in range(max_iters):
        # ---- X-update ----
        c = M - L - U
        X = prox_l1_nonneg_linear(H, c, lam, rho, X0=X,
                                   max_iters=x_inner_iters,
                                   HtH=HtH, Ht=Ht)
 
        # ---- L-update ----
        b = M - H @ X - U
        if l_method == "admm":
            L = prox_nuclear_nonneg(b, tau=1.0 / rho,
                                     inner_iters=l_inner_iters,
                                     inner_rho=1.0)
        elif l_method == "pg":
            L = prox_nuclear_nonneg_pg(b, tau=1.0 / rho,
                                        iters=l_inner_iters,
                                        step=l_pg_step)
        else:
            raise ValueError("l_method must be 'admm' or 'pg'")
 
        # ---- dual update ----
        r = H @ X + L - M          # primal residual
        U = U + r
 
        # ---- residuals for stopping ----
        s = rho * (Ht @ (L - L_prev))  # dual residual (approx, w.r.t. X-block)
        L_prev = L.copy()
 
        r_norm = np.linalg.norm(r)
        s_norm = np.linalg.norm(s)
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
 
        if verbose and (k % 100 == 0 or k == max_iters - 1):
            obj = lam * np.sum(X) + np.linalg.norm(L, 'nuc')
            print(f"iter {k:4d}  obj={obj:10.4f}  "
                  f"||r||={r_norm:.3e}  ||s||={s_norm:.3e}")
 
        if r_norm < tol_pri and s_norm < tol_dual:
            break
 
    return X, L, history

def nn_gmsEDA(signal, fs, num_of_chops, overlap, lmethod = 'pg'):
    """
    Decompose EDA signal(s) into a sparse component (X) and a baseline
    component (baseline) via nonnegative generalized matrix separation.

    Parameters
    ----------
    signal : ndarray, shape (n, m)
        Each column is an EDA signal.

    Returns
    -------
    X : ndarray
        Reconstructed (decomposed) signal component.
    baseline : ndarray
        Reconstructed baseline component.
    """

    #num_of_chops = 5
    #overlap = 0.8
    signal = np.array(signal)
    if signal.ndim == 1:
        signal = signal.reshape([-1,1])

    Y = chop_data(signal, num_of_chops, overlap)
    N, K = Y.shape  # N is new signal length after chopping

    # parameter for generating filter matrix H
    tau1 = 2
    tau2 = 0.75
    H = make_H(N, fs, tau1, tau2, 'tril')

    # decompose via GMS-P
    para = {
        'rho_outer': 1,
        'lasso_rho': 1,
        'max_iter': 100,
        'lasso_max_iter': 50,
        'tol_outer': 1e-8,
        'lasso_tol': 1e-5,
        'lasso_decomp': 'svd',
        'lasso_method': 'FISTA',
    }

    lam = 1 / np.sqrt(max(N, K)) * 5

    X_hat, L_hat, hist = admm_nn_separation(H, Y, lam, rho=1.0,
                                            max_iters=200,
                                            tol_pri=1e-6, tol_dual=1e-6,
                                            l_method=lmethod,
                                            verbose=False)
    
    X = reconstr_data(X_hat, signal.shape, num_of_chops, overlap)
    baseline = reconstr_data(L_hat, signal.shape, num_of_chops, overlap)

    return X, baseline

# ---------------------------------------------------------------------------
# Chopping / reconstruction
# ---------------------------------------------------------------------------

def chop_data(data, num_of_chops=2, overlap=0):
    """
    Slices each column of input into segments with optional overlap.
    Processed per column, as different columns represent different participants.

    Parameters
    ----------
    data : ndarray, shape (num_rows, num_cols)
        Input matrix.
    num_of_chops : int
        Number of segments per column if overlap = 0.
    overlap : float
        Overlap ratio, at most 0.5.

    Returns
    -------
    Y : ndarray
        Matrix of chopped data.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    num_rows, num_cols = data.shape

    # Y_nrows will determine the length of columns in Y
    Y_nrows = int(np.ceil(num_rows / num_of_chops))
    overlap_length = int(np.floor(Y_nrows * overlap))
    nonlap_length = Y_nrows - overlap_length
    temp = int(np.floor((num_rows - Y_nrows) / nonlap_length))

    if temp < (num_rows - Y_nrows) / nonlap_length:
        multiplier = temp + 2
    else:
        multiplier = temp + 1

    Y = np.zeros((Y_nrows, num_cols * multiplier))

    for col in range(num_cols):
        y = np.zeros((Y_nrows, multiplier))

        if temp < (num_rows - Y_nrows) / nonlap_length:
            for i in range(multiplier - 1):
                start_idx = nonlap_length * i
                end_idx = nonlap_length * i + Y_nrows
                y[:, i] = data[start_idx:end_idx, col]
            # last col of y is different
            y[:, multiplier - 1] = data[(num_rows - Y_nrows):num_rows, col]
        else:
            for i in range(multiplier):
                start_idx = nonlap_length * i
                end_idx = nonlap_length * i + Y_nrows
                y[:, i] = data[start_idx:end_idx, col]

        s_idx = col * multiplier
        e_idx = (col + 1) * multiplier
        Y[:, s_idx:e_idx] = y

    return Y


def reconstr_data(Y, data_size, num_of_chops, overlap):
    """
    This is the reverse of chop_data().
    Takes the average at overlap area.
    created on 8/16/2025 by Xuemei Chen (converted to Python)
    """
    num_rows = data_size[0]
    num_cols = data_size[1]

    Y_nrows, Y_ncols = Y.shape

    multiplier = int(np.ceil(Y_ncols / num_cols))

    idx_input = np.arange(1, num_rows + 1).reshape(-1, 1).astype(float)
    idx = chop_data(idx_input, num_of_chops, overlap)
    idx = np.round(idx).astype(int)  # idx holds 1-based original row positions

    data = np.zeros(data_size)

    idx_diff = idx[0, :] - idx[0, 0]  # length = multiplier

    for ci in range(num_cols):
        j = 0  # corresponds to MATLAB j = 1
        col_j = idx[:, j]
        col_j1 = idx[:, j + 1]

        nonoverlap_idx = np.setdiff1d(col_j, col_j1)
        Y_col = j + ci * multiplier
        data[nonoverlap_idx - 1, ci] = Y[nonoverlap_idx - 1, Y_col]

        overlap_idx = np.intersect1d(col_j, col_j1)
        Y_col2 = (j + 1) + ci * multiplier
        data[overlap_idx - 1, ci] = 0.5 * (
            Y[overlap_idx - idx_diff[j] - 1, Y_col]
            + Y[overlap_idx - idx_diff[j + 1] - 1, Y_col2]
        )

        for j in range(1, multiplier - 1):  # MATLAB j = 2:(multiplier-1)
            col_j = idx[:, j]
            col_j1 = idx[:, j + 1]
            col_jm1 = idx[:, j - 1]

            nonoverlap_idx = np.setdiff1d(col_j, col_j1)
            # discard if already covered by previous j
            nonoverlap_idx2 = np.setdiff1d(nonoverlap_idx, col_jm1)
            Y_col = j + ci * multiplier
            data[nonoverlap_idx2 - 1, ci] = Y[nonoverlap_idx2 - idx_diff[j] - 1, Y_col]

            overlap_idx = np.intersect1d(col_j, col_j1)
            Y_col2 = (j + 1) + ci * multiplier
            data[overlap_idx - 1, ci] = 0.5 * (
                Y[overlap_idx - idx_diff[j] - 1, Y_col]
                + Y[overlap_idx - idx_diff[j + 1] - 1, Y_col2]
            )

        j = multiplier - 1  # corresponds to MATLAB j = multiplier
        col_j = idx[:, j]
        col_jm1 = idx[:, j - 1]
        need_idx = np.setdiff1d(col_j, col_jm1)
        Y_col = j + ci * multiplier
        data[need_idx - 1, ci] = Y[need_idx - idx_diff[j] - 1, Y_col]

    return data

