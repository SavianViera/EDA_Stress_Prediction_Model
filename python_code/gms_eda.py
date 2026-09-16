"""
gms_eda.py

Generalized Matrix Separation (GMS) decomposition for EDA (electrodermal
activity) signals.

Converted from the MATLAB implementation by Xuemei Chen. Consolidates the
following notebook cells into a single importable/runnable module:

    gmsEDA               - main entry point
    chop_data            - segment a signal into overlapping chunks
    reconstr_data         - reverse of chop_data, averages overlaps back
    make_H               - build the (Toeplitz-based) convolution/filter matrix
    gen_matrix_sep_con    - preconditioned generalized matrix separation
    gen_matrix_sep        - ADMM solver for L + H S = M, min ||L||_* + lam||S||_1
    SVT                   - singular value thresholding
    my_lasso              - lasso solver (ADMM / ISTA / FISTA)
    SoftThresh            - soft-thresholding operator
    pref                  - SVD-based preconditioning helper
    is_circulant          - check whether a matrix is circulant

Usage
-----
As a library:

    from gms_eda import gmsEDA
    X, baseline = gmsEDA(signal)

As a script (signal read from a CSV file, one EDA signal per column):

    python gms_eda.py input.csv --outdir results/
"""

import argparse
import os

import numpy as np
from scipy.linalg import cholesky, solve_triangular, toeplitz


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gmsEDA(signal, fs):
    """
    Decompose EDA signal(s) into a sparse component (X) and a baseline
    component (baseline) via generalized matrix separation.

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
    # 5/16/2026
    # by Xuemei Chen (converted to Python)

    num_of_chops = 5
    overlap = 0.8

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

    lam = 1 / np.sqrt(max(N, K)) * 3

    output = gen_matrix_sep_con(Y, H, lam, para)
    S = output['S']
    # Force non-negativity
    S[S < 0] = 0

    B = Y - H @ S

    X = reconstr_data(S, signal.shape, num_of_chops, overlap)
    baseline = reconstr_data(B, signal.shape, num_of_chops, overlap)

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


# ---------------------------------------------------------------------------
# Filter matrix construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generalized matrix separation
# ---------------------------------------------------------------------------

def gen_matrix_sep_con(M, H, lam, para):
    """
    Preconditioned generalized matrix separation.
    min ||L||_* + lam ||S||_1, subject to L + H S = M

    Parameters
    ----------
    M : ndarray, shape (m, n)
        Given matrix.
    H : ndarray, shape (m, p)
        Given matrix, not necessarily circulant.
    lam : float
    para : dict
        Fields may include:
        - rho_outer
        - rho_inner
        - N_outer
        - N_inner
        - tol_outer
        - tol_inner
        - decomp: 'svd' or 'chol'

    Returns
    -------
    output : dict with fields
        - L: ndarray, m x n
        - S: ndarray, p x n
        - count_outer
        - para
    """
    # created on 7/18/2025, Xuemei Chen
    # updated on 8/5/2025 (converted to Python)

    u, s_diag, vh = np.linalg.svd(H)
    v = vh.T  # MATLAB's svd returns V (not V'), so v = vh.T here

    k = np.linalg.matrix_rank(H)

    uk = u[:, 0:k]           # m by k
    vk = v[:, 0:k]           # p by k

    s = s_diag[0:k]          # singular values vector, truncated to rank k
    s_inv = 1.0 / s

    C = uk @ np.diag(s_inv) @ uk.T
    CH = uk @ vk.T

    para = dict(para)  # avoid mutating caller's dict, mirror MATLAB's copy semantics
    para['preconditioned'] = True
    para['V_H'] = v

    p = H.shape[1]
    ss = np.zeros(p)
    ss[0:k] = 1
    para['S2_H'] = ss

    outputC = gen_matrix_sep(C @ M, CH, lam, para)

    output = {}
    output['S'] = outputC['S']
    output['L'] = M - H @ outputC['S']
    output['count_outer'] = outputC['count_outer']
    # output['isCirc'] = outputC['isCirc']
    output['para'] = outputC['para']

    return output


def gen_matrix_sep(M, H, lam, para):
    """
    Generalized matrix separation.
    Use ADMM to solve:
        min ||L||_* + lam ||S||_1, subject to L + H S = M

    Parameters
    ----------
    M : ndarray, shape (m, n)
        Given matrix.
    H : ndarray, shape (m, p)
        Given matrix, not necessarily circulant.
    lam : float
        Positive scalar.
    para : dict
        Fields:
          - rho_outer: step size for ADMM, provided by user
          - max_iter: max number of iterations for ADMM
          - tol_outer: convergence tolerance for ADMM
          - preconditioned: True or False. If True, H has been
            preconditioned with all singular values equal to 1
          - lasso_decomp: {'svd', 'chol'}
          - lasso_method: {'ADMM', 'FISTA'}
          - lasso_rho: step size for lasso if lasso_method is ADMM
          - lasso_max_iter: max num of iterations for lasso
          - lasso_tol: convergence tolerance for lasso

    para['preconditioned'] must be provided by the user!

    Returns
    -------
    output : dict with fields
        - L: m x n
        - S: p x n
        - count_outer
        - para: similar to input with possibly more fields
    """
    # updated on 4/9/2025
    # updated on 4/21/2025
    # updated on 7/5/2025: precompute A^Tb outside of loop
    # updated on 8/5/2025: use my_lasso() and take advantage of V if
    # preconditioning H
    # converted to Python

    para = dict(para)  # shallow copy to avoid mutating caller's dict

    m, n = M.shape
    _, p = H.shape

    # default parameter values
    rho_outer_default = 1
    max_iter_default = 200
    tol_outer_default = 1e-7
    lasso_method_default = 'FISTA'
    lasso_decomp_default = 'svd'
    lasso_rho_default = 1
    lasso_tol_default = 1e-5

    # pass the parameters
    if 'preconditioned' not in para:
        raise ValueError('Please specify "para[\'preconditioned\']" value.')

    if 'rho_outer' not in para:
        para['rho_outer'] = rho_outer_default

    if 'max_iter' not in para:
        para['max_iter'] = max_iter_default

    if 'tol_outer' not in para:
        para['tol_outer'] = tol_outer_default

    if 'lasso_method' not in para:
        para['lasso_method'] = lasso_method_default

    if 'lasso_decomp' not in para:
        para['lasso_decomp'] = lasso_decomp_default

    if 'lasso_tol' not in para:
        para['lasso_tol'] = lasso_tol_default

    rho_outer = para['rho_outer']
    N_outer = para['max_iter']
    tol_outer = para['tol_outer']
    lasso_method = para['lasso_method']
    lasso_decomp = para['lasso_decomp']

    # parameters for my_lasso()
    para_lasso = {}
    para_lasso['max_iter'] = para['lasso_max_iter']
    para_lasso['tol'] = para['lasso_tol']
    para_lasso['method'] = lasso_method
    para_lasso['decomp'] = lasso_decomp  # default value

    # para_lasso['isCirculant'] is determined later

    if lasso_method == 'FISTA':
        # FISTA
        # H circulant does not matter
        isCirculant = None
        if para['preconditioned']:
            para_lasso['L'] = 1  # this is biggest singular value of H'H
        else:
            s = np.linalg.svd(H.T @ H, compute_uv=False)
            para_lasso['L'] = s[0]
    elif lasso_method == 'ADMM':
        # ADMM
        if 'lasso_rho' not in para:
            # if para['lasso_rho'] not given
            para['lasso_rho'] = lasso_rho_default
        para_lasso['rho'] = para['lasso_rho']

        if is_circulant(H):
            isCirculant = True
            # para_lasso['isCirculant'] = isCirculant
            d = np.abs(np.fft.fft(H[:, 0]))
            para_lasso['coef'] = d ** 2
        else:
            # more common case for ADMM
            isCirculant = False
            # para_lasso['isCirculant'] = isCirculant
            if (para_lasso['decomp'] == 'svd') and para['preconditioned']:
                # singular values of H'H
                para_lasso['S2_A'] = para['S2_H']  # p by 1
                # singular vectors of H or H'H
                para_lasso['V_A'] = para['V_H']  # p by p
            elif (para_lasso['decomp'] == 'svd') and (not para['preconditioned']):
                para_lasso['V_A'], para_lasso['S2_A'] = pref(H)
            elif para_lasso['decomp'] == 'chol':
                # chol(A, 'lower') in MATLAB returns lower-triangular L s.t. A = L*L'
                A_chol = H.T @ H + para['lasso_rho'] * np.eye(p)
                L_H = cholesky(A_chol, lower=True)
                para_lasso['L'] = L_H
    else:
        raise ValueError(f"Unknown lasso_method: {lasso_method}")

    para_lasso['isCirculant'] = isCirculant
    para['isCirculant'] = isCirculant

    # initialization and main loop
    L = np.zeros((m, n))
    S = np.zeros((p, n))
    U = np.zeros((m, n))

    RelChg = 1
    count_outer = 0
    while RelChg > tol_outer and count_outer < N_outer:
        Slast = S
        Llast = L
        L = SVT(M - H @ S - U, 1 / rho_outer)
        # L[L < 0] = 0  # this line is added on 10/12/2025 (kept commented, as in MATLAB)
        S, _ = my_lasso(H, M - U - L, lam / rho_outer, para_lasso)
        # S[S < 0] = 0  # this line is added on 8/12/2025 (kept commented, as in MATLAB)

        U = U + L + H @ S - M
        count_outer += 1

        # Check convergence
        # finding the relative error
        Ldn = np.linalg.norm(L - Llast, 'fro')
        Sdn = np.linalg.norm(S - Slast, 'fro')
        Ln = np.linalg.norm(Llast, 'fro')
        Sn = np.linalg.norm(Slast, 'fro')

        # updating stopping criteria
        RelChg = (Ldn ** 2 + Sdn ** 2) ** 0.5 / ((Ln ** 2 + Sn ** 2) ** 0.5 + 1)

    output = {}
    output['L'] = L
    output['S'] = S
    output['count_outer'] = count_outer
    output['para'] = para

    return output


def SVT(A, a):
    """Singular value thresholding."""
    U, S, Vh = np.linalg.svd(A, full_matrices=False)  # 'econ' equivalent
    S2 = SoftThresh(S, a)
    B = U @ np.diag(S2) @ Vh
    return B


# ---------------------------------------------------------------------------
# Lasso solver
# ---------------------------------------------------------------------------

def SoftThresh(x, kappa):
    """
    Soft thresholding operator, works for vectors or matrices.
    z(i) = x(i) - kappa if x(i) > kappa
    z(i) = x(i) + kappa if x(i) < -kappa
    z(i) = 0            otherwise
    """
    x = np.asarray(x)
    z = np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)
    return z


def my_lasso(A, b, lam, para):
    """
    Returns argmin_x { 0.5*|Ax-b|^2 + lam*|x|_1 }

    Parameters
    ----------
    A : ndarray, shape (m, p)
    b : ndarray, shape (m, k)
    lam : float
        Positive scalar.
    para : dict
        Fields:
          - method: 'ADMM', 'ISTA', 'FISTA'
          - tol: stopping criteria
          - max_iter
          - V_A, S2_A, L_A
          - rho: step size if method is ADMM
          - L: Lipschitz constant if method is FISTA
          - decomp: 'svd' or 'chol'
          - isCirculant: whether A is circulant. Only used when
            gen_matrix_sep is called.

    Returns
    -------
    x : ndarray, shape (p, k)
    num_iter : int
        Number of iterations run.

    Notes
    -----
    For para['method']: choose ADMM or FISTA. ISTA is slow in general.
    Reference for FISTA: Beck, Amir, and Marc Teboulle.
    "A fast iterative shrinkage-thresholding algorithm for linear inverse
    problems." SIAM journal on imaging sciences 2.1 (2009): 183-202.
    written by Xuemei Chen 8/4/2025 (converted to Python)
    """
    para = dict(para)  # shallow copy, mirrors MATLAB struct pass-by-value

    m, p = A.shape
    _, k = b.shape

    # pass parameter
    max_iter_default = 100
    tol_default = 1e-5
    method_default = 'FISTA'
    decomp_default = 'svd'
    rho_default = 1

    if 'max_iter' not in para:
        para['max_iter'] = max_iter_default
    if 'tol' not in para:
        para['tol'] = tol_default
    if 'method' not in para:
        para['method'] = method_default

    max_iter = para['max_iter']
    tol = para['tol']
    method = para['method']

    if method == 'ADMM':
        if 'rho' not in para:
            para['rho'] = rho_default
        if 'decomp' not in para:
            para['decomp'] = decomp_default
        if 'isCirculant' not in para:
            para['isCirculant'] = is_circulant(A)

        rho = para['rho']
        decomp = para['decomp']

        if para['isCirculant'] and ('coef' not in para):
            d = np.abs(np.fft.fft(A[:, 0]))
            para['coef'] = d ** 2

        if para['isCirculant']:
            coef = para['coef']
        else:  # not circulant
            if decomp == 'svd' and ('V_A' not in para):
                para['V_A'], para['S2_A'] = pref(A)
            if decomp == 'svd':
                V = para['V_A']
                sig = para['S2_A']
            if decomp == 'chol' and ('L_A' not in para):
                para['L_A'] = cholesky(A.T @ A + rho * np.eye(p), lower=True)
            if decomp == 'chol':
                L_A = para['L_A']

    elif method == 'FISTA':
        if 'L' not in para:
            s = np.linalg.svd(A.T @ A, compute_uv=False)
            para['L'] = s[0]
        L = para['L']

    # main part
    x = np.zeros((p, k))
    num_iter = 0

    if method == 'ADMM':
        Atb = A.T @ b  # m by k -- Atb precomputed
        z = np.zeros((p, k))
        u = np.zeros((p, k))
        for j in range(1, max_iter + 1):
            xlast = x

            # update x
            rhs = Atb + rho * (z - u)
            if para['isCirculant']:
                # [m,n]./[m,1] will divide each col
                x = np.fft.ifft(np.fft.fft(rhs, axis=0) / (coef + rho)[:, None], axis=0)
                x = np.real(x)
            elif decomp == 'svd':
                x = (V.T @ rhs) / (sig + rho)[:, None]
                x = V @ x
            elif decomp == 'chol':
                y = solve_triangular(L_A, rhs, lower=True)
                x = solve_triangular(L_A.T, y, lower=False)

            # update z
            z = SoftThresh(x + u, lam / rho)

            # update u
            u = u + x - z

            # check convergence
            x_change = np.linalg.norm(x - xlast, 'fro') / (np.linalg.norm(xlast, 'fro') + 1)
            if x_change < tol:
                break

        num_iter = j

    elif method == 'ISTA':
        s = np.linalg.svd(A.T @ A, compute_uv=False)
        s = s[0]
        t = 1 / s
        for j in range(1, max_iter + 1):
            xlast = x
            y = x - t * (A.T @ (A @ x - b))
            x = SoftThresh(y, lam * t)
            x_change = np.linalg.norm(x - xlast, 'fro') / (np.linalg.norm(xlast, 'fro') + 1)
            if x_change < tol:
                break
        num_iter = j

    elif method == 'FISTA':
        t = 1
        y = x
        for j in range(1, max_iter + 1):
            xlast = x
            tlast = t
            x = SoftThresh(y - (1 / L) * (A.T @ (A @ y - b)), lam / L)

            t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y = x + (tlast - 1) / t * (x - xlast)
            x_change = np.linalg.norm(x - xlast, 'fro') / (np.linalg.norm(xlast, 'fro') + 1)
            if x_change < tol:
                break
        num_iter = j

    return x, num_iter


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def pref(A):
    """
    [~, sig, V] = svd(A'*A)
    """
    p_loc, m_loc = A.shape  # matches MATLAB's [p, m] = size(A)

    if m_loc < p_loc:
        _, sig, Vh = np.linalg.svd(A.T @ A)  # V is m_loc x m_loc
        V = Vh.T
        # sig is already a length-m_loc vector of singular values (>= 0)
    else:
        _, s, Vh = np.linalg.svd(A)  # V is m_loc x m_loc
        V = Vh.T
        sig = np.zeros(m_loc)
        sig[:len(s)] = np.abs(s) ** 2

    return V, sig


def is_circulant(A):
    m, n = A.shape

    # we need A to be square if circulant
    if m != n:
        return False

    first_row = A[0, :]

    # check if each row is a shift of the first row
    for i in range(1, m):
        expected_row = np.roll(first_row, i)
        if np.any(A[i, :] != expected_row):
            return False

    return True


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description="Run GMS EDA decomposition on a signal file (CSV, one "
                     "EDA signal per column, no header)."
    )
    parser.add_argument("input", help="Path to input CSV file.")
    parser.add_argument(
        "--outdir", default=".", help="Directory to write output CSVs to (default: current dir)."
    )
    args = parser.parse_args()

    signal = np.loadtxt(args.input, delimiter=",")
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    X, baseline = gmsEDA(signal)

    os.makedirs(args.outdir, exist_ok=True)
    x_path = os.path.join(args.outdir, "gmsEDA_signal.csv")
    b_path = os.path.join(args.outdir, "gmsEDA_baseline.csv")

    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(b_path, baseline, delimiter=",")

    print(f"Signal component written to: {x_path}")
    print(f"Baseline component written to: {b_path}")


if __name__ == "__main__":
    _main()