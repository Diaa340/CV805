"""Exact best threshold-additive rule [u_q + w_i > 0] on an arbitrary set of weighted cells (MILP, HiGHS): the exact
additive ceiling of Prop. 2 (the logistic fit used before is only a lower bound on it). Usage: from exact_additive import exact_ceiling"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def exact_ceiling(q, i, y, w=None, M=201.0, time_limit=600):
    """q, i: cell indices; y: 0/1 labels; w: cell weights. Returns (weighted accuracy of the best additive rule, status)."""
    q, i, y = map(np.asarray, (q, i, y)); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    nq, ni, n = q.max() + 1, i.max() + 1, len(y); nv = nq + ni + n
    A = lil_matrix((n, nv)); lb = np.zeros(n); ub = np.zeros(n)
    for r in range(n):
        A[r, q[r]] = 1; A[r, nq + i[r]] = 1
        if y[r]:
            A[r, nq + ni + r] = M; lb[r], ub[r] = 1, np.inf
        else:
            A[r, nq + ni + r] = -M; lb[r], ub[r] = -np.inf, -1
    c = np.r_[np.zeros(nq + ni), w]
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.r_[np.zeros(nq + ni), np.ones(n)],
               bounds=Bounds(np.r_[-50 * np.ones(nq + ni), np.zeros(n)], np.r_[50 * np.ones(nq + ni), np.ones(n)]), options={"time_limit": time_limit})
    z = np.round(res.x[nq + ni:]) if res.x is not None else None
    return (1 - (w * z).sum() / w.sum() if z is not None else np.nan), res.message, getattr(res, "mip_gap", None)


def exact_table_ceiling(T, cols=None, time_limit=600):
    """Exact additive ceiling of a complete table T (rows = questions, columns = images), optionally on a subset of
    columns. Identical columns are merged (they can share one image score without loss). Returns (best accuracy found,
    upper bound, status message)."""
    X = T[:, cols] if cols is not None else T
    pats, cnt = np.unique(X.T, axis=0, return_counts=True)        # distinct column patterns
    R = X.shape[0]; q = np.repeat(np.arange(R), len(pats)); i = np.tile(np.arange(len(pats)), R)
    y = pats.T.ravel(); w = np.tile(cnt, R)
    qq, ii, yy, ww = q, i, y, w
    nq, ni, n = R, len(pats), len(yy); M = 201.0
    A = lil_matrix((n, nq + ni + n)); lb = np.zeros(n); ub = np.zeros(n)
    for r in range(n):
        A[r, qq[r]] = 1; A[r, nq + ii[r]] = 1
        if yy[r]:
            A[r, nq + ni + r] = M; lb[r], ub[r] = 1, np.inf
        else:
            A[r, nq + ni + r] = -M; lb[r], ub[r] = -np.inf, -1
    res = milp(np.r_[np.zeros(nq + ni), ww], constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.r_[np.zeros(nq + ni), np.ones(n)],
               bounds=Bounds(np.r_[-50 * np.ones(nq + ni), np.zeros(n)], np.r_[50 * np.ones(nq + ni), np.ones(n)]), options={"time_limit": time_limit})
    tot = ww.sum(); best = 1 - (ww * np.round(res.x[nq + ni:])).sum() / tot if res.x is not None else np.nan
    dual = getattr(res, "mip_dual_bound", None); upper = 1 - dual / tot if dual is not None else np.nan
    return best, upper, res.message


def exact_partial_ceiling(T, W=None, time_limit=900):
    """Exact additive ceiling of a partial table T (-1 = cell not asked), with optional cell weights W (same shape).
    Columns with identical (pattern, weight) are merged. Returns (best accuracy found, upper bound, status)."""
    W = np.where(T >= 0, 1.0, 0.0) if W is None else np.where(T >= 0, W, 0.0)
    key = np.concatenate([T, W], 0).T; pats, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    R = T.shape[0]; Tp, Wp = pats[:, :R].T, pats[:, R:].T * cnt[None]
    qq, ii = np.where(Tp >= 0); yy = Tp[qq, ii]; ww = Wp[qq, ii]; nq, ni, n = R, Tp.shape[1], len(yy); M = 201.0
    A = lil_matrix((n, nq + ni + n)); lb = np.zeros(n); ub = np.zeros(n)
    for r in range(n):
        A[r, qq[r]] = 1; A[r, nq + ii[r]] = 1
        if yy[r]:
            A[r, nq + ni + r] = M; lb[r], ub[r] = 1, np.inf
        else:
            A[r, nq + ni + r] = -M; lb[r], ub[r] = -np.inf, -1
    res = milp(np.r_[np.zeros(nq + ni), ww], constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.r_[np.zeros(nq + ni), np.ones(n)],
               bounds=Bounds(np.r_[-50 * np.ones(nq + ni), np.zeros(n)], np.r_[50 * np.ones(nq + ni), np.ones(n)]), options={"time_limit": time_limit})
    tot = ww.sum(); best = 1 - (ww * np.round(res.x[nq + ni:])).sum() / tot if res.x is not None else np.nan
    dual = getattr(res, "mip_dual_bound", None); return best, (1 - dual / tot if dual is not None else np.nan), res.message


def exact_partial_fit(T, W=None, time_limit=600):
    """As exact_partial_ceiling, but also returns the per-cell prediction matrix of the best additive rule found
    (same shape as T; -1 where T is -1)."""
    W = np.where(T >= 0, 1.0, 0.0) if W is None else np.where(T >= 0, W, 0.0)
    key = np.concatenate([T, W], 0).T; pats, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); inv = np.ravel(inv)
    R = T.shape[0]; Tp, Wp = pats[:, :R].T, pats[:, R:].T * cnt[None]
    qq, ii = np.where(Tp >= 0); yy = Tp[qq, ii]; ww = Wp[qq, ii]; nq, ni, n = R, Tp.shape[1], len(yy); M = 201.0
    A = lil_matrix((n, nq + ni + n)); lb = np.zeros(n); ub = np.zeros(n)
    for r in range(n):
        A[r, qq[r]] = 1; A[r, nq + ii[r]] = 1
        if yy[r]:
            A[r, nq + ni + r] = M; lb[r], ub[r] = 1, np.inf
        else:
            A[r, nq + ni + r] = -M; lb[r], ub[r] = -np.inf, -1
    res = milp(np.r_[np.zeros(nq + ni), ww], constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.r_[np.zeros(nq + ni), np.ones(n)],
               bounds=Bounds(np.r_[-50 * np.ones(nq + ni), np.zeros(n)], np.r_[50 * np.ones(nq + ni), np.ones(n)]), options={"time_limit": time_limit})
    u, w = res.x[:nq], res.x[nq:nq + ni]; Pp = ((u[:, None] + w[None]) > 0).astype(int)   # prediction per (row, column pattern)
    P = np.where(T >= 0, Pp[:, inv], -1); tot = ww.sum(); best = 1 - (ww * np.round(res.x[nq + ni:])).sum() / tot
    dual = getattr(res, "mip_dual_bound", None)
    return P, best, (1 - dual / tot if dual is not None else np.nan), res.message
