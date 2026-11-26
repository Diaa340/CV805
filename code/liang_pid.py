"""Wrapper around Liang et al.'s verbatim CVX estimator (liang_cvx.py): solves with ECOS (the option max_iters in their
code is an ECOS option; newer cvxpy defaults to Clarabel), silences output, and returns atoms in bits."""
import contextlib, io
import numpy as np
import cvxpy as cp
import liang_cvx as L
_solve = cp.Problem.solve
def _ecos(self, *a, **k):
    k["verbose"] = False; k.setdefault("solver", cp.ECOS); return _solve(self, *a, **k)
cp.Problem.solve = _ecos


def pid_bits(P):
    with contextlib.redirect_stdout(io.StringIO()):
        Q = L.solve_Q_new(P)
    f = 1 / np.log(2)
    return dict(R=L.CoI(Q) * f, U1=L.UI(Q, cond_id=1) * f, U2=L.UI(Q, cond_id=0) * f, S=L.CI(P, Q) * f)
