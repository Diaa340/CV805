"""Small numpy-only helpers shared across environments."""
import numpy as np


def mmi(T):
    """MMI-PID synergy (bits) of a deterministic table under uniform cells (-1 = missing cell)."""
    H = lambda p: -sum(x * np.log2(x) for x in (p, 1 - p) if 0 < x < 1)
    m = T >= 0; p = T[m].mean(); pq = m.sum(1) / m.sum(); pi = m.sum(0) / m.sum()
    rq = np.array([T[r][m[r]].mean() if m[r].any() else 0 for r in range(T.shape[0])])
    ri = np.array([T[:, c][m[:, c]].mean() if m[:, c].any() else 0 for c in range(T.shape[1])])
    I12 = H(p); Iq = H(p) - sum(pq[r] * H(rq[r]) for r in range(len(rq))); Ii = H(p) - sum(pi[c] * H(ri[c]) for c in range(len(ri)))
    return I12 - max(Iq, Ii)
