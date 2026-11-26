"""Analysis of the exception curricula (paper/preregistration_curric.md). Writes paper/results_curric.txt.
Per condition: exception accuracy and rule-cell accuracy, both averaged over answers with weights pi_y = share of
exceptions with answer y (so answer imbalance cannot drive differences), and overall accuracy. Differences against the
baseline (random / uniform_300) with 95% bootstrap intervals over units (GQA: base units)."""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []; NB = 2000; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)


def sums(D, conds):
    """Per cluster: correct counts and cell counts by (type, answer) for each condition, plus overall."""
    cl = sorted(D.cluster.unique()); k = {c: i for i, c in enumerate(cl)}; S = np.zeros((len(cl), 2, 2, len(conds))); N = np.zeros((len(cl), 2, 2))
    for (c, g, y), x in D.groupby(["cluster", "nonadd", "y"]):
        S[k[c], int(g), int(y)] = x[conds].astype(float).sum().values; N[k[c], int(g), int(y)] = len(x)
    return S, N


def metrics(S, N, w):
    s = np.einsum("u,ugyc->gyc", w, S); n = np.einsum("u,ugy->gy", w, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = s / n[..., None]; pi = n[1] / n[1].sum()
        E = np.nansum(pi[:, None] * A[1], 0); R = np.nansum(pi[:, None] * A[0], 0); All = s.sum((0, 1)) / n.sum()
    return E, R, All


def compare(D, conds, base, label):
    S, N = sums(D, conds); nu = S.shape[0]; P = metrics(S, N, np.ones(nu))
    B = [metrics(S, N, np.bincount(rng.integers(0, nu, nu), minlength=nu).astype(float)) for _ in range(NB)]
    b = conds.index(base); say(f"\n=== {label}: {nu} units; exception / rule (answer-matched) / overall accuracy, and difference from {base} (points) ===")
    out = {}
    for j, c in enumerate(conds):
        d = {}
        for t, nm in ((0, "E"), (1, "R"), (2, "All")):
            v = 100 * (P[t][j] - P[t][b]); ci = 100 * np.percentile([x[t][j] - x[t][b] for x in B], [2.5, 97.5]); d[nm] = (v, ci)
        out[c] = d; out[c]["abs"] = (100 * P[0][j], 100 * P[1][j], 100 * P[2][j])
        say(f"  {c:12s} {100 * P[0][j]:5.1f} / {100 * P[1][j]:5.1f} / {100 * P[2][j]:5.1f} | exceptions {d['E'][0]:+5.1f} [{d['E'][1][0]:+5.1f},{d['E'][1][1]:+5.1f}] | "
            f"rule {d['R'][0]:+5.1f} [{d['R'][1][0]:+5.1f},{d['R'][1][1]:+5.1f}] | overall {d['All'][0]:+5.1f} [{d['All'][1][0]:+5.1f},{d['All'][1][1]:+5.1f}]")
    def diff(c1, c2):   # exception-accuracy difference between two conditions
        i, j = conds.index(c1), conds.index(c2); v = 100 * (P[0][i] - P[0][j]); ci = 100 * np.percentile([x[0][i] - x[0][j] for x in B], [2.5, 97.5]); return v, ci
    return out, diff


C1 = C2 = 0; STEP1 = ["random", "misfit", "easy_first", "hard_first", "focal", "loss_select"]
for b in ("tables", "clevr", "gqa"):
    fs = sorted(glob.glob(os.path.join(ROOT, "runs", "curric", f"test_{b}_*.csv.gz")))
    if not fs:
        continue
    D = pd.concat([pd.read_csv(f) for f in fs]); D["cluster"] = D.unit.str.rsplit("_s", n=1).str[0] if b == "gqa" else D.unit
    out, diff = compare(D, STEP1, "random", f"STEP 1, {b}")
    m = out["misfit"]; c1 = m["E"][0] >= 3 and m["E"][1][0] > 0 and m["All"][1][0] > -1
    vs = {c: diff("misfit", c) for c in ("focal", "loss_select")}; c2 = all(v[0] >= 2 and v[1][0] > 0 for v in vs.values())
    say(f"  misfit vs focal: exceptions {vs['focal'][0]:+.1f} [{vs['focal'][1][0]:+.1f},{vs['focal'][1][1]:+.1f}] | misfit vs loss_select: "
        f"{vs['loss_select'][0]:+.1f} [{vs['loss_select'][1][0]:+.1f},{vs['loss_select'][1][1]:+.1f}]  -> C1 {c1}, C2 {c2}")
    C1 += c1; C2 += c2
say(f"\nSTEP 1 VERDICTS: C1 (misfit beats random on exceptions, no overall cost) {C1}/3 -> {'PASS' if C1 >= 2 else 'FAIL'} | "
    f"C2 (misfit beats focal AND loss-based selection on exceptions) {C2}/3 -> {'PASS' if C2 >= 2 else 'FAIL'}")

fs = sorted(f for f in glob.glob(os.path.join(ROOT, "runs", "vlm_curric", "task_*.csv.gz")) if "smoke" not in f)
if fs:
    D = pd.concat([pd.read_csv(f) for f in fs]); D["cluster"] = D.unit
    STEP2 = ["zeroshot", "uniform_300", "misfit_300", "loss_300", "uniform_600", "misfit_600"]
    out, diff = compare(D, STEP2, "uniform_300", f"STEP 2, VLM nonce-word CLEVR ({D.task.nunique()} tasks; test sets are half exceptions)")
    m = out["misfit_300"]; v1 = m["E"][0] >= 3 and m["E"][1][0] > 0 and m["R"][1][0] > -2; v2 = diff("misfit_300", "loss_300")
    v6 = diff("misfit_600", "uniform_600")
    say(f"  misfit_300 vs loss_300: exceptions {v2[0]:+.1f} [{v2[1][0]:+.1f},{v2[1][1]:+.1f}] | misfit_600 vs uniform_600: {v6[0]:+.1f} [{v6[1][0]:+.1f},{v6[1][1]:+.1f}]")
    say(f"\nSTEP 2 VERDICTS: V1 (misfit_300 beats uniform_300 on exceptions, rule cost < 2 points) -> {'PASS' if v1 else 'FAIL'} | "
        f"V2 (misfit_300 beats loss_300 on exceptions) -> {'PASS' if v2[1][0] > 0 else 'FAIL'}")
open(os.path.join(ROOT, "paper", "results_curric.txt"), "w").write("\n".join(lines) + "\n")
