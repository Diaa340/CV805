"""Pre-registered analysis of the VLM six-task nonce-word CLEVR study (paper/preregistration_nonce.md). Same-answer metrics
(Strat copied from analyze_strat.py, which runs on import). Writes paper/results_clevr_vlm.txt, paper/figs/clevr_vlm_metrics.csv"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []; NB = 1000
def say(s=""): print(s, flush=True); lines.append(s)


class Strat:
    """Per-unit sums by (type, answer) for pairs of columns; bootstrap over units."""
    def __init__(self, X, pairs, seed=0):
        self.units = sorted(X.unit.unique()); k = {u: i for i, u in enumerate(self.units)}; self.pairs = pairs; cols = sorted({c for p in pairs.values() for c in p})
        self.ci = {c: i for i, c in enumerate(cols)}; nu = len(self.units); self.S = np.zeros((nu, 2, 2, len(cols))); self.N = np.zeros((nu, 2, 2))
        for (u, g, y), x in X.groupby(["unit", "nonadd", "y"]):
            self.S[k[u], g, y] = x[cols].astype(float).sum().values; self.N[k[u], g, y] = len(x)
        self.rng = np.random.default_rng(seed)

    def stats(self, w):
        S = np.einsum("u,ugyc->gyc", w, self.S); N = np.einsum("u,ugy->gy", w, self.N); out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            A = S / N[..., None]; pi = N[1] / N[1].sum()
            Ag = S.sum(1) / N.sum(1)[:, None]; At = S.sum((0, 1)) / N.sum()
        for name, (after, ref) in self.pairs.items():
            d = A[..., self.ci[after]] - A[..., self.ci[ref]]   # [type, answer]
            ok = ~np.isnan(d[0]) & ~np.isnan(d[1]) & (N[1] > 0); p = pi * ok; p = p / p.sum()
            out[name] = dict(R=Ag[0, self.ci[after]] - Ag[0, self.ci[ref]], E=Ag[1, self.ci[after]] - Ag[1, self.ci[ref]],
                             d_raw=(Ag[1, self.ci[after]] - Ag[1, self.ci[ref]]) - (Ag[0, self.ci[after]] - Ag[0, self.ci[ref]]),
                             d_strat=float(np.nansum(p * (d[1] - d[0]))), all=At[self.ci[after]] - At[self.ci[ref]],
                             R_yes=d[0, 1], R_no=d[0, 0], E_yes=d[1, 1], E_no=d[1, 0])
        return out

    def run(self):
        nu = len(self.units); self.P = self.stats(np.ones(nu))
        self.B = [self.stats(np.bincount(self.rng.integers(0, nu, nu), minlength=nu).astype(float)) for _ in range(NB)]; return self

    def ci_(self, f):
        return np.nanpercentile([f(b) for b in self.B], [2.5, 97.5])


V = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "nonce_vlm", "order_[0-9].csv.gz")))])
late = V[V.pos >= 2].groupby("order").mean(numeric_only=True); allr = V.groupby("order").mean(numeric_only=True); pick = {}
for r in late.index:   # matched plasticity for EWC
    ok = [l for l in ("100", "10000") if late.loc[r, f"ewc_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]; pick[r] = max(ok, key=lambda l: allr.loc[r, f"ewc_{l}__final"]) if ok else "100"
for s in ("own", "final"):
    V[f"ewc__{s}"] = 0.0
    for l in ("100", "10000"):
        m = V.order.map(pick) == l; V.loc[m, f"ewc__{s}"] = V.loc[m, f"ewc_{l}__{s}"]
M = ["finetune", "ewc", "lwf_1", "er", "derpp", "er_classbal", "lora_expand"]
SB = Strat(V[V.pos <= 5], {m: (f"{m}__final", f"{m}__own") for m in M}).run()
SF = Strat(V[V.pos >= 2], {m: (f"{m}__own", "scratch") for m in M}).run()
zs = V.groupby("nonadd").zeroshot.mean(); say(f"=== Nonce-word CLEVR, Qwen2.5-VL-3B + LoRA: {V.order.nunique()} orders, {V.unit.nunique()} units; zero-shot accuracy rule {zs[0]:.3f}, exceptions {zs[1]:.3f}; EWC picked {pick} ===")
say("  method        BWT_R   BWT_E | dBWT raw | dBWT_s [95%]           | E yes/no  R yes/no            | dFWT_s [95%]           | final acc R / E")
rows = []
for m in M:
    p, f = SB.P[m], SF.P[m]; a = SB.ci_(lambda x: x[m]["d_strat"]); c = SF.ci_(lambda x: x[m]["d_strat"])
    acc = V.groupby("nonadd")[f"{m}__final"].mean(); rows.append(dict(method=m, BWT_R=p["R"], BWT_E=p["E"], dBWT_raw=p["d_raw"], dBWT_s=p["d_strat"], lo=a[0], hi=a[1], dFWT_s=f["d_strat"], flo=c[0], fhi=c[1]))
    say(f"  {m:12s} {p['R']:+.3f} {p['E']:+.3f} | {p['d_raw']:+.3f} | {p['d_strat']:+.3f} [{a[0]:+.3f},{a[1]:+.3f}] | {p['E_yes']:+.3f}/{p['E_no']:+.3f} {p['R_yes']:+.3f}/{p['R_no']:+.3f} | "
        f"{f['d_strat']:+.3f} [{c[0]:+.3f},{c[1]:+.3f}] | {acc[0]:.3f} / {acc[1]:.3f}")
R = pd.DataFrame(rows).set_index("method"); R.round(4).to_csv(os.path.join(ROOT, "paper", "figs", "nonce_vlm_metrics.csv"))
zs_all, sc_all = V.zeroshot.mean(), V[V.pos >= 1].scratch.mean(); gate = bool(zs_all <= 0.80 and sc_all >= zs_all + 0.10)
say(f"\nGATES: zero-shot accuracy {zs_all:.3f} (must be <= 0.80); task-only accuracy {sc_all:.3f} (must be >= zero-shot + 0.10) -> {'MET' if gate else 'NOT MET: tests below are not interpretable'}")
vc1 = all(R.loc[m, "dBWT_s"] <= -0.03 and R.loc[m, "hi"] < 0 for m in ("er", "derpp")); vc2 = bool(R.loc["derpp", "dFWT_s"] <= -0.03 and R.loc["derpp", "fhi"] < 0)
# dBWT_s(expand) - dBWT_s(er), bootstrap with the same unit draws
diff = SB.P["lora_expand"]["d_strat"] - SB.P["er"]["d_strat"]; dci = np.nanpercentile([b["lora_expand"]["d_strat"] - b["er"]["d_strat"] for b in SB.B], [2.5, 97.5]); vc3 = bool(dci[0] > 0)
say(f"\nVC1 ER and DER++ lose exceptions more within the same answer: {'PASS' if vc1 else 'FAIL'}")
say(f"VC2 DER++ hinders new exceptions more (dFWT_s {R.loc['derpp', 'dFWT_s']:+.3f} [{R.loc['derpp', 'flo']:+.3f},{R.loc['derpp', 'fhi']:+.3f}]): {'PASS' if vc2 else 'FAIL'}")
say(f"VC3 LoRA-expansion dBWT_s minus ER's: {diff:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]: {'PASS' if vc3 else 'FAIL'}")
open(os.path.join(ROOT, "paper", "results_nonce_vlm.txt"), "w").write("\n".join(lines) + "\n")
