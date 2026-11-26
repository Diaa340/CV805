"""Pre-registered analysis for MMBench / MMStar (paper/preregistration_mm.md). Writes paper/results_mm.txt"""
import json, os, re, subprocess
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); X = os.path.join(ROOT, "code", "xiu"); lines = []
def say(s=""): print(s); lines.append(s)


def recombine(tr, va, keys):
    """Conditional logit over options: score(o) = sum_k a_k * log p_k(o) + c_letter(o); fit on tr, accuracy on va."""
    def tens(rows):
        K = max(r["k"] for r in rows); F = torch.full((len(rows), K, len(keys)), 0.0); M = torch.zeros(len(rows), K, dtype=torch.bool)
        for n, r in enumerate(rows):
            for j, key in enumerate(keys):
                F[n, :r["k"], j] = torch.log_softmax(torch.tensor(r[key], dtype=torch.float32), 0)
            M[n, :r["k"]] = True
        return F, M, torch.tensor([r["answer"] for r in rows])
    Ft, Mt, yt = tens(tr); Fv, Mv, yv = tens(va); a = torch.ones(len(keys), requires_grad=True); c = torch.zeros(Ft.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([a, c], max_iter=500)
    def s(F, M):
        z = (F * a).sum(-1) + c; return z.masked_fill(~M, -1e9)
    def closure():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(s(Ft, Mt), yt); l.backward(); return l
    opt.step(closure)
    with torch.no_grad():
        return (s(Fv, Mv).argmax(1) == yv).float().mean().item(), a.detach().tolist()


V = {}
for ds in ("mmbench", "mmstar"):
    R = [json.loads(l) for l in open(os.path.join(X, "results", f"mm_scores_{ds}.jsonl"))]; tr = [r for r in R if r["split"] == "train"]; va = [r for r in R if r["split"] == "val"]
    acc = lambda rows, key: np.mean([int(np.argmax(r[key]) == r["answer"]) for r in rows]); chance = np.mean([1 / r["k"] for r in va])
    a_add, w = recombine(tr, va, ["img", "txt"]); a_full = acc(va, "full")
    gap = (a_full - a_add) / (a_full - chance); V[f"MM-1 {ds}"] = "PASS" if gap <= 0.30 else "FAIL"
    say(f"=== {ds}: {len(va)} held-out items (chance {chance:.3f}) ===")
    say(f"  options kept: full {a_full:.3f} | text-only {acc(va, 'txt'):.3f} | image + options (no question) {acc(va, 'img'):.3f} | "
        f"additive recombination {a_add:.3f} (weights img {w[0]:.2f}, txt {w[1]:.2f})")
    say(f"  MM-1: normalised fusion gap (full - additive) / (full - chance) = {gap:.3f} (<= 0.30) -> {V[f'MM-1 {ds}']}")
    cats = pd.DataFrame([dict(cat=r["category"], full=int(np.argmax(r["full"]) == r["answer"]), txt=int(np.argmax(r["txt"]) == r["answer"]), img=int(np.argmax(r["img"]) == r["answer"])) for r in va])
    say("  by category (full / text-only / image+options):\n" + cats.groupby("cat").mean().round(2).to_string())
    E = {sp: [json.loads(json.dumps(c)) for it in json.load(open(os.path.join(X, "results", "embeds", ds, sp, "qwen25_7.json"))) for c in it["conversations"] if c["from"] == "gpt"] for sp in ("train", "val")}
    if all(E.values()):
        ids = {sp: [it["id"] for it in json.load(open(os.path.join(X, "data", ds, f"{ds}_{sp}.json")))] for sp in ("train", "val")}
        lab = {r["id"]: r["answer"] for r in R}
        conv = lambda sp: [dict(k=int(np.sum(np.array(c["vl_prob"]) > 0)) or 4, answer=lab[i], v=c["v_prob"], l=c["l_prob"], vl=c["vl_prob"]) for i, c in zip(ids[sp], E[sp])]
        for rows in (conv("train"), conv("val")):
            for r in rows:
                r["k"] = max(2, sum(1 for p in r["vl"] if p > 0)); r["v"], r["l"] = [max(p, 1e-6) for p in r["v"][:r["k"]]], [max(p, 1e-6) for p in r["l"][:r["k"]]]
        ctr, cva = conv("train"), conv("val")
        for rr in (ctr, cva):
            for r in rr:
                r["k"] = max(2, sum(1 for p in r["vl"] if p > 0)); r["v"] = list(np.log(np.maximum(r["v"][:r["k"]], 1e-6))); r["l"] = list(np.log(np.maximum(r["l"][:r["k"]], 1e-6)))
        a_n, _ = recombine(ctr, cva, ["v", "l"]); accn = lambda rows, key: np.mean([int(np.argmax(r[key]) == r["answer"]) for r in rows])
        say(f"  MM-2 (Xiu et al.'s Gaussian-noise removal): image-side (text noised) {accn(cva, 'v'):.3f} | language-side (image noised) {accn(cva, 'l'):.3f} | "
            f"additive recombination {a_n:.3f} | full {accn(cva, 'vl'):.3f}")
        out = subprocess.run([os.path.join(ROOT, ".venv_xiu", "bin", "python"), "batch_vlm_final_mmbench_drop.py", "--directory", f"results/embeds/{ds}/", "--file_name", "qwen25_7.json"],
                             cwd=X, capture_output=True, text=True).stdout
        p = {k: float(re.search(rf"{k}: ([0-9.eE+-]+)", out).group(1)) for k in ("Redundancy", "Unique1", "Unique2", "Synergy")}; tot = sum(p.values())
        say(f"  MM-3 reproduced decision-level PID (their BATCH): R {p['Redundancy']:.3f} U_vis {p['Unique1']:.3f} U_lang {p['Unique2']:.3f} S {p['Synergy']:.3f} bits (S share {p['Synergy'] / tot:.2f})")
say("\nVERDICTS: " + " | ".join(f"{k} {v}" for k, v in V.items()))
open(os.path.join(ROOT, "paper", "results_mm.txt"), "w").write("\n".join(lines) + "\n")
