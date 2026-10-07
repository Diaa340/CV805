"""CLOVE-scene exact answer tables (paper/preregistration_clove.md). Outcome-free.
Per scene task (a-f): yes/no existence questions from that task (train and val files) whose program clove_exec
supports. Rules as in the GQA build: a question is kept if no named object is a superordinate category (HYPER, copied
from gqa_build.py), every named object occurs >= 50 times in GQA scene graphs, and the executor reproduces its answer on
its own image; a family is kept if the executor agrees on >= 99% of these questions over all six tasks. Distinct
programs (object ids removed) are grouped 10 per unit within a family, up to 8 units per task, families taken
round-robin. Per unit: 150 training images from the task's CLOVE train split and 50 test images from its val split,
each sampled among images whose scene graph contains at least one object named by the unit's questions; kept if
both pools are large enough and the yes-share of the full table is in [0.10, 0.90].
Writes runs/clove/units.npz, runs/clove/units.csv"""
import collections, glob, json, os, re
import numpy as np, pandas as pd
from clove_exec import execute, norm, prog, scenes, supported
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clove"); os.makedirs(OUT, exist_ok=True)
SRC = open(os.path.join(ROOT, "gqa_build.py")).read(); HYPER = eval(re.search(r"HYPER = (\{.*?\})\n", SRC, re.S).group(1))   # not imported: gqa_build runs on import
import sys
POOL_ALL = "--pool-all" in sys.argv; TAG = "_v2" if POOL_ALL else ""
TASKS = ["a", "b", "c", "d", "e", "f"]; S = scenes(); L = lambda x: x if isinstance(x, (list, dict)) else eval(x)
vocab = collections.Counter(o["name"] for sc in S.values() for o in sc["objects"].values())
names = lambda p: {norm(s["argument"]) for s in p if s["operation"] == "select"} | {norm(s["argument"]).rsplit(",", 2)[0].strip() for s in p if s["operation"] == "relate"}
Q, imgs = {}, {}
for t in TASKS:
    for split in ("train", "val"):
        D = json.load(open(glob.glob(os.path.join(ROOT, "data", "raw", "clove", "json", "scene", f"scene_{t}#*_{split}.json"))[0]))
        imgs[t, split] = sorted({str(d["image_id"]) for d in D if str(d["image_id"]) in S})
        for d in D:
            if d["answer"] in ("yes", "no") and str(d["image_id"]) in S:
                p = prog(d["program"])
                if supported(p) and not names(p) & HYPER and all(vocab[n] >= 50 and n != "_" for n in names(p)):
                    Q[d["question_id"]] = dict(task=t, p=p, fam=L(d["raw_question_type"])["detailed"], ok=execute(p, S[str(d["image_id"])]) == d["answer"], text=d["question"])
agree = pd.DataFrame(Q.values()).groupby("fam").ok.agg(["mean", "size"]); print(agree.round(3).to_string())
fams = sorted(agree[agree["mean"] >= 0.99].index); print("families kept:", fams)
rows, store = [], {}
for ti, t in enumerate(TASKS):
    progs = collections.defaultdict(dict)
    for qid, q in sorted(Q.items()):
        if q["task"] == t and q["fam"] in fams and q["ok"]:
            progs[q["fam"]].setdefault(json.dumps([(s["operation"], norm(s["argument"]), s["dependencies"]) for s in q["p"]]), qid)
    groups = {f: [v[k] for k in np.random.default_rng(100 + ti).permutation(sorted(v))] for f, v in progs.items()}
    groups = {f: [ids[10 * g:10 * g + 10] for g in range(len(ids) // 10)] for f, ids in groups.items()}
    order = [(f, g) for g in range(max(map(len, groups.values()))) for f in sorted(groups) if g < len(groups[f])]   # round-robin over families
    nk = 0
    for f, g in order:
        if nk == 8:
            break
        ids = groups[f][g]; nm = set().union(*[names(Q[i]["p"]) for i in ids]); r = dict(task=t, unit=f"{t}_{f}_{g}", family=f)
        if POOL_ALL and any(s["operation"] == "relate" for i in ids for s in Q[i]["p"]):   # addendum 1: relation units sample images containing all named objects of some question
            pools = {sp: [m for m in imgs[t, sp] if any(names(Q[i]["p"]) <= {o["name"] for o in S[m]["objects"].values()} for i in ids)] for sp in ("train", "val")}
        else:
            pools = {sp: [m for m in imgs[t, sp] if any(o["name"] in nm for o in S[m]["objects"].values())] for sp in ("train", "val")}
        r.update(n_train_pool=len(pools["train"]), n_val_pool=len(pools["val"]))
        if len(pools["train"]) < 150 or len(pools["val"]) < 50:
            r["keep"] = False; rows.append(r); continue
        rng = np.random.default_rng(200 + 10 * ti + g)
        tr = sorted(rng.choice(pools["train"], 150, replace=False)); te = sorted(rng.choice(pools["val"], 50, replace=False)); ui = tr + te
        T = np.array([[int(execute(Q[i]["p"], S[m]) == "yes") for m in ui] for i in ids], np.int8)
        r.update(yes=T.mean(), keep=bool(0.10 <= T.mean() <= 0.90)); rows.append(r)
        if r["keep"]:
            nk += 1; store[r["unit"]] = dict(T=T, imgs=np.array(ui), text=np.array([Q[i]["text"] for i in ids]), qids=np.array(ids))
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, f"units{TAG}.csv"), index=False)
np.savez(os.path.join(OUT, f"units{TAG}.npz"), **{f"{k}__{u}": v for u, d in store.items() for k, v in d.items()})
print(D.groupby("task").keep.sum().to_dict(), "units kept;", D[D.keep].family.value_counts().to_dict())
