"""GQA exact tables (paper/preregistration_gqa.md). Outcome-free. Existence families whose questions are well-posed on
any image (select / filter / exist / and / or; no relations to a definite referent). Families kept if the executor
agrees with GQA's answers on the question's own image for >= 99% of questions; questions kept if it agrees on their own
image and no selected name is a superordinate category (to avoid hypernym mismatches on other images).
Units: up to 60 distinct programs per family, in groups of 10; per unit, 200 val images sampled from those whose scene
graph contains at least one object name selected by the unit's questions (as POPE samples images where objects
plausibly occur); yes-share in [0.10, 0.90].
Writes runs/gqa/units.npz and runs/gqa/units.csv"""
import collections, json, os, re
import numpy as np, pandas as pd
from gqa_exec import SUPPORTED, execute, load, norm
from pid_utils import mmi
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "gqa"); os.makedirs(OUT, exist_ok=True)
HYPER = {"animal", "vehicle", "furniture", "device", "utensil", "clothing", "food", "fruit", "vegetable", "appliance", "kitchen appliance",
         "electronics", "container", "drink", "dessert", "weapon", "bag", "toy", "accessory", "fast food", "baked good", "meat", "seafood",
         "plant", "flower", "tree", "person", "people", "bird", "dish", "sauce", "beverage", "dairy", "computer", "screen", "sign", "vegetables", "fruits"}
Q, S = load(); fam = collections.defaultdict(list)
for qid, q in Q.items():
    if {s["operation"].split(" ")[0] for s in q["semantic"]} <= SUPPORTED and q["imageId"] in S:
        fam[q["types"]["detailed"]].append(qid)
imgs = sorted(S); imgs = [imgs[k] for k in np.sort(np.random.default_rng(81).choice(len(imgs), 1000, replace=False))]
rows, store = [], {}
for f, ids in sorted(fam.items()):
    agree = np.mean([execute(Q[i]["semantic"], S[Q[i]["imageId"]]) == Q[i]["answer"] for i in ids])
    if agree < 0.99:
        print(f"skip family {f}: own-image agreement {agree:.3f}"); continue
    progs = {}
    for i in ids:
        q = Q[i]; names = {norm(s["argument"]) for s in q["semantic"] if s["operation"] == "select"}
        if execute(q["semantic"], S[q["imageId"]]) == q["answer"] and not names & HYPER:
            progs.setdefault(json.dumps([(s["operation"], norm(s["argument"]), s["dependencies"]) for s in q["semantic"]]), i)
    keys = sorted(progs); pick = np.random.default_rng(82).choice(len(keys), min(60, len(keys)), replace=False); qids = [progs[keys[p]] for p in pick]
    for g in range(len(qids) // 10):
        sl = slice(10 * g, 10 * g + 10); names = {norm(s["argument"]) for i in qids[sl] for s in Q[i]["semantic"] if s["operation"] == "select"}
        pool = [m for m in sorted(S) if any(o["name"] in names for o in S[m]["objects"].values())]
        ui = [pool[k] for k in np.sort(np.random.default_rng(83 + g).choice(len(pool), min(200, len(pool)), replace=False))]
        Tg = np.array([[int(execute(Q[i]["semantic"], S[m]) == "yes") for m in ui] for i in qids[sl]], np.int8); y = Tg.mean(); keep = 0.10 <= y <= 0.90 and len(ui) == 200
        r = dict(unit=f"{f}_{g}", family=f, yes=y, keep=keep, own_agree=agree)
        if keep:
            r["syn_mmi_exact"] = mmi(Tg); store[r["unit"]] = dict(T=Tg, qids=np.array(qids[sl]), text=np.array([Q[i]["question"] for i in qids[sl]]), imgs=np.array(ui))
        rows.append(r)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "units.csv"), index=False)
np.savez(os.path.join(OUT, "units.npz"), **{f"{k}__{u}": v for u, d in store.items() for k, v in d.items()})
print(f"kept {D.keep.sum()} units from {D[D.keep].family.nunique()} families:", D[D.keep].family.value_counts().to_dict())
