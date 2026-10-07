"""MMBench (en dev) and MMStar in Xiu et al.'s input format (paper/preregistration_mm.md), with option CONTENTS kept for
our scorer. Prompt = [hint] + question + lettered options + "Answer with the option's letter from the given choices
directly." Random 3:1 split (seed 0), as in Xiu et al. Writes xiu/data/<ds>/<ds>_{train,val}.json and runs/mm/<ds>_items.csv"""
import base64, io, json, re
import numpy as np, pandas as pd
HF = "/home/diaaabuhani/.cache/huggingface/hub"
def mmbench():
    D = pd.read_parquet(f"{HF}/datasets--lmms-lab--MMBench/snapshots/56ba1af8954932c4804bd3f522e05ed96e63b654/en/dev-00000-of-00001.parquet")
    for _, r in D.iterrows():
        opts = [(L, str(r[L])) for L in "ABCD" if str(r[L]) not in ("nan", "None", "")]
        hint = str(r["hint"]) if str(r["hint"]) not in ("nan", "None") else ""
        yield dict(id=str(r["index"]), image=r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"], question=str(r["question"]), hint=hint,
                   options=opts, answer=str(r["answer"]), category=str(r["category"]))
def mmstar():
    D = pd.read_parquet(f"{HF}/datasets--Lin-Chen--MMStar/snapshots/bc98d668301da7b14f648724866e57302778ab27/mmstar.parquet")
    for _, r in D.iterrows():
        text = str(r["question"])
        if "\nChoices:" in text:  # MathVista-style items: "Hint: ...\nQuestion: ...\nChoices:\n(A) ...\n(B) ..."
            q, _, o = text.partition("\nChoices:"); q = q.split("Question:", 1)[-1]; opts = re.findall(r"\(([A-D])\) (.*?)(?=\n\([A-D]\) |$)", o.strip(), flags=re.S)
        else:
            q, _, o = text.partition("\nOptions:"); opts = re.findall(r"([A-D]): (.*?)(?=, [A-D]: |$)", o.strip(), flags=re.S)
        yield dict(id=str(r["index"]), image=r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"], question=q.strip(), hint="",
                   options=[(L, t.strip()) for L, t in opts], answer=str(r["answer"]), category=str(r["category"]))
for ds, gen in (("mmbench", mmbench), ("mmstar", mmstar)):
    items = [it for it in gen() if len(it["options"]) >= 2 and it["answer"] in [L for L, _ in it["options"]]]
    out = []
    for it in items:
        b = it["image"] if isinstance(it["image"], (bytes, bytearray)) else bytes(it["image"])
        prompt = (it["hint"] + "\n" if it["hint"] else "") + it["question"] + "".join(f"\n{L}. {t}" for L, t in it["options"]) + "\nAnswer with the option's letter from the given choices directly."
        out.append({"id": it["id"], "image": base64.b64encode(b).decode(), "num_options": len(it["options"]), "category": it["category"],
                    "options": it["options"], "question": it["question"], "hint": it["hint"],
                    "conversations": [{"from": "human", "value": prompt + "\n<image>"}, {"from": "gpt", "value": it["answer"]}]})
    perm = np.random.default_rng(0).permutation(len(out)); n = int(0.75 * len(out))
    json.dump([out[i] for i in sorted(perm[:n])], open(f"xiu/data/{ds}/{ds}_train.json", "w")); json.dump([out[i] for i in sorted(perm[n:])], open(f"xiu/data/{ds}/{ds}_val.json", "w"))
    pd.DataFrame([dict(id=o["id"], n_opt=o["num_options"], category=o["category"], answer=o["conversations"][1]["value"]) for o in out]).to_csv(f"runs/mm/{ds}_items.csv", index=False)
    print(ds, len(out), "items ->", n, "/", len(out) - n, "| options per item", pd.Series([o["num_options"] for o in out]).value_counts().to_dict())
