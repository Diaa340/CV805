"""Nonce-word CLEVR continual learning with a VLM (paper/preregistration_nonce.md). Identical to vlm_seq_clevr.py (same
units, tasks, cells, methods, learner) except that each task's questions use that task's own made-up attribute words
(nonce.py), so every task teaches new word-image bindings. Optional --steps sets the steps per task.
Usage: python vlm_seq_nonce.py <order 0|1> [--smoke] [--steps N]"""
import os, sys
import vlm_seq as VS
import vlm_seq_clevr as VC
from nonce import rename, vocab
VC.OUT = os.path.join(VC.ROOT, "runs", "nonce_vlm"); os.makedirs(VC.OUT, exist_ok=True); _orig = VC.tasks_and_cells


def nonce_tasks():
    T = _orig(); V = vocab()
    for tk, d in T.items():
        for sp in ("train", "test"):
            d[sp]["text_orig"] = d[sp]["text"]; d[sp]["text"] = [rename(x, V[int(tk)]) for x in d[sp]["text"]]
            d[sp]["renamed"] = (d[sp]["text"] != d[sp]["text_orig"]).astype(int)
    return T


VC.tasks_and_cells = nonce_tasks
if __name__ == "__main__":
    if "--steps" in sys.argv:
        k = sys.argv.index("--steps"); VS.STEPS = int(sys.argv[k + 1]); del sys.argv[k:k + 2]
    VC.main()
