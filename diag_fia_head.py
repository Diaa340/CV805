"""POST-HOC (not pre-registered): where does task-1 forgetting live after task 2 -- encoders or head?
For each method/lambda, S and R retention of: full new model; new encoders + OLD head; OLD encoders + new head.
Writes paper/results_diag_fia_head.txt"""
import copy, os, sys
import numpy as np, torch
from fia import train_task2, masked_logits
from route_exps import Net, load_unit, train
from v3_drift import model_types
dev = "cuda"; lines = []
def say(s=""): print(s, flush=True); lines.append(s)
class A_: pass
for data in sys.argv[1:] or ["bridge_cifar2v", "cifar2v", "fv_cremad"]:
    a = A_(); a.data = data; a.snr = [4.5, 3.0]; a.shift = "12"; a.class_split = 0; a.order = "normal"
    seed = 23; U = load_unit(a, seed); c1 = U["c1"]; c = 1 / len(c1); torch.manual_seed(seed)
    net1 = Net(U["tr1"][0].shape[1], U["tr1"][1].shape[1], U["C"]).to(dev); train(net1, U["tr1"], 30, 1e-3, w=U["w1"])
    for p in net1.parameters(): p.requires_grad_(False)
    exact = data in ("synth", "fv_cremad") or data.startswith("bridge_")
    types = U["types"] if exact else model_types(net1, U["te1"], c1, seed)
    x1, x2, y = U["te1"]
    ev = lambda e1, e2, h: (masked_logits(h(torch.cat([e1(x1), e2(x2)], 1)), c1).argmax(1) == y).cpu().numpy()
    with torch.no_grad(): before = ev(net1.e1, net1.e2, net1.head)
    def ret(ok, t):
        k = types == t
        return np.nan if k.sum() < 30 or before[k].mean() - c < 0.3 else (ok[k].mean() - c) / (before[k].mean() - c)
    say(f"\n[{data}] seed {seed}: S retention (R retention) -- full model | new enc + OLD head | OLD enc + new head")
    for meth, lam in [("finetune", 0.0)] + [(m, l) for m in ("l2", "fia", "fia_joint", "lwf") for l in (0.1, 1.0, 10.0)]:
        torch.manual_seed(1000 + seed); net = copy.deepcopy(net1)
        for p in net.parameters(): p.requires_grad_(True)
        train_task2(net, net1, U, meth, lam)
        with torch.no_grad():
            X1, X2, Y2 = U["te2"]; t2 = (masked_logits(net(X1, X2), U["c2"]).argmax(1) == Y2).float().mean().item()
            r = [ev(net.e1, net.e2, net.head), ev(net.e1, net.e2, net1.head), ev(net1.e1, net1.e2, net.head)]
        say(f"  {meth:9s} lam {lam:5.1f} task2 {t2:.3f} | " + " | ".join(f"S {ret(o, 'S'):.3f} (R {ret(o, 'R'):.3f})" for o in r))
open("paper/results_diag_fia_head.txt", "w").write("\n".join(lines) + "\n")
