"""RPSnet (Rajasegaran et al., "Random Path Selection for Incremental Learning", NeurIPS 2019; official code
github.com/brjathu/RPSnet, read in ext_src/RPSnet, not executed) on the route_exps.Net fusion net, under our protocol.
Port of the MLP variant (rps_net.RPS_net_mlp + learner.Learner + mnist.py settings):
  * every hidden Linear of Net (e1: 3, e2: 3, head: 1 -> L = 7 layers) becomes M = 8 parallel modules whose outputs are
    summed over the modules on the inference path; the 2-way classifier is shared and always trained (final_layers[-1]).
  * task 1: path = module 0 in every layer. Module 0 is a copy of the shared task-1 model (net1), so task 1 is identical
    to every other method; modules 1..7 get PyTorch's default init.
  * every later task (jump = 1, as mnist.py): NPATH = 8 candidate paths (mnist.py runs 8 test cases), each N = 1 random
    module per layer (util.get_path); train_path = path minus the frozen fixed path, infer_path = fixed + path; modules
    off train_path are frozen. The best candidate is kept and its path joins the fixed path.
  * loss (learner.train): CE + cx/B * sum KL(log_softmax(out/2) || softmax(old/2)), clamped at 0; old = previous model.
    Domain-incremental (both tasks share the 2 classes), so RPS's target vector is the old model's logits in full.
    Controller: cx = 1 for sessions <= jump, else rigidness 2.5 * (session - jump) (session = task index - 1).
Deviations, for comparability with the paper's other methods:
  * optimisation as every other method here: Adam lr 1e-3, weight decay 1e-5, E2 epochs of BS = 256 new-task draws,
    plus 32 memory samples concatenated to each batch (as cl_seq's iCaRL). RPS uses its own epochs / LR schedule.
  * memory: the same random 1000-sample memory as ER / DER++ (RPS draws random exemplars per class; ours is random over
    both answers).
  * path choice by accuracy on data the learner holds (the memory plus 2000 new-task training draws), never on test
    cells (the official get_best_model picks by test accuracy).
  * the old model is evaluated on its own inference path (the paper's distillation); the official code passes the new
    infer path to old_model, so its teacher includes the untrained new modules.
All RPS randomness uses its own generators (module init in build() under fork_rng, path draws, batch indices, validation
draws), so methods run before or after it in the same process see the global RNG unchanged."""
import copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

M, NPATH, JUMP, RIGID, TEMP, RB, NVAL = 8, 8, 1, 2.5, 2.0, 32, 2000


class Multi(nn.Module):
    """M parallel Linear modules; output = sum of the modules on this layer's row of the path."""
    def __init__(self, base, gen):
        super().__init__(); self.mods = nn.ModuleList([copy.deepcopy(base)])
        for _ in range(1, M):
            lin = nn.Linear(base.in_features, base.out_features)   # on CPU, built inside build()'s fork_rng
            with torch.no_grad():   # nn.Linear.reset_parameters, drawn from gen (CPU)
                nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5, generator=gen); bd = 1 / base.in_features ** 0.5
                lin.bias.uniform_(-bd, bd, generator=gen)
            self.mods.append(lin.to(base.weight.device))

    def forward(self, x, row):
        return sum(self.mods[i](x) for i in np.flatnonzero(row))


class RPSNet(nn.Module):
    def __init__(self, net1, gen):
        super().__init__()
        lin = lambda seq: [m for m in seq if isinstance(m, nn.Linear)]
        self.e1 = nn.ModuleList([Multi(l, gen) for l in lin(net1.e1)]); self.e2 = nn.ModuleList([Multi(l, gen) for l in lin(net1.e2)])
        self.h0 = Multi(net1.head[0], gen); self.cls = copy.deepcopy(net1.head[2]); self.L = len(self.e1) + len(self.e2) + 1
        self.infer = np.zeros((self.L, M), int); self.infer[:, 0] = 1

    def layers(self):
        return list(self.e1) + list(self.e2) + [self.h0]

    def hidden(self, a, b):
        P, n1 = self.infer, len(self.e1)
        for k, m in enumerate(self.e1):
            a = m(a, P[k]); a = F.relu(a) if k < n1 - 1 else a
        for k, m in enumerate(self.e2):
            b = m(b, P[n1 + k]); b = F.relu(b) if k < len(self.e2) - 1 else b
        return F.relu(self.h0(torch.cat([a, b], 1), P[-1]))

    def forward(self, a, b):
        return self.cls(self.hidden(a, b))

    def feat(self, a, b):
        """Herding features, as cl_seq.hfeat (normalised output of head[:2]) on the inference path."""
        return F.normalize(self.hidden(a, b), dim=1)


def build(net1, seed):
    """RPSNet from the task-1 model; the global torch RNG is left as it was."""
    with torch.random.fork_rng(devices=[]):
        g = torch.Generator(); g.manual_seed(seed * 1000 + 7); return RPSNet(net1, g)


def get_path(L, rng):
    """util.get_path with N = 1: one random module per layer."""
    p = np.zeros((L, M), int); p[np.arange(L), rng.integers(0, M, L)] = 1; return p


def session(prev, fixed, sess, x1, x2, y, mem, steps, seed, bs=256, pool=None):
    """One RPS session (task index sess >= 1) from model prev with frozen modules `fixed`. x1, x2, y: new-task training
    draws; mem: dict x1, x2, y of the replay memory. pool (optional): dict x1, x2, logit of exception cells; adds
    0.5 MSE to the stored logits on RB pool draws per step from its own generator (as cl_metrics_eap.derpp_pool "logit").
    Returns (best net, new fixed path, log)."""
    dev = y.device; n = len(y); cx = 1.0 if sess <= JUMP else RIGID * (sess - JUMP); prev = prev.eval(); out, log = None, []
    gv = torch.Generator(device=dev); gv.manual_seed(seed * 1000 + sess); iv = torch.randint(0, n, (NVAL,), device=dev, generator=gv)
    vx1, vx2, vy = torch.cat([mem["x1"], x1[iv]]), torch.cat([mem["x2"], x2[iv]]), torch.cat([mem["y"], y[iv]])
    for k in range(NPATH):
        path = get_path(prev.L, np.random.default_rng(seed * 1000 + sess * 10 + k)); train = path * (1 - fixed); net = copy.deepcopy(prev)
        net.infer = np.maximum(fixed, path)
        for l, layer in enumerate(net.layers()):
            for i, mod in enumerate(layer.mods):
                mod.requires_grad_(bool(train[l, i]))
        params = [p for p in net.parameters() if p.requires_grad]; opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
        g = torch.Generator(device=dev); g.manual_seed(seed * 100 + sess); net.train()   # same batches for every candidate
        gp = torch.Generator(device=dev); gp.manual_seed(seed * 100 + sess + 50)
        for _ in range(steps):
            i = torch.randint(0, n, (bs,), device=dev, generator=g); j = torch.randint(0, len(mem["y"]), (RB,), device=dev, generator=g)
            a, b, t = torch.cat([x1[i], mem["x1"][j]]), torch.cat([x2[i], mem["x2"][j]]), torch.cat([y[i], mem["y"][j]])
            o = net(a, b)
            with torch.no_grad():
                old = prev(a, b)
            kd = F.kl_div(F.log_softmax(o / TEMP, 1), F.softmax(old / TEMP, 1), reduction="none").clamp(min=0).sum() / len(t)
            loss = F.cross_entropy(o, t) + cx * kd
            if pool is not None and len(pool["logit"]):
                jp = torch.randint(0, len(pool["logit"]), (RB,), device=dev, generator=gp)
                loss = loss + 0.5 * F.mse_loss(net(pool["x1"][jp], pool["x2"][jp]), pool["logit"][jp])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(vx1, vx2).argmax(1) == vy).float().mean())
        log.append(dict(sess=sess, cand=k, val_acc=acc, new_modules=int(train.sum()), path="".join(map(str, path.argmax(1)))))
        if out is None or acc > out[1]:
            out = (net, acc, path)
    for p in out[0].parameters():
        p.requires_grad_(True)
    return out[0], np.maximum(fixed, out[2]), log
