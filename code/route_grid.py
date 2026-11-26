"""Scheduler for route_exps.py on GPU 2 (pre-registered grid). Marker files runs/<tag>.{started,done,failed}."""
import os, subprocess, time, itertools
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2")
jobs = []
for d in ("synth", "cifar2v", "avmnist", "cremad"):   # discovery batch: seeds 0-2, default config
    jobs.append((f"disc_{d}", f"--data {d} --seeds 0 1 2 --batch discovery"))
ARCH = [(2, 256), (1, 128), (3, 512)]
for (snr, sh, (dp, w)) in itertools.product(("4.5 3.0", "4.5 4.5", "3.5 3.5"), ("12", "1", "2"), ARCH):
    jobs.append((f"ho_synth_{snr.replace(' ', '-')}_s{sh}_d{dp}w{w}", f"--data synth --snr {snr} --shift {sh} --depth {dp} --width {w} --seeds 3 4 5 6 7"))
for (d, sp, od, (dp, w)) in itertools.product(("cremad", "avmnist", "cifar2v"), (0, 1), ("normal", "swap"), ARCH):
    jobs.append((f"ho_{d}_sp{sp}_{od}_d{dp}w{w}", f"--data {d} --class_split {sp} --order {od} --depth {dp} --width {w} --seeds 3 4 5 6 7"))
has = lambda t, e: os.path.exists(f"runs/{t}.{e}")
MAX = int(os.environ.get("MAXPAR", 6))
while True:
    todo = [j for j in jobs if not has(j[0], "started")]
    running = [j for j in jobs if has(j[0], "started") and not has(j[0], "done") and not has(j[0], "failed")]
    while todo and len(running) < MAX:
        tag, a = todo.pop(0); open(f"runs/{tag}.started", "w").close(); running.append((tag, a))
        subprocess.Popen(f".venv/bin/python -u code/route_exps.py {a} --tag {tag} > runs/{tag}.log 2>&1 && touch runs/{tag}.done || touch runs/{tag}.failed", shell=True, env=env)
        print(time.strftime("%H:%M:%S"), "launch", tag, flush=True)
    if not todo and not running:
        break
    time.sleep(20)
print(time.strftime("%H:%M:%S"), "ALL DONE; failed:", [j[0] for j in jobs if has(j[0], "failed")], flush=True)
