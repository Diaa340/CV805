"""Scheduler for P2 (preregistration_p2_model_types.md) on GPU 2."""
import os, subprocess, time, itertools
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2")
jobs = [(f"p2_{d}_sp{sp}_{od}", f"--data {d} --class_split {sp} --order {od} --seeds 13 14 15 16 17 --model_types")
        for d, sp, od in itertools.product(("cifar2v", "avmnist", "cremad", "food101", "n24news"), (0, 1), ("normal", "swap"))]
has = lambda t, e: os.path.exists(f"runs/{t}.{e}")
while True:
    todo = [j for j in jobs if not has(j[0], "started")]
    running = [j for j in jobs if has(j[0], "started") and not has(j[0], "done") and not has(j[0], "failed")]
    while todo and len(running) < 10:
        tag, a = todo.pop(0); open(f"runs/{tag}.started", "w").close(); running.append((tag, a))
        subprocess.Popen(f".venv/bin/python -u v3_drift.py {a} --tag {tag} > runs/{tag}.log 2>&1 && touch runs/{tag}.done || touch runs/{tag}.failed", shell=True, env=env)
    if not todo and not running:
        break
    time.sleep(15)
print(time.strftime("%H:%M:%S"), "P2 DONE; failed:", [j[0] for j in jobs if has(j[0], "failed")], flush=True)
