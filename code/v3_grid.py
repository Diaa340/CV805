"""Scheduler for v3_drift.py (preregistration_v3.md) on GPU 2."""
import os, subprocess, time, itertools
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2")
jobs = []
for snr, sh, (dp, w) in itertools.product(("4.5 3.0", "4.5 4.5", "3.5 3.5"), ("12", "1", "2"), [(2, 256), (1, 128), (3, 512)]):
    jobs.append((f"v3_synth_{snr.replace(' ', '-')}_s{sh}_d{dp}w{w}", f"--data synth --snr {snr} --shift {sh} --depth {dp} --width {w} --seeds 13 14 15 16 17"))
for d, sp, od in itertools.product(("cifar2v", "avmnist", "cremad", "food101", "n24news"), (0, 1), ("normal", "swap")):
    jobs.append((f"v3_{d}_sp{sp}_{od}", f"--data {d} --class_split {sp} --order {od} --seeds 3 4 5 6 7"))
has = lambda t, e: os.path.exists(f"runs/{t}.{e}")
while True:
    todo = [j for j in jobs if not has(j[0], "started")]
    running = [j for j in jobs if has(j[0], "started") and not has(j[0], "done") and not has(j[0], "failed")]
    while todo and len(running) < 8:
        tag, a = todo.pop(0); open(f"runs/{tag}.started", "w").close(); running.append((tag, a))
        subprocess.Popen(f".venv/bin/python -u code/v3_drift.py {a} --tag {tag} > runs/{tag}.log 2>&1 && touch runs/{tag}.done || touch runs/{tag}.failed", shell=True, env=env)
    if not todo and not running:
        break
    time.sleep(15)
print(time.strftime("%H:%M:%S"), "V3 DONE; failed:", [j[0] for j in jobs if has(j[0], "failed")], flush=True)
