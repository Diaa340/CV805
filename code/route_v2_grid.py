"""Scheduler for route_v2.py (preregistration_v2.md grid) on GPU 2."""
import os, subprocess, time, itertools
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2")
jobs = [("v2disc", "--seeds 0 1 2 --batch discovery")]
for snr, sh, (dp, w) in itertools.product(("4.5 3.0", "4.5 4.5", "3.5 3.5"), ("12", "1", "2"), [(2, 256), (1, 128), (3, 512)]):
    jobs.append((f"v2_{snr.replace(' ', '-')}_s{sh}_d{dp}w{w}", f"--snr {snr} --shift {sh} --depth {dp} --width {w} --seeds 3 4 5 6 7"))
has = lambda t, e: os.path.exists(f"runs/{t}.{e}")
while True:
    todo = [j for j in jobs if not has(j[0], "started")]
    running = [j for j in jobs if has(j[0], "started") and not has(j[0], "done") and not has(j[0], "failed")]
    while todo and len(running) < 6:
        tag, a = todo.pop(0); open(f"runs/{tag}.started", "w").close(); running.append((tag, a))
        subprocess.Popen(f".venv/bin/python -u code/route_v2.py {a} --tag {tag} > runs/{tag}.log 2>&1 && touch runs/{tag}.done || touch runs/{tag}.failed", shell=True, env=env)
    if not todo and not running:
        break
    time.sleep(20)
print(time.strftime("%H:%M:%S"), "V2 DONE; failed:", [j[0] for j in jobs if has(j[0], "failed")], flush=True)
