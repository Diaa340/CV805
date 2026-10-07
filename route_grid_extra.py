"""Extra held-out data sources for route_exps.py (pre-registered 'if available'): same real-data configs as route_grid.py."""
import os, subprocess, sys, time, itertools
env = dict(os.environ, CUDA_VISIBLE_DEVICES="2")
datasets = sys.argv[1:] or ["food101"]
ARCH = [(2, 256), (1, 128), (3, 512)]
jobs = [(f"ho_{d}_sp{sp}_{od}_d{dp}w{w}", f"--data {d} --class_split {sp} --order {od} --depth {dp} --width {w} --seeds 3 4 5 6 7")
        for d, sp, od, (dp, w) in itertools.product(datasets, (0, 1), ("normal", "swap"), ARCH)]
has = lambda t, e: os.path.exists(f"runs/{t}.{e}")
MAX = int(os.environ.get("MAXPAR", 6))
while True:
    todo = [j for j in jobs if not has(j[0], "started")]
    running = [j for j in jobs if has(j[0], "started") and not has(j[0], "done") and not has(j[0], "failed")]
    while todo and len(running) < MAX:
        tag, a = todo.pop(0); open(f"runs/{tag}.started", "w").close(); running.append((tag, a))
        subprocess.Popen(f".venv/bin/python -u route_exps.py {a} --tag {tag} > runs/{tag}.log 2>&1 && touch runs/{tag}.done || touch runs/{tag}.failed", shell=True, env=env)
        print(time.strftime("%H:%M:%S"), "launch", tag, flush=True)
    if not todo and not running:
        break
    time.sleep(20)
print(time.strftime("%H:%M:%S"), "EXTRA DONE; failed:", [j[0] for j in jobs if has(j[0], "failed")], flush=True)
