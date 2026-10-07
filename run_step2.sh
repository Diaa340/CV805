#!/usr/bin/env bash
# Step 2 chain: waits for step 1; runs only if R1 passed and A or B passed (preregistration_step1.md). GPU 2 only.
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2; L=runs/clevr2/chain.log
until grep -q "STEP1 DONE" runs/tables/step1_analysis.log 2>/dev/null; do sleep 60; done
V=$(grep "VERDICTS" runs/tables/step1_analysis.log)
echo "step 1: $V" >> $L
if echo "$V" | grep -q "R1 STRUCTURE WINS" && echo "$V" | grep -Eq "(A|B) STRUCTURE WINS"; then
  echo "step 2 starts $(date)" >> $L
  for s in 0 1 2 3; do .venv/bin/python -u step2_train.py --shard $s 4 --out train_s$s > runs/clevr2/train_shard$s.log 2>&1 & done
  wait
  .venv/bin/python - <<'PY'
import glob, pandas as pd
pd.concat([pd.read_csv(f) for f in sorted(glob.glob("runs/clevr2/train_s*.csv"))]).to_csv("runs/clevr2/train.csv", index=False)
PY
  .venv/bin/python analyze_step2.py > runs/clevr2/analysis.log 2>&1
  echo "STEP2 DONE" >> runs/clevr2/analysis.log
else
  echo "step 2 NOT run: step-1 condition not met" >> $L; echo "STEP2 SKIPPED" >> runs/clevr2/analysis.log
fi
