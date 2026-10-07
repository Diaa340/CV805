#!/usr/bin/env bash
# Step 1 pipeline (preregistration_step1.md). GPU 2 only.
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2; L=runs/tables/step1_pipeline.log
( .venv_dit/bin/python tables_pid2.py fresh 12 && .venv_dit/bin/python tables_pid2.py paircand 12 && .venv/bin/python step1_pairs.py ) >> $L 2>&1 &
PIDJOB=$!
for s in 0 1 2 3; do
  ( for set in selected fresh; do .venv/bin/python -u tables_step1.py --set $set --shard $s 4 --out ${set}_s$s; done ) > runs/tables/step1_shard$s.log 2>&1 &
done
wait $PIDJOB; echo "PID + pairs done" >> $L
( for s in 0 1; do .venv/bin/python -u tables_step1.py --set pairs --shard $s 2 --out pairs_s$s > runs/tables/step1_pairs_shard$s.log 2>&1 & done; wait ) &
wait
.venv/bin/python analyze_step1.py > runs/tables/step1_analysis.log 2>&1
echo STEP1 DONE >> runs/tables/step1_analysis.log
