#!/usr/bin/env bash
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
until [ -f runs/unrepair/structure.npz ]; do sleep 30; done
for a in "tables 0 2" "tables 1 2" "clevr 0 2" "clevr 1 2" "gqa 0 1" "pope 0 1"; do
  set -- $a; .venv/bin/python -u unrepair.py $1 $2 $3 > runs/unrepair/log_$1_$2.txt 2>&1 &
done
wait; .venv/bin/python analyze_unrepair.py > runs/unrepair/analysis.log 2>&1; echo "UNREPAIR DONE" >> runs/unrepair/analysis.log
