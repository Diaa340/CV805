#!/usr/bin/env bash
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
until [ -f runs/clm/structure.npz ]; do sleep 30; done
for a in "tables 0 4" "tables 1 4" "tables 2 4" "tables 3 4" "clevr 0 3" "clevr 1 3" "clevr 2 3" "gqa 0 3" "gqa 1 3" "gqa 2 3" "pope 0 1"; do set -- $a; .venv/bin/python -u cl_metrics.py $1 $2 $3 > runs/clm/log_$1_$2.txt 2>&1 & done
wait; .venv/bin/python analyze_clm.py > runs/clm/analysis.log 2>&1; echo "CLM DONE" >> runs/clm/analysis.log
