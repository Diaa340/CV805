#!/usr/bin/env bash
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2
for a in "clevr 0 3" "clevr 1 3" "clevr 2 3" "gqa 0 3" "gqa 1 3" "gqa 2 3" "pope 0 1"; do set -- $a; .venv/bin/python -u code/cl_metrics.py $1 $2 $3 > runs/clm/log_$1_$2.txt 2>&1 & done
wait
until [ $(grep -l "^DONE" runs/clm/log_tables_*.txt 2>/dev/null | wc -l) -eq 4 ]; do sleep 60; done
.venv/bin/python code/analyze_clm.py > runs/clm/analysis.log 2>&1; echo "CLM DONE" >> runs/clm/analysis.log
