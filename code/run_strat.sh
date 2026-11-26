#!/usr/bin/env bash
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2; mkdir -p runs/clbal
for a in "tables 0 4" "tables 1 4" "tables 2 4" "tables 3 4" "clevr 0 2" "clevr 1 2" "gqa 0 2" "gqa 1 2"; do set -- $a; .venv/bin/python -u code/cl_bal.py $1 $2 $3 > runs/clbal/log_$1_$2.txt 2>&1 & done
for r in $(seq 0 11); do .venv/bin/python -u code/cl_seq.py $r --bal > runs/clove_cl/log_bal_$r.txt 2>&1 & done
wait; .venv/bin/python code/analyze_strat.py > runs/clbal/analysis.log 2>&1; echo "STRAT DONE" >> runs/clbal/analysis.log
