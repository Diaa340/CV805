#!/usr/bin/env bash
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2
for a in "tables 0 2" "tables 1 2" "gqa 0 1" "pope 0 1" "clevr 0 2" "clevr 1 2"; do set -- $a; .venv/bin/python -u code/unrepair2.py $1 $2 $3 > runs/unrepair2/log_$1_$2.txt 2>&1 & done
for s in 0 1 2; do .venv/bin/python -u code/cl_exceptions2.py tables $s 3 > runs/clx2/log_tables_$s.txt 2>&1 & done
until [ -f runs/clx2/train_structure.npz ]; do sleep 30; done
for s in 0 1; do .venv/bin/python -u code/cl_exceptions2.py clevr $s 2 > runs/clx2/log_clevr_$s.txt 2>&1 & done
wait; .venv/bin/python code/analyze_f1f2.py > runs/f1f2_analysis.log 2>&1; echo "F1F2 DONE" >> runs/f1f2_analysis.log
