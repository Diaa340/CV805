#!/usr/bin/env bash
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
.venv/bin/python rev_null.py > runs/rev/null.log 2>&1 &
for a in "tables 0 4" "tables 1 4" "tables 2 4" "tables 3 4" "clevr 0 2" "clevr 1 2" "gqa 0 2" "gqa 1 2"; do set -- $a; .venv_cu128/bin/python -u cl_difficulty.py $1 $2 $3 > runs/rev/difficulty/log_$1_$2.txt 2>&1 & done
wait
.venv/bin/python analyze_difficulty.py > runs/rev/difficulty_analysis.log 2>&1
until [ -f runs/rev/unique.csv ]; do sleep 60; done
.venv_cu128/bin/python rev_unique_analysis.py > runs/rev/unique_analysis.log 2>&1
echo "REVISION DONE" >> runs/rev/difficulty_analysis.log
