#!/usr/bin/env bash
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
for a in "tables 0 3" "tables 1 3" "tables 2 3" "clevr 0 2" "clevr 1 2" "gqa 0 3" "gqa 1 3" "gqa 2 3"; do set -- $a; .venv_cu128/bin/python -u curric.py $1 $2 $3 > runs/curric/log_$1_$2.txt 2>&1 & done
wait
.venv_cu128/bin/python -u vlm_curric.py 0,1,2 > runs/vlm_curric/log_a.txt 2>&1 &
.venv_cu128/bin/python -u vlm_curric.py 3,4,5 > runs/vlm_curric/log_b.txt 2>&1 &
wait
.venv/bin/python analyze_curric.py > runs/curric/analysis.log 2>&1; echo "CURRIC DONE" >> runs/curric/analysis.log
