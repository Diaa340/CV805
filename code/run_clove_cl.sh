#!/usr/bin/env bash
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2
for r in $(seq 0 11); do .venv/bin/python -u code/cl_seq.py $r > runs/clove_cl/log_$r.txt 2>&1 & done
wait; .venv/bin/python code/analyze_clove_cl.py > runs/clove_cl/analysis.log 2>&1; echo "CLOVE CL DONE" >> runs/clove_cl/analysis.log
