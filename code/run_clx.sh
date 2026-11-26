#!/usr/bin/env bash
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2
for b in tables clevr; do for s in 0 1 2; do .venv/bin/python -u code/cl_exceptions.py $b $s 3 > runs/clx/log_${b}_$s.txt 2>&1 & done; done
wait; .venv/bin/python code/analyze_clx.py > runs/clx/analysis.log 2>&1; echo "CLX DONE" >> runs/clx/analysis.log
