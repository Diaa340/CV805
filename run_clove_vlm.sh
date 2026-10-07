#!/usr/bin/env bash
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
for o in 0 1; do .venv/bin/python -u vlm_seq.py $o > runs/clove_vlm/log_$o.txt 2>&1 & done
wait; .venv/bin/python analyze_clove_vlm.py > runs/clove_vlm/analysis.log 2>&1; echo "CLOVE VLM DONE" >> runs/clove_vlm/analysis.log
