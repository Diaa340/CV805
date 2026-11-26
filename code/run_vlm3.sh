#!/usr/bin/env bash
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2
.venv/bin/python -u code/vlm3_score.py qwen > runs/vlm3/log_qwen.txt 2>&1 &
.venv/bin/python -u code/vlm3_score.py internvl > runs/vlm3/log_internvl.txt 2>&1 &
wait
.venv_llava15/bin/python -u code/vlm3_score.py llava > runs/vlm3/log_llava.txt 2>&1
until ! pgrep -f vlm3_refine.py > /dev/null; do sleep 60; done
.venv/bin/python code/analyze_vlm3.py > runs/vlm3/analysis.log 2>&1; echo "VLM3 DONE" >> runs/vlm3/analysis.log
