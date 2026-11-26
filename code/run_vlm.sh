#!/usr/bin/env bash
# VLM scale test (preregistration_vlm.md): after step 2 finishes. GPU 2 only.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2
until grep -qE "STEP2 (DONE|SKIPPED)" runs/clevr2/analysis.log 2>/dev/null; do sleep 60; done
.venv/bin/python -u code/vlm_test.py --bs 16 > runs/clevr2/vlm_run.log 2>&1
.venv/bin/python code/analyze_vlm.py > runs/clevr2/vlm_analysis.log 2>&1
echo "VLM DONE" >> runs/clevr2/vlm_analysis.log
