#!/usr/bin/env bash
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
for o in 0 1; do .venv/bin/python -u vlm_seq_nonce.py $o --resume > runs/nonce_vlm/log_resume_$o.txt 2>&1; done
.venv/bin/python analyze_nonce_vlm.py > runs/nonce_vlm/analysis.log 2>&1; echo "NONCE DONE" >> runs/nonce_vlm/analysis.log
