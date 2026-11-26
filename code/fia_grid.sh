#!/usr/bin/env bash
# FIA grid (preregistration_v4_fia.md), GPU 2, seeds 23-25; starts after the v4 grid; then both analyses.
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=2; S="--seeds 23 24 25"
until grep -q "V4 GRID DONE" runs/v4_grid.log; do sleep 30; done
.venv/bin/python code/analyze_v4.py > runs/v4_analysis.log 2>&1
launch () { tag=$1; shift; .venv/bin/python -u code/fia.py "$@" $S --tag $tag > runs/$tag.log 2>&1 & }
for d in cifar2v food101; do launch fia_bridge_$d --data bridge_$d --shift 12; done
for d in cifar2v cremad food101 n24news avmnist fv_cremad clevr; do for od in normal swap; do launch fia_${d}_${od} --data $d --class_split 0 --order $od; done; done
wait
.venv/bin/python code/analyze_fia.py > runs/fia_analysis.log 2>&1
echo FIA GRID DONE >> runs/fia_analysis.log
