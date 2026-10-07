#!/usr/bin/env bash
# v4 measurement grid (preregistration_v4_fia.md + addendum), GPU 2, seeds 23-25.
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2; S="--seeds 23 24 25"
launch () { tag=$1; shift; .venv/bin/python -u v4_measure.py "$@" $S --tag $tag > runs/$tag.log 2>&1 & }
for a in "2 256" "3 512"; do set -- $a; launch v4_synth_d$1w$2 --data synth --snr 4.5 3.0 --shift 12 --depth $1 --width $2; done
for d in cifar2v food101; do for a in "2 256" "3 512"; do set -- $a; launch v4_bridge_${d}_d$1w$2 --data bridge_$d --shift 12 --depth $1 --width $2; done; done
wait
for d in cifar2v avmnist cremad food101 n24news clevr fv_cremad; do for od in normal swap; do launch v4_${d}_${od} --data $d --class_split 0 --order $od; done; done
wait
echo V4 GRID DONE
