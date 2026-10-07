#!/usr/bin/env bash
# P4 pipeline: wait for downloads -> extract features -> run jobs on GPU 2 -> pre-registered analysis.
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
run_jobs () {  # $1 = data, then arg sets
  d=$1; shift
  for spec in "$@"; do set -- $spec; tag="p4_${d}_sp$1_$2_d$3w$4"
    .venv/bin/python -u p3_readout.py --data $d --class_split $1 --order $2 --depth $3 --width $4 --seeds 13 14 15 16 17 --tag $tag > runs/$tag.log 2>&1 &
  done
}
( until grep -q vqa-done data/raw/vqa/download.log; do sleep 30; done
  .venv/bin/python -u extract_features.py vqa > runs/extract_vqa.log 2>&1 && run_jobs vqa "0 normal 2 256" "0 swap 2 256" "1 normal 2 256" "1 swap 2 256"; wait ) &
( until grep -q clevr-done data/raw/clevr/download.log; do sleep 60; done
  .venv/bin/python -u extract_features.py clevr > runs/extract_clevr.log 2>&1 && run_jobs clevr "0 normal 2 256" "0 swap 2 256" "0 normal 3 512" "0 swap 3 512"; wait ) &
wait
until [ $(ps -u $USER -o cmd= | grep -c "p3_readout[.]py --data") -eq 0 ]; do sleep 30; done
.venv/bin/python analyze_p4.py > runs/p4_analysis.log 2>&1
echo "P4 CHAIN DONE" >> runs/p4_analysis.log
