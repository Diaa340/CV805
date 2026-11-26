#!/usr/bin/env bash
# MMBench / MMStar chain (preregistration_mm.md), after the POPE Xiu chain. GPU 2.
cd "$(dirname "$0")/xiu"; export CUDA_VISIBLE_DEVICES=2; PY=../../.venv_xiu/bin/python
until [ -f chain_done.txt ]; do sleep 60; done
for ds in mmbench mmstar; do
  mkdir -p results/devis_vector/$ds results/embeds/$ds/train results/embeds/$ds/val
  $PY -u qwen25_small_devi.py --read_dir data/$ds/${ds}_ --image_dir "" --model_size 7 --data_mode train --vector_save_dir results/devis_vector/$ds > logs_devi_$ds.txt 2>&1
  for m in train val; do $PY -u qwen25_small_embed.py --read_dir data/$ds/${ds}_ --image_dir "" --save_dir results/embeds/$ds --num_choices 4 --model_qwen_size 7 --data_mode $m --devi_vector_dir results/devis_vector/$ds > logs_embed_${ds}_$m.txt 2>&1; done
  $PY -u mm_scores.py $ds > logs_scores_$ds.txt 2>&1
done
cd ../..; .venv/bin/python code/analyze_mm.py > runs/mm/analysis.log 2>&1; echo "MM DONE" >> runs/mm/analysis.log
