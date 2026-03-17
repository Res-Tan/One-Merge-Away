#!/usr/bin/env bash
set -euo pipefail

tasks=(gsm8k codealpaca codeEvol dolly alpaca)

for ((i=0; i<${#tasks[@]}; i++)); do
  for ((j=i+1; j<${#tasks[@]}; j++)); do
    qsub -g tga-mdl -ar 5829 -N ${tasks[i]}_${tasks[j]} -v MERGE_TASKS=${tasks[i]}_${tasks[j]} run_eval_merge.sh
    sleep 1
  done
done
