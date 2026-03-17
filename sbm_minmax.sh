#!/usr/bin/env bash
set -euo pipefail

tasks=(gsm8k codealpaca codeEvol dolly alpaca)
lr=0.01

for ((i=0; i<${#tasks[@]}-2; i++)); do
  for ((j=i+1; j<${#tasks[@]}-1; j++)); do
    for ((k=j+1; k<${#tasks[@]}; k++)); do
      
      merge_name="${tasks[i]}_${tasks[j]}_${tasks[k]}"
      
      qsub -g tga-mdl \
           -N qw_50_${merge_name} \
           -v MERGE_TASKS=${merge_name},LR=${lr} \
           run_minmax_mutation.sh
      
      sleep 1
    done
  done
done