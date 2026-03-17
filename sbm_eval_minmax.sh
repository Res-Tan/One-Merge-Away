#!/usr/bin/env bash

set -eu

tasks=(gsm8k codealpaca codeEvol dolly alpaca)

REPO_ROOT="/"
DEFAULT_DIR="$REPO_ROOT/results/baj_mutation/advbench/deepseek_7b/task_arithmetic_lr0.05_te10"
TARGET_DIR="${1:-$DEFAULT_DIR}"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Directory not found: $TARGET_DIR" >&2
  exit 1
fi

find "$TARGET_DIR" -maxdepth 1 -type f -name '*_0_100.json' | sort | while IFS= read -r file_path; do
  file_name="$(basename "$file_path")"
  task_name="${file_name%_0_100.json}"

  IFS='_' read -r task1 task2 task3 <<< "$task_name"
  task_list=("$task1" "$task2" "$task3")

  remaining=()

  for t in "${tasks[@]}"; do
      [[ " ${task_list[*]} " =~ " $t " ]] || remaining+=("$t")
  done

  merge_tasks=$(IFS=_; echo "${remaining[*]}")
  echo "$merge_tasks"
  # echo $DEFAULT_DIR

  qsub -g tga-mdl -N ds_0.05 -v MERGE_TASKS=${merge_tasks},MINMAX_TASKS=${task_name},TARGET_PATH=${DEFAULT_DIR} run_eval_merge_minmax.sh
    
  sleep 1
done

