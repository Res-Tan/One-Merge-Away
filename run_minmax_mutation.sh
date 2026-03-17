#!/bin/sh

# task=$TASK # pretrain gsm8k codealpaca codeEvol dolly alpaca

model_name=qwen_7b # deepseek_7b gemma_7b llama2_7b llama2_13b llama3_8b qwen_7b
# merge_tasks=codealpaca_dolly_alpaca
merge_tasks=$MERGE_TASKS
merge_method=task_arithmetic
dataset=advbench
start_idx=0
end_idx=50
# lr=0.01
lr=$LR

python minmax_mutation.py \
    --model_name ${model_name} \
    --merge_tasks ${merge_tasks} \
    --merge_method ${merge_method} \
    --dataset ${dataset} \
    --learning_rate ${lr} \
    --idx ${start_idx} ${end_idx} \
