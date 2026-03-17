#!/bin/sh

# task=$TASK # pretrain gsm8k codealpaca codeEvol dolly alpaca

model_name=llama2_7b # deepseek_7b gemma_7b llama2_7b llama2_13b llama3_8b qwen_7b
merge_tasks=gsm8k_codealpaca
# merge_tasks=$MERGE_TASKS
merge_method=task_arithmetic
dataset=advbench
start_idx=0
end_idx=50

# for i in $(seq 1 9)
# do
#     w=$(echo "scale=1; $i/10" | bc)
#     w2=$(echo "scale=1; 1-$w" | bc)
    
#     python eval_merge.py \
#         --model_name ${model_name} \
#         --merge_tasks ${merge_tasks} \
#         --merge_method ${merge_method} \
#         --merge_args $w $w2 \
#         --attack_method ${attack_method} \
#         --dataset ${dataset} \

#     wait
# done

python minmax_gcg.py \
    --model_name ${model_name} \
    --merge_tasks ${merge_tasks} \
    --merge_method ${merge_method} \
    --dataset ${dataset} \
    --idx ${start_idx} ${end_idx} \
