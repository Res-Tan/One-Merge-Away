import json
import copy
import gc
import torch
import argparse
from utils import (
    create_dir,
    load_model_and_tokenizer,
    check_jailbreak_success,
    ask_judge_harmbench,
)
from merge_utils import get_eval_merged_model

model_judge_path = '.../models/HarmBench-Llama-2-13b-cls'
base_path = '/'

model_dict = {
    # "llama2_7b_base": "llama-2-7b-hf",
    "llama2_7b": "llama-2-7b-chat-hf",
    # "llama2_13b_base": "llama-2-13b-hf",
    "llama2_13b": "llama-2-13b-chat-hf",
    "llama3_8b": "llama-3.1-8B-Instruct",
    "mistral_7b": "Mistral-7B-Instruct-v0.2",
    "vicuna_7b": "vicuna-7b-v1.5",
    "zephyr_7b_beta": "zephyr-7b-beta",
    "zephyr_7b_alpha": "zephyr-7b-alpha",
    "qwen_7b": "Qwen-7B-chat",
    "qwen_14b_chat": "Qwen-14B-Chat",
    "deepseek_7b": "deepseek_7b_chat",
    "baichuan2_7b": "Baichuan2-7B-Chat",
    "falcon_7b": "falcon-7b-instruct",
    "gemma_7b": "gemma-7b-it",
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="llama2_7b")
parser.add_argument('--merge_tasks', type=str, default="alpaca_dolly")
parser.add_argument('--merge_method', type=str, default="linear")
parser.add_argument('--merge_args', type=float, nargs='+', required=True)
parser.add_argument('--attack_method', type=str, default="direct")
parser.add_argument('--dataset', type=str, default="advbench")

args = parser.parse_args()

merge_task_list = args.merge_tasks.split('_')

# Load eval datalist
eval_data_path = f'{base_path}/results_new/gcg/{args.dataset}/{args.model_name}/space_seed42/space_pretrain_0_100.json'
with open(eval_data_path, 'r') as f:
    eval_data_list = json.load(f)
# eval_data_list = eval_data_list[:100]

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(f".../models/{model_dict[args.model_name]}", False, "cpu")
model.eval()
model.requires_grad_(False)  # Save memory
model_judge, tokenizer_judge = load_model_and_tokenizer(model_judge_path, False, "cuda:0")
model_judge.eval()
model_judge.requires_grad_(False)

# Load merge model
merge_model_list = [model]
for task_name in merge_task_list:
    task_path = f".../finetuned_models/{task_name}/{args.model_name}/inst_epoch1_safe0"
    task_model = load_model_and_tokenizer(task_path, True, "cpu")
    task_model.eval()
    task_model.requires_grad_(False)
    merge_model_list.append(task_model)

merged_model = get_eval_merged_model(args.model_name, merge_model_list, args.merge_method, args.merge_args)

# Clear Task Models
del merge_model_list
gc.collect()

# Create Save Filder
save_file_name = "_".join(f"{n}{v}" for n, v in zip(merge_task_list, args.merge_args))
target_save_path = f'{base_path}/minmax/results/{args.attack_method}/{args.dataset}/{args.model_name}/{args.merge_method}'
create_dir(target_save_path)

# Evaluate Attacks
if len(eval_data_list) != 0:
    save_data_list = []
    for data in eval_data_list:
        adv_prompt = data['prompt']
        adv_string = data['adv_string']
        if args.attack_method == 'direct':
            adv_examples = f"{adv_prompt}"
        else:
            adv_examples = f"{adv_prompt} {adv_string}"

        is_jailbrokens, responses = check_jailbreak_success(
            model=merged_model,
            tokenizer=tokenizer,
            prompt=adv_examples,
            generate_length=512,
            return_response=True,
        )
        ratings = ask_judge_harmbench(
            model_judge,
            tokenizer_judge,
            adv_prompt,
            responses,
        )

        is_jailbroken = ratings[0] == 1

        save_data = copy.deepcopy(data)
        # save_data['eval_responses'].append(responses[0])
        # save_data['eval_jailbroken'].append(is_jailbroken)
        save_data['merged_responses'] = responses
        save_data['merged_jailbroken'] = is_jailbroken
        # save_data['ft_response'] = [responses[0]]
        # save_data['harmful_judge'] = is_jailbroken
        # save_data['eval_tasks'].append(args.task)
        save_data_list.append(save_data)

    with open(f'{target_save_path}/{save_file_name}.json', 'w') as f:
        json.dump(save_data_list, f, indent=4)

