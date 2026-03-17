import json
import copy
import gc
import torch
import argparse
import sys
import pandas as pd
from tqdm import tqdm
import time
from accelerate.utils import find_executable_batch_size
from opt_utils import (
    get_score_autodan,
    autodan_sample_control,
    autodan_sample_control_hga,
)
from utils import (
    set_seed,
    create_dir,
    load_dataset,
    load_model_and_tokenizer,
    check_jailbreak_success,
    ask_judge_harmbench,
    get_not_allowed_tokens,
)
from merge_utils import ModelMergingTA

model_judge_path = 'models/HarmBench-Llama-2-13b-cls'
base_path = '/'
# dataset path
dataset_path = {
    'advbench': 'data/advbench/harmful_behaviors.csv',
    'malicious': 'data/malicious/malicious.csv',
    'harmbench': 'data/harmbench/harmbench_gcg.csv',
}

probing_dataset = {
    'benign': 'data/prompt-driven_benign.txt',
    'harmful': 'data/prompt-driven_harmful.txt',
}

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama2_7b")
    parser.add_argument('--merge_tasks', type=str, default="alpaca_dolly")
    parser.add_argument('--merge_method', type=str, default="task_arithmetic")
    # parser.add_argument('--merge_args', type=float, nargs='+', required=True)
    # parser.add_argument('--attack_method', type=str, default="direct")
    parser.add_argument('--dataset', type=str, default="advbench")
    parser.add_argument("--idx", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--training_epoch", type=int, default=10)
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--sampling_number",
        type=int,
        default=512,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--mutation",
        type=float,
        default=0.01,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=5,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=5,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Output directory for individual results",
    )
    parser.add_argument(
        "--crossover",
        type=float,
        default=0.5,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--num_elites",
        type=float,
        default=0.05,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of candidates to sample (B in the algorithm)",
    )
    parser.add_argument(
        "--init_prompt_path",
        type=str,
        default=f"assets/autodan_initial_prompt.txt",
        help="Output directory for individual results",
    )

    args = parser.parse_args()
    return args

def log_init():
    log_dict = {"loss": [], "suffix": [],
                "time": [], "respond": [], "jailbroken": []}
    return log_dict

def get_developer(model_name):
    developer_dict = {"llama2_7b": "Meta", "llama2_13b": "Meta", "llama3_8b": "Meta", "vicuna": "LMSYS", "qwen_7b": "Alibaba", "gemma_7b": "Google",
                      "guanaco": "TheBlokeAI", "WizardLM": "WizardLM", "deepseek_7b": "Deepseek",
                      "mpt-chat": "MosaicML", "mpt-instruct": "MosaicML", "falcon": "TII"}
    return developer_dict[model_name]

def get_score_scav_prob(merged_model, instruction, target, test_controls, embedding_origin):
    losses = []
    # import ipdb
    # ipdb.set_trace()
    for item in test_controls:
        if item is not None:
            adv_string = item.replace('[REPLACE]', instruction.lower())

        messages = [
            {"role": "user", "content": adv_string},
        ]
        prompt = merged_model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        test_rep = merged_model.get_single_rep(prompt)

        hidden_state_start_point = embedding_origin[-1][:, -1, :]
        hidden_state_start_point = hidden_state_start_point.view(
            hidden_state_start_point.shape[0], -1
        )
        hidden_state = test_rep[-1][:, -1, :]
        vector_from_start_to_here_batch = hidden_state - hidden_state_start_point

        projected_distance_from_start_batch = torch.sum(
            vector_from_start_to_here_batch * merged_model.probing_direction, dim=1
        )
        projected_distance_from_start = torch.mean(projected_distance_from_start_batch)
        loss = -projected_distance_from_start
        
        losses.append(loss)

    return torch.stack(losses)

def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")
    set_seed(args.seed)

    merge_task_list = args.merge_tasks.split('_')
    learning_rate = args.learning_rate
    training_epoch = args.training_epoch
    adv_string_init = open(args.init_prompt_path, 'r').readlines()
    adv_string_init = ''.join(adv_string_init)
    template_name = args.model_name
    num_steps = args.num_steps
    batch_size = args.batch_size
    num_elites = max(1, int(args.batch_size * args.num_elites))
    crossover = args.crossover
    num_points = args.num_points
    mutation = args.mutation
    prefix_string_init = None

    # Load datasets
    dataset = pd.read_csv(dataset_path[args.dataset], usecols=[0, 1]).to_numpy()
    if args.idx is not None:
        lower_bound = max(0, args.idx[0])
        lower_bound = min(len(dataset), lower_bound)
        upper_bound = min(len(dataset), args.idx[1])
        upper_bound = max(0, upper_bound)
        dataset = dataset[lower_bound:upper_bound]
    else:
        lower_bound = 0
        upper_bound = len(dataset)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(f"models/{model_dict[args.model_name]}", False, "cpu")
    model.eval()
    model.requires_grad_(False)  # Save memory
    model_judge, tokenizer_judge = load_model_and_tokenizer(model_judge_path, False, "cuda:0")
    model_judge.eval()
    model_judge.requires_grad_(False)

    # Load probing datasets
    dataset_probe_benign = load_dataset(probing_dataset['benign'])
    dataset_probe_harmful = load_dataset(probing_dataset['harmful'])

    # Load merge model
    merge_model_list = [model]
    for task_name in merge_task_list:
        task_path = f"finetuned_models/{task_name}/{args.model_name}/inst_epoch1_safe0"
        task_model = load_model_and_tokenizer(task_path, True, "cpu")
        task_model.eval()
        task_model.requires_grad_(False)
        merge_model_list.append(task_model)

    if args.merge_method == 'task_arithmetic':
        merged_model = ModelMergingTA(args.model_name, merge_model_list, tokenizer, model_judge, tokenizer_judge, dataset_probe_benign, dataset_probe_harmful, learning_rate, training_epoch, 'tanh', args.seed)

    # Clear Task Models
    del merge_model_list
    gc.collect()

    create_dir(f'{base_path}/results/baj_mutation/{args.dataset}/{args.model_name}/{args.merge_method}_lr{str(args.learning_rate)}_te{str(args.training_epoch)}')
    result_filename = f'{base_path}/results/baj_mutation/{args.dataset}/{args.model_name}/{args.merge_method}_lr{str(args.learning_rate)}_te{str(args.training_epoch)}/{args.merge_tasks}_{args.idx[0]}_{args.idx[1]}.json'

    # Run the jailbreak attack
    final_prompt_list = []
    final_response_list = []
    infos = []
    for idx, (goal, target) in tqdm(enumerate(dataset, start=lower_bound), total=len(dataset)):
        merged_model.init_lambda()
        reference = torch.load(f'assets/prompt_group.pth', map_location='cpu')

        log = log_init()
        info = {"idx": idx, "prompt": "", "target": "", "adv_string": "",
                "response": "", "total_time": 0, "jailbroken": False, "log": log}
        info["prompt"] = info["prompt"].join(goal)
        info["target"] = info["target"].join(target)
        start_time = time.time()

        user_prompt = goal
        target = target
        for o in range(len(reference)):
            reference[o] = reference[o].replace('[MODEL]', template_name.title())
            reference[o] = reference[o].replace('[KEEPER]', get_developer(template_name))
        new_adv_suffixs = reference[:batch_size]
        word_dict = {}

        messages = [
            {"role": "user", "content": "{instruction}"},
        ]
        original_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        emb_origin = merged_model.get_single_rep(original_prompt)

        for j in range(num_steps):
            with torch.no_grad():
                epoch_start_time = time.time()
                losses = get_score_scav_prob(merged_model, user_prompt, target, new_adv_suffixs, emb_origin)
                score_list = losses.cpu().numpy().tolist()

                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffixs[best_new_adv_suffix_id]

                current_loss = losses[best_new_adv_suffix_id]

                if isinstance(prefix_string_init, str):
                    best_new_adv_suffix = prefix_string_init + best_new_adv_suffix
                adv_suffix = best_new_adv_suffix

                is_success, gen_str = merged_model.check_jailbreak_success(user_prompt, adv_suffix)

                if j % args.iter == 0:
                    unfiltered_new_adv_suffixs = autodan_sample_control(control_suffixs=new_adv_suffixs,
                                                                        score_list=score_list,
                                                                        num_elites=num_elites,
                                                                        batch_size=batch_size,
                                                                        crossover=crossover,
                                                                        num_points=num_points,
                                                                        mutation=mutation,
                                                                        API_key=None,
                                                                        reference=reference)
                else:
                    unfiltered_new_adv_suffixs, word_dict = autodan_sample_control_hga(word_dict=word_dict,
                                                                                        control_suffixs=new_adv_suffixs,
                                                                                        score_list=score_list,
                                                                                        num_elites=num_elites,
                                                                                        batch_size=batch_size,
                                                                                        crossover=crossover,
                                                                                        mutation=mutation,
                                                                                        API_key=None,
                                                                                        reference=reference)

                new_adv_suffixs = unfiltered_new_adv_suffixs

                epoch_end_time = time.time()
                epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)

                print(
                    "################################\n"
                    f"Current Data: {idx}/{len(dataset)}\n"
                    f"Current Epoch: {j}/{num_steps}\n"
                    f"Passed:{is_success}\n"
                    f"Loss:{current_loss.item()}\n"
                    f"Epoch Cost:{epoch_cost_time}\n"
                    f"Current Suffix:\n{best_new_adv_suffix}\n"
                    f"Current Response:\n{gen_str}\n"
                    f"Current Jailbroken:\n{str(is_success)}\n"
                    "################################\n")

                info["log"]["time"].append(epoch_cost_time)
                info["log"]["loss"].append(current_loss.item())
                info["log"]["suffix"].append(best_new_adv_suffix)
                info["log"]["respond"].append(gen_str)
                info["log"]["jailbroken"].append(is_success)

                last_loss = current_loss.item()

                if is_success:
                    # Training Merged Parameters
                    print('################################ Finding Max Merged Parameters')
                    is_end = merged_model.train_merged_param(goal, adv_suffix)
                    if is_end:
                        break
                
                gc.collect()
                torch.cuda.empty_cache()

        end_time = time.time()
        cost_time = round(end_time - start_time, 2)
        info["total_time"] = cost_time
        info["adv_string"] = adv_suffix
        info["response"] = gen_str
        info["jailbroken"] = is_success
        
        final_prompt_list.append(adv_suffix)
        final_response_list.append(gen_str)

        infos.append(info)
    
        with open(result_filename, 'w') as json_file:
            json.dump(infos, json_file, indent=4)
        
if __name__ == "__main__":
    main()
