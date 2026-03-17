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

model_judge_path = '.../models/HarmBench-Llama-2-13b-cls'
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
    parser.add_argument("--init_adv_string", type=str, default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", help="Initial adversarial string")
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

    args = parser.parse_args()
    return args

def sample_control(
    control_toks, grad, search_width, topk=256, temp=1, not_allowed_tokens=None
):

    if not_allowed_tokens is not None:
        # grad[:, not_allowed_tokens.to(grad.device)] = np.infty
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / search_width, device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (search_width, 1), device=grad.device),
    )
    # Honestly you should use .scatter here instead of .scatter_ but that's how GCG's source code does it so I'll leave it here
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks

def filter_candidates(sampled_top_indices, tokenizer):
    sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
    new_sampled_top_indices = []
    count = 0
    for j in range(len(sampled_top_indices_text)):
        # tokenize again
        tmp = tokenizer(
            sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False
        ).to(sampled_top_indices.device)["input_ids"][0]
        # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
        if not torch.equal(tmp, sampled_top_indices[j]):
            count += 1
            continue
        else:
            new_sampled_top_indices.append(sampled_top_indices[j])

    if len(new_sampled_top_indices) == 0:
        raise ValueError("All candidates are filtered out.")

    sampled_top_indices = torch.stack(new_sampled_top_indices)
    return sampled_top_indices

@find_executable_batch_size(starting_batch_size=256)
def second_forward(candidate_batch_size, model, full_embed, hidden_states_start_point, probing_direction):
    losses_batch = []
    for i in range(0, full_embed.shape[0], candidate_batch_size):
        with torch.no_grad():
            full_embed_this_batch = full_embed[i : i + candidate_batch_size]
            outputs = model(
                inputs_embeds=full_embed_this_batch, output_hidden_states=True
            )

            hidden_states_batch = outputs.hidden_states[-1][:, -1, :]
            hidden_states_batch = hidden_states_batch.view(
                hidden_states_batch.shape[0], -1
            )
            
            vector_from_start_to_here_batch = hidden_states_batch - hidden_states_start_point
            
            projected_distance_from_start_batch = torch.sum(
                vector_from_start_to_here_batch * probing_direction, dim=1
            )
            loss = -projected_distance_from_start_batch
            losses_batch.append(loss)

    losses_batch = torch.cat(losses_batch, dim=0)
    return losses_batch

def jailbreak_this_prompt(
    prompt: str,
    merged_model,
    tokenizer,
    model_judge,
    tokenizer_judge,
    init_adv_string="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    max_epochs: int = 512,
    sampling_number: int = 512,
    tqdm_desc: str = "Jailbreaking",
    prompt_idx: int = -1,
):
    prompt_start_time = time.time()
    merged_model.merged_model.eval()

    not_allowed_tokens = get_not_allowed_tokens(tokenizer).to(merged_model.merged_model.device)
    hidden_states_start_point, adv_tokenids, full_before_adv_ids, full_after_adv_ids = merged_model.get_init_states(prompt, init_adv_string)

    # Attack loop
    for epoch in tqdm(range(max_epochs), desc=tqdm_desc):
        grad, loss, full_before_adv_embed, full_after_adv_embed = merged_model.compute_grad(adv_tokenids, hidden_states_start_point, full_before_adv_ids, full_after_adv_ids)

        # Sample candidates
        sampled_tokenids = sample_control(
            adv_tokenids.squeeze(0),
            grad,
            search_width=sampling_number,
            topk=256,
            temp=1,
            not_allowed_tokens=not_allowed_tokens,
        )
        try:
            sampled_tokenids = filter_candidates(sampled_tokenids, tokenizer)
        except ValueError as e:
            # No candidates left. Attack cannot proceed anymore.
            print(f"FAILED: {e}")
            result = {
                "idx": prompt_idx,
                "prompt": repr(prompt),
                "adv_string": repr(init_adv_string),
                "response": "FAILED",
                "jailbroken": False,
                "epoch": -1,
                "loss": -1,
                "attack_time": "00:00:00",
            }
            return result

        # Forward pass #2 (not requires grad): Calculate candidates loss
        merged_embed_layer = merged_model.merged_model.get_input_embeddings()
        sampled_embeds = merged_embed_layer(sampled_tokenids)
        full_embed = torch.cat(
            [
                full_before_adv_embed.repeat(sampled_tokenids.shape[0], 1, 1),
                sampled_embeds,
                full_after_adv_embed.repeat(sampled_tokenids.shape[0], 1, 1),
            ],
            dim=1,
        )

        losses_batch = second_forward(
            merged_model.merged_model,
            full_embed,
            hidden_states_start_point,
            merged_model.probing_direction
        )

        adv_tokenids = sampled_tokenids[losses_batch.argmin()]
        current_loss = losses_batch.min().item()
        adv_string = tokenizer.decode(adv_tokenids)

        # Check if the adversarial example is successful
        is_jailbroken, response = check_jailbreak_success(
            model=merged_model.merged_model,
            tokenizer=tokenizer,
            prompt=f"{prompt} {adv_string}",
            generate_length=32,
            return_response=True,
        )
        ##### EPOCH LOG #####
        print()
        # print(f"==>> loss_batch.shape: {losses_batch.shape}")
        print(f"==>> loss: {current_loss}")
        print(f"==>> adv_string: {repr(adv_string)}")
        print(f"==>> is_jailbroken: {is_jailbroken}")
        print(f"==>> response: {repr(response)}")
        sys.stdout.flush()
        
        ##### EPOCH LOG #####
        if is_jailbroken:
            print(f"==>> Jailbreak successful at epoch {epoch}. Double-checking...")
            # Double-check
            _, response = check_jailbreak_success(
                model=merged_model.merged_model,
                tokenizer=tokenizer,
                prompt=f"{prompt} {adv_string}",
                generate_length=512,
                return_response=True,
            )
            rating, _ = ask_judge_harmbench(
                model_judge,
                tokenizer_judge,
                prompt,
                response,
            )
            is_jailbroken = rating == 1
            if is_jailbroken:
                print(f"==>> Jailbreak indeed successful at epoch {epoch}")
                print(f"==>> Updating Merge Model")
                merged_model.train_merged_param(prompt, adv_string, 0.01, 10)


                break

    eval_responses = []
    eval_jailbroken = [] 

    prompt_end_time = time.time()
    prompt_time = prompt_end_time - prompt_start_time
    prompt_time_strf = time.strftime("%H:%M:%S", time.gmtime(prompt_time))

    result = {
        "idx": prompt_idx,
        "prompt": prompt,
        "adv_string": adv_string,
        "response": response,
        "jailbroken": is_jailbroken,
        "epoch": epoch,
        "loss": current_loss,
        "attack_time": prompt_time_strf,
        "eval_responses": eval_responses,
        "eval_jailbroken": eval_jailbroken,
    }

    ##### ATTACK LOG #####
    print("Prompt Result".center(50, "-"))
    print(f"==>> Time: {prompt_time_strf}")
    print(f"==>> prompt: {repr(prompt)}")
    print(f"==>> adv_string: {repr(adv_string)}")
    print(f"==>> response: {repr(response)}")
    print(f"==>> is_jailbroken: {is_jailbroken}")
    print(f"==>> epoch: {epoch}")
    print(f"==>> loss: {loss.item()}")
    print("Prompt Result".center(50, "-"))
    sys.stdout.flush()

    return result

def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")
    set_seed(args.seed)

    merge_task_list = args.merge_tasks.split('_')

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
    model, tokenizer = load_model_and_tokenizer(f".../models/{model_dict[args.model_name]}", False, "cpu")
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
        task_path = f".../finetuned_models/{task_name}/{args.model_name}/inst_epoch1_safe0"
        task_model = load_model_and_tokenizer(task_path, True, "cpu")
        task_model.eval()
        task_model.requires_grad_(False)
        merge_model_list.append(task_model)

    if args.merge_method == 'task_arithmetic':
        merged_model = ModelMergingTA(args.model_name, merge_model_list, tokenizer, dataset_probe_benign, dataset_probe_harmful, 'tanh', args.seed)
    merged_model.update_merged_model()

    # Clear Task Models
    del merge_model_list
    gc.collect()

    create_dir(f'{base_path}/results/baj_gcg/{args.dataset}/{args.model_name}/{args.merge_method}')
    result_filename = f'{base_path}/results/baj_gcg/{args.dataset}/{args.model_name}/{args.merge_method}/{args.merge_tasks}.json'

    # Run the jailbreak attack
    results = []
    for idx, (goal, target) in tqdm(enumerate(dataset, start=lower_bound), total=len(dataset)):
        prompt_result = jailbreak_this_prompt(
            prompt=goal,
            merged_model=merged_model,
            tokenizer=tokenizer,
            init_adv_string=args.init_adv_string,
            max_epochs=args.max_epochs,
            sampling_number=args.sampling_number,
            model_judge=model_judge,
            tokenizer_judge=tokenizer_judge,
            tqdm_desc=f"Jailbreaking {idx}/{lower_bound}-{upper_bound}",
            prompt_idx=idx,
        )
        results.append(prompt_result)

        with open(result_filename, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
