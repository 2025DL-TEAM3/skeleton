import traceback, random

import numpy as np
import torch
import torch.nn.functional as F

from typing import Callable, List
from transformers import PreTrainedTokenizer, PreTrainedModel, GenerationConfig
from peft import PeftModel

from . import data_augmentation, arc_utils
from .datatypes import *

def _infer_test_shape(datapoint: DataPointDict) -> tuple[int, int]:
    train_input = np.array(datapoint['train'][0]['input'])
    train_output = np.array(datapoint['train'][0]['output'])
    test_input = np.array(datapoint['test'][0]['input'])
    
    if train_input.shape == train_output.shape:
        x, y = test_input.shape
    else:
        x = (train_output.shape[0] * test_input.shape[0] // train_input.shape[0])
        y = (train_output.shape[1] * test_input.shape[1] // train_input.shape[1])
    
    return x, y

def _generate_grids_batch(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    datapoints: List[DataPointDict],
    generation_config: GenerationConfig,
    parse_grid_fn: Callable[[List[int]], Grid],
) -> List[Grid]:
    prompt_messages = [arc_utils.format_prompt_messages(dp) for dp in datapoints]
    prompt_strs = [
        tokenizer.apply_chat_template(
            prompt_msg,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=False,
        )
        for prompt_msg in prompt_messages
    ]
    
    model_inputs = tokenizer(
        text=prompt_strs,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True, 
    ).to(model.device)

    prompt_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).long().sum(dim=1)

    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            generation_config=generation_config,
        ) # (batch_size, total_seq_len)

    grids = []
    for seq, prompt_len, datapoint in zip(output, prompt_lens, datapoints):
        new_ids = seq[prompt_len:].cpu().tolist()
        try:
            parsed_grid = parse_grid_fn(new_ids)
            grids.append(np.array(parsed_grid))
        except Exception as e:
            print(f"Error parsing grid, using random grid")
            print("Parsed grid:")
            arc_utils.print_grid(parsed_grid)
            traceback.print_exc()
            x, y = _infer_test_shape(datapoint)
            grids.append(np.random.randint(0, 10, (x, y)))
    return grids

def predict_single(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    parse_grid_fn: Callable[[List[int]], Grid],
    base_datapoint: DataPointDict,
) -> Grid:
    return _generate_grids_batch(
        model, tokenizer, [base_datapoint], generation_config, parse_grid_fn
    )


def predict_naive_v1(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    parse_grid_fn: Callable[[List[int]], Grid],
    augmented_datapoints_and_params_map: List[tuple[DataPointDict, dict]],
    batch_size_generation: int,
) -> Grid:
    """
    Naive policy
    1. Augment the test grid
    2. Generate output grids for each augmented test grid
    3. Reverse augment the generated grids
    4. Select by voting
    """

    inferred_grids = []
    batch_size = batch_size_generation or 1
    for batch in arc_utils.chunked(augmented_datapoints_and_params_map, batch_size):
        datapoints, params_maps = zip(*batch) # list[tuple[DataPointDict, dict]]
        inferred = _generate_grids_batch(
            model, tokenizer, datapoints, generation_config, parse_grid_fn
        )
        inferred_grids.extend(
            zip(datapoints, params_maps, inferred)
        )
    
    for dp, _, grid in inferred_grids:
        dp["test"][0]["output"] = grid

    reversed_datapoints = [
        data_augmentation.revesre_datapoint_augmentation(dp, params_map)
        for dp, params_map, _ in inferred_grids
    ]
    
    final_grid = _select_best_grid_by_voting(reversed_datapoints)
    return final_grid

def _select_best_grid_by_voting(datapoint_candidates: List[DataPointDict]) -> Grid:
        # vote for the most common grid
        inferred_grids = [
            datapoint["test"][0]["output"]
            for datapoint in datapoint_candidates
        ]

        grid_counts = {}
        for grid in inferred_grids:
            grid_tuple = tuple(map(tuple, grid))
            if grid_tuple in grid_counts:
                grid_counts[grid_tuple] += 1
            else:
                grid_counts[grid_tuple] = 1
        
        # if all grides are distinct, return a random one
        if len(grid_counts) == len(inferred_grids):
            print("All grids are distinct, returning a random one.")
            return random.choice(inferred_grids)
    
        # find the grid with the highest count
        best_grid = max(grid_counts, key=grid_counts.get)
        return best_grid

def select_best_grid_from_logits_grid_wise_selection(
    candidates: List[tuple[DataPointDict, dict, List[int], torch.FloatTensor]],
    parse_grid_fn: Callable[[List[int]], Grid],
) -> Grid:
    best_score = float("-inf")
    best_grid  = None

    for dp, params, ids, logits in candidates:
        # compute per‐token log‐probs
        log_probs = F.log_softmax(logits, dim=-1) # (gen_len, vocab_size)
        token_ids = torch.tensor(ids).unsqueeze(-1) # (gen_len, 1), sampled token ids
        
        # gather log‐probs of the actually generated tokens
        # Note: score computes a confidence [log p(augmented testgrid | augmented train grid)]
        token_log_probs = log_probs.gather(1, token_ids).squeeze(-1)  # (gen_len,)
        score = token_log_probs.sum().item()

        # parse and reverse‐augment
        try:
            grid_aug = parse_grid_fn(ids)
            grid_np = np.array(grid_aug)
        except Exception as e:
            print("Grid parding failed. Excluding this candidate.")
            continue
    
        dp["test"][0]["output"] = grid_np
        grid = data_augmentation.reverse_grid_augmentation(
            dp, params
        )

        if score > best_score:
            best_score = score
            best_grid  = grid

    return best_grid

def select_best_grid_from_logits_cell_wise_argmax(
    candidates: List[tuple[DataPointDict, dict, List[int], torch.FloatTensor]],
    parse_grid_fn: Callable[[List[int]], Grid],
    tokenizer: PreTrainedTokenizer,
) -> Grid:
    reversed_grids = []
    revsered_log_probs = []
    
    for dp, params, ids, logits in candidates:
        log_probs = F.log_softmax(logits, dim=-1) # (gen_len, vocab_size)
        
        digit_ids, digit_log_probs = [], []
        for token_id, log_prob in zip(ids, log_probs):
            token = tokenizer.decode([token_id], skip_special_tokens=True)
            if len(token) == 1 and token.isdigit():
                digit_ids.append(int(token_id))
                digit_log_probs.append(log_prob[token_id].item())
        
        try:
            grid_aug = np.array(parse_grid_fn(ids))
            h_aug, w_aug = grid_aug.shape
        except Exception as e:
            print("Grid parding failed. Excluding this candidate.")
            continue
        
        # grid_aug = np.array(grid_np, dtype=int).reshape(h_aug, w_aug)
        log_prob_aug = np.array(digit_log_probs, dtype=float).reshape(h_aug, w_aug)
        
        # reverse‐augment
        reversed_grid = data_augmentation.reverse_grid_augmentation(
            grid_aug, params
        )
        reversed_log_prob = data_augmentation.reverse_grid_augmentation(
            log_prob_aug, params_map=params, skip_names=["color"] # skip color, since float grids
        )
        
        reversed_grids.append(reversed_grid)
        revsered_log_probs.append(reversed_log_prob)
    
    if not reversed_grids:
        print("No valid grids found. Returning random grid.")
        x, y = _infer_test_shape(candidates[0][0])
        return np.random.randint(0, 10, (x, y))

    # TODO: what if candidates have different shapes?
    grids_arr = np.stack(reversed_grids, axis=0) # (num_candidates, h_orig, w_orig), int array
    log_prob_maps_arr = np.stack(revsered_log_probs, axis=0) # (num_candidates, h_orig, w_orig), float array
    
    num_candidates, h_orig, w_orig = grids_arr.shape
    final_grid = np.zeros((h_orig, w_orig), dtype=grids_arr.dtype)
    
    winners = np.argmax(log_prob_maps_arr, axis=0) # (h_orig, w_orig), values in [0, num_candidates)
    for i in range(h_orig):
        for j in range(w_orig):
            final_grid[i, j] = grids_arr[winners[i, j], i, j]
    return final_grid

def predict_logits(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    parse_grid_fn: Callable[[List[int]], Grid],
    augmented_datapoints_and_params_map: List[tuple[DataPointDict, dict]],
    batch_size_generation: int,
    grid_select_policy: str,
) -> Grid:
    """
    Utilize logits
    1. Augment the test grid
    2. Generate output ids and logits for each augmented test grids
    3. Select the best grid based on configured grid_select_policy
    """

    all_candidates = []
    batch_size = batch_size_generation or 1
    for batch in arc_utils.chunked(augmented_datapoints_and_params_map, batch_size):
        # TODO how can i reverse the augmentation?
        datapoints, params_maps = zip(*batch) # list[tuple[DataPointDict, dict]]
        out_batch = _generate_logits_batch(
            model, tokenizer, list(datapoints), generation_config,
        )
        # out_batch: list[(ids, logits)]
        for (output_ids, logits), datapoint, params_map in zip(out_batch, datapoints, params_maps):
            all_candidates.append((datapoint, params_map, output_ids, logits))

    final_grid = _select_best_grid_from_logits(
        all_candidates,
        grid_select_policy,
        parse_grid_fn,
        tokenizer,
    )
    return final_grid

def _generate_logits_batch(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    datapoints: List[DataPointDict],
    generation_config: GenerationConfig,
) -> List[tuple[List[int], torch.FloatTensor]]:
    prompt_messages = [arc_utils.format_prompt_messages(dp) for dp in datapoints]
    prompt_strs = [
        tokenizer.apply_chat_template(
            prompt_msg,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=False,
        )
        for prompt_msg in prompt_messages
    ]

    model_inputs = tokenizer(
        text=prompt_strs,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True, 
    ).to(model.device)

    prompt_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).long().sum(dim=1) # (batch_size,)

    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    # output.sequences: (batch_size, total_seq_len)
    # output.scores: Tuples of (batch_size, vocab_size), one per generation step
    # scores: (batch_size, gen_len, vocab_size)
    scores = torch.stack(output.scores, dim=1)

    results = []
    for i, seq in enumerate(output.sequences):
        prompt_len = prompt_lens[i]
        output_ids = seq[prompt_len:].cpu().tolist()
        logits = scores[i].cpu() # (gen_len, vocab_size)
        results.append((output_ids, logits))
    return results

def _select_best_grid_from_logits(
    candidates: List[tuple[DataPointDict, dict, List[int], torch.FloatTensor]],
    grid_select_policy: str,
    parse_grid_fn: Callable[[List[int]], Grid],
    tokenizer: PreTrainedTokenizer,
) -> Grid:
    if grid_select_policy == "grid-wise":
        return select_best_grid_from_logits_grid_wise_selection(
            candidates, parse_grid_fn
        )
    elif grid_select_policy == "cell-wise-argmax":
        return select_best_grid_from_logits_cell_wise_argmax(
            candidates, parse_grid_fn, tokenizer
        )
    else:
        raise ValueError(f"Unknown grid selection policy: {grid_select_policy}")
    
def route_predict(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    parse_grid_fn: Callable[[List[int]], Grid],
    base_datapoint: DataPointDict,
    num_augmentations: int,
    batch_size_generation: int,
    grid_select_policy: str,
) -> Grid:
    
    augmented_datapoints_and_params_map = [
        data_augmentation.random_datapoint_augmentation(base_datapoint, swap_train_and_test=False)
        for _ in range(num_augmentations)
    ]

    if grid_select_policy == "naive":
        return predict_naive_v1(
            model, tokenizer, generation_config, parse_grid_fn, augmented_datapoints_and_params_map, batch_size_generation
        )
    else:
        return predict_logits(
            model, tokenizer, generation_config, parse_grid_fn, augmented_datapoints_and_params_map, batch_size_generation, grid_select_policy
        )