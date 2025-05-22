import numpy as np
import torch
import torch.nn.functional as F

from typing import Callable, List
from transformers import PreTrainedTokenizer

from . import data_augmentation
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

    grids_arr = np.stack(reversed_grids, axis=0) # (num_candidates, h_orig, w_orig), int array
    log_prob_maps_arr = np.stack(revsered_log_probs, axis=0) # (num_candidates, h_orig, w_orig), float array
    
    num_candidates, h_orig, w_orig = grids_arr.shape
    final_grid = np.zeros((h_orig, w_orig), dtype=grids_arr.dtype)
    
    winners = np.argmax(log_prob_maps_arr, axis=0) # (h_orig, w_orig), values in [0, num_candidates)
    for i in range(h_orig):
        for j in range(w_orig):
            final_grid[i, j] = grids_arr[winners[i, j], i, j]
    return final_grid