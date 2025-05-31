import traceback, random

import numpy as np
import torch
import torch.nn.functional as F

from typing import Callable, List
from transformers import PreTrainedTokenizer, PreTrainedModel, GenerationConfig
from peft import PeftModel

from . import data_augmentation, arc_utils
from .datatypes import *

class ARCInferencer:
    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig,
        parse_grid_fn: Callable[[List[int]], Grid],
        fmt_opts: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.model.generation_config = self.generation_config
        self.parse_grid_fn = parse_grid_fn
        self.device = model.device
        self.fmt_opts = fmt_opts
        self.allowed_tokens = self._get_allowed_tokens()

    def _get_allowed_tokens(self):
        digit_tokens = []
        for i in range(10):
            token = self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            digit_tokens.append(token)
            
        newline_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        
        allowed_tokens = set(digit_tokens + [newline_token, self.tokenizer.eos_token_id])
        return allowed_tokens

    def parse_grid(self, ids: List[int]) -> Grid:
        return self.parse_grid_fn(ids)

    def _infer_test_shape(self, datapoint: DataPointDict) -> tuple[int, int]:
        train_input = np.array(datapoint['train'][0]['input'])
        train_output = np.array(datapoint['train'][0]['output'])
        test_input = np.array(datapoint['test'][0]['input'])
        
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] * test_input.shape[0] // train_input.shape[0])
            y = (train_output.shape[1] * test_input.shape[1] // train_input.shape[1])
        
        return x, y
    
    def _augment_datapoint(
        self, 
        base_datapoint: DataPointDict,
        num_augmentations: int = 1,
    ) -> List[tuple[DataPointDict, dict]]:
        return [
            data_augmentation.random_datapoint_augmentation(base_datapoint, swap_train_and_test=False)
            for _ in range(num_augmentations)
        ]
    
    def _reverse_grids(self, grids: List[Grid], params_maps: List[dict]) -> List[Grid]:
        reversed_grids = []
        for grid, params_map in zip(grids, params_maps):
            reversed_grid = data_augmentation.reverse_grid_augmentation(grid, params_map)
            reversed_grids.append(reversed_grid)
        return reversed_grids

    def print_logits(self, output, prompt_lens, scores, batch_size, num_return_sequences):
        allowed_ids = sorted(self.allowed_tokens)
        RED = "\033[91m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        print(len(output.sequences))
        print(len(output.sequences_scores))
        print(batch_size)
        

        for i in range(batch_size):
            prompt_len = prompt_lens[i]
            for j in range(num_return_sequences):
                seq = output.sequences[i * num_return_sequences + j]
                output_ids = seq[prompt_len:].cpu().tolist()
                logits = scores[i * num_return_sequences + j]
                probs = F.softmax(logits, dim=-1)    # [gen_len, vocab_size]
            
                print(f"\n=== 샘플 {i} === {j+1}번째 output sequnce scores:", output.sequences_scores[i * num_return_sequences + j])
                for token_idx, step_probs in enumerate(probs):
                    token_id = output_ids[token_idx]
                    decoded_token = self.tokenizer.decode([token_id])
                    decoded_token = decoded_token if decoded_token != "\n" else "sep"
                    print(f"{BLUE} Token {token_idx+1}:{RESET} {decoded_token}")

                    # allowed token별로 출력
                    for allowed_id in allowed_ids:
                        token_str = self.tokenizer.decode([allowed_id])
                        p = step_probs[allowed_id].item()
                        color = RED if token_id == allowed_id else ""
                        print(f"   {color}{repr(token_str)} (id={allowed_id}): {p:.4f}{RESET}", end=" ")
                    print()

    def _generate(
        self, 
        batch_datapoints: List[DataPointDict], 
        return_logits: bool = False,
    ) -> List[List[int]] | List[tuple[List[int], torch.Tensor]]:
        input_start = self.fmt_opts.get("input_start", "")
        input_end = self.fmt_opts.get("input_end", "")
        output_end = self.fmt_opts.get("output_end", "")
        preprompt = self.fmt_opts.get("preprompt", "")

        batch_size = len(batch_datapoints)
        num_return_sequences = 1  # Number of sequences to return per input
        
        prompt_messages = [arc_utils.format_prompt_messages(dp, self.tokenizer, input_start, input_end, output_end, preprompt) for dp in batch_datapoints]

        model_inputs = self.tokenizer(
            text=prompt_messages,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        prompt_lens = [len(ids) for ids in model_inputs["input_ids"]]

        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                return_dict_in_generate=return_logits,
                output_scores=return_logits,
                num_beams=num_return_sequences,
                num_return_sequences=num_return_sequences,
            )
    
        if not return_logits:
            # When using num_return_sequences, outputs are reshaped to (batch_size * num_return_sequences, seq_len)
            # Reshape results to get a list of sequences for each datapoint
            result = []
            for i in range(batch_size):
                # Get all sequences for this batch item
                batch_sequences = []
                for j in range(num_return_sequences):
                    idx = i * num_return_sequences + j
                    if idx < len(output):  # Safety check
                        batch_sequences.append(output[idx][prompt_lens[i]:].cpu().tolist())
                result.append(batch_sequences[0])
            return result
    
        # output.sequences: (batch_size * num_return_sequences, total_seq_len)
        # output.scores: Tuples of (batch_size * num_return_sequences, vocab_size), one per generation step
        # scores: (batch_size * num_return_sequences, gen_len, vocab_size)
        scores = torch.stack(output.scores, dim=1)

        self.print_logits(output, prompt_lens, scores, batch_size, num_return_sequences)
    
        # Reshape results for return_logits=True case
        result = []
        for i in range(batch_size):
            batch_sequences = []
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j
                if idx < len(output.sequences):  # Safety check
                    output_ids = output.sequences[idx][prompt_lens[i]:].cpu().tolist()
                    try:
                        parsed_grid = self.parse_grid(output_ids)
                        grid_aug = np.array(parsed_grid)
                        candidate = np.array(grid_aug, dtype=np.uint8)
                        print("cand     :", candidate.tolist())
                    except:
                        continue
                    
                    batch_sequences.append(output_ids)
            result.append(batch_sequences[0])
        return result
    
    def _vote_select(
        self, 
        grid_candidates: List[Grid]
    ) -> Grid:
        grid_counts = {}
        for grid in grid_candidates:
            grid_tuple = tuple(map(tuple, grid))
            grid_counts[grid_tuple] = grid_counts.get(grid_tuple, 0) + 1
        
        if len(grid_counts) == len(grid_candidates):
            return random.choice(grid_candidates)
    
        best = max(grid_counts, key=grid_counts.get)
        return best
    
    def _grid_wise_select(
        self,
        candidates: List[tuple[List[int], torch.Tensor]],
        params_maps: List[dict], 
    ) -> Grid:
        best_score, best_grid = float("-inf"), None

        for (ids, logits), params_map in zip(candidates, params_maps):
            # compute per‐token log‐probs
            log_probs = F.log_softmax(logits, dim=-1) # (gen_len, vocab_size)
            token_ids = torch.tensor(ids).unsqueeze(-1) # (gen_len, 1), sampled token ids

            # gather log‐probs of the actually generated tokens
            # Note: score computes a confidence [log p(augmented testgrid | augmented train grid)]
            token_log_probs = log_probs.gather(1, token_ids).squeeze(-1)  # (gen_len,)
            score = token_log_probs.sum().item()

            try:
                parsed_grid = self.parse_grid(ids)
                grid_aug = np.array(parsed_grid)
            except Exception as e:
                print("Grid parding failed. Excluding this candidate.")
                continue

            grid_orig = data_augmentation.reverse_grid_augmentation(grid_aug, params_map)
            if score > best_score:
                best_score = score
                best_grid = grid_orig
        return best_grid
    
    def _cell_wise_argmax_select(
        self,
        candidates: List[tuple[List[int], torch.Tensor]],
        params_maps: List[dict],
    ) -> Grid:
        grid_origs, log_prob_origs = [], []
        for (ids, logits), params_map in zip(candidates, params_maps):
            log_probs = F.log_softmax(logits, dim=-1) # (gen_len, vocab_size)

            digit_ids, digit_log_probs = [], []
            for token_id, log_prob in zip(ids, log_probs):
                token = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if len(token) == 1 and token.isdigit():
                    digit_ids.append(int(token_id))
                    digit_log_probs.append(log_prob[token_id].item())
            
            try:
                parsed_grid = self.parse_grid(digit_ids)
                grid_aug = np.array(parsed_grid)
                h_aug, w_aug = grid_aug.shape
            except Exception as e:
                print("Grid parsing failed. Excluding this candidate.")
                continue
        
            log_prob_aug = np.array(digit_log_probs, dtype=float).reshape(h_aug, w_aug)

            grid_orig = data_augmentation.reverse_grid_augmentation(grid_aug, params_map)
            log_prob_orig = data_augmentation.reverse_grid_augmentation(
                log_prob_aug, params_map, skip_names=["color"] # skip color, since float grids
            )

            grid_origs.append(grid_orig)
            log_prob_origs.append(log_prob_orig)
        
        if not grid_origs:
            print("No valid grids found. Returning random grid.")
            x, y = self._infer_test_shape(candidates[0][0]) # TODO: unmatch type. will raise error
            return np.random.randint(0, 10, (x, y))
    
        # TODO: what if candidates have different shapes?
        grids_arr = np.stack(grid_origs, axis=0) # (num_candidates, h_orig, w_orig), int array
        log_prob_maps_arr = np.stack(log_prob_origs, axis=0) # (num_candidates, h_orig, w_orig), float array
    
        num_candidates, h_orig, w_orig = grids_arr.shape
        final_grid = np.zeros((h_orig, w_orig), dtype=grids_arr.dtype)
        
        winners = np.argmax(log_prob_maps_arr, axis=0) # (h_orig, w_orig), values in [0, num_candidates)
        for i in range(h_orig):
            for j in range(w_orig):
                final_grid[i, j] = grids_arr[winners[i, j], i, j]
        return final_grid
    
    def predict(
        self,
        base_datapoint: DataPointDict,
        num_augmentations: int = 1,
        batch_size_generation: int = 1,
        grid_select_policy: str = "naive",
    ) -> Grid:
        augmented = self._augment_datapoint(base_datapoint, num_augmentations)
        datapoints, params_maps = zip(*augmented)
        datapoints, params_maps = list(datapoints), list(params_maps)

        if grid_select_policy == "naive":
            candidate_ids = []
            for batch in arc_utils.chunked(augmented, batch_size_generation):
                batch_datpoints = [dp for dp, _ in batch]
                batch_output_ids = self._generate(batch_datpoints, return_logits=False)
                candidate_ids.extend(batch_output_ids)
            grids = []
            for ids in candidate_ids:
                try:
                    parsed_grid = self.parse_grid(ids)
                    grid_aug = np.array(parsed_grid)
                    grids.append(grid_aug)
                except Exception as e:
                    continue
            if not grids:
                print("No valid grids found. Returning random grid.")
                x, y = self._infer_test_shape(base_datapoint)
                return np.random.randint(0, 10, (x, y))
            return self._vote_select(self._reverse_grids(grids, params_maps))
        else:
            candidate_ids_logits = []
            for batch in arc_utils.chunked(augmented, batch_size_generation):
                batch_datpoints = [dp for dp, _ in batch]
                batch_output_ids_logits = self._generate(batch_datpoints, return_logits=True)
                candidate_ids_logits.extend(batch_output_ids_logits)

            if grid_select_policy == "grid-wise":
                selected_grid = self._grid_wise_select(candidate_ids_logits, params_maps)
            elif grid_select_policy == "cell-wise-argmax":
                selected_grid = self._cell_wise_argmax_select(candidate_ids_logits, params_maps)
            else:
                raise ValueError(f"Unknown grid select policy: {grid_select_policy}")
        
            return selected_grid
        
    def predict_single(
        self,
        base_datapoint: DataPointDict,
    ) -> Grid:
        output_ids = self._generate([base_datapoint], return_logits=False)[0]
        try:
            parsed_grid = self.parse_grid(output_ids)
            grid_aug = np.array(parsed_grid)
        except Exception as e:
            print("Grid parsing failed. Returning random grid.")
            traceback.print_exc()
            x, y = self._infer_test_shape(base_datapoint)
            return np.random.randint(0, 10, (x, y))
        return grid_aug