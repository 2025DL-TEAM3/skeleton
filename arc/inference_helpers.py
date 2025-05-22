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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.parse_grid_fn = parse_grid_fn
        self.device = model.device

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

    def _generate(
        self, 
        batch_datapoints: List[DataPointDict], 
        return_logits: bool = False
    ) -> List[List[int]] | List[tuple[List[int], torch.Tensor]]:
        prompt_messages = [arc_utils.format_prompt_messages(dp) for dp in batch_datapoints]
        prompt_strs = [
            self.tokenizer.apply_chat_template(
                prompt_msg,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
                enable_thinking=False,
            )
            for prompt_msg in prompt_messages
        ]
        
        model_inputs = self.tokenizer(
            text=prompt_strs,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
        ).to(self.device)

        prompt_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).long().sum(dim=1)

        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=return_logits,
                output_scores=return_logits,
            )
        
        if not return_logits:
            # output: (batch_size, total_seq_len)
            return [
                seq[prompt_lens[i]:].cpu().tolist()
                for i, seq in enumerate(output)
            ]
        
        # output.sequences: (batch_size, total_seq_len)
        # output.scores: Tuples of (batch_size, vocab_size), one per generation step
        # scores: (batch_size, gen_len, vocab_size)
        scores = torch.stack(output.scores, dim=1)

        results = []
        for i, seq in enumerate(output.sequences):
            prompt_len = prompt_lens[i]
            output_ids = seq[prompt_len:].cpu().tolist()
            logits = scores[i].cpu()
            results.append((output_ids, logits))
        return results
    
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
                    print(f"Error parsing grid, using random grid")
                    print("Parsed grid:")
                    arc_utils.print_grid(parsed_grid)
                    traceback.print_exc()
                    x, y = self._infer_test_shape(base_datapoint)
                    grids.append(np.random.randint(0, 10, (x, y)))
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