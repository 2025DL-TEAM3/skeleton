import traceback, random

import numpy as np
import torch
import torch.nn.functional as F

from typing import Callable, List
from transformers import PreTrainedTokenizer, PreTrainedModel, GenerationConfig
from peft import PeftModel

from . import data_augmentation, arc_utils
from .datatypes import *
from torch.nn.utils.rnn import pad_sequence

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
        self.model.generation_config = generation_config
        self.parse_grid_fn = parse_grid_fn
        self.device = model.device
        self.fmt_opts = fmt_opts

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
        input_start = self.fmt_opts.get("input_start", "")
        input_end = self.fmt_opts.get("input_end", "")
        output_end = self.fmt_opts.get("output_end", "")
        preprompt = self.fmt_opts.get("preprompt", "")
        prompt_messages = [arc_utils.format_prompt_messages(dp, self.tokenizer, input_start, input_end, output_end, preprompt) for dp in batch_datapoints]

        model_inputs = self.tokenizer(
            text=prompt_messages,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            padding_side="left",
        ).to(self.device)

        prompt_lens = [len(ids) for ids in model_inputs["input_ids"]]

        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=return_logits,
                output_scores=return_logits,
                use_cache=True,
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
    
    def _vote_select_with_log_probs(
        self,
        candidates: List[tuple[List[int], torch.Tensor]],
        params_maps: List[dict],
    ) -> Grid:
        grid_info = {} # { (grid_tuple): (count, log_prob_sum, grid)}
        
        for (ids, logits), params_map in zip(candidates, params_maps):
            log_probs = F.log_softmax(logits, dim=-1) # (gen_len, vocab_size)
            token_ids = torch.tensor(ids).unsqueeze(-1) # (gen_len, 1)
            token_log_probs = log_probs.gather(1, token_ids).squeeze(-1)  # (gen_len,)
            score = token_log_probs.sum().item()
            
            try:
                parsed_grid = self.parse_grid(ids)
                grid_aug = np.array(parsed_grid)
            except Exception as e:
                print("Grid parsing failed. Excluding this candidate.")
                continue
            
            grid_orig = data_augmentation.reverse_grid_augmentation(grid_aug, params_map)
            grid_tuple = tuple(map(tuple, grid_orig))
            
            if grid_tuple in grid_info:
                count, best_score, _ = grid_info[grid_tuple]
                grid_info[grid_tuple] = (count + 1, max(best_score, score), grid_orig)
            else:
                grid_info[grid_tuple] = (1, score, grid_orig)
        
        if not grid_info:
            print("No valid grids found. Returning random grid.")
            x, y = self._infer_test_shape(candidates[0][0])
            return np.random.randint(0, 10, (x, y))
        
        max_votes = max(count for count, _, _ in grid_info.values())
        top_candidates = [(count, score, grid) for grid_tuple, (count, score, grid) in grid_info.items() if count == max_votes]
        
        if len(top_candidates) == 1:
            # if there's a single top candidate, return it
            return top_candidates[0][2]
        else:
            # if there are multiple candidates with the same max votes, select the one with the highest score
            _, best_score, best_grid = max(top_candidates, key=lambda x: x[1])
            return best_grid
    
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
        grid_select_policy: str = "naive",
    ) -> Grid:
        augmented = self._augment_datapoint(base_datapoint, num_augmentations)
        datapoints, params_maps = zip(*augmented)
        datapoints, params_maps = list(datapoints), list(params_maps)

        if grid_select_policy == "naive":
            candidate_ids: List[List[int]] = self._generate(datapoints, return_logits=False)
            grids = []
            for ids in candidate_ids:
                try:
                    parsed_grid = self.parse_grid(ids)
                    grid_aug = np.array(parsed_grid)
                    grids.append(grid_aug)
                except Exception as e:
                    print("/// Parsing grid failed. Excluding this candidate.")
                    print(parsed_grid)
                    continue
            if not grids:
                print("No valid grids found. Returning random grid.")
                x, y = self._infer_test_shape(base_datapoint)
                return np.random.randint(0, 10, (x, y))
            return self._vote_select(self._reverse_grids(grids, params_maps))
        else:
            candidate_ids_logits: List[tuple[List[int], torch.Tensor]] = self._generate(datapoints, return_logits=True)

            if grid_select_policy == "grid-wise":
                selected_grid = self._grid_wise_select(candidate_ids_logits, params_maps)
            elif grid_select_policy == "cell-wise-argmax":
                selected_grid = self._cell_wise_argmax_select(candidate_ids_logits, params_maps)
            elif grid_select_policy == "voted-gridwise":
                selected_grid = self._vote_select_with_log_probs(candidate_ids_logits, params_maps)
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

    def _dfs_generate(
        self,
        datapoint: DataPointDict,
        indices: List[int],
        output_ids: List[int],
    ) -> List[List[int]]:
        # 1) build the one‐and‐only prompt IDs
        start, end, out_marker, pre = (
            self.fmt_opts["input_start"],
            self.fmt_opts["input_end"],
            self.fmt_opts["output_end"],
            self.fmt_opts["preprompt"],
        )
        prompt = arc_utils.format_prompt_messages(
            datapoint, self.tokenizer, start, end, out_marker, pre
        )
        base_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        pad_id = self.tokenizer.pad_token_id

        # 2) for each split‐point, prepend the prefix
        seq_tensors = []
        for idx in indices:
            seq = torch.tensor(base_ids + output_ids[:idx], device=self.device)
            seq_tensors.append(seq)

        padded = pad_sequence(
            seq_tensors,
            batch_first=True,
            padding_value=pad_id,
            padding_side="left",
        )
        padded_len = padded.size(1)
        attention_mask = (padded != pad_id).long()

        # 3) generate all branches in one batch
        with torch.no_grad():
            outs = self.model.generate(
                input_ids=padded,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                use_cache=True,
            )

        # 4) get the output_ids[:idx] + newly generated tokens
        results = []
        for i, idx in enumerate(indices):
            new_tokens = outs[i][padded_len:].cpu().tolist()
            completed_output_ids = output_ids[:idx] + new_tokens
            results.append(completed_output_ids)

        return results

    def apply_grid_augmentation(self, grid: Grid, params_map: dict) -> Grid:
        return data_augmentation.grid_augmentation(grid, params_map, list(params_map.keys()))

    def _dfs_select_highest_confidence(
        self,
        datapoint: DataPointDict,
        batch_output_ids: List[List[int]],
    ) -> int:
        scores = []
        input_start = self.fmt_opts.get("input_start", "")
        input_end   = self.fmt_opts.get("input_end", "")
        output_end  = self.fmt_opts.get("output_end", "")
        preprompt   = self.fmt_opts.get("preprompt", "")

        # for each candidate sequence
        for output_ids in batch_output_ids:
            aug_scores = []

            # parse the original grid
            orig_grid = self.parse_grid(output_ids)

            for aug_dp, params in self._augment_datapoint(datapoint, num_augmentations=4):
                # apply grid augmentation
                aug_grid = self.apply_grid_augmentation(orig_grid, params)

                # re-tokenize
                aug_grid_str = arc_utils.stringify_grid_output(aug_grid, end=output_end)
                aug_ids = self.tokenizer.encode(aug_grid_str, add_special_tokens=False)
                cand_tensor = torch.tensor(aug_ids, device=self.device).unsqueeze(0)

                # build prompt for this aug_dp
                prompt = arc_utils.format_prompt_messages(
                    aug_dp, self.tokenizer, input_start, input_end, output_end, preprompt
                )
                toks = self.tokenizer(
                    [prompt], add_special_tokens=False,
                    return_tensors="pt", padding=True, truncation=False, # no truncation to get full scores
                ).to(self.device)

                # append the candidate
                prompt_len = toks["input_ids"].size(1)
                toks["input_ids"] = torch.cat([toks["input_ids"], cand_tensor], dim=1)
                toks["attention_mask"] = torch.cat([
                    toks["attention_mask"],
                    torch.ones_like(cand_tensor)
                ], dim=1)

                # single forward for scoring
                with torch.no_grad():
                    out = self.model(
                        input_ids      = toks["input_ids"],
                        attention_mask = toks["attention_mask"],
                        return_dict    = True,
                    )
                logits = out.logits[0]  # (total_len, vocab_size)
                scores_tensor = logits[prompt_len : prompt_len + len(aug_ids)]

                # sum log-probs
                logp       = F.log_softmax(scores_tensor, dim=-1)
                ids_t      = cand_tensor.squeeze(0).unsqueeze(-1)
                token_logp = logp.gather(1, ids_t).squeeze(-1)
                aug_scores.append(token_logp.sum().item())

            # average across the 4 augmentations
            scores.append(sum(aug_scores) / len(aug_scores))

        # pick the candidate with highest average score
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    def predict_dfs(
        self,
        base_datapoint: DataPointDict,
    ) -> Grid:
        output_ids, logits = self._generate([base_datapoint], return_logits=True)[0]
        log_probs = F.log_softmax(logits, dim=-1) # (gen_len, vocab_size)
        token_log_probs = log_probs.gather(1, torch.tensor(output_ids).unsqueeze(-1)).squeeze(-1) # (gen_len, )

        indicies = []
        for i in range(len(output_ids)):
            if torch.exp(token_log_probs[i]) < 0.9:
                indicies.append(i)
                if len(indicies) >= 8:
                    break

        # select the highest confidence grid
        if len(indicies) > 0:
            batch_output_ids = [output_ids] + self._dfs_generate(base_datapoint, indicies, output_ids)
            best_idx = self._dfs_select_highest_confidence(base_datapoint, batch_output_ids)
            output_ids = batch_output_ids[best_idx]

        try:
            parsed_grid = self.parse_grid(output_ids)
            return np.array(parsed_grid)
        except Exception as e:
            print("Grid parsing failed. Returning random grid.")
            traceback.print_exc()
            x, y = self._infer_test_shape(base_datapoint)
            return np.random.randint(0, 10, (x, y))