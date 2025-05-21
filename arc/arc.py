import os, glob, json, time, random, copy, gc, datetime, traceback
from wasabi import msg
from omegaconf import DictConfig, OmegaConf
import numpy as np

from trl import SFTTrainer, SFTConfig
from transformers import GenerationConfig, TrainingArguments, EarlyStoppingCallback
import torch
import torch.nn.functional as F
from typing import List, Union, Literal, Any, TypedDict, Callable, Optional
from datasets import load_dataset, Dataset as HFDataset
from pprint import pprint

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import LoraConfig, PeftModel

from . import arc_utils, data_transform, data_augmentation
from .datatypes import *

def load_config(config_path: str) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    
    if cfg.artifact_name is  None:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cfg.artifact_name = f"train-{now}"
    train_artifacts_dir = os.path.join(cfg.artifacts_dir, cfg.artifact_name)
    
    os.makedirs(train_artifacts_dir, exist_ok=True)
    cfg.train_artifacts_dir = train_artifacts_dir
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return cfg_dict

class ARCSolver:
    def __init__(
        self, 
        token: str = None,
        config_path: str = "artifacts/config.yaml",
    ):
        cfg = load_config(config_path)
        train_artifacts_dir = cfg.get("train_artifacts_dir", None)
        cache_dir = cfg.get("cache_dir", None)
        model_id = cfg.get("model", {}).get("model_id", "Qwen/Qwen3-4B")
        use_custom_head = cfg.get("model", {}).get("use_custom_head", False)
        lora_rank = cfg.get("model", {}).get("lora_rank", 16)
        lora_alpha = cfg.get("model", {}).get("lora_alpha", 32)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        if train_artifacts_dir is not None:
            self.train_artifacts_dir = train_artifacts_dir
        else:
            self.train_artifacts_dir = os.path.join("artifacts", "train-default")
        self.checkpoint_save_path = os.path.join(self.train_artifacts_dir, "checkpoints")
        self.logging_save_path = os.path.join(self.train_artifacts_dir, "logs")

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        
        self.model_args = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": True,  # Allow the model to use custom code from the repository
            "quantization_config": bnb_config,  # Apply the 4-bit quantization configuration
            "attn_implementation": "sdpa",  # Use scaled-dot product attention for better performance
            "torch_dtype": torch.float16,  # Set the data type for the model
            "use_cache": False,  # Disable caching to save memory
            "token": token,
            "tie_word_embeddings": not use_custom_head,
            # "device_map": "auto",  # Automatically map the model to available devices
        }
        if cache_dir is not None:
            print(f"Using cache dir: {cache_dir}")
            self.model_args["cache_dir"] = cache_dir
            self.cache_dir = cache_dir
        else:
            print("No cache dir found, using default cache location.")
            self.cache_dir = None
        
        # Load tokenizer first so it's available for model optimization
        tokenizer_args = {
            "pretrained_model_name_or_path": model_id,
            "token": token,
        }
        if cache_dir:
            tokenizer_args["cache_dir"] = cache_dir
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
        self.tokenizer.bos_token_id = 151643 # Default for Qwen3
        
        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            **self.model_args,
        ).to(self.device)
        print(f"✓ Model loaded: {self.model_id}")

        # Optimize model vocabulary for ARC tasks if enabled
        if use_custom_head:
            from .custom_head import apply_custom_head
            # Only keep necessary tokens (digits, thinking tokens, special tokens)
            apply_custom_head(self.base_model, self.tokenizer)
            print(f"✓ Model vocabulary optimization applied")
        else:
            print("Model vocabulary optimization skipped.")

        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if use_custom_head:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"]

        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_rank, 
            lora_alpha=lora_alpha,  
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
        self.peft_model = None
        
        self.enable_ttt = False
        
        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,   
        )
        self.batch_size_generation = 1
        
        
    def parse_grid(self, ids: List[int]) -> Grid:
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        grid = arc_utils.gridify_grid(decoded)
        return grid

    def train(
        self, 
        *,
        train_dataset: HFDataset,
        eval_dataset: HFDataset= None,
        **train_args_dict,
    ):
        """
        Train a model with train_dataset.
        """
        os.makedirs(self.checkpoint_save_path, exist_ok=True)
        os.makedirs(self.logging_save_path, exist_ok=True)
        use_data_augmentation = train_args_dict.pop("use_data_augmentation", False)
        patience = train_args_dict.pop("patience", 5)
        callbacks = []
        if eval_dataset is not None and patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=0.0,
                )
            )

        sft_training_args = SFTConfig(
            output_dir=self.checkpoint_save_path,
            logging_dir=self.logging_save_path,
            log_level="debug",
            max_length=None, # avoid truncation
            label_names=["labels"], # TODO: check if needed
            **train_args_dict,
        )
        
        transform = data_transform.get_data_transform(use_data_augmentation)
        transform_name = transform.__class__.__name__
        msg.info(f"Using transform: {transform_name}")  
        
        train_dataset = train_dataset.map(
            transform,
            remove_columns=train_dataset.column_names,
            desc=f"Applying train dataset transform ({transform_name})",
            num_proc=sft_training_args.dataset_num_proc,
        )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                transform,
                remove_columns=eval_dataset.column_names,
                desc=f"Applying eval dataset transform ({transform_name})",
                num_proc=sft_training_args.dataset_num_proc,
            )
        
        start_time = time.time()
        msg.info("Using SFTTrainer for training.")
        trainer = SFTTrainer(
            model=self.base_model if self.peft_model is None else self.peft_model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_training_args,
            peft_config=self.peft_config,
            callbacks=callbacks if callbacks else None,
        )
        
        # preemptively save the model before training, to check error
        trainer.save_model(os.path.join(self.checkpoint_save_path, "checkpoint-initial"))
        trainer.train()
        trainer.save_model(os.path.join(self.checkpoint_save_path, "checkpoint-final"))
        end_time = time.time()
        msg.good(f"Training completed in {end_time - start_time:.2f} seconds.")

    def test_time_training(self, examples: List[ExampleDict]):
        pass

    def predict(
        self, 
        examples: List[ExampleDict], 
        questions_input: Grid
    ) -> Grid:
        if self.enable_ttt:
            self.test_time_training(examples)
        
        base_datapoint = {
            "train": examples,
            "test": [
                {
                    "input": questions_input,
                    "output": None,
                }
            ],
        }
        
        if not self.use_data_augmentation_for_eval or self.num_augmentations < 1:
            (output_ids, logits) = self._predict_logits_batch([base_datapoint])[0]
            return self._select_best_grid_from_logits(
                [(base_datapoint, None, output_ids, logits)]
            )
        
        augmented_datapoints_and_params_map = [
            data_augmentation.random_datapoint_augmentation(base_datapoint, swap_train_and_test=False)
            for _ in range(self.num_augmentations)
        ]
        
        all_candidates = []
        batch_size = self.batch_size_generation or 1
        for batch in arc_utils.chunked(augmented_datapoints_and_params_map, batch_size):
            # TODO how can i reverse the augmentation?
            datapoints, params_maps = zip(*batch) # list[tuple[DataPointDict, dict]]
            out_batch = self._predict_logits_batch(list(datapoints))
            # out_batch: list[(ids, logits)]
            for (output_ids, logits), datapoint, params_map in zip(out_batch, datapoints, params_maps):
                all_candidates.append((datapoint, params_map, output_ids, logits))

        final_grid = self._select_best_grid_from_logits(all_candidates)
        return final_grid
    
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
    
    def _predict_logits_batch(
        self, 
        datapoints: List[DataPointDict]
    ) -> List[tuple[List[int], torch.FloatTensor]]:
        prompt_messages = [arc_utils.format_prompt_messages(dp) for dp in datapoints]
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
        
        prompt_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).long().sum(dim=1) # (batch_size,)
        
        with torch.no_grad():
            output = self.base_model.generate(
                **model_inputs,
                generation_config=self.generation_config,
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
        self, 
        candidates: List[tuple[DataPointDict, dict, List[int], torch.FloatTensor]]
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
                grid_aug = self.parse_grid(ids)
                grid_np = np.array(grid_aug)
            except Exception as e:
                print("Grid parding failed. Excluding this candidate.")
                continue
        
            dp["test"][0]["output"] = grid_np
            grid = data_augmentation.revesre_datapoint_augmentation(
                dp, params
            )["test"][0]["output"]

            if score > best_score:
                best_score = score
                best_grid  = grid

        return best_grid

    
    def _predict_grid_batch(self, datapoints: List[DataPointDict]) -> List[Grid]:
        prompt_messages = [arc_utils.format_prompt_messages(dp) for dp in datapoints]
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
            output = self.base_model.generate(
                **model_inputs,
                generation_config=self.generation_config,
            ) # (batch_size, total_seq_len)
        
        grids = []
        for seq, prompt_len, datapoint in zip(output, prompt_lens, datapoints):
            new_ids = seq[prompt_len:].cpu().tolist()
            try:
                parsed_grid = self.parse_grid(new_ids)
                grids.append(np.array(parsed_grid))
            except Exception as e:
                print(f"Error parsing grid, using random grid")
                print("Parsed grid:")
                arc_utils.print_grid(parsed_grid)
                traceback.print_exc()
                x, y = self._infer_test_shape(datapoint)
                grids.append(np.random.randint(0, 10, (x, y)))
        return grids
    
    def _select_best_grid(self, datapoint_candidates: List[DataPointDict]) -> Grid:
        # for now, just vote for the most common grid
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

    def prepare_evaluation(
        self,
        checkpoint_path: str = "artifacts/checkpoint-final",
        enable_ttt: bool = True,
        use_data_augmentation_for_eval: bool = True,
        num_augmentations: int = 5,
        batch_size_generation: int = 1,
    ):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.enable_ttt = enable_ttt
        self.use_data_augmentation_for_eval = use_data_augmentation_for_eval
        self.num_augmentations = num_augmentations
        self.batch_size_generation = batch_size_generation
        
        try:
            # Note: checkpoint config should be match with the model config used in intialization
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint_path, 
                is_trainable=enable_ttt,
            )
            print("Loaded LoRA adapter and tokenizer from checkpoint.")
            self.peft_model.eval()
        except Exception as e:
            print(f"No LoRA adapter found or incompatible: {e}")
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    solver = ARCSolver()




