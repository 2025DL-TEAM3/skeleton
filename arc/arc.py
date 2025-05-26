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

from . import arc_utils, data_transform, data_augmentation, inference_helpers
from .datatypes import *

def load_config(config_path: str) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    
    assert cfg.artifact_name, "Artifact name must be specified in the config file."
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
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
        target_modules = cfg.get("model", {}).get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        augmented_dataset_dir = cfg.get("augmented_dataset_dir", "augmented_dataset")
        use_cached_augmented_dataset = cfg.get("use_cached_augmented_dataset", True)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        if train_artifacts_dir is not None:
            self.train_artifacts_dir = train_artifacts_dir
        else:
            self.train_artifacts_dir = os.path.join("artifacts", "train-default")
        self.checkpoint_save_path = os.path.join(self.train_artifacts_dir, "checkpoints")
        self.logging_save_path = os.path.join(self.train_artifacts_dir, "logs")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True,  
            bnb_4bit_quant_type="nf4",  
            bnb_4bit_compute_dtype=torch.float16, 
        )
        
        self.model_args = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": True, 
            "quantization_config": bnb_config,  
            "attn_implementation": "sdpa", 
            "torch_dtype": torch.float16,  
            "use_cache": False,  
            "token": token,
            "tie_word_embeddings": not use_custom_head,
            # "device_map": "auto",  
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
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.base_model.config.eos_token_id = self.tokenizer.eos_token_id

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
            max_new_tokens=200,
            do_sample=True,
            temperature=0.5,
            top_p=0.8,
            top_k=30,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.grid_select_policy = "naive"
        
        self.augmented_dataset_dir = augmented_dataset_dir
        os.makedirs(self.augmented_dataset_dir, exist_ok=True)
        self.use_cached_augmented_dataset = use_cached_augmented_dataset
        
        
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
        
        transform = data_transform.get_data_transform()
        transform_name = transform.__class__.__name__
        msg.info(f"Using transform: {transform_name}")  
        
        if self.use_cached_augmented_dataset and os.path.isfile(os.path.join(self.augmented_dataset_dir, "train.jsonl")):
            # load from augmented dataset
            print(f"Loading train dataset from cached augmented dataset path: {self.augmented_dataset_dir}")
            train_dataset = arc_utils.load_augmented_dataset_from_jsonl(
                os.path.join(self.augmented_dataset_dir, "train.jsonl"),
            )
            print(f"Loaded {len(train_dataset)} examples from cached augmented dataset.")
        else:
            print("Using cached augmented dataset is disabled or no cached dataset found, applying transform to train dataset.")
            train_dataset = train_dataset.map(
                transform,
                remove_columns=train_dataset.column_names,
                desc=f"Applying train dataset transform ({transform_name})",
                num_proc=sft_training_args.dataset_num_proc,
            )
            arc_utils.save_augmented_dataset_to_jsonl(
                train_dataset, 
                os.path.join(self.augmented_dataset_dir, "train.jsonl"),
            )
        
        if eval_dataset is not None:
            if self.use_cached_augmented_dataset and os.path.isfile(os.path.join(self.augmented_dataset_dir, "eval.jsonl")):
                # load from augmented dataset
                print(f"Loading eval dataset from cached augmented dataset path: {self.augmented_dataset_dir}")
                eval_dataset = arc_utils.load_augmented_dataset_from_jsonl(
                    os.path.join(self.augmented_dataset_dir, "eval.jsonl"),
                )
                print(f"Loaded {len(eval_dataset)} examples from cached augmented dataset.")
            else:
                print("Using cached augmented dataset is disabled or no cached dataset found, applying transform to eval dataset.")
                eval_dataset = eval_dataset.map(
                    transform,
                    remove_columns=eval_dataset.column_names,
                    desc=f"Applying eval dataset transform ({transform_name})",
                    num_proc=sft_training_args.dataset_num_proc,
                )
                arc_utils.save_augmented_dataset_to_jsonl(
                    eval_dataset, 
                    os.path.join(self.augmented_dataset_dir, "eval.jsonl"),
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
        # save
        trainer.save_model(os.path.join(self.checkpoint_save_path, "checkpoint-final"))
        trainer.state.save_to_json(os.path.join(self.logging_save_path, "trainer_state.json"))
        with open(os.path.join(self.logging_save_path, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)
        end_time = time.time()
        msg.good(f"Training completed in {end_time - start_time:.2f} seconds.")
        msg.info(f"Best checkpoint was {trainer.state.best_model_checkpoint}")

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

        inferencer = inference_helpers.ARCInferencer(
            self.peft_model if self.peft_model else self.base_model,
            self.tokenizer,
            self.generation_config,
            parse_grid_fn=self.parse_grid,
        )
        
        if not self.use_data_augmentation_for_generation or self.num_augmentations < 1:
            return inferencer.predict_single(base_datapoint)
        
        return inferencer.predict(
            base_datapoint,
            num_augmentations=self.num_augmentations,
            grid_select_policy=self.grid_select_policy,
        )

    def prepare_evaluation(
        self,
        checkpoint_path: str = "artifacts/checkpoint-final",
        enable_ttt: bool = False,
        use_data_augmentation_for_generation: bool = True,
        num_augmentations: int = 10,
        grid_select_policy: Literal["naive", "grid-wise", "cell-wise-argmax"] = "naive",
        **kwargs,
    ):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.enable_ttt = enable_ttt
        self.use_data_augmentation_for_generation = use_data_augmentation_for_generation
        self.num_augmentations = num_augmentations
        self.grid_select_policy = grid_select_policy

        generation_config_msg = (
            f"--------------- Generation Config ---------------\n"
            f"Using generation config:\n"
            f"- enable_ttt: {enable_ttt}\n"
            f"- use_data_augmentation_for_generation: {use_data_augmentation_for_generation}\n"
            f"- num_augmentations: {num_augmentations}\n"
            f"- grid_select_policy: {grid_select_policy}\n"
        )
        print(generation_config_msg)
        
        try:
            # Note: checkpoint config should be match with the model config used in intialization
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint_path, 
                is_trainable=enable_ttt,
            )
            print("------------------ LoRA adapter loaded -----------------")
            print(f"Model ID: {self.model_id}")
            for adapter_name, config in self.peft_model.peft_config.items():
                print(f"Adapter: {adapter_name}")
                print(f"  Rank: {config.r}")
                print(f"  Alpha: {config.lora_alpha}")
                print(f"  Target Modules: {config.target_modules}")

            print("Loaded LoRA adapter and tokenizer from checkpoint.")
            self.peft_model.eval()
        except Exception as e:
            print(f"No LoRA adapter found or incompatible: {e}")
            traceback.print_exc()
            raise e


if __name__ == "__main__":
    solver = ARCSolver()




