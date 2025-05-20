import os, glob, json, time, random, copy, gc, datetime, traceback
from wasabi import msg
from omegaconf import DictConfig, OmegaConf
import numpy as np

from trl import SFTTrainer, SFTConfig
from transformers import GenerationConfig, TrainingArguments, EarlyStoppingCallback
import torch
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
    
    return cfg

class ARCSolver:
    def __init__(
        self, 
        token: str = None,
        config_path: str = "artifacts/config.yaml",
    ):
        cfg = load_config(config_path)
        train_artifacts_dir = cfg.train_artifacts_dir
        cache_dir = cfg.cache_dir
        sep_str = cfg.sep_str
        model_id = cfg.model.model_id
        use_custom_head = cfg.model.use_custom_head
        lora_rank = cfg.model.lora_rank
        lora_alpha = cfg.model.lora_alpha
        target_modules = cfg.model.target_modules
        
        
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

        # Optimize model vocabulary for ARC tasks if enabled
        if use_custom_head:
            from .custom_head import apply_custom_head
            # Only keep necessary tokens (digits, thinking tokens, special tokens)
            apply_custom_head(self.base_model, self.tokenizer, keep_digits=True, keep_thinking_tokens=True)
            print(f"✓ Model vocabulary optimization applied")
        else:
            print("Model vocabulary optimization skipped.")

        
        # Use default target modules if none provided
        if target_modules is None:
            # Include lm_head and embeddings for training with LoRA
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

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        
        self.sep_str = sep_str
        self.sep_token_id = self.tokenizer.encode(self.sep_str, add_special_tokens=False)[0]
        
        self.enable_ttt = False
        
        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,   
        )
        
        
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
        )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                transform,
                remove_columns=eval_dataset.column_names,
                desc=f"Applying eval dataset transform ({transform_name})",
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
        
        datapoint = {
            "train": examples,
            "test": [
                {
                    "input": questions_input,
                    "output": None,
                }
            ],
        }
        
        if not self.use_data_augmentation_for_eval or self.num_augmentations < 1:
            inferred_grid = self._predict_grid(datapoint)
            return inferred_grid
        
        augmented_datapoints_and_params_map = [
            data_augmentation.random_datapoint_augmentation(datapoint, swap_train_and_test=False)
            for _ in range(self.num_augmentations)
        ]
        
        # run inference on augmented datapoints
        inferred_grids = [
            self._predict_grid(aug_dp_and_pm[0])
            for aug_dp_and_pm in augmented_datapoints_and_params_map
        ]
        
        for (augmented_datapoint, _), inferred_grid in zip(augmented_datapoints_and_params_map, inferred_grids):
            augmented_datapoint["test"][0]["output"] = inferred_grid
        
        # reverse the augmented datapoints
        reversed_datapoints = [
            data_augmentation.revesre_datapoint_augmentation(aug_dp_and_pm[0], aug_dp_and_pm[1])
            for aug_dp_and_pm in augmented_datapoints_and_params_map
        ]
        
        final_grid = self._select_best_grid(reversed_datapoints)
        return final_grid

    def _predict_grid(self, datapoint: DataPointDict) -> Grid:
        prompt_messages = arc_utils.format_prompt_messages(datapoint)
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=False,
        )
        
        model_inputs = self.tokenizer(
            text=prompt_str,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.base_model.generate(
                **model_inputs,
                generation_config=self.generation_config,
            ).squeeze(0).cpu()
            output_ids = output_ids[len(model_inputs["input_ids"][0]):].tolist()
            
            train_input = np.array(datapoint['train'][0]['input'])
            train_output = np.array(datapoint['train'][0]['output'])
            test_input = np.array(datapoint['test'][0]['input'])
            
            if train_input.shape == train_output.shape:
                x, y = test_input.shape
            else:
                x = (train_output.shape[0] * test_input.shape[0] // train_input.shape[0])
                y = (train_output.shape[1] * test_input.shape[1] // train_input.shape[1])
            
            try:
                parsed_grid = self.parse_grid(output_ids)
                grid = np.array(parsed_grid)
                # grid = grid[:x, :y]
                
            except Exception as e:
                print(f"Error parsing grid, using random grid")
                print("Parsed grid:")
                arc_utils.print_grid(parsed_grid)
                print("Output ids:")
                print(output_ids)
                traceback.print_exc()
                grid = np.random.randint(0, 10, (x, y))

            return grid


    def prepare_evaluation(
        self,
        checkpoint_path: str = "artifacts/checkpoint-final",
        enable_ttt: bool = True,
        use_data_augmentation_for_eval: bool = True,
        num_augmentations: int = 5,
    ):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.enable_ttt = enable_ttt
        self.use_data_augmentation_for_eval = use_data_augmentation_for_eval
        self.num_augmentations = num_augmentations
        
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




