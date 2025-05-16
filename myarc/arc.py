import os, glob, json, time, random

from trl import SFTTrainer, SFTConfig
from transformers import GenerationConfig, TrainingArguments
import torch
from typing import List, Union, Literal, Any, TypedDict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from pprint import pprint


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, PeftMixedModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

from .arc_utils import format_prompt_messages
from .datatypes import *

class ARCSolver:
    def __init__(
        self, 
        token: str | None = None,
        model_id: str = "Qwen/Qwen3-4B",
        train_artifacts_dir: str | None = None,
        enable_gradient_checkpointing: bool =False,
        sep_str: str = "\n",
        cache_dir: str | None = None,
        lora_rank: int = 8,
    ):
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
        
        model_args = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": True,  # Allow the model to use custom code from the repository
            "quantization_config": bnb_config,  # Apply the 4-bit quantization configuration
            "attn_implementation": "sdpa",  # Use scaled-dot product attention for better performance
            "torch_dtype": torch.float16,  # Set the data type for the model
            "use_cache": False,  # Disable caching to save memory
            "token": token,
            "device_map": "auto",  # Automatically map the model to available devices
        }
        if cache_dir is not None:
            print(f"Using cache dir: {cache_dir}")
            model_args["cache_dir"] = cache_dir
        else:
            print("No cache dir found, using default cache location.")
        
        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            **model_args,
        )
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_rank, 
            lora_alpha=32,  
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.peft_model = None
        
        if enable_gradient_checkpointing:
            print("Enabling gradient checkpointing for memory efficiency.")
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        tokenizer_args = {
            "pretrained_model_name_or_path": model_id,
            "token": token,
        }
        if cache_dir:
            tokenizer_args["cache_dir"] = cache_dir
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
        self.tokenizer.bos_token_id = 151643 # Default for Qwen3
        self.enable_thinking = False

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        
        self.sep_str = sep_str
        self.sep_token_id = self.tokenizer.encode(self.sep_str, add_special_tokens=False)[0]
        
        self.enable_ttt = False
        
        
    def parse_grid(self, ids: List[int]) -> Grid:
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep_token_id:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                if idx == self.tokenizer.eos_token_id:
                    break
                row.append(inv_map.get(idx, 0))
        if len(row) > 0:
            grid.append(row)
        return grid

    def train(
        self, 
        *,
        train_dataset: HFDataset,
        eval_dataset: HFDataset = None,
        num_epochs: int = 5,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 4,
        batch_size: int = 2,
        warmup_ratio: float = 0.03,
        resume_from_checkpoint: str | None = None,
        optimizer: str = "adamw",
        max_grad_norm: float = 0.3,
        lr_scheduler_type: str = "linear",
        fp16: bool = True,
    ):
        """
        Train a model with train_dataset.
        """
        os.makedirs(self.checkpoint_save_path, exist_ok=True)
        os.makedirs(self.logging_save_path, exist_ok=True)

        training_args = SFTConfig(
            output_dir=self.checkpoint_save_path,
            resume_from_checkpoint=resume_from_checkpoint,

            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            
            eval_strategy="steps",
            logging_dir=self.logging_save_path,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=5,
                                
            learning_rate=learning_rate,
            optim=optimizer,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            
            fp16=fp16,
            
            max_length=None, # avoid truncation
            label_names=["labels"], # TODO: check if needed
        )

        # Note: should not attach eos, since the model is trained with ChatML-style prompts
        trainer = SFTTrainer(
            model=self.base_model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            peft_config=self.peft_config,
        )
        
        start_time = time.time()
        trainer.train()
        trainer.save_model(os.path.join(self.checkpoint_save_path, "checkpoint-final"))
        self.tokenizer.save_pretrained(os.path.join(self.checkpoint_save_path, "checkpoint-final"))

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")

    def test_time_training(self, examples: List[ExampleDict]):
        pass

    def predict(
        self, 
        examples: List[ExampleDict], 
        questions_input: Grid
    ) -> Grid:
        if self.enable_ttt:
            self.test_time_training(examples)
        
        datapoint: DataPointDict = {
            "train": examples,
            "test": [
                {
                    "input": questions_input
                }
            ]
        }

        prompt_message = format_prompt_messages(datapoint)
        prompt = self.tokenizer.apply_chat_template(
            prompt_message,
            add_generation_prompt=True, 
            enable_thinking=self.enable_thinking,
            tokenize=False,
        )
        
        model_inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)

        config = GenerationConfig(
            # temperature=0.7, top_p=0.8, top_k=20,    # 권장 값
            # bos_token_id=self.tokenizer.bos_token_id,
            # eos_token_id=self.tokenizer.eos_token_id,
            # pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=32786 if self.enable_thinking else 150,
            do_sample=True,   
        )
        
        output_ids = self.base_model.generate(
            **model_inputs,
            generation_config=config,
        ).squeeze(0).cpu()

        output_ids = output_ids[len(model_inputs.input_ids[0]):].tolist() # generated portion
        
        if self.enable_thinking:
            try:
                think_close_idx = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                think_close_idx = 0
            
            think_content = self.tokenizer.decode(output_ids[:think_close_idx], skip_special_tokens=True).strip()
            print(f"Thinking content: {think_content}")
            
            output_ids = output_ids[think_close_idx:]
            
        train_input = np.array(examples[0]['input'])
        train_output = np.array(examples[0]['output'])
        test_input = np.array(questions_input)

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] * test_input.shape[0] // train_input.shape[0])
            y = (train_output.shape[1] * test_input.shape[1] // train_input.shape[1])

        try:
            grid = np.array(self.parse_grid(output_ids))
            # grid = grid[:x, :y]
            
        except Exception as e:
            print(f"Error parsing grid: {e}")
            grid = np.random.randint(0, 10, (x, y))

        return grid


    def prepare_evaluation(
        self,
        checkpoint_name: str = "checkpoint-final",
        enable_ttt: bool = False,
        enable_thinking: bool = False,
    ):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        checkpoint_path = os.path.join(self.checkpoint_save_path, checkpoint_name)
        self.enable_ttt = enable_ttt
        self.enable_thinking = enable_thinking
        
        try:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint_path,
                is_trainable=self.enable_ttt,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path
            )
            print("Loaded LoRA adapter and tokenizer from checkpoint.")
        except Exception as e:
            print(f"No LoRA adapter found or incompatible: {e}")
            
            
        self.peft_model.eval()

if __name__ == "__main__":
    solver = ARCSolver()




