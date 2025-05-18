import os, glob, json, time, random, copy
from wasabi import msg

from trl import SFTTrainer, SFTConfig
from transformers import GenerationConfig, TrainingArguments
import torch
from typing import List, Union, Literal, Any, TypedDict, Callable, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from pprint import pprint
import datasets

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
from transformers.trainer import seed_worker

from .arc_utils import format_prompt_messages
from . import arc_utils, data_transform
from .datatypes import *
from .arc_dataset import TaskBatchSampler, ARCTrainDataset, ARCValidationDataset

class ARCSFTTrainer:
    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        train_dataset_builder: Callable[[], HFDataset],
        train_dataset_transform: Callable[[DataPointDict], PromptCompletionPair] = None,
        eval_dataset: HFDataset = None,
        eval_dataset_transform: Callable[[DataPointDict], PromptCompletionPair] = None,
        processing_class: PreTrainedTokenizer = None,
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        peft_config: Optional[PeftConfig] = None,
        use_task_batch_sampler: bool = True,
    ):  
        self.model = model
        self.train_dataset_builder = train_dataset_builder
        if train_dataset_transform is None:
            self.train_dataset_transform = data_transform.DefaultFormatMessages()
        else:
            self.train_dataset_transform = train_dataset_transform

        self.processing_class = processing_class

        self.eval_dataset = eval_dataset
        if eval_dataset is not None:
            if eval_dataset_transform is None:
                eval_dataset_transform = data_transform.DefaultFormatMessages()
            self.eval_dataset = eval_dataset.map(
                eval_dataset_transform,
                remove_columns=eval_dataset.column_names,
                desc="Applying eval dataset transform",
            )

        if args is None:
            args = SFTConfig(f"{model.config._name_or_path.split('/')[-1]}-SFT")
        self.args = args
        if isinstance(model, PeftModel):
            self.peft_config = None
        else:
            self.peft_config = peft_config
        self.use_task_batch_sampler = use_task_batch_sampler
        self.num_epochs = args.num_train_epochs
        
        # TODO: oom handling
        self.train_datasets: list[HFDataset] = [
            self.train_dataset_builder() for _ in range(self.num_epochs)
        ]
        
        # TODO: make it configurable
        self.eval_strategy = args.eval_strategy
        self.logging_strategy = args.logging_strategy
        self.logging_dir = args.logging_dir
        self.logging_file = os.path.join(self.logging_dir, "train.log")
        self.save_strategy = args.save_strategy 
    
    def log(self, message: str, trainer: SFTTrainer = None, msg_type: str = "info"):
        if self.logging_strategy == "no":
            return
        
        with open(self.logging_file, "a") as f:
            f.write(f"{message}\n")
        try:
            msg_fn = getattr(msg, msg_type)
            msg_fn(message)
        except Exception as e:
            # fallback to msg.info
            msg.info(message)
        if trainer is not None:
            trainer.log(message)
    
    def train(self):
        start_time = time.time()
        train_dataset = None
        for epoch in range(self.num_epochs):
            self.log(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            epoch_start_time = time.time()
            # train_dataset = self.train_dataset_builder()
            train_dataset = self.train_datasets[epoch]
            
            train_dataset = train_dataset.map(
                self.train_dataset_transform,
                remove_columns=train_dataset.column_names,
                desc=f"Applying train dataset transform for epoch {epoch + 1}"
            )

            training_arguments = self.prepare_training_arguments(self.args, epoch)

            trainer = TaskBatchSamplingSFTTrainer(
                model=self.model,
                processing_class=self.processing_class,
                train_dataset=train_dataset,
                eval_dataset=self.eval_dataset,
                args=training_arguments,
                peft_config=self.peft_config,
                use_task_batch_sampler=self.use_task_batch_sampler,
            )

            trainer.train()

            if self.eval_dataset is not None and self.eval_strategy != "no":
                eval_metrics = trainer.evaluate()
                self.log(f"[Eval][Epoch {epoch + 1} / {self.num_epochs}] {eval_metrics}")

            self.model = trainer.model
            model_checkpoint_dir = os.path.join(
                training_arguments.output_dir, f"checkpoint-{epoch + 1}"
            )
            trainer.save_model(model_checkpoint_dir)

            epoch_end_time = time.time()
            self.log(f"Epoch {epoch + 1} completed in {epoch_end_time - epoch_start_time:.2f} seconds")
        
            if epoch + 1 == self.num_epochs:
                # save final model
                final_model_dir = os.path.join(training_arguments.output_dir, "checkpoint-final")
                trainer.save_model(final_model_dir)
        
        end_time = time.time()
        self.log(f"Training completed in {end_time - start_time:.2f} seconds", msg_type="good")
    
    def prepare_training_arguments(self, base_args: SFTConfig, epoch: int) -> SFTConfig:
        args = copy.deepcopy(base_args)

        args.logging_strategy = "no"
        args.save_strategy = "no"
        args.eval_strategy = "no"

        args.num_train_epochs = 1
        return args

class TaskBatchSamplingSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        train_dataset: HFDataset = None,
        eval_dataset: HFDataset = None,
        processing_class: PreTrainedTokenizer = None,
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        peft_config: Optional[PeftConfig] = None,
        use_task_batch_sampler: bool = True,
    ):    
        # args.dataset_kwargs = {
        #     'skip_prepare_dataset': True, # Customize dataset
        # }
        super().__init__(
            model=model,
            processing_class=processing_class,
            # TODO: set to None, will be set in get_train_dataloader
            # for now, just set to huggingface dataset
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            args=args,
            peft_config=peft_config,
        )
        self.use_task_batch_sampler = use_task_batch_sampler
    
    def get_train_dataloader(self) -> DataLoader:
        if not self.use_task_batch_sampler:
            return super().get_train_dataloader()
    
        raise ValueError("TaskBatchSampler is not supported yet.")
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # train_dataset = self.train_dataset_builder()
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            # Removed "batch_size" because we use batch_sampler below
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            dataloader_params["batch_sampler"] = TaskBatchSampler(
                dataset=train_dataset,
                batch_size=self._train_batch_size,
            )
        print(f"Dataloader loaded with {len(train_dataset)} samples.")
        dataloader = DataLoader(train_dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)

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
        
        self.model_args = {
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
            self.model_args["cache_dir"] = cache_dir
            self.cache_dir = cache_dir
        else:
            print("No cache dir found, using default cache location.")
            self.cache_dir = None
        
        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            **self.model_args,
        )
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_rank, 
            lora_alpha=64,  
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.peft_model = None
        
        if enable_gradient_checkpointing:
            print("Enabling gradient checkpointing for memory efficiency.")
            self.base_model.gradient_checkpointing_enable()
            self.base_model.config.use_cache = False
            if hasattr(self.base_model, "enable_input_require_grads"):
                self.base_model.enable_input_require_grads()

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
        # train_dataset: ARCTrainDataset,
        train_dataset_builder: Callable[[], ARCTrainDataset] | Callable[[], HFDataset],
        eval_dataset: ARCValidationDataset | HFDataset= None, # TODO
        num_epochs: int = 5,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 4,
        batch_size: int = 1,
        use_task_batch_sampler: bool = True,
        warmup_ratio: float = 0.05,
        resume_from_checkpoint: str | None = None,
        optimizer: str = "adamw",
        max_grad_norm: float = 1.0,
        lr_scheduler_type: str = "linear",
        fp16: bool = True,
        eval_strategy: str = "epoch",
        save_strategy: str = "epoch",
        logging_strategy: str = "epoch",
    ):
        """
        Train a model with train_dataset.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, HFDataset):
            raise ValueError("eval_dataset must be a HuggingFace dataset for now.")

        os.makedirs(self.checkpoint_save_path, exist_ok=True)
        os.makedirs(self.logging_save_path, exist_ok=True)

        peft_config = self.peft_config
        if resume_from_checkpoint is not None:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            is_peft_checkpoint = arc_utils.is_peft_checkpoint_path(resume_from_checkpoint)
            if is_peft_checkpoint:
                print(f"Loading LoRA adapter from {resume_from_checkpoint}")
                self.peft_model = PeftModel.from_pretrained(
                    self.base_model,
                    resume_from_checkpoint,
                    is_trainable=True,
                )
                peft_config = None # already loaded
            else:
                print(f"Loading model from {resume_from_checkpoint}")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    resume_from_checkpoint,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )

        training_args = SFTConfig(
            output_dir=self.checkpoint_save_path,

            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            
            eval_strategy=eval_strategy,
            logging_strategy=logging_strategy,
            logging_dir=self.logging_save_path,
            save_strategy=save_strategy,
            
            learning_rate=learning_rate,
            optim=optimizer,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            
            fp16=fp16,
            
            max_length=None, # avoid truncation
            label_names=["labels"], # TODO: check if needed
        )

        trainer = ARCSFTTrainer(
            model=self.base_model if self.peft_model is None else self.peft_model,
            processing_class=self.tokenizer,
            train_dataset_builder=train_dataset_builder,
            train_dataset_transform=data_transform.RandomAugmentationTransform(), # TODO: make it configurable
            eval_dataset=eval_dataset,
            eval_dataset_transform=data_transform.RandomAugmentationTransform(), # TODO: make it configurable
            args=training_args,
            peft_config=peft_config,
            use_task_batch_sampler=use_task_batch_sampler,
        )
        
        trainer.train()


    def test_time_training(self, examples: List[ExampleDict]):
        """
        Test time training (TTT) for the model.
        """
        msg.info("Test time training...")
        new_dataset = arc_utils.create_n_minus_1_dataset(examples)
        new_hf_dataset = HFDataset.from_list(new_dataset)
        
        self.train(
            train_dataset_builder=lambda: new_hf_dataset,
            num_epochs=1,
            batch_size=1,
            use_task_batch_sampler=False,
            learning_rate=8e-5,
            gradient_accumulation_steps=4,
            warmup_ratio=0.0,
            fp16=True,
            optimizer="paged_adamw_8bit",
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
        )
        msg.good("Test time training completed.")

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
            max_new_tokens=32786 if self.enable_thinking else 8196,
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




