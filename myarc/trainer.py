import os, glob, json, time, random, copy, gc
import warnings
from dataclasses import dataclass
from wasabi import msg
from tqdm import trange

from trl import SFTTrainer, SFTConfig, apply_chat_template
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from transformers import GenerationConfig, TrainingArguments
import torch
from typing import List, Union, Literal, Any, TypedDict, Callable, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from pprint import pprint
import datasets
import bitsandbytes as bnb

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

def tokenize(
    example: FormattedPromptCompletionPair,
    processing_class: PreTrainedTokenizer,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    processed_prompt = processing_class(
        text=example["prompt"],
        add_special_tokens=add_special_tokens,
    )

    processed = processing_class(
        text=example["prompt"] + example["completion"],
        add_special_tokens=add_special_tokens,
    )

    prompt_ids = processed_prompt["input_ids"]
    prompt_completion_ids = processed["input_ids"]
    if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
        warnings.warn(
            "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
            "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
            "token handling. Verify that the tokenizer is processing text consistently."
        )

    # Create a completion mask
    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    processed = {**processed, "completion_mask": completion_mask}

    return processed


    
class ARCSFTTrainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        processing_class: PreTrainedTokenizer,
        train_dataset: HFDataset | ARCTrainDataset,
        eval_dataset: HFDataset | ARCValidationDataset,
        augment_transform: Callable[[DataPointDict], PromptCompletionPair],
        args: SFTConfig,
        peft_config: LoraConfig,
        patience: int = 0,
    ):
        self.model = model
        self.processing_class = processing_class
        self.train_dataset = self._prepare_dataset(train_dataset, "train dataset")
        self.eval_dataset = self._prepare_dataset(eval_dataset, "eval dataset")
        self.args = args
        self.peft_config = peft_config
        self.patience = patience
        
        self.augment_transform = augment_transform
        
        self.logging_file = os.path.join(args.output_dir, "train_log.txt")
        self.metric_file = os.path.join(args.output_dir, "metrics.jsonl")
        
        pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
        pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
        if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
        self.data_collator = DataCollatorForLanguageModeling(pad_token_id, completion_only_loss=True)
        
        print(len(self.train_dataset), len(self.eval_dataset))
        print(len(self._get_train_dataloader()))
        print(len(self._get_eval_dataloader()))
    
    def _prepare_dataset(self, dataset: HFDataset | Dataset, dataset_name: str) -> HFDataset | Dataset:
        if isinstance(dataset, HFDataset):
            dataset = dataset.map(
                self.augment_transform,
                remove_columns=dataset.column_names,
                desc=f"Applying data augmentation to {dataset_name}",
            )
            
            column_names = next(iter(dataset)).keys()
            dataset = dataset.map(
                apply_chat_template,
                fn_kwargs={"tokenizer": self.processing_class},
                remove_columns="messages" if "messages" in column_names else None,
                desc=f"Applying chat template to {dataset_name}",
            )
            
            dataset = dataset.map(
                tokenize,
                fn_kwargs={
                    "processing_class": self.processing_class,
                    "add_special_tokens": False,
                },
                desc=f"Tokenizing {dataset_name}",
            )

            return dataset
        elif isinstance(dataset, ARCTrainDataset) or isinstance(dataset, ARCValidationDataset):
            def transform_fn(datapoint: DataPointDict) -> Any:
                prompt_completion_pair = self.augment_transform(datapoint)
                chat_template_applied = apply_chat_template(
                    prompt_completion_pair,
                    tokenizer=self.processing_class,
                )
                processed = tokenize(chat_template_applied, self.processing_class, add_special_tokens=False)
                return processed
            dataset.transform = transform_fn
            return dataset
    
    def log(self, message: str, type: Literal["info", "good", "print"] = "info"):
        """
        Log a message to the console and to the logging file.
        """
        msg_fn_map = {
            "info": msg.info,
            "good": msg.good,
            "print": print,
        }
        msg_fn = msg_fn_map.get(type, msg.info)
        msg_fn(message)
        
        with open(self.logging_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
    
    def log_metrics(self, metrics: dict):
        """
        Log metrics to the console and to the metrics file.
        """
        with open(self.metric_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
            f.flush()
    
    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        return outputs.loss
    
    def train(self):
        start_time = time.time()
        self.log(f"===== Training started =====")
        # self.log(f"Training arguments: {json.dumps(self.args.to_dict(), indent=2)}")
        # self.log(f"PEFT config: {json.dumps(self.peft_config.to_dict(), indent=2)}")
        
        peft_config = self.peft_config
        self.model = get_peft_model(self.model, peft_config)
        
        self.log(f"Model: {self.model.config._name_or_path}")
        self.model.print_trainable_parameters()
        
        # TODO: configurable optimizer and scheduler
        optimizer = bnb.optim.PagedAdam8bit(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )
        
        steps_per_epoch = (len(self.train_dataset) // self.args.per_device_train_batch_size) // self.args.gradient_accumulation_steps
        total_optimizer_steps = steps_per_epoch * self.args.num_train_epochs
        warmup_steps = int(total_optimizer_steps * self.args.warmup_ratio)
        
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        self.log(f"Warmup steps: {warmup_steps}, total optimizer steps: {total_optimizer_steps}", type="print")
        
        start_epoch, global_step = 0, 0
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.args.output_dir, "checkpoint-best")
        
        if self.args.resume_from_checkpoint:
            self.log(f"Resuming from checkpoint: {self.args.resume_from_checkpoint}")
            loded_info = self.load_checkpoint(self.args.resume_from_checkpoint, optimizer, scheduler)
            self.log(f"Loaded info: {loded_info}")
            
            start_epoch = loded_info["start_epoch"]
            global_step = loded_info["global_step"]
            best_val_loss = loded_info["best_val_loss"]
        
        if self.args.resume_from_checkpoint and os.path.exists(best_model_path):
            self.log(f"Loading best model from: {best_model_path}")
            loaded_info = self.load_checkpoint(best_model_path, None, None)
            best_checkpoint_loss = loaded_info["best_val_loss"]

            if best_checkpoint_loss < best_val_loss:
                best_val_loss = best_checkpoint_loss
            
        self.model.train()
        self.log(f"Training for {self.args.num_train_epochs} epochs", type="print")
        for epoch in trange(start_epoch, self.args.num_train_epochs):
            total_loss = 0.0
            dataloader = self._get_train_dataloader()
            for step, batch in enumerate(dataloader):
                global_step += 1
                batch_device = {k: v.to(self.model.device) for k, v in batch.items()}
                loss = self.compute_loss(self.model, batch_device)
                
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                
                if (global_step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                current_lr = scheduler.get_last_lr()[0]
                
                if global_step % self.args.logging_steps == 0:
                    self.log(
                        f"[Train] [Epoch {epoch + 1}/{self.args.num_train_epochs}] "
                        f"step {step + 1} "
                        f"train loss: {total_loss / (step + 1):.4f} "
                        f"current lr: {current_lr:.6f} "
                        , type="print"
                    )
                
                if global_step % self.args.eval_steps == 0:
                    val_loss = self.evaluate()
                    self.log(
                        f"[Validation] [Epoch {epoch + 1}/{self.args.num_train_epochs}] "
                        f"step {global_step} "
                        f"validation loss: {val_loss:.4f} "
                        , type="print"
                    )
                    
                    self.log_metrics({
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_loss": total_loss / (step + 1),
                        "val_loss": val_loss,
                        "current_lr": current_lr,
                    })
                    
                    improved = False
                    improvement_message = []

                    if val_loss < best_val_loss:
                        improvement_message.append(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                        best_val_loss = val_loss
                        improved = True
                    
                    if improved:
                        self.log(", ".join(improvement_message), type="print")
                        self.save_model(
                            best_model_path,
                            optimizer,
                            scheduler,
                            epoch + 1,
                            global_step,
                            val_loss,
                        )
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        self.log(f"Validation metrics did not improve. Patience: {patience_counter}/{self.patience}", type="print")
                    
                    if self.patience > 0 and patience_counter >= self.patience:
                        self.log(f"Early stopping triggered after {self.patience} validation checks without improvement", type="print")
                        self.load_checkpoint(best_model_path, optimizer, scheduler)
                        return
                
                    self.model.train()
                
                del batch_device, loss
            
            self.save_model(
                os.path.join(self.args.output_dir, f"checkpoint-{epoch + 1}"),
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
            )
            
            avg_epoch_loss = total_loss / len(dataloader)
            self.log(
                f"[Epoch {epoch + 1}/{self.args.num_train_epochs}] "
                f"average training loss: {avg_epoch_loss:.4f} "
                , type="print"
            )
            
            torch.cuda.empty_cache()
            gc.collect()
            
            self.log_metrics({
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": avg_epoch_loss,
            })
        

        self.model.eval()
        self.log(f"===== Training completed =====", type="good")
        end_time = time.time()
        self.log(f"Total training time: {end_time - start_time:.2f} seconds", type="print")
        
        self.save_model(
            os.path.join(self.args.output_dir, "checkpoint-final"),
            optimizer,
            scheduler,
            self.args.num_train_epochs,
            global_step,
            best_val_loss,
        )
        
    def evaluate(self):
        self.log(f"===== Evaluating =====")
        self.model.eval()
        total_loss = 0.0
        dataloader = self._get_eval_dataloader()
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch_device = {k: v.to(self.model.device) for k, v in batch.items()}
                loss = self.compute_loss(self.model, batch_device)
                total_loss += loss.item()
                
                if (step + 1) % self.args.logging_steps == 0:
                    self.log(
                        f"[Eval] step {step + 1} "
                        f"validation loss: {total_loss / (step + 1):.4f} "
                        , type="print"
                    )
                
                del batch_device, loss
        avg_loss = total_loss / len(dataloader)
        self.log(f"Average validation loss: {avg_loss:.4f}", type="print")
        return avg_loss
    
    def _get_train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
            drop_last=False,
        )
        return train_dataloader
    
    def _get_eval_dataloader(self):
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
            drop_last=False,
        )
        return eval_dataloader

            
    def save_model(
        self,
        output_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        val_loss: Optional[float] = None,
    ):
        self.log(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processing_class.save_pretrained(output_dir)
        
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        state = dict()
        if epoch is not None:
            state["epoch"] = epoch
        if global_step is not None:
            state["global_step"] = global_step
        if val_loss is not None:
            state["val_loss"] = val_loss
        with open(os.path.join(output_dir, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)
        
        model_config = {
            "base": self.model.config._name_or_path,
            "type": self.model.config.model_type,
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": int(self.model.config.vocab_size),
        }
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
    
    def load_checkpoint(
        self,
        checkpoint_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    ) -> dict:
        self.log(f"Loading checkpoint from {checkpoint_dir}")
        self.model = PeftModel.from_pretrained(self.model, checkpoint_dir, is_trainable=True)
        
        if optimizer is not None and os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        
        if scheduler is not None and os.path.exists(os.path.join(checkpoint_dir, "scheduler.pt")):
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        
        state_pth = os.path.join(checkpoint_dir, "training_state.json")
        state = {
            "epoch": 0,
            "global_step": 0,
            "best_val_loss": float("inf"),
        }
        
        if os.path.isfile(state_pth):
            _st = json.load(open(state_pth, "r"))
            state["epoch"] = _st.get("epoch", 0)
            state["global_step"] = _st.get("global_step", 0)
            state["best_val_loss"] = _st.get("val_loss", float("inf"))
        return state
