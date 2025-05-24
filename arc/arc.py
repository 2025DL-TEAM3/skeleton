import os, glob, json, time, random, copy, gc, datetime, traceback
from wasabi import msg
from omegaconf import DictConfig, OmegaConf
import numpy as np

from trl import SFTTrainer, SFTConfig
from transformers import GenerationConfig, TrainingArguments, EarlyStoppingCallback, TrainerCallback
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
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, PeftModel

from . import arc_utils, data_transform, data_augmentation, inference_helpers
from .datatypes import *


class SaveModelCallback(TrainerCallback):
    """
    Custom callback to periodically save full LoRA adapter + LM Head weights.
    """
    def __init__(self, solver, save_steps: int = 100):
        self.solver = solver
        self.save_steps = save_steps
        os.makedirs(self.solver.checkpoint_save_path, exist_ok=True)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step > 0 and step % self.save_steps == 0:
            path = os.path.join(self.solver.checkpoint_save_path, f"checkpoint-step-{step}")
            # Save LoRA adapter
            if self.solver.peft_model is not None:
                self.solver.peft_model.save_pretrained(path)
            else:
                self.solver.base_model.save_pretrained(path)
            # Save LM head weights
            lm_head_path = os.path.join(path, "lm_head_weights.pt")
            self.solver.save_lm_head(lm_head_path)
            print(f"✓ Saved adapter + LM Head at step {step} to {path}")
        return control


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
        finetune_lm_head = cfg.get("model", {}).get("finetune_lm_head", False)
        lora_rank = cfg.get("model", {}).get("lora_rank", 16)
        lora_alpha = cfg.get("model", {}).get("lora_alpha", 32)
        target_modules = cfg.get("model", {}).get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        
        # LM head를 직접 fine-tune하는 경우 LoRA 타겟 모듈에서 제외
        if finetune_lm_head and "lm_head" in target_modules:
            target_modules = [m for m in target_modules if m != "lm_head"]
            print(f"✓ lm_head will be fine-tuned directly (removed from LoRA target_modules)")
        
        self.finetune_lm_head = finetune_lm_head
        
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
        self.grid_select_policy = "naive"
        
        
    def save_lm_head(self, lm_head_path: str = None):
        model = self if not hasattr(self, 'base_model') else self.base_model
        
        # 만약 모델이 DataParallel로 래핑되어 있는 경우 원본 모델 가져오기
        if hasattr(model, 'module'):
            model = model.module
        
        # lm_head 가중치 추출
        lm_head_state = {}
        for name, param in model.named_parameters():
            if "lm_head" in name:
                lm_head_state[name] = param.cpu().clone()
        
        if lm_head_state:
            if lm_head_path is None:
                raise ValueError("lm_head_path must be specified")
                
            # 디렉토리가 없는 경우 생성
            os.makedirs(os.path.dirname(lm_head_path), exist_ok=True)
            
            torch.save(lm_head_state, lm_head_path)
            print(f"✓ Saved LM head weights to {lm_head_path}")
            return True
        else:
            print("No LM head weights found to save.")
            return False
            
    def _save_lm_head(self, checkpoint_path: str):
        """
        이전 호환성을 위한 메서드
        """
        lm_head_path = os.path.join(checkpoint_path, "lm_head_weights.pt")
        return self.save_lm_head(lm_head_path)
    
    def _load_from_checkpoint(self, checkpoint_path: str):
        """
        체크포인트에서 모델을 로드하고 필요시 LM head 가중치도 로드
        """
        from peft import PeftModel
        
        # LoRA 모델 로드
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            checkpoint_path,
            is_trainable=True
        )
        print(f"✓ Loaded LoRA weights from {checkpoint_path}")
        
        # LM head 가중치 로드 시도
        lm_head_path = os.path.join(checkpoint_path, "lm_head_weights.pt")
        if os.path.exists(lm_head_path) and self.finetune_lm_head:
            lm_head_state = torch.load(lm_head_path, map_location=self.device)
            
            # 기본 모델에 LM head 가중치 적용
            base_model = self.peft_model.base_model
            
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            
            # 가중치 로드 및 적용
            for name, param in base_model.named_parameters():
                if name in lm_head_state:
                    if param.shape != lm_head_state[name].shape:
                        error_msgs.append(f"Size mismatch for {name}: {param.shape} vs {lm_head_state[name].shape}")
                    else:
                        with torch.no_grad():
                            param.copy_(lm_head_state[name].to(param.device))
                elif "lm_head" in name:
                    missing_keys.append(name)
            
            # 로드되지 않은 키 출력
            for name in lm_head_state:
                if not any(name == param_name for param_name, _ in base_model.named_parameters()):
                    unexpected_keys.append(name)
            
            if error_msgs:
                print("\nError(s) in loading LM head weights:")
                for msg in error_msgs:
                    print(f"  {msg}")
            if missing_keys:
                print(f"Some LM head weights were not found in the checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Some weights from the checkpoint were not used: {unexpected_keys}")
                
            print(f"✓ Loaded LM head weights from {lm_head_path}")
        elif self.finetune_lm_head:
            print(f"LM head weights file not found at {lm_head_path}. Using initialized weights.")
    
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
        
        # 트레이닝 설정 추출
        use_data_augmentation = train_args_dict.pop("use_data_augmentation", False)
        patience = train_args_dict.pop("patience", 5)
        lm_head_lr = train_args_dict.pop("lm_head_learning_rate", None)
        # save_steps는 SFTConfig에서 자동으로 처리하므로 여기서 추출하지 않음
        
        callbacks = []
        if eval_dataset is not None and patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=0.0
                )
            )
        # Use our custom callback
        save_steps = train_args_dict.get("save_steps", 100)
        callbacks.append(SaveModelCallback(self, save_steps=save_steps))

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
        
        self.peft_model = trainer.model
        
        if self.finetune_lm_head and lm_head_lr is not None:
            for n, p in self.peft_model.base_model.named_parameters():
                p.requires_grad = False

            lora_params = []
            lm_head_params = []
            for n, p in self.peft_model.named_parameters():
                if "lora" in n.lower():
                    p.requires_grad = True
                    lora_params.append(p)
                elif "lm_head" in n:
                    p.requires_grad = True
                    lm_head_params.append(p)
                else:
                    p.requires_grad = False

            optimizer_grouped_parameters = []
            if lora_params:
                optimizer_grouped_parameters.append({
                    "params": lora_params,
                    "lr": train_args_dict.pop("lora_learning_rate", sft_training_args.learning_rate)
                })
            if lm_head_params:
                print("lm_head_params", len(lm_head_params))
                optimizer_grouped_parameters.append({
                    "params": lm_head_params,
                    "lr": lm_head_lr
                })

            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0
            )
            trainer.optimizers = (optimizer, None)
            print(trainer.optimizers)
            print(trainer.model)
        # preemptively save the model before training, to check error
        trainer.save_model(os.path.join(self.checkpoint_save_path, "checkpoint-initial"))
        trainer.train()
        # save
        final_checkpoint_path = os.path.join(self.checkpoint_save_path, "checkpoint-final")
        trainer.save_model(final_checkpoint_path)
        trainer.state.save_to_json(os.path.join(self.logging_save_path, "trainer_state.json"))
        with open(os.path.join(self.logging_save_path, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)
            
        # LM head가 직접 fine-tune 되었을 경우 별도로 저장
        if self.finetune_lm_head:
            self._save_lm_head(final_checkpoint_path)
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
            batch_size_generation=self.batch_size_generation,
            grid_select_policy=self.grid_select_policy,
        )

    def prepare_evaluation(
        self,
        checkpoint_path: str = "artifacts/checkpoint-final",
        enable_ttt: bool = False,
        use_data_augmentation_for_generation: bool = True,
        num_augmentations: int = 10,
        batch_size_generation: int = 5,
        grid_select_policy: Literal["naive", "grid-wise", "cell-wise-argmax"] = "naive",
        **kwargs,
    ):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.enable_ttt = enable_ttt
        self.use_data_augmentation_for_generation = use_data_augmentation_for_generation
        self.num_augmentations = num_augmentations
        self.batch_size_generation = batch_size_generation
        self.grid_select_policy = grid_select_policy

        generation_config_msg = (
            f"--------------- Generation Config ---------------\n"
            f"Using generation config:\n"
            f"- enable_ttt: {enable_ttt}\n"
            f"- use_data_augmentation_for_generation: {use_data_augmentation_for_generation}\n"
            f"- num_augmentations: {num_augmentations}\n"
            f"- batch_size_generation: {batch_size_generation}\n"
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
            
            # LM head 가중치 로드 시도 (finetune_lm_head가 True인 경우)
            if self.finetune_lm_head:
                lm_head_path = os.path.join(checkpoint_path, "lm_head_weights.pt")
                if os.path.exists(lm_head_path):
                    lm_head_state = torch.load(lm_head_path, map_location=self.device)
                    
                    base_model = self.peft_model.base_model
                    
                    missing_keys = []
                    unexpected_keys = []
                    error_msgs = []
                    
                    # 가중치 로드 및 적용
                    for name, param in base_model.named_parameters():
                        if name in lm_head_state:
                            if param.shape != lm_head_state[name].shape:
                                error_msgs.append(f"Size mismatch for {name}: {param.shape} vs {lm_head_state[name].shape}")
                            else:
                                with torch.no_grad():
                                    param.copy_(lm_head_state[name].to(param.device))
                        elif "lm_head" in name:
                            missing_keys.append(name)
                    
                    for name in lm_head_state:
                        if not any(name == param_name for param_name, _ in base_model.named_parameters()):
                            unexpected_keys.append(name)
                    
                    if error_msgs:
                        print("\nError(s) in loading LM head weights:")
                        for msg in error_msgs:
                            print(f"  {msg}")
                    if unexpected_keys:
                        print(f"Some weights from the checkpoint were not used: {unexpected_keys}")
                    
                    print(f"✓ Loaded LM head weights from {lm_head_path}")
                else:
                    print(f"LM head weights file not found at {lm_head_path}. Using initialized weights.")
            
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