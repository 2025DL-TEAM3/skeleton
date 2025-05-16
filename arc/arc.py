import os, glob, json, time, random

from transformers import GenerationConfig
import torch
from typing import List, Union, Literal, Any, TypedDict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pprint import pprint
from bitsandbytes.optim import AdamW8bit


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

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from .datatypes import *

def train_test_example_to_input_target_ids(
    train_examples: List[ExampleDict],
    test_example: ExampleDict,
    solver: "ARCSolver",
    keep_batch_dim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    datapoint: DataPointDict = {
        "train": train_examples,
        "test": [
            {
                "input": test_example["input"],
            }
        ]
    }
    input_ids = solver.datapoint_to_input(datapoint, keep_batch_dim=False)
    
    target_ids = torch.tensor(
        solver.format_grid(test_example["output"]), dtype=torch.long
    )
    
    # concat eos
    eos = solver.tokenizer.eos_token_id
    if target_ids[-1] != eos:
        target_ids = torch.cat([target_ids, torch.tensor([eos], dtype=torch.long, device=target_ids.device)])
    
    if keep_batch_dim:
        input_ids = input_ids.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)
    
    return input_ids, target_ids
    

class ARCDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        solver: "ARCSolver",
        num_samples_per_task: int = 4,
        num_steps_per_task: int = 50,
    ):
        """
        Args:
            dataset_path (str): Path to the dataset directory
            solver (ARCSolver): Instance of the ARCSolver class
        """
        self.dataset_path = dataset_path
        self.solver = solver
        self.all_tasks: list[TaskDict] = []
        self.load_dataset()
        
        self.num_samples_per_task = num_samples_per_task
        self.num_steps_per_task = num_steps_per_task
        self.total_num_steps = len(self.all_tasks) * num_steps_per_task
    
    def load_dataset(self):
        json_file_paths = glob.glob(f"{self.dataset_path}/*.json")
        if not json_file_paths:
            raise ValueError(f"No JSON files found in {self.dataset_path}")
        
        print(f"Found {len(json_file_paths)} JSON files.")
        
        for json_file_path in json_file_paths:
            task_id = os.path.basename(json_file_path).split(".")[0]
            try:
                with open(json_file_path, 'r') as f:
                    task_json = json.load(f)
                    if isinstance(task_json, list) and len(task_json) > 0:
                        self.all_tasks.append({
                            "file_path": json_file_path,
                            "task_id": task_id,
                            "examples": task_json
                        })
            except Exception as e:
                print(f"Error loading file: {json_file_path} - {e}")
        
        if not self.all_tasks:
            raise ValueError("No valid examples found in JSON files.")
        
        print(f"Successfully loaded {len(self.all_tasks)} JSON files.")
    
    def __len__(self):
        return self.total_num_steps
    
    def __getitem__(self, idx):
        task = random.choice(self.all_tasks)
        examples = task["examples"]
        
        # select self.num_samples_per_task examples
        sampled_examples = random.sample(examples, self.num_samples_per_task)
        
        train_examples = sampled_examples[:self.num_samples_per_task - 1]
        test_example = sampled_examples[self.num_samples_per_task - 1]
        
        input_ids, target_ids = train_test_example_to_input_target_ids(
            train_examples,
            test_example,
            self.solver,
            keep_batch_dim=False,
        )
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }
        
class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(
        self, 
        token=None,
        checkpoint_save_path=None,
        enable_gradient_checkpointing=False,
        sep_str="\n",
    ):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "Qwen/Qwen3-4B"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint_save_path = checkpoint_save_path if checkpoint_save_path else "artifacts"

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
        
        self.model: Union[PreTrainedModel, PeftModel, PeftMixedModel] = AutoModelForCausalLM.from_pretrained(
            **model_args,
        )
        
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
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
        self.tokenizer.bos_token_id = 151643 # Default for Qwen3

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        
        self.sep_str = sep_str
        self.sep_token_id = self.tokenizer.encode(self.sep_str, add_special_tokens=False)[0]
        
        
    def parse_grid(self, ids: List[int]) -> Grid:
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (Grid): parsed 2D grid
        """
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

    # TODO: is col_sep needed?
    def format_grid(self, grid: Grid) -> List[int]:
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (Grid): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep_token_id)
        return ids

    def grid_to_str(self, grid: Grid) -> str:
        """Convert 2D grid to string format
        Args:
            grid (Grid): 2D grid
                [
                    [1, 2],
                    [3, 4]
                ]
        Returns:
            str: String representation of the grid
                12 
                34
        """
        return self.sep_str.join("".join(str(c) for c in row) for row in grid) + self.sep_str

    def format_prompt(self, datapoint: DataPointDict) -> FormattedPrompt:
        # Build example block
        n = len(datapoint['train'])
        plural = 's' if n != 1 else ''
        examples_block = ''
        for i, ex in enumerate(datapoint['train'], start=1):
            examples_block += f"Example {i} Input:\n"
            examples_block += self.grid_to_str(ex['input'])
            examples_block += f"Example {i} Output:\n"
            examples_block += self.grid_to_str(ex['output'])
        template1 = user_message_template1.format(n=n, plural=plural, examples=examples_block)

        # Build test input block
        test_input = f"Test Input:\n{self.grid_to_str(datapoint['test'][0]['input'])}"
        template2 = user_message_template2.format(test_grid=test_input)

        # Assemble messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": template1},
            {"role": "assistant", "content": "Understood"},
            {"role": "user",   "content": template2},
            {"role": "assistant", "content": "Sure!"},
            {"role": "user",   "content": user_message_template3}
        ]

        # 3) Apply chat template without tokenizing
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False
        )

        # 4) Manually tokenize the resulting prompt text
        inputs = self.tokenizer(text, return_tensors="pt")
        # Extract the first sequence in the batch
        input_ids = inputs["input_ids"][0]

        return {
            'input_ids': input_ids,
            'input': datapoint['test'][0]['input'],
            'train': datapoint['train'],
        }
        
    def dynamic_collate(
        self, 
        batch: List[dict],
        padding_side: Literal["left", "right"] = "right",
    ) -> dict:
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_ids"] for item in batch]
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side=padding_side)
        padded_target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side=padding_side)
        
        return {
            "input_ids": padded_input_ids,
            "target_ids": padded_target_ids,
        }


    def seq2seq_loss(self, prompt_ids: torch.LongTensor, target_ids: torch.LongTensor) -> torch.Tensor:
        """
        prompt_ids  : [B, L]  ← 문제 설명(프롬프트)
        target_ids  : [B, T]  ← 정답 토큰 시퀀스
        ------------------------------------------
        inp   = [B, L+T]      ←  [프롬프트][정답] 한줄로 연결
        labels= same shape     (프롬프트 부분은 -100으로 마스킹)
        ------------------------------------------
        model(input_ids=inp, labels=labels)  →  .loss
        """

        inp = torch.cat([prompt_ids, target_ids], dim=1)
        attn_mask = inp.ne(self.tokenizer.pad_token_id).long()
        
        labels = inp.clone()
        labels[:, : prompt_ids.size(1)] = -100
        labels[inp == self.tokenizer.pad_token_id] = -100

        outputs = self.model(input_ids=inp, labels=labels, attention_mask=attn_mask)
        return outputs.loss
    
    def set_peft_model(self):
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16, 
            lora_alpha=32,  
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(self.model, peft_config)

    def train(
        self, 
        train_dataset: ARCDataset,
        num_epochs: int = 5,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 4,
        batch_size: int = 1,
        warmup_rate: float = 0.1,
        checkpoint_name_to_resume_from: str = None,
        
        validation_dataset: ARCDataset = None,
        patience: int = 5,
        val_batch_size: int = 1,
        val_steps: int = 500,
    ):
        """
        Train a model with train_dataset.
        """
        
        # 로깅 파일 설정
        log_file = os.path.join(self.checkpoint_save_path, "training_log.txt")
        
        # 로깅 함수 정의
        def log_message(message, print_to_console=True):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
            if print_to_console:
                print(message)
        
        # 학습 시작 정보 로깅
        timestamp = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
        log_message(f"===== Training started at {timestamp} =====")
        log_message(f"Batch size: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
        log_message(f"Grad. accum: {gradient_accumulation_steps}, Warmup rate: {warmup_rate}")
        
        self.set_peft_model()
        self.model.print_trainable_parameters()
        
        dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, 
            collate_fn=lambda b: self.dynamic_collate(b),
        )
        
        val_loader = None
        if validation_dataset is not None:
            val_loader = DataLoader(validation_dataset, val_batch_size, shuffle=False, pin_memory=True, collate_fn=self.dynamic_collate)
            log_message(f"Total validation batches: {len(val_loader)}")
            
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
    
        # 1) 한 에폭당 optimizer step 수
        steps_per_epoch = (len(train_dataset) // batch_size) // gradient_accumulation_steps

        # 2) 전체 optimizer step 수
        total_optimizer_steps = num_epochs * steps_per_epoch

        # 3) 워밍업 스텝 수
        warmup_steps = int(total_optimizer_steps * warmup_rate)
        log_message(f"Warmup steps: {warmup_steps}, Total optimizer steps: {total_optimizer_steps}")
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        start_epoch, global_step = 0, 0
        # Early stopping variables
        best_val_accuracy = 0
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(self.checkpoint_save_path, "checkpoint-best")

        if checkpoint_name_to_resume_from:
            start_epoch, global_step, best_val_accuracy, best_val_loss = self.load_checkpoint(checkpoint_name_to_resume_from, optimizer, scheduler)
            log_message(f"Resuming from {checkpoint_name_to_resume_from}: epoch {start_epoch} and global step {global_step}")
        
        
        # Set model to training model
        start_time = time.time()
        self.model.train()
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            # steps = len(dataset) / batch_size
            for step, batch in enumerate(dataloader):
                global_step += 1
                
                # Move batch to device and compute loss
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                loss = self.seq2seq_loss(input_ids, target_ids) / gradient_accumulation_steps
                # loss = self.seq2seq_loss_with_regularization(input_ids, target_ids, epoch=epoch) / steps_accum
                
                # Backpropagation
                loss.backward()
                
                # Gradient accumulation and optimization
                if global_step % gradient_accumulation_steps == 0:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()  # Update model parameters
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()  # Clear gradients
                
                # Track and display training progress
                total_loss += loss.item()
                current_lr = scheduler.get_last_lr()[0]
                
                if step % 100 == 0:
                    log_message(f"[Epoch {epoch+1}] step {step} loss {loss.item():.4f} lr {current_lr:.6f}")

                # Validation check - 첫 번째 에폭에서는 검증 건너뛰기
                if val_loader is not None and global_step % val_steps == 0 and epoch >= 0:
                    val_loss, val_accuracy = self.validate(val_loader)
                    log_message(f"[Validation] global_step {global_step} loss {val_loss:.4f} accuracy {val_accuracy:.4f}")

                    # 손실과 정확도를 모두 고려한 early stopping
                    improved = False
                    improvement_message = []
                    
                    # 정확도 향상 확인
                    if val_accuracy > best_val_accuracy:
                        improvement_message.append(f"Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}")
                        best_val_accuracy = val_accuracy
                        improved = True
                    
                    # 손실 향상 확인 (정확도가 동일할 때도 손실이 감소했으면 개선으로 간주)
                    if val_loss < best_val_loss:
                        improvement_message.append(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                        best_val_loss = val_loss
                        improved = True
                    
                    if improved:
                        log_message(", ".join(improvement_message))
                        # Save best model
                        self.save_model(best_model_path, optimizer, scheduler, epoch, global_step, val_accuracy, val_loss)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        log_message(f"Validation metrics did not improve. Patience: {patience_counter}/{patience}")
                    
                    # Early stopping
                    if patience_counter >= patience:
                        log_message(f"Early stopping triggered after {patience} validation checks without improvement")
                        # Load best model
                        self.load_checkpoint(best_model_path, optimizer, scheduler)
                        return
                    
                    # Set model back to training mode after validation
                    self.model.train()

                # Save checkpoint every 5000 steps
                if global_step % 5000 == 0:
                    self.save_model(os.path.join(self.checkpoint_save_path, f"checkpoint-{global_step}"), optimizer, scheduler, epoch, global_step, best_val_accuracy, best_val_loss)
            
            # Print average loss for the epoch
            avg_epoch_loss = total_loss/len(dataloader)
            log_message(f"Epoch {epoch+1} avg loss {avg_epoch_loss:.4f}")
        
        self.model.eval()  # Set model to evaluation mode after training
        log_message("===== Training completed =====")

        # save final model
        self.save_model(os.path.join(self.checkpoint_save_path, "checkpoint-final"), optimizer, scheduler, epoch, global_step, best_val_accuracy, best_val_loss)

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    def validate(self, val_loader):
        """
        Validate the model on the validation dataset
        
        Args:
            val_loader: DataLoader for validation dataset
            
        Returns:
            avg_loss (float): Average loss on validation dataset
            accuracy (float): Accuracy on validation dataset
        """
        print(f"=== Starting validation (total {len(val_loader)} batches) ===")
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0.0
        
        # 검증 시에는 Greedy decoding으로 일관된 출력 생성
        val_config = GenerationConfig(
            do_sample=False,  # 샘플링 없이 확정적 생성
            bos_token_id=151643,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 진행 상황 출력
                if batch_idx == (len(val_loader) // 2) or batch_idx == len(val_loader) - 1:
                    print(f"Validation progress: {batch_idx+1}/{len(val_loader)} batches")
                
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Compute validation loss
                loss = self.seq2seq_loss(input_ids, target_ids)
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

                # get attention mask (should ignore padding tokens when generating)
                attn_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
                
                # Generate outputs for accuracy calculation
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    generation_config=val_config,
                )
                
                # Calculate accuracy by comparing predictions with targets
                # This is a simple token-level accuracy metric
                for i in range(input_ids.size(0)):
                    pred_tokens = outputs[i, input_ids.size(1):].tolist()
                    target_tokens = target_ids[i].tolist()
                    
                    # Remove padding tokens from target
                    target_tokens = [t for t in target_tokens if t != self.tokenizer.pad_token_id]
                    
                    # Compare predicted grid with target grid
                    pred_grid = self.parse_grid(pred_tokens)
                    target_grid = self.parse_grid(target_tokens)
                    
                    try:
                        # numpy 배열로 변환하여 shape 비교 및 내용 확인을 간소화
                        pred_np = np.array(pred_grid)
                        target_np = np.array(target_grid)
                        
                        # 점수 계산 방식:
                        # 1. shape 일치 시 기본적으로 0.5점
                        # 2. 내용까지 완벽히 일치하면 1.0점
                        # 3. shape 불일치 시 0점
                        
                        # 두 배열의 shape 비교
                        if pred_np.shape == target_np.shape:
                            # Shape이 일치하면 기본 0.5점
                            score = 0.5
                            
                            # 내용까지 모두 일치하면 1.0점으로 업그레이드
                            if np.array_equal(pred_np, target_np):
                                score = 1.0
                                
                            correct_predictions += score
                        else:
                            # Shape 불일치
                            correct_predictions += 0.0
                    except (ValueError, TypeError) as e:
                        # 배열 변환 실패 시 0점 처리
                        correct_predictions += 0.0
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        print(f"=== Validation complete: loss {avg_loss:.4f}, accuracy {accuracy:.4f} ===")
        return avg_loss, accuracy
    

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """
        Load the model and its configuration
        """
        # 1) model + PEFT adapter load
        self.model = PeftModel.from_pretrained(self.model, path, is_trainable=True).to(self.device)
        # 2) optimizer / scheduler state load
        opt_path = os.path.join(path, "optimizer.pth")
        if optimizer is not None and os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        sch_path = os.path.join(path, "scheduler.pth")
        if scheduler is not None and os.path.isfile(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location=self.device))
        # 3) epoch / global_step restore (optional)
        state_path = os.path.join(path, "training_state.json")
        start_epoch, start_step = 0, 0
        val_accuracy, val_loss = 0, float('inf')
        if os.path.isfile(state_path):
            st = json.load(open(state_path))
            start_epoch = st.get('epoch', 0)
            start_step  = st.get('global_step', 0)
            val_accuracy = st.get('val_accuracy', 0)
            val_loss = st.get('val_loss', float('inf'))
        return start_epoch, start_step, val_accuracy, val_loss

    def save_model(self, path=None, optimizer=None, scheduler=None, epoch=None, global_step=None, val_accuracy=None, val_loss=None):
        """
        Save the model and its configuration
        """
        if path is None:
            path = "artifacts/qwen3-4b-lora/checkpoint-final"
        os.makedirs(path, exist_ok=True)
        # save model weight + PEFT adapter
        self.model.save_pretrained(path)
        # save optimizer + scheduler state
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pth"))
        # training state
        state = {}
        if epoch is not None:
            state['epoch'] = epoch
        if global_step is not None:
            state['global_step'] = global_step
        if val_accuracy is not None:
            state['val_accuracy'] = val_accuracy
        if val_loss is not None:
            state['val_loss'] = val_loss
        if state:
            with open(os.path.join(path, "training_state.json"), "w") as f:
                json.dump(state, f, indent=2)
        # save config metadata
        info = {
            'base': self.model.config._name_or_path,
            'type': self.model.config.model_type,
            'hidden_size': int(self.model.config.hidden_size),
            'vocab_size': int(self.model.config.vocab_size),
        }
        with open(os.path.join(path, "model_config.json"), 'w') as f:
            json.dump(info, f, indent=2)
    
    def test_time_training(self, examples: List[ExampleDict], num_epochs: int = 1):
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        original_examples = examples.copy()
        
        for epoch in range(num_epochs):
            running = 0.0
            for i, curernt_train_on_example in enumerate(original_examples):
                few_shot_examples = [
                    ex for idx, ex in enumerate(original_examples) if idx != i
                ]
                                
                input_ids, target_ids = train_test_example_to_input_target_ids(
                    few_shot_examples,
                    curernt_train_on_example,
                    self,
                    keep_batch_dim=True,
                )
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                optimizer.zero_grad()
                loss = self.seq2seq_loss(input_ids, target_ids)
                loss.backward()
                optimizer.step()
                running += loss.item()
        
        self.model.eval()
            

    def predict(self, examples: List[ExampleDict], questions_input: Grid) -> Grid:
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        # --- BEGIN TTT (Test-Time Training) with Leave-One-Out and consistent formatting ---
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        import time

        start = time.time()
        max_train_time = 40
        stop_ttt = False

        if not trainable_params:
            print(
                "Warning: No trainable LoRA parameters found for TTT. LoRA adapter might not be configured for training or no adapter loaded."
            )
        else:
            # optimizer = AdamW(
            #     trainable_params, lr=5e-5
            # )  # Lowered learning rate for TTT
            optimizer = AdamW8bit(trainable_params, lr=5e-5)
            ttt_epochs = 1

            original_ttt_examples = examples

            for epoch in range(ttt_epochs):
                if stop_ttt:
                    break
                epoch_loss = 0.0
                gradient_accumulation_steps = 3
                for i, current_train_on_example in enumerate(original_ttt_examples):
                    ttt_start = time.time()
                    few_shot_context_for_ttt = [
                        e for idx, e in enumerate(original_ttt_examples) if idx != i
                    ]

                    ttt_question_input = current_train_on_example["input"]

                    ttt_datapoint_for_prompt = {
                        "train": few_shot_context_for_ttt,
                        "test": [{"input": ttt_question_input}],
                    }

                    prompt_data_ttt = self.format_prompt(ttt_datapoint_for_prompt)
                    prompt_ids_ttt = prompt_data_ttt["input_ids"].unsqueeze(0)

                    target_grid_for_ttt = current_train_on_example["output"]
                    target_tokens_ttt = self.format_grid(target_grid_for_ttt)
                    target_tokens_ttt.append(self.tokenizer.eos_token_id)
                    target_ids_ttt = torch.tensor(
                        [target_tokens_ttt], dtype=torch.long, device=self.device
                    )

                    try:
                        loss = self.seq2seq_loss(prompt_ids_ttt, target_ids_ttt)
                        loss.backward()
                        if (i + 1) % gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                        epoch_loss += loss.item()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"[OOM WARNING] Skipping TTT step {i+1} due to OOM.")
                            torch.cuda.empty_cache()  # GPU 메모리 클리어
                            stop_ttt = True
                            break
                        else:
                            raise e  # OOM 외 오류는 그대로 던짐
                    ttt_end = time.time()
                    train_time = ttt_end - start
                    if(train_time > max_train_time):
                        stop_ttt = True
                        break
                    print(f"TTT {i+1}/{len(original_ttt_examples)} loss {loss.item():.4f} time {ttt_end-ttt_start:.2f}s")

        ttt_end = time.time()
        print(f"TTT total time {ttt_end-start:.2f}s")

        self.model.eval()
        # --- END TTT ---
        datapoint = {"train": examples, "test": [{"input": questions_input}]}

        prompt = self.format_prompt(datapoint)
        # input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).to(self.device).view(1, -1)
        input_ids = prompt["input_ids"].unsqueeze(0)

        attn_mask = torch.ones_like(input_ids)

        # Qwen3 모델은 더 많은 토큰을 생성할 수 있도록 설정
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=20,  # 권장 값
            bos_token_id=151643,  # Qwen3 모델의 내부 기본값 명시적 사용
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150,
            do_sample=True,
        )

        # shape 계산용
        train_input = np.array(prompt["train"][0]["input"])
        train_output = np.array(prompt["train"][0]["output"])
        test_input = np.array(prompt["input"])

        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = train_output.shape[0] * test_input.shape[0] // train_input.shape[0]
            y = train_output.shape[1] * test_input.shape[1] // train_input.shape[1]

        try:
            output = (
                self.model.generate(
                    input_ids=input_ids,
                    generation_config=config,
                    attention_mask=attn_mask,
                )
                .squeeze()
                .cpu()
            )
            N_prompt = input_ids.size(1)
            output = output[N_prompt:].tolist()

            grid = np.array(self.parse_grid(output))
            if grid.shape[0] == 0 or grid.shape[1] == 0:
                print("[Warning] Empty grid parsed. Falling back to random.", flush=True)
                grid = np.random.randint(0, 10, (x, y))

        except Exception as e:
            print(f"[Error during prediction] {e}. Returning random grid.", flush=True)
            
            grid = np.random.randint(0, 10, (x, y))

        predict_end = time.time()
        print(f"Predict total time {predict_end - ttt_end:.2f}s", flush=True)

        return grid

    def prepare_evaluation(
        self,
        checkpoint_name: str = "checkpoint-final",
        enable_ttt: bool = True,
    ):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        checkpoint_path = os.path.join(self.checkpoint_save_path, checkpoint_name)
        self.enable_ttt = enable_ttt
        
        # LoRA 어댑터 로드
        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                is_trainable=self.enable_ttt,
            )
            print("Loaded LoRA adapter")
        except Exception as e:
            print(f"No LoRA adapter found or incompatible: {e}")
            
            
        self.model.eval()
        
    def datapoint_to_input(self, datapoint: DataPointDict, keep_batch_dim: bool=False) -> torch.Tensor:
        """
        Convert a datapoint to input format for the model.
        """
        prompt = self.format_prompt(datapoint)
        if isinstance(prompt["input_ids"], torch.Tensor):
            input_ids = prompt["input_ids"]
        else:
            input_ids = torch.tensor(prompt["input_ids"], dtype=torch.long)
        
        if keep_batch_dim:
            input_ids = input_ids.unsqueeze(0)
        return input_ids

if __name__ == "__main__":
    solver = ARCSolver()




