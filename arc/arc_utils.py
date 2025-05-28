
import os, glob, json, time, random
from itertools import islice

from typing import List, Dict, Any, Tuple, Union, Optional, Iterator

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM

from .datatypes import *

def load_tasks_from_paths(json_paths: list[str]) -> list[TaskDict]:
    all_tasks = []
    for json_file_path in json_paths:
        task_id = os.path.basename(json_file_path).rsplit(".", 1)[0]
        try:
            with open(json_file_path, 'r') as f:
                task_json = json.load(f)
                if isinstance(task_json, list) and len(task_json) > 0:
                    all_tasks.append({
                        "file_path": json_file_path,
                        "task_id": task_id,
                        "examples": task_json
                    })
        except Exception as e:
            print(f"Error loading file: {json_file_path} - {e}")
    
    if not all_tasks:
        raise ValueError("No valid examples found in JSON files.")
        
    print(f"Successfully loaded {len(all_tasks)} JSON files.")
    return all_tasks

def load_json_normal_tasks(dataset_path: str) -> list[TaskDict]:
    json_file_paths = glob.glob(os.path.join(dataset_path, '*.json'))
    if not json_file_paths:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")

    print(f"Found {len(json_file_paths)} JSON files for normal tasks.")
    
    return load_tasks_from_paths(json_file_paths)

def sample_datapoints_from_normal_task(
    task: TaskDict,
    num_samples: int = 4,
) -> DataPointDict:
    examples = task["examples"]
    if len(examples) < num_samples:
        raise ValueError(f"Not enough examples in the task. Required: {num_samples}, Available: {len(examples)}")

    sampled_examples = random.sample(examples, num_samples)
    train_examples = sampled_examples[:num_samples - 1]
    test_example = sampled_examples[num_samples - 1]

    return {
        "train": train_examples,
        "test": [test_example],
    }
    
def sample_datapoints_from_normal_task_no_replacement(
    task: TaskDict,
    num_samples: int = 4,
    num_datapoints: int = 100000,
    num_overlapping: int = 0,
) -> List[DataPointDict]:
    assert 0 <= num_overlapping < num_samples, "num_overlapping must be in [0, num_samples)"
    
    examples = task["examples"]
    total_available = len(examples)
    
    stride = num_samples - num_overlapping
    max_datapoints = (total_available - num_samples) // stride + 1
    actual_datapoints = min(num_datapoints, max_datapoints)
    
    # Shuffle before sliding window
    shuffled = random.sample(examples, total_available)
    
    datapoints: List[DataPointDict] = []
    
    for i in range(actual_datapoints):
        start_index = i * stride
        end_index = start_index + num_samples
        
        if end_index > total_available:
            break  # avoid index error in last window
        
        sampled_examples = shuffled[start_index:end_index]
        train_examples = sampled_examples[:num_samples - 1]
        test_example = sampled_examples[num_samples - 1]
        
        datapoints.append({
            "train": train_examples,
            "test": [test_example],
        })
    
    return datapoints
def sample_from_multiple_normal_tasks(
    tasks: List[TaskDict],
    num_samples: int = 4,
    num_datapoints_per_task: int = 50,
    num_overlapping: int = 0,
    replace: bool = False,
) -> List[DataPointDict]:
    if replace:
        return [
            sample_datapoints_from_normal_task(task, num_samples=num_samples)
            for task in tasks
            for _ in range(num_datapoints_per_task)
        ]
    else:
        all_datapoints = []
        for task in tasks:
            sampled_datapoints = sample_datapoints_from_normal_task_no_replacement(
                task, num_samples=num_samples, num_datapoints=num_datapoints_per_task, num_overlapping=num_overlapping
            )
            all_datapoints.extend(sampled_datapoints)
        return all_datapoints
    
def datapoint_to_prompt_completion_pair(
    datapoint: DataPointDict,
    tokenizer: PreTrainedTokenizer,
    input_start: str,
    input_end: str,
    output_end: str,
    preprompt: str = "",
) -> PromptCompletionPair:
    input_text = format_prompt_messages(datapoint, tokenizer, input_start, input_end, output_end, preprompt)
    
    if datapoint["test"][0]["output"] is not None:
        test_output_grid = datapoint["test"][0]["output"]
        label_text = stringify_grid_output(test_output_grid, end = output_end)
    else:
        label_text = ""
    
    full_text = input_text + label_text

    ret = dict()
    ret["train"] = input_text
    ret["reply"] = label_text
    ret["text"] = full_text

    return ret

def save_augmented_dataset_to_jsonl(
    dataset: HFDataset,
    save_path: str,
):
    with open(save_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
            
def load_augmented_dataset_from_jsonl(
    dataset_path: str,
) -> HFDataset:
    example_list = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            example_list.append(json.loads(line.strip()))
    return HFDataset.from_list(example_list)
    # return HFDataset.from_json(dataset_path, keep_in_memory=True)

def stringify_grid_input(grid: Grid, start: str = "\n", end: str = "\n") -> str:
    grid_str = "\n".join("".join(str(cell) for cell in row) for row in grid)
    return start + grid_str + end

def stringify_grid_output(grid: Grid, end: str = "\n") -> str:
    grid_str = "\n".join("".join(str(cell) for cell in row) for row in grid)
    return grid_str + end

def stringify_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def gridify_grid(grid: str) -> Grid:
    data = [[int(x) for x in row if x.isdigit()] for row in grid.strip().split("\n")]
    return [row for row in data if len(row)]

def format_prompt_messages(
    datapoint: DataPointDict,
    tokenizer: PreTrainedTokenizer,
    input_start: str,
    input_end: str,
    output_end: str,
    preprompt: str = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz",
) -> list[ChatEntry]:
    examples_block = preprompt
    for i, ex in enumerate(datapoint["train"], start=1):
        in_txt  = stringify_grid_input(ex["input"], start = input_start, end = input_end)
        out_txt = stringify_grid_output(ex["output"], end = output_end)
        examples_block += in_txt + out_txt

    test_in_txt = stringify_grid_input(datapoint["test"][0]["input"], start = input_start, end = input_end)

    return examples_block + test_in_txt

def is_peft_checkpoint_path(checkpoint_path: str) -> bool:
    """
    Check if the checkpoint path is a PEFT checkpoint.
    """
    return os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json"))

def create_n_minus_1_dataset(examples: List[ExampleDict]) -> List[DataPointDict]:
    new_dataset = []
    for i in range(len(examples)):
        new_example = {
            "train": examples[:i] + examples[i+1:],
            "test": [examples[i]],
        }
        new_dataset.append(new_example)
    return new_dataset

def print_grid(grid: Grid):
    for i, row in enumerate(grid):
        prefix = "[[" if i == 0 else " ["
        suffix = "]]" if i == len(grid) - 1 else "]"
        print(prefix + " ".join(str(cell) for cell in row) + suffix)
    print()
    
def chunked(it: List, n: int) -> Iterator[List]:
    it = iter(it)
    return iter(lambda: list(islice(it, n)), [])

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
            
class InputMaskingDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, mask_first_n_examples=0, **kwargs):
        super().__init__(**kwargs)
        self.mask_first_n_examples = mask_first_n_examples

    def torch_call(self, examples):
        batch = super().torch_call(examples)  # call super, masking all inputs
        eos_id = self.tokenizer.eos_token_id
        eos_mask = batch["input_ids"] == eos_id
        batch["labels"][eos_mask] = eos_id
        
        for i in range(len(batch['labels'])):
            # 마스킹되지 않은 값(-100이 아닌 값)이 있는지 확인
            if (batch['labels'][i] != -100).any():
                for _ in range(self.mask_first_n_examples):
                    # mask first still unmasked output block
                    nonzero_indices = (batch['labels'][i] != -100).nonzero()
                    if nonzero_indices.numel() == 0:  # 모든 값이 -100인 경우
                        break
                    
                    beg_pos = nonzero_indices.min().item()
                    
                    # 첫 번째 마스킹된 위치 찾기
                    masked_indices = (batch['labels'][i][beg_pos:] == -100).nonzero()
                    if masked_indices.numel() == 0:  # 뒷부분에 -100이 없는 경우
                        batch['labels'][i][beg_pos:] = -100  # 모두 마스킹
                        break
                        
                    mid_pos = masked_indices.min().item() + beg_pos
                    
                    # 마지막 마스킹되지 않은 위치 찾기
                    last_nonzero = (batch['labels'][i] != -100).nonzero()
                    if last_nonzero.numel() == 0:  # 모든 값이 -100인 경우
                        break
                        
                    end_pos = last_nonzero.max().item() + 1
                    
                    if mid_pos < end_pos:
                        batch['labels'][i][beg_pos:mid_pos] = -100
        return batch