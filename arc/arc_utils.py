
import os, glob, json, time, random
from itertools import islice

from typing import List, Dict, Any, Tuple, Union, Optional, Iterator

from datasets import Dataset as HFDataset

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

def load_json_reasoning_tasks(
    dataset_path: str,
    ignore_wrong_teacher_output: bool = True,
) -> list[ReasoningTaskDict]:
    json_file_paths = glob.glob(os.path.join(dataset_path, '*.json'))
    if not json_file_paths:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")

    print(f"Found {len(json_file_paths)} JSON files for reasoning tasks.")

    task_id_datapoint_map: dict[str, List[ReasoningDataPointDict]] = {}
    for json_file_path in json_file_paths:
        try:
            with open(json_file_path, 'r') as f:
                task_json = json.load(f)
                task_id = task_json["task_id"]
                if task_id not in task_id_datapoint_map:
                    task_id_datapoint_map[task_id] = []
                
                train_indices = task_json["train_indices"]
                test_index = task_json["test_index"]
                datapoint = task_json["datapoint"]
                input_messages = task_json["input_messages"]
                output_text = task_json["output_text"]
                reasoning = task_json["reasoning"]
                test_output_text = task_json["test_output_text"]
                correct = task_json["correct"]

                datapoint["test"][0]["output"] = gridify_grid(test_output_text)
                reasoning_datapoint = {
                    **datapoint,
                    "reasoning": [reasoning],
                }

                if not correct:
                    print(f"Warning: Teacher output is incorrect. json_file_path: {json_file_path}", end="")
                    if ignore_wrong_teacher_output:
                        print(" - Ignoring this example.")
                        continue
                    else:
                        print(" - Adding this example anyway.")
                        task_id_datapoint_map[task_id].append(reasoning_datapoint)
                else:
                    task_id_datapoint_map[task_id].append(reasoning_datapoint)
        except Exception as e:
            print(f"Error loading file: {json_file_path} - {e}")
    if not task_id_datapoint_map:
        raise ValueError("No valid examples found in JSON files.")

    all_tasks = [
        {
            "task_id": task_id,
            "datapoints": task_id_datapoint_map[task_id],
        } 
        for task_id in task_id_datapoint_map
    ]

    print(f"Successfully loaded {len(all_tasks)} JSON files for reasoning tasks.")
    return all_tasks

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
) -> List[DataPointDict]:
    examples = task["examples"]
    total_available = len(examples)
    max_datapoints = total_available // num_samples
    
    actual_datapoints = min(num_datapoints, max_datapoints)
    # actual_datapoints = max_datapoints # XXX: use all available datapoints
    total_required = actual_datapoints * num_samples
    
    shuffled = random.sample(examples, total_required)
    datapoints: List[DataPointDict] = []
    
    for i in range(actual_datapoints):
        start_index = i * num_samples
        end_index = start_index + num_samples
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
                task, num_samples=num_samples, num_datapoints=num_datapoints_per_task
            )
            all_datapoints.extend(sampled_datapoints)
        return all_datapoints
    
def datapoint_to_prompt_completion_pair(
    datapoint: DataPointDict,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
) -> PromptCompletionPair:
    # TODO : for now, use only the first one
    test_output_grid = datapoint["test"][0]["output"]
    test_output_text = stringify_grid(test_output_grid)

    if "reasoning" in datapoint:
        reasoning = datapoint["reasoning"][0]
        reasoning_content = f"{reasoning_start_token} {reasoning} {reasoning_end_token}"
        test_output_text = f"{reasoning_content}\n{test_output_text}"

    input_messages = format_prompt_messages(datapoint)
    output_message = [
        {"role": "assistant", "content": test_output_text}
    ]

    return {
        "prompt": input_messages,
        "completion": output_message,
    }

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

def stringify_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def gridify_grid(grid: str) -> Grid:
    return [[int(cell) for cell in row.split()] for row in grid.strip().split("\n")]

# def gridify_grid(grid: str) -> Grid:
#     return [list(map(int, str(row))) for row in grid.strip().split("\n")]

def format_prompt_messages(datapoint: DataPointDict) -> list[ChatEntry]:
    train_examples = datapoint["train"]
    test_input = datapoint["test"][0]["input"] # TODO : for now, use only the first one

    n = len(train_examples)
    plural = 's' if n != 1 else ''
    examples_block = ''
    for i, ex in enumerate(train_examples, start=1):
        examples_block += f"Example {i} Input:\n"
        examples_block += stringify_grid(ex['input'])
        examples_block += f"\nExample {i} Output:\n"
        examples_block += stringify_grid(ex['output'])
        examples_block += "\n----------------------------------------\n"
    template1 = user_message_template1.format(n=n, plural=plural, examples=examples_block)

    test_input_block = f"Test Input:\n{stringify_grid(test_input)}"
    template2 = user_message_template2.format(test_grid=test_input_block)

    user_message = (
        f"{template1}\n"
        f"{template2}\n"
        f"{user_message_template3}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return messages


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