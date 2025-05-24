
import os, glob, json, time, random
from itertools import islice

from typing import List, Dict, Any, Tuple, Union, Optional, Iterator

from .datatypes import *

# system prompt
system_prompt = """\
You are an expert ARC solver. 
Given a few input→output grid examples, infer the transformation rule and apply it to the test grid.
Grids are up to 10×10, colors encoded as digits 0–9, background is 0."""

# ─── 2) User Prompt: Examples ────────────────────────────────────────────────
user_message_template1 = """\
Here are {n} example{plural}:

{examples_block}"""

# ─── 3) User Prompt: Test Input ─────────────────────────────────────────────
user_message_template2 = """\
Now apply the learned rule to this test input:
{test_grid}"""

# ─── 4) User Prompt: Output Format ───────────────────────────────────────────
user_message_template3 = "Only return the resulting grid as rows of digits (no spaces, no extra text):"

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

def datapoint_to_prompt_completion_pair(
    datapoint: DataPointDict,
) -> PromptCompletionPair:
    # TODO : for now, use only the first one
    test_output_grid = datapoint["test"][0]["output"]
    test_output_text = stringify_grid(test_output_grid)

    input_messages = format_prompt_messages(datapoint)
    output_message = [
        {"role": "assistant", "content": test_output_text}
    ]

    return {
        "prompt": input_messages,
        "completion": output_message,
    }

def stringify_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def gridify_grid(grid: str) -> Grid:
    return [[int(cell) for cell in row.split()] for row in grid.strip().split("\n")]

def format_prompt_messages(datapoint: DataPointDict) -> list[ChatEntry]:
    examples_block = ""
    for i, ex in enumerate(datapoint["train"], start=1):
        in_txt  = stringify_grid(ex["input"])
        out_txt = stringify_grid(ex["output"])
        examples_block += f"Example {i}:\nInput:\n{in_txt}\nOutput:\n{out_txt}\n\n"

    test_in_txt = stringify_grid(datapoint["test"][0]["input"])

    user_msg = "\n".join([
        user_message_template1.format(
            n=len(datapoint["train"]),
            plural="s" if len(datapoint["train"]) != 1 else "",
            examples_block=examples_block
        ),
        user_message_template2.format(test_grid=test_in_txt),
        user_message_template3
    ])

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg}
    ]


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