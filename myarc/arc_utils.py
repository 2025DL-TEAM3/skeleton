
import os, glob, json, time, random

from .datatypes import *

# system prompt
system_prompt = """You are a helpful AI assistant. Your job is to solve tasks from the Abstraction and Reasoning Challenge (ARC). 
The user will present you with sample input and output grids for each task. 
Your job will be to understand the transformation between the input and the output and apply it to the last input grid given by the user. 
The puzzle-like inputs and outputs present a grid where each square can be one of ten colors. A grid can be any height or width between 1x1 and 30x30.
The background of the grid is typically colored with 0.
The tasks from ARC are based on the following priors:

- Objectness: Objects persist and cannot appear or disappear without reason. Objects can interact or not depending on the circumstances.
- Goal-directed: Objects can be animate or inanimate. Some objects are "agents" - they have intentions and they pursue goals.
- Numbers & counting: Objects can be counted or sorted by their shape, appearance, or movement using basic mathematics like addition, subtraction, and comparison.
- Basic geometry & topology: Objects can be shapes like rectangles, triangles, and circles which can be mirrored, rotated, translated, deformed, combined, repeated, etc. Differences in distances can be detected.

The transformations between input and output should be based on these priors.
"""

# user prompt 1: examples
user_message_template1 = """Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.
There are {n} example{plural} in total. Here they are:

{examples}
"""

# user prompt 2: test input
user_message_template2 = (
    "Now apply that rule to this test input grid:\n"
    "{test_grid}"
)

# user prompt 3: output format
user_message_template3 = (
    "Only return the output grid (rows as digit sequences; each ending with a newline; no extra text or spaces):"
)

def load_json_normal_tasks(dataset_path: str) -> list[TaskDict]:
    json_file_paths = glob.glob(os.path.join(dataset_path, '*.json'))
    if not json_file_paths:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")

    print(f"Found {len(json_file_paths)} JSON files for normal tasks.")
    
    all_tasks = []
    for json_file_path in json_file_paths:
        task_id = os.path.basename(json_file_path).split(".")[0]
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
        
    print(f"Successfully loaded {len(all_tasks)} JSON files for normal tasks.")
    return all_tasks

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

def stringify_grid(grid: Grid) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)

def gridify_grid(grid: str) -> Grid:
    return [list(map(int, row)) for row in grid.split("\n") if row.strip()]

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