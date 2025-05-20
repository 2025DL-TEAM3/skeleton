import glob
import json
import os
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
from transformers import PreTrainedTokenizerBase, AutoTokenizer

import numpy as np
from copy import deepcopy

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arc import arc_utils, data_augmentation
from arc.datatypes import *

def plot_grid(ax, grid: Grid, title: str):
    ax.imshow(grid, cmap='tab10', interpolation='none', vmin=0, vmax=9)
    ax.set_title(title)
    ax.axis('off')

def visualize_all_datapoints(
    datapoints: list[tuple[DataPointDict, str]],
    title: str = "Datapoint Variations Visualization"
):
    num_versions = len(datapoints)
    rows = 2 * num_versions
    cols = 4

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 1.5))
    fig.suptitle(title, fontsize=18)

    for i, (datapoint, label) in enumerate(datapoints):
        row_offset = i * 2

        for j in range(3):  # Train examples
            if j < len(datapoint["train"]):
                example = datapoint["train"][j]
                plot_grid(axes[row_offset, j], example["input"], f"{label} Train {j} In")
                plot_grid(axes[row_offset + 1, j], example["output"], f"{label} Train {j} Out")
            else:
                axes[row_offset, j].axis('off')
                axes[row_offset + 1, j].axis('off')

        # Test example
        if len(datapoint["test"]) > 0:
            test_example = datapoint["test"][0]
            plot_grid(axes[row_offset, 3], test_example["input"], f"{label} Test In")
            plot_grid(axes[row_offset + 1, 3], test_example["output"], f"{label} Test Out")
        else:
            axes[row_offset, 3].axis('off')
            axes[row_offset + 1, 3].axis('off')

    # Add horizontal lines to divide sectors
    for i in range(1, num_versions):
        y = (i * 2)  # Row between each version
        fig.subplots_adjust(hspace=0.5)
        for ax in axes[y]:
            ax.axhline(-0.5, color='black', linewidth=1.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _apply_chat_template(
    prompt_completion_pair: PromptCompletionPair,
    tokenizer: PreTrainedTokenizerBase
) -> FormattedPromptCompletionPair:
    last_role = prompt_completion_pair["prompt"][-1]["role"]
    if last_role == "user":
        add_generation_prompt = True
        continue_final_message = False
    elif last_role == "assistant":
        add_generation_prompt = False
        continue_final_message = True
    else:
        raise ValueError(f"Invalid role in the last message: {last_role}")
    prompt = tokenizer.apply_chat_template(
        prompt_completion_pair["prompt"],
        continue_final_message=continue_final_message,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    
    prompt_completion = tokenizer.apply_chat_template(
        prompt_completion_pair["prompt"] + prompt_completion_pair["completion"], tokenize=False
    )
    completion = prompt_completion[len(prompt):]
    
    return {
        "prompt": prompt,
        "completion": completion,
    }

def pretty_print_prompt_completion_pair(prompt_completion_pair: FormattedPromptCompletionPair):
    print("============ Prompt Completion Pair ============")
    print("------------ Prompt ------------")
    for message in prompt_completion_pair["prompt"]:
        print(f"{message['role']}: {message['content']}")
    print("\n------------ Completion ------------")
    for message in prompt_completion_pair["completion"]:
        print(f"{message['role']}: {message['content']}")
    print()

def pretty_print_chat_template_applied(concat: str):
    print("============ Chat Template Applied ============")
    print("------------ Prompt + Completion ------------")
    print(concat)
    print()

def pretty_print_tokenized(tokenized: dict):
    print("============ Tokenized ============")
    print("------------ Input IDs ------------")
    print(tokenized["input_ids"])
    print("\n------------ Attention Mask ------------")
    print(tokenized["attention_mask"])
    print("\n------------ Completion Mask ------------")
    print(tokenized["completion_mask"])
    print()

def pretty_print_tokenized_ids(input_ids: list[int], tokenizer: PreTrainedTokenizerBase):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print("============ convert_ids_to_tokens ============")
    for token, token_id in zip(tokens, input_ids):
        print(f"({token!r}: {token_id})", end=" ")
    print()
    print("\n------------ decode ------------")
    for token_id in input_ids:
        print(f"({tokenizer.decode([token_id])!r:>5}: {token_id})", end=" ")

def tokenize(
    formatted_prompt_completion_pair: FormattedPromptCompletionPair,
    tokenizer: PreTrainedTokenizerBase, 
    add_special_tokens: bool = True
):
    proccessed_prompt = tokenizer(
        text=formatted_prompt_completion_pair["prompt"],
        add_special_tokens=add_special_tokens,
    )
    processed = tokenizer(
        text=formatted_prompt_completion_pair["prompt"] + formatted_prompt_completion_pair["completion"],
        add_special_tokens=add_special_tokens,
    )
    
    prompt_ids = proccessed_prompt["input_ids"]
    prompt_completion_ids = processed["input_ids"]
    if not prompt_completion_ids[:len(prompt_ids)] == prompt_ids:
        raise ValueError("Prompt IDs do not match")
    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    processed = {**processed, "completion_mask": completion_mask}
    return processed

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    dataset_path = cfg.dataset_dir
    seed = 42
    num_train_examples_per_task = 3
    num_datapoints_per_task = 50
    visualize = False
    verbose = False
    random.seed(seed)
    
    json_file_paths = sorted(glob.glob(f"{dataset_path}/*.json"))
    tasks = arc_utils.load_tasks_from_paths(json_file_paths)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    
    datapoints = [
        arc_utils.sample_datapoints_from_normal_task(task, num_samples=num_train_examples_per_task + 1)
        for task in tasks
        for _ in range(num_datapoints_per_task)
    ]

    
    for i, datapoint in enumerate(datapoints):
        
        augmented_datapoint, params_map = data_augmentation.random_datapoint_augmentation(datapoint=datapoint, swap_train_and_test=False)
        
        reversed_datapoint = data_augmentation.revesre_datapoint_augmentation(augmented_datapoint, params_map)
        
        assert datapoint == reversed_datapoint, "Reversed datapoint does not match original"
        
        if visualize:
            visualize_all_datapoints(
                datapoints=[
                    (datapoint, "Original"),
                    (augmented_datapoint, "Augmented"),
                    (reversed_datapoint, "Reversed"),
                ],
                title=f"Datapoint {i + 1} Variations"
            )
        
        prompt_completion_pair = arc_utils.datapoint_to_prompt_completion_pair(datapoint)
        
        if verbose: pretty_print_prompt_completion_pair(prompt_completion_pair)
        
        chat_template_applied = _apply_chat_template(prompt_completion_pair, tokenizer)
        if verbose: pretty_print_chat_template_applied(chat_template_applied['prompt'] + chat_template_applied['completion'])
        
        tokenized = tokenize(chat_template_applied, tokenizer, False)
        if verbose: pretty_print_tokenized(tokenized)
        if verbose: pretty_print_tokenized_ids(tokenized["input_ids"], tokenizer)
        
        decoded_with_special_tokens = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)
        # pretty_print_chat_template_applied(decoded_with_special_tokens)
        assert decoded_with_special_tokens == chat_template_applied['prompt'] + chat_template_applied['completion'], "Decoded text does not match original"

    # plt.tight_layout()
    # plt.show() 

if __name__ == "__main__":
    main()