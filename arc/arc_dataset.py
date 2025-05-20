import random
from typing import Literal, Callable
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, Sampler
import glob
import json
import os
from . import arc_utils
from .datatypes import *

def build_hf_train_val_dataset(
    dataset_path: str,
    num_train_examples_per_normal_task: int = 3,
    num_datapoints_per_task: int = 50,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[HFDataset, HFDataset]:
    json_file_paths = sorted(glob.glob(f"{dataset_path}/*.json"))
    random.shuffle(json_file_paths)
    split = int(len(json_file_paths) * (1 - val_ratio))
    train_file_paths, val_file_paths = json_file_paths[:split], json_file_paths[split:]
    
    print(f"Train tasks: {len(train_file_paths)}, Validation tasks: {len(val_file_paths)}")
    
    train_tasks = arc_utils.load_tasks_from_paths(train_file_paths)
    val_tasks = arc_utils.load_tasks_from_paths(val_file_paths)
    
    train_datapoints = [
        arc_utils.sample_datapoints_from_normal_task(task, num_samples=num_train_examples_per_normal_task + 1)
        for task in train_tasks
        for _ in range(num_datapoints_per_task)
    ]
    
    val_datapoints = [
        arc_utils.sample_datapoints_from_normal_task(task, num_samples=num_train_examples_per_normal_task + 1)
        for task in val_tasks
        for _ in range(num_datapoints_per_task)
    ]
    
    train_dataset = HFDataset.from_list(train_datapoints)
    val_dataset = HFDataset.from_list(val_datapoints)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Reset random seed
    random.seed()
    
    return train_dataset, val_dataset


def build_train_val_dataset(
    dataset_path: str | None = None,
    num_train_examples_per_task: int = 3,
    num_datapoints_per_task: int = 50,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[HFDataset, HFDataset]:
    return build_hf_train_val_dataset(
        dataset_path=dataset_path,
        num_train_examples_per_normal_task=num_train_examples_per_task,
        num_datapoints_per_task=num_datapoints_per_task,
        val_ratio=val_ratio,
        seed=seed,
    )
