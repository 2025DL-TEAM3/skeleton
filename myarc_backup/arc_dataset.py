import random
from datasets import Dataset as HFDataset

from . import arc_utils
from .datatypes import *

def build_hf_dataset(
    dataset_path: str, 
    reasoning_task_path: str,
    num_samples_per_normal_task: int = 4, # Note: reasoning tasks are already sampled with 4
    # num_steps_per_task: int = 50, # TODO
) -> HFDataset:
    if dataset_path is not None:
        normal_tasks = arc_utils.load_json_normal_tasks(dataset_path)
        def normal_datapoint_sampler(task: TaskDict) -> DataPointDict:
            return arc_utils.sample_datapoints_from_normal_task(task, num_samples=num_samples_per_normal_task)

        normal_datapoints = [
            normal_datapoint_sampler(task)
            for task in normal_tasks
        ]
    else:
        normal_datapoints = []
    print(f"Loaded {len(normal_datapoints)} normal datapoints from {dataset_path}")
        
    if reasoning_task_path is not None:
        reasoning_datapoints = [
            datapoint
            for task in arc_utils.load_json_reasoning_tasks(
                reasoning_task_path,
                ignore_wrong_teacher_output=False, # TODO
            )
            for datapoint in task["datapoints"]
        ]
    else:
        reasoning_datapoints = []
    print(f"Loaded {len(reasoning_datapoints)} reasoning datapoints from {reasoning_task_path}")

    all_datapoints = normal_datapoints + reasoning_datapoints
    random.shuffle(all_datapoints)

    hf_dataset = HFDataset.from_list([
        arc_utils.datapoint_to_prompt_completion_pair(datapoint)
        for datapoint in all_datapoints
    ])
    
    hf_dataset = hf_dataset.shuffle()

    return hf_dataset

