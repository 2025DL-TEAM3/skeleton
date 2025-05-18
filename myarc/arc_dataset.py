import random
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, Sampler

from . import arc_utils
from .datatypes import *

NORMAL_TASKS = []

def build_hf_dataset(
    dataset_path: str | None = None,
    reasoning_task_path: str | None = None,
    num_train_examples_per_normal_task: int = 3,
    num_datapoints_per_task: int = 50,
) -> HFDataset:
    global NORMAL_TASKS
    if dataset_path is not None:
        if not NORMAL_TASKS:
            NORMAL_TASKS = arc_utils.load_json_normal_tasks(dataset_path)
        normal_tasks = NORMAL_TASKS
        def normal_datapoint_sampler(task: TaskDict) -> DataPointDict:
            return arc_utils.sample_datapoints_from_normal_task(task, num_samples=num_train_examples_per_normal_task + 1) # 1 for test

        normal_datapoints = [
            normal_datapoint_sampler(task)
            for task in normal_tasks
            for _ in range(num_datapoints_per_task)
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

    # hf_dataset = HFDataset.from_list([
    #     arc_utils.datapoint_to_prompt_completion_pair(datapoint)
    #     for datapoint in all_datapoints
    # ])
    
    # hf_dataset = hf_dataset.shuffle()

    hf_dataset = HFDataset.from_list(all_datapoints) # Note: return datapoint instead of prompt-completion pair

    return hf_dataset

def build_hf_train_val_dataset(
    dataset_path: str | None = None,
    reasoning_task_path: str | None = None,
    num_train_examples_per_normal_task: int = 3,
    num_datapoints_per_task: int = 50,
    val_ratio: float = 0.1,
) -> tuple[HFDataset, HFDataset]:
    hf_dataset = build_hf_dataset(
        dataset_path=dataset_path,
        reasoning_task_path=reasoning_task_path,
        num_train_examples_per_normal_task=num_train_examples_per_normal_task,
        num_datapoints_per_task=num_datapoints_per_task,
    )
    splitted = hf_dataset.train_test_split(test_size=val_ratio)
    train_dataset = splitted["train"]
    val_dataset = splitted["test"]
    return train_dataset, val_dataset

def build_torch_train_val_dataset(
    dataset_path: str | None = None,
    reasoning_task_path: str | None = None,
    num_train_examples_per_normal_task: int = 3,
    num_datapoints_per_task_task: int = 50,
    val_ratio: float = 0.1,
) -> tuple["ARCTrainDataset", "ARCValidationDataset"]:
    train_dataset = ARCTrainDataset(
        dataset_path=dataset_path,
        num_train_examples_per_normal_task=num_train_examples_per_normal_task,
        num_datapoints_per_task=num_datapoints_per_task_task,
    )

    val_dataset = ARCValidationDataset(
        dataset_path=dataset_path,
        num_train_examples_per_normal_task=num_train_examples_per_normal_task,
        num_datapoints_per_task=num_datapoints_per_task_task,
        val_ratio=val_ratio,
    )

    return train_dataset, val_dataset

def build_train_val_dataset(
    dataset_path: str | None = None,
    reasoning_task_path: str | None = None,
    num_train_examples_per_normal_task: int = 3,
    num_datapoints_per_task: int = 50,
    val_ratio: float = 0.1,
    return_type: Literal["pt", "hf"] = "pt",
) -> tuple[HFDataset, HFDataset] | tuple["ARCTrainDataset", "ARCValidationDataset"]:
    if return_type == "pt":
        return build_torch_train_val_dataset(
            dataset_path=dataset_path,
            reasoning_task_path=reasoning_task_path,
            num_train_examples_per_normal_task=num_train_examples_per_normal_task,
            num_datapoints_per_task_task=num_datapoints_per_task,
            val_ratio=val_ratio,
        )
    elif return_type == "hf":
        return build_hf_train_val_dataset(
            dataset_path=dataset_path,
            reasoning_task_path=reasoning_task_path,
            num_train_examples_per_normal_task=num_train_examples_per_normal_task,
            num_datapoints_per_task=num_datapoints_per_task,
            val_ratio=val_ratio,
        )
    else:
        raise ValueError(f"Unknown return type: {return_type}")

class TaskBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_datapoints_per_task = self.dataset.num_datapoints_per_task
        self.num_tasks = len(self.dataset.normal_tasks)

    def __iter__(self):
        task_indices = list(range(self.num_tasks))
        random.shuffle(task_indices)
        for task_idx in task_indices:
            datapoint_indices = list(range(
                task_idx * self.num_datapoints_per_task,
                (task_idx + 1) * self.num_datapoints_per_task,
            ))
            random.shuffle(datapoint_indices)
            for i in range(0, len(datapoint_indices), self.batch_size):
                batch_indices = datapoint_indices[i:i + self.batch_size]
                yield batch_indices

    def __len__(self):
        per_task = (self.num_datapoints_per_task + self.batch_size - 1) // self.batch_size
        return self.num_tasks * per_task

class ARCTrainDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | None = None,
        num_train_examples_per_normal_task: int = 3,
        num_datapoints_per_task: int = 50,
    ):
        self.normal_tasks = arc_utils.load_json_normal_tasks(dataset_path)
        self.num_train_examples_per_normal_task = num_train_examples_per_normal_task
        self.num_datapoints_per_task = num_datapoints_per_task
        self.total_num_datapoints = len(self.normal_tasks) * num_datapoints_per_task

        print(f"Loaded total {len(self.normal_tasks)} tasks from {dataset_path}")
        print(f"Total # datapoints per epoch: {self.total_num_datapoints}")
    
    def __len__(self):
        return self.total_num_datapoints

    def __getitem__(self, idx: int):
        """
        Given an index, return a prompt-completion pair.

        idx: [0, total_num_datapoints)

        [0, 1, ..., 49] [50, 51, ..., 99] ...
        <--- task 0---> <--- task 1  ---> ...
        """
        task_idx = (idx // self.num_datapoints_per_task) % len(self.normal_tasks)
        task = self.normal_tasks[task_idx]

        # choose another task if insufficient examples
        # Note: should not happen
        while len(task["examples"]) < self.num_train_examples_per_normal_task + 1:
            random_task_idx = random.randint(0, len(self.normal_tasks) - 1)
            task = self.normal_tasks[random_task_idx]
            
        datapoint = arc_utils.sample_datapoints_from_normal_task(
            task,
            num_samples=self.num_train_examples_per_normal_task + 1, # 1 for test
        )
        # prompt_completion_pair = arc_utils.datapoint_to_prompt_completion_pair(datapoint)

        # return prompt_completion_pair

        return datapoint

class ARCValidationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | None = None,
        num_train_examples_per_normal_task: int = 3,
        num_datapoints_per_task: int = 50,
        val_ratio: float = 0.03,
        seed: int = 42,
    ):
        # no instance variable: free after init
        normal_tasks = arc_utils.load_json_normal_tasks(dataset_path) 
        self.num_train_examples_per_normal_task = num_train_examples_per_normal_task
        self.num_datapoints_per_task = num_datapoints_per_task
        self.total_num_datapoints = len(normal_tasks) * num_datapoints_per_task

        print(f"Loaded total {len(normal_tasks)} tasks from {dataset_path}")
        print(f"Total # datapoints per epoch: {self.total_num_datapoints}")

        self.seed = seed
        self.max_val_tasks = int(len(normal_tasks) * self.num_datapoints_per_task * val_ratio)
        self.validation_datapoints = self._prepare_validation_datapoints(normal_tasks)
    
    def _prepare_validation_datapoints(self, normal_tasks: list[TaskDict]) -> list[DataPointDict]:
        random.seed(self.seed)
        validation_datapoints = []

        sampled_task_indices = list(range(len(normal_tasks)))
        random.shuffle(sampled_task_indices)
        sampled_task_indices = sampled_task_indices[:self.max_val_tasks]
        
        for task_idx in sampled_task_indices:
            task = normal_tasks[task_idx]
            if len(task["examples"]) < self.num_train_examples_per_normal_task + 1:
                print(f"Warning: Task {task['task_id']} has insufficient examples. Skipping.")
                continue

            for _ in range(self.num_datapoints_per_task):
                datapoint = arc_utils.sample_datapoints_from_normal_task(
                    task,
                    num_samples=self.num_train_examples_per_normal_task + 1, # 1 for test
                )
                # prompt_completion_pair = arc_utils.datapoint_to_prompt_completion_pair(datapoint)
                # validation_datapoints.append(prompt_completion_pair)
                validation_datapoints.append(datapoint)
        
        random.seed()

        print(f"Loaded {len(validation_datapoints)} validation datapoints.")
        return validation_datapoints
    
    def __len__(self):
        return len(self.validation_datapoints)

    def __getitem__(self, idx: int):
        safe_idx = idx % len(self.validation_datapoints)
        return self.validation_datapoints[safe_idx]