import random
from abc import ABC, abstractmethod
from datasets import Dataset as HFDataset, concatenate_datasets

from . import arc_utils
from .datatypes import *
from .data_augmentation import *

class DataTransform(ABC):
    @abstractmethod
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, datapoint: DataPointDict) -> PromptCompletionPair:
        return self.transform(datapoint)

class DefaultFormatMessages(DataTransform):
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        return arc_utils.datapoint_to_prompt_completion_pair(datapoint)
    

class RandomAugmentationTransform(DataTransform):
    def __init__(
        self, 
        augmentation_names: Optional[list[str]] = None,
        swap_train_and_test: bool = True, 
        apply_task_augmentation_probability: float = 0.5
    ):
        self.augmentation_names = augmentation_names
        self.swap_train_and_test = swap_train_and_test
        self.apply_task_augmentation_probability = apply_task_augmentation_probability

    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        augmented_datapoint, _ = random_datapoint_augmentation(datapoint, self.swap_train_and_test, self.augmentation_names)
        # if random.random() < self.apply_task_augmentation_probability: # Note: padd/upscale does not help training
        #     augmented_datapoint = random_task_augmentation(augmented_datapoint)
        return arc_utils.datapoint_to_prompt_completion_pair(augmented_datapoint)

def augment_and_expand(
    dataset: HFDataset,
    augmentation_names: Optional[list[str]] = None,
    swap_train_and_test: bool = True,
    num_proc: int = 4
):
    if augmentation_names is None:
        augmentation_names = ["geometric", "color"]
    
    # geometric augmentation
    geometrically_augmented_dataset = dataset.map(
        RandomAugmentationTransform(
            augmentation_names=["geometric"],
            swap_train_and_test=swap_train_and_test
        ),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Geometric augmentation"
    )
    
    # color augmentation
    color_permuted_dataset = dataset.map(
        RandomAugmentationTransform(
            augmentation_names=["color"],
            swap_train_and_test=swap_train_and_test
        ),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Color augmentation"
    )
    
    original_dataset = dataset.map(
        DefaultFormatMessages(),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Original dataset formatting"
    )
    
    # combine original, geometric, and color augmented datasets
    expanded_dataset = concatenate_datasets([original_dataset, geometrically_augmented_dataset, color_permuted_dataset])
    
    # shuffle the dataset
    expanded_dataset = expanded_dataset.shuffle()
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Expanded dataset size: {len(expanded_dataset)}")
    return expanded_dataset