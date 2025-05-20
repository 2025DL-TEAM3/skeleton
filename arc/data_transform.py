from abc import ABC, abstractmethod

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
    def __init__(self, swap_train_and_test: bool = True, apply_task_augmentation_probability: float = 0.5):
        self.swap_train_and_test = swap_train_and_test
        self.apply_task_augmentation_probability = apply_task_augmentation_probability

    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        augmented_datapoint, _ = random_datapoint_augmentation(datapoint, self.swap_train_and_test)
        if random.random() < self.apply_task_augmentation_probability:
            augmented_datapoint = random_task_augmentation(augmented_datapoint)
        return arc_utils.datapoint_to_prompt_completion_pair(augmented_datapoint)

def get_data_transform(use_data_augmentation: bool) -> DataTransform:
    if use_data_augmentation:
        return RandomAugmentationTransform()
    else:
        return DefaultFormatMessages()