from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer

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
    def __init__(self, tokenizer: PreTrainedTokenizer, fmt_opts: dict):
        self.tokenizer = tokenizer
    
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        input_start = self.fmt_opts.get("input_start", "")
        input_end = self.fmt_opts.get("input_end", "")
        output_end = self.fmt_opts.get("output_end", "")
        preprompt = self.fmt_opts.get("preprompt", "")
        return arc_utils.datapoint_to_prompt_completion_pair(datapoint, tokenizer=self.tokenizer, input_start=input_start, input_end=input_end, output_end=output_end, preprompt=preprompt)
    

class RandomAugmentationTransform(DataTransform):
    def __init__(self, tokenizer: PreTrainedTokenizer, fmt_opts: dict, swap_train_and_test: bool = True, apply_task_augmentation_probability: float = 0.5):
        self.tokenizer = tokenizer
        self.swap_train_and_test = swap_train_and_test
        self.apply_task_augmentation_probability = apply_task_augmentation_probability
        self.fmt_opts = fmt_opts

    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        augmented_datapoint, _ = random_datapoint_augmentation(datapoint, self.swap_train_and_test)
        # if random.random() < self.apply_task_augmentation_probability: # Note: padd/upscale does not help training
        #     augmented_datapoint = random_task_augmentation(augmented_datapoint)
        input_start = self.fmt_opts.get("input_start", "")
        input_end = self.fmt_opts.get("input_end", "")
        output_end = self.fmt_opts.get("output_end", "")
        preprompt = self.fmt_opts.get("preprompt", "")
        return arc_utils.datapoint_to_prompt_completion_pair(augmented_datapoint, tokenizer=self.tokenizer, input_start=input_start, input_end=input_end, output_end=output_end, preprompt=preprompt)

def get_data_transform(use_data_augmentation: bool, tokenizer: PreTrainedTokenizer, fmt_opts: dict) -> DataTransform:
    if use_data_augmentation:
        return RandomAugmentationTransform(tokenizer, fmt_opts)
    else:
        return DefaultFormatMessages(tokenizer, fmt_opts)