import random
from abc import ABC, abstractmethod
from datasets import Dataset as HFDataset, concatenate_datasets
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
        self.fmt_opts = fmt_opts
    
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        input_start = self.fmt_opts.get("input_start", "")
        input_end = self.fmt_opts.get("input_end", "")
        output_end = self.fmt_opts.get("output_end", "")
        preprompt = self.fmt_opts.get("preprompt", "")
        return arc_utils.datapoint_to_prompt_completion_pair(datapoint, tokenizer=self.tokenizer, input_start=input_start, input_end=input_end, output_end=output_end, preprompt=preprompt)
    

class RandomAugmentationTransform(DataTransform):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        fmt_opts: dict, 
        augmentation_names: Optional[list[str]] = None,
        swap_train_and_test: bool = True, 
        apply_task_augmentation_probability: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.swap_train_and_test = swap_train_and_test
        self.apply_task_augmentation_probability = apply_task_augmentation_probability
        self.fmt_opts = fmt_opts
        self.augmentation_names = augmentation_names

    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        augmented_datapoint, _ = random_datapoint_augmentation(datapoint, self.swap_train_and_test, self.augmentation_names)
        input_start = self.fmt_opts.get("input_start", "")
        input_end = self.fmt_opts.get("input_end", "")
        output_end = self.fmt_opts.get("output_end", "")
        preprompt = self.fmt_opts.get("preprompt", "")
        return arc_utils.datapoint_to_prompt_completion_pair(augmented_datapoint, tokenizer=self.tokenizer, input_start=input_start, input_end=input_end, output_end=output_end, preprompt=preprompt)

def augment_and_expand(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    fmt_opts: dict,
    augmentation_names: Optional[list[str]] = None,
    swap_train_and_test: bool = True,
    num_proc: int = 4
):
    if augmentation_names is None:
        augmentation_names = ["geometric", "color"]
    
    # geometric augmentation
    geometrically_augmented_dataset = dataset.map(
        RandomAugmentationTransform(
            tokenizer=tokenizer,
            fmt_opts=fmt_opts,
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
            tokenizer=tokenizer,
            fmt_opts=fmt_opts,
            augmentation_names=["color"],
            swap_train_and_test=swap_train_and_test
        ),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Color augmentation"
    )
    
    original_dataset = dataset.map(
        DefaultFormatMessages(
            tokenizer=tokenizer,
            fmt_opts=fmt_opts
        ),
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