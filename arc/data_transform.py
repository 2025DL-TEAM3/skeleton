from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer

from . import arc_utils
from .datatypes import *
from .data_augmentation import *
from datasets import Dataset as HFDataset, concatenate_datasets

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
    def __init__(self, tokenizer: PreTrainedTokenizer, fmt_opts: dict, augmentations_names: Optional[List[str]] = None, swap_train_and_test: bool = True, apply_task_augmentation_probability: float = 0.5):
        self.tokenizer = tokenizer
        self.swap_train_and_test = swap_train_and_test
        self.apply_task_augmentation_probability = apply_task_augmentation_probability
        if augmentations_names is None:
            augmentations_names = ["geometric", "color"]
        self.augmentations_names = augmentations_names
        self.fmt_opts = fmt_opts

    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        augmented_datapoint, _ = random_datapoint_augmentation(datapoint, self.augmentations_names, self.swap_train_and_test)
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

def augment_for_ttt(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    fmt_opts: dict,
    swap_train_and_test: bool = True,
    num_proc: int = 4,
    num_repeat: int = 2,
) -> HFDataset:
    original_dataset = dataset.map(
        DefaultFormatMessages(
            tokenizer=tokenizer,
            fmt_opts=fmt_opts
        ),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    aug_datasets = [original_dataset]
    for _ in range(num_repeat):
        geometrically_augmented_dataset = dataset.map(
            RandomAugmentationTransform(
                tokenizer=tokenizer,
                fmt_opts=fmt_opts,
                augmentations_names=["geometric"],
                swap_train_and_test=swap_train_and_test
            ),
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )
        aug_datasets.append(geometrically_augmented_dataset)
    
    expanded_dataset = concatenate_datasets(aug_datasets)
    expanded_dataset = expanded_dataset.shuffle()

    return expanded_dataset