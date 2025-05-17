from abc import ABC, abstractmethod

from . import arc_utils
from .datatypes import *

class DataTransform(ABC):
    @abstractmethod
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, datapoint: DataPointDict) -> PromptCompletionPair:
        return self.transform(datapoint)

class DefaultFormatMessages(DataTransform):
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        return arc_utils.datapoint_to_prompt_completion_pair(datapoint)