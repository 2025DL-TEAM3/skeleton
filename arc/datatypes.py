from typing import List, Union, Literal, Any, TypedDict
import torch

Grid = List[List[int]]

class ExampleDict(TypedDict):
    input: Grid
    output: Grid

class TestExampleDict(TypedDict):
    input: Grid
    

class DataPointDict(TypedDict):
    train: List[ExampleDict]
    test: List[ExampleDict]

class FormattedPrompt(TypedDict):
    input_ids: torch.Tensor
    input: Grid
    train: List[ExampleDict]

class TaskDict(TypedDict):
    file_path: str
    task_id: str
    examples: List[ExampleDict]

class ChatEntry(TypedDict):
    role: Literal["user", "assistant"]
    content: str

class PromptCompletionPair(TypedDict):
    prompt: List[ChatEntry]
    completion: List[ChatEntry]
    
class FormattedPromptCompletionPair(TypedDict):
    prompt: str
    completion: str