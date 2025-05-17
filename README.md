# Intro to Deep Learning Term Project
Team 03

# Quick Start

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Configuration

### Directory Hierarchy
```
workspace/
    - dataset/
    - skeleton/
        - conf/
        - artifacts/
            - <artifact_name>
                - checkpoints/
                    - checkpoint-0
                    - ...
                    - checkpoint-final
                - logs/
                - config.yaml
        - evaluation
        - arc/
        - train.py
        - predict.py
        - evaluate.py
```

### Artifact_name name
```yaml
artifact_name: train-test
```

### Paths
```yaml
# conf/config.yaml

# Define the root workspace directory
# can be used for kubernetes
workspace: /home/<username>/code/intro_dl/term_project
cache_dir: null

# Paths based on workspace
dataset_dir: ${workspace}/dataset
artifacts_dir: ${workspace}/skeleton/artifacts
evaluation_dir: ${workspace}/skeleton/evaluation
```

### Trains

See `conf/train.yaml`

## 3. Train

```bash
python train.py
python train.py artifact_name=my-train
```

# Code Structure

## Dataset

`task json` to `DatapointDict`
- Select task
- Sample 3 `ExampleDict` and 1 `ExampleDict` for train examples and test input, respectively

### Interface
```python
from datasets import Dataset as HFDataset
dataset: HFDataset
dataset[0]: DatapointDict # see datatypes.py
```

## Transformation

You should transform `DataPointDict` into `PromptCompletionPair` to train/ineference

```python
datapoint = {
    "train": [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [5, 6]]
        },
        ...
    ],
    "test": [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [5, 6]]
        }
    ]
}

prompt_completion_pair = {
    'prompt': {
        'role': 'user', 
        'content': "You're a smart puzzle solver. ... Test Input: 12\n34\n"
    }
    'completion': {
        'role': 'assistant',
        'content': '34\n56\n'
    }
}
```

### `DataTransform` class

You can acheive this in any way, by implementing a subclass for `DataTransform`

```python
class DataTransform(ABC):
    @abstractmethod
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, datapoint: DataPointDict) -> PromptCompletionPair:
        return self.transform(datapoint)

class DefaultFormatMessages(DataTransform):
    def transform(self, datapoint: DataPointDict) -> PromptCompletionPair:
        return arc_utils.datapoint_to_prompt_completion_pair(datapoint)
```

`DefaultFormatMessages` just concatenate train examples into prompt string (예전에 하던 `format_prmopt`와 유사)

You can:
- augment datapoints and sample train examples in them
- ... other techniques

### Apply custom transform class
You can apply your customized transform class by adding it into a paramter
```python
# ARCSolver.train() @ arc.py

...
trainer = ARCSFTTrainer(
    model=self.base_model if self.peft_model is None else self.peft_model,
    processing_class=self.tokenizer,
    train_dataset_builder=train_dataset_builder,
    train_dataset_transform=None, # TODO
    eval_dataset=eval_dataset,
    eval_dataset_transform=None, # TODO
    args=training_args,
    peft_config=peft_config,
    use_task_batch_sampler=use_task_batch_sampler,
)
```

## Trainer

1. map dataset using your transform (or default one)
2. **reload train dataset for each epoch**
3. instantiate `SFTTrainer`
4. *TODO* : Task Batch Sampling 구현하기