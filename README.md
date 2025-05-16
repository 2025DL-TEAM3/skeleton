# Intro to Deep Learning Term Project
Team 03

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