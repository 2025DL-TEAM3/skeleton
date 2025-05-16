import os
import datetime
import hydra
from wasabi import msg
from omegaconf import DictConfig, OmegaConf
from myarc import ARCSolver
from myarc.arc_dataset import TaskBatchSampler, ARCTrainDataset, ARCValidationDataset

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    msg.info("--- Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    if cfg.artifact_name is not None:
        train_artifacts_dir = os.path.join(cfg.artifacts_dir, cfg.artifact_name)
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        train_artifacts_dir = os.path.join(cfg.artifacts_dir, f"train-{now}")
    
    os.makedirs(train_artifacts_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(train_artifacts_dir, "config.yaml"))

    msg.info("Loading dataset...")
    train_dataset = ARCTrainDataset(
        dataset_path=cfg.dataset_dir,
        num_train_examples_per_normal_task=cfg.dataset.num_train_examples_per_task,
        num_datapoints_per_task=cfg.dataset.num_datapoints_per_task,
    )
    train_dataset_size = len(train_dataset)
    valdation_task_size = int(train_dataset_size * cfg.dataset.val_ratio / cfg.dataset.num_datapoints_per_task)
    validation_dataset = ARCValidationDataset(
        dataset_path=cfg.dataset_dir,
        num_train_examples_per_normal_task=cfg.dataset.num_train_examples_per_task,
        num_datapoints_per_task=cfg.dataset.num_datapoints_per_task,
        max_val_tasks=valdation_task_size,
    )
    msg.good(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(validation_dataset)}")

    msg.info("Initializing model...")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    solver = ARCSolver(
        # token=cfg.token, # TODO
        train_artifacts_dir=train_artifacts_dir,
        **model_config,
    )
    msg.good(f"Model: {solver.model_id}")

    msg.info("Starting training...")
    train_config = OmegaConf.to_container(cfg.train, resolve=True)
    solver.train(
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        **train_config,
    )
    msg.good("Training completed!")

if __name__ == "__main__":
    main()
