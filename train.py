import os
import datetime
import hydra
from datasets import Dataset as HFDataset

from wasabi import msg
from omegaconf import DictConfig, OmegaConf
from myarc import ARCSolver
from myarc.arc_dataset import build_train_val_dataset

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
    train_dataset, val_dataset = build_train_val_dataset(
        dataset_path=cfg.dataset_dir,
        num_train_examples_per_normal_task=cfg.dataset.num_train_examples_per_task,
        num_datapoints_per_task=cfg.dataset.num_datapoints_per_task * cfg.train.num_epochs, # TODO
        val_ratio=cfg.dataset.val_ratio,
        return_type="hf", # TODO
    )
    msg.good(f"Train dataset size: {len(train_dataset)} ({100 * (1 - cfg.dataset.val_ratio):.2f}% of total)")
    msg.good(f"Validation dataset size: {len(val_dataset)} ({100 * cfg.dataset.val_ratio:.2f}% of total)")

    msg.info("Initializing model...")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    solver = ARCSolver(
        # token=cfg.token, # TODO
        train_artifacts_dir=train_artifacts_dir,
        cache_dir=cfg.cache_dir,
        **model_config,
    )
    msg.good(f"Model: {solver.model_id}, Lora Rank: {cfg.model.lora_rank}")

    msg.info("Starting training...")
    train_config = OmegaConf.to_container(cfg.train, resolve=True)
    # def train_dataset_builder() -> HFDataset: # TODO
    #     msg.info("Loading Training dataset...")
    #     train_dataset, _ = build_train_val_dataset(
    #         dataset_path=cfg.dataset_dir,
    #         num_train_examples_per_normal_task=cfg.dataset.num_train_examples_per_task,
    #         num_datapoints_per_task=cfg.dataset.num_datapoints_per_task,
    #         val_ratio=cfg.dataset.val_ratio,
    #         return_type="hf", # TODO
    #     )
    #     msg.good(f"Train dataset size: {len(train_dataset)} ({100 * (1 - cfg.dataset.val_ratio):.2f}% of total)")
    #     return train_dataset
    solver.train(
        train_dataset=train_dataset,
        # train_dataset_builder=train_dataset_builder,
        eval_dataset=val_dataset,
        **train_config,
    )
    msg.good("Training completed!")

if __name__ == "__main__":
    main()
