import os, sys
import datetime
import hydra

from wasabi import msg
from omegaconf import DictConfig, OmegaConf
from arc import ARCSolver
from arc.arc_dataset import build_train_val_dataset
from arc.arc_utils import Tee

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    assert cfg.artifact_name, "Artifact name must be specified in the config file."
    
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    train_artifacts_dir = os.path.join(cfg.artifacts_dir, cfg.artifact_name)
    os.makedirs(train_artifacts_dir, exist_ok=True)
    
    log_file_path = os.path.join(
        train_artifacts_dir, f"train-log-redirected.txt"
    )
    
    sys.stdout = sys.stderr = Tee(sys.stdout, open(log_file_path, "a"))
    
    msg.info("--- Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))

    OmegaConf.save(config=cfg, f=os.path.join(train_artifacts_dir, "config.yaml"))

    msg.info("Loading dataset...")
    datset_config = OmegaConf.to_container(cfg.dataset, resolve=True)
    train_dataset, val_dataset = build_train_val_dataset(
        dataset_path=cfg.dataset_dir,
        **datset_config,
    )
    msg.good(f"Train dataset size: {len(train_dataset)} ({100 * (1 - cfg.dataset.val_ratio):.2f}% of total)")
    msg.good(f"Validation dataset size: {len(val_dataset)} ({100 * cfg.dataset.val_ratio:.2f}% of total)")

    msg.info("Initializing model...")
    solver = ARCSolver(
        config_path=os.path.join(train_artifacts_dir, "config.yaml"),
    )
    msg.good(f"Model: {solver.model_id}, Lora Rank: {cfg.model.lora_rank}, Lora Alpha: {cfg.model.lora_alpha}")

    msg.info("Starting training...")
    train_config = OmegaConf.to_container(cfg.train, resolve=True)
    solver.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        **train_config,
    )
    msg.good("Training completed!")

if __name__ == "__main__":
    main()
