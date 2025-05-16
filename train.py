import os
import datetime
import hydra
from wasabi import msg
from omegaconf import DictConfig, OmegaConf
from myarc import ARCSolver
from myarc.arc_dataset import build_hf_dataset

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    msg.info("--- Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    if cfg.train_name is not None:
        train_artifacts_dir = os.path.join(cfg.artifacts_dir, cfg.train_name)
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        train_artifacts_dir = os.path.join(cfg.artifacts_dir, f"train-{now}")
    
    os.makedirs(train_artifacts_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(train_artifacts_dir, "config.yaml"))

    msg.info("Loading dataset...")
    hf_dataset = build_hf_dataset(
        dataset_path=cfg.dataset_dir,
        num_train_examples_per_normal_task=cfg.dataset.num_train_examples_per_task,
        num_steps_per_task=cfg.dataset.num_steps_per_task,
    )
    hf_dataset_splitted = hf_dataset.train_test_split(test_size=cfg.dataset.val_ratio)
    msg.good(f"Train dataset size: {len(hf_dataset_splitted['train'])}, Test dataset size: {len(hf_dataset_splitted['test'])}")

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
        train_dataset=hf_dataset_splitted["train"],
        eval_dataset=hf_dataset_splitted["test"],
        **train_config,
    )
    msg.good("Training completed!")

if __name__ == "__main__":
    main()
