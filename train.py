import os
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from myarc import ARCSolver
from myarc.arc_dataset import build_hf_dataset

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("--- Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))

    if cfg.train_name is not None:
        train_artifacts_dir = os.path.join(cfg.artifacts_dir, cfg.train_name)
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        train_artifacts_dir = os.path.join(cfg.artifacts_dir, f"train-{now}")

    print("Initializing model...")
    solver = ARCSolver(
        # token=cfg.token, # TODO
        model_id=cfg.model.model_id,
        train_artifacts_dir=train_artifacts_dir,
        cache_dir=cfg.cache_dir,
        lora_rank=cfg.model.lora_rank,
    )

    print("Loading dataset...")
    hf_dataset = build_hf_dataset(
        dataset_path=cfg.dataset,
        num_train_examples_per_normal_task=cfg.dataset.num_train_examples_per_task,
        num_steps_per_task=cfg.dataset.num_steps_per_task,
    )
    hf_dataset_splitted = hf_dataset.train_test_split(test_size=cfg.train.val_ratio)

    print("Starting training...")
    train_config = OmegaConf.to_container(cfg.train, resolve=True)
    solver.train(
        train_dataset=hf_dataset_splitted["train"],
        eval_dataset=hf_dataset_splitted["test"],
        **train_config,
    )

    print("Training completed!")

if __name__ == "__main__":
    main()
