import numpy as np
from tqdm.auto import tqdm
import os
import multiprocessing
import traceback
import hydra
from omegaconf import DictConfig, OmegaConf
from wasabi import msg

from transformers import set_seed
from datasets import load_dataset
import pandas as pd
import json

from arc import arc_utils

import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def check_match(pred, truth):
    pred = np.array(pred, dtype=np.uint8)
    truth = np.array(truth, dtype=np.uint8)

    if len(pred.shape) != 2 or pred.shape != truth.shape:
        return 0
    else:
        return int(np.all(pred == truth))

def load_single_file(file_path):
    try:
        with open(file_path) as fp:
            return json.load(fp)
    except Exception as e:
        print(f"파일 로드 오류: {file_path} - {e}")
        return None

def load_data(base_dir, num_samples=1000, num_workers=1):
    filenames = os.listdir(base_dir)
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(load_single_file, data_files), total=len(data_files), desc="파일 로딩"))
        dataset = [data for data in results if data is not None]
    else:
        dataset = []
        for fn in tqdm(data_files, desc="파일 로딩"):
            try:
                with open(fn) as fp:
                    data = json.load(fp) 
                dataset.append(data)    
            except Exception as e:
                print(f"파일 로드 오류: {fn} - {e}")

    filenames = [fn.split(".")[0] for fn in filenames]  
    data = []
    MAX_LEN = num_samples  
    rng = np.random.default_rng(42)  

    N = len(dataset)

    while len(data) < MAX_LEN:
        task_idx = rng.integers(0, N)
        task = dataset[task_idx]
        file_name = filenames[task_idx]

        n_task = len(task) 
        grids_idx =  rng.choice(n_task, size=4, replace=True) 
        train_grids = [task[i] for i in grids_idx[:3]]       
        test_grids = [task[i] for i in grids_idx[3:]]         

        test_inputs = [{'input': grid['input']} for grid in test_grids]  
        test_outputs = [grid['output'] for grid in test_grids]           
        test_outputs_transformed = [{'output': grid} for grid in test_outputs]  
        combined_tests = []
        for test_input, test_output in zip(test_inputs, test_outputs_transformed):
            combined_tests.append({'input': test_input['input'], 'output': test_output['output']})  

        data.append({
            'task': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs,
            'test': combined_tests,
        })

    df = pd.DataFrame(data) 
    return df


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from arc import ARCSolver
    from datasets import Dataset

    artifacts_dir = cfg.artifacts_dir
    checkpoint_path = cfg.predict.checkpoint_path
    # rel_path = os.path.relpath(checkpoint_path, artifacts_dir)
    # artifact_name = rel_path.split(os.sep)[0]
    # checkpoint_name = rel_path.split(os.sep)[-1]
    eval_name = cfg.evaluate.eval_name
    log_file_path = os.path.join(
        cfg.evaluation_dir, f"{cfg.evaluate.eval_name}-log.txt"
    )

    sys.stdout = sys.stderr = Tee(sys.stdout, open(log_file_path, "a"))
    
    print("--- Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))
    
    if hasattr(cfg.evaluate, "seed") and cfg.evaluate.seed is not None:
        set_seed(cfg.evaluate.seed)

    msg.info("Loading data...")
    df = load_data(cfg.dataset_dir, num_samples=cfg.evaluate.num_samples, num_workers=cfg.evaluate.num_workers)
    msg.good(f"Loaded {len(df)} samples.")
    
    msg.info("Initializing solver...")
    
    pparent_dir = os.path.dirname(os.path.dirname(checkpoint_path)) # train_artifacts_dir
    config_path = os.path.join(pparent_dir, "config.yaml")
    print(f"Config path: {config_path}")
    solver = ARCSolver( # TODO : should support init without arguments
        config_path=config_path,
    )
    predict_config = OmegaConf.to_container(cfg.predict, resolve=True)
    solver.prepare_evaluation(
        **predict_config
    )
    
    eval_dataset = Dataset.from_pandas(df).shuffle(seed=42).select(range(min(cfg.evaluate.num_samples, len(df))))

    scores = []
    print(f"{len(eval_dataset)}개의 샘플에 대해 평가를 시작합니다...")
    for eval_data in tqdm(eval_dataset, desc="평가 진행"):
        try:
            test_input = eval_data["test"][0]["input"]
            print("Test Input:")
            arc_utils.print_grid(test_input)
            
            print("Ground Truth:")
            arc_utils.print_grid(eval_data["test"][0]["output"])
            
            if not isinstance(test_input, np.ndarray):
                test_input = np.array(test_input)
                
            preds = solver.predict(
                eval_data["train"],   
                test_input,   
            )
            print("Predictions:")
            arc_utils.print_grid(preds)
            s = check_match(preds, eval_data["test"][0]["output"])  
            scores.append(s)
            print(f"Score: {s}")
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            traceback.print_exc()
            continue
    
    if scores:
        score = np.array(scores).mean() * 100  
        print(f"Evaluation scores: {score:.2f}%", flush=True)
        print(f"성공한 평가 수: {len(scores)}/{len(eval_dataset)}")
    else:
        print("오류로 인해 평가 결과가 없습니다.")
    
    print("Evaluation Success")


if __name__ == "__main__":
    main()