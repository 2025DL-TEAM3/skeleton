import argparse
import traceback
import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
import json
import random

from myarc import ARCSolver
from myarc import arc_utils

def shape_accuracy(prediction, ground_truth):
    """
    Calculate shape accuracy - shapes of prediction and ground truth must match
    """
    pred_shape = np.array(prediction).shape
    true_shape = np.array(ground_truth).shape
    
    return pred_shape == true_shape

def is_correct(prediction, ground_truth):
    """
    Calculate grid accuracy - if prediction and ground truth are the same
    """
    pred_np = np.array(prediction)
    true_np = np.array(ground_truth)
    
    if len(pred_np.shape) != 2 or pred_np.shape != true_np.shape:
        return 0
    else:
        return int(np.all(pred_np == true_np))

def cell_accuracy(prediction, ground_truth):
    """
    Calculate cell-wise accuracy - average accuracy of each cell
    """
    pred_np = np.array(prediction).flatten()
    true_np = np.array(ground_truth).flatten()
    
    # If shapes are different, truncate to the smaller one
    min_len = min(len(pred_np), len(true_np))
    pred_np = pred_np[:min_len]
    true_np = true_np[:min_len]
    
    # Calculate cell-wise accuracy
    return accuracy_score(true_np, pred_np)

def visualize_example(task_id, test_input, ground_truth, prediction):
    """Visualize the results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input grid
    axes[0].imshow(test_input, cmap='viridis', vmin=0, vmax=9)
    axes[0].set_title('Input')
    for i in range(len(test_input)):
        for j in range(len(test_input[0])):
            axes[0].text(j, i, str(test_input[i][j]), 
                         ha='center', va='center', color='white')
    
    # Ground truth
    true_array = np.array(ground_truth)
    axes[1].imshow(true_array, cmap='viridis', vmin=0, vmax=9)
    axes[1].set_title('Ground Truth')
    for i in range(len(true_array)):
        for j in range(len(true_array[0])):
            axes[1].text(j, i, str(true_array[i][j]), 
                         ha='center', va='center', color='white')
    
    # Prediction
    pred_array = np.array(prediction)
    axes[2].imshow(pred_array, cmap='viridis', vmin=0, vmax=9)
    axes[2].set_title('Prediction')
    for i in range(len(pred_array)):
        for j in range(len(pred_array[0])):
            axes[2].text(j, i, str(pred_array[i][j]), 
                         ha='center', va='center', color='white')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{task_id}_visualization.png")
    plt.close()

def evaluate(cfg: DictConfig):
    train_artifacts_dir = os.path.join(cfg.artifacts_dir, cfg.train_name)

    solver = ARCSolver(
        # token=args.token, # TODO
        train_artifacts_dir=train_artifacts_dir,
    )
    solver.prepare_evaluation(
        checkpoint_name=cfg.predict.checkpoint_name,
        enable_ttt=cfg.predict.enable_ttt,
        enable_thinking=cfg.predict.enable_thinking,
    )
    
    all_tasks = arc_utils.load_json_normal_tasks(cfg.dataset)
    
    random.seed(42)
    
    num_tasks_to_evaluate = min(cfg.evaluate.num_tasks_to_evaluate, len(all_tasks))
    print(f"Evaluating {num_tasks_to_evaluate} examples...")
    
    selected_tasks = random.sample(all_tasks, num_tasks_to_evaluate)
    
    results = []
    accruacies = []
    shape_accuracies = []
    cell_accuracies = []
    
    for i, task in enumerate(selected_tasks):
        print(f"#### Evaluating example {i+1}/{num_tasks_to_evaluate}...")
        task_id = task["task_id"]
        print(f"--- Task ID: {task_id} ---")

        sampled_examples = random.sample(task["examples"], 4)
        train_examples = sampled_examples[:3]
        test_example = sampled_examples[3]
        
        test_input = test_example['input']
        ground_truth = test_example['output']
        ground_truth = np.array(ground_truth)
        print(f"ground_truth: \n{ground_truth}")
        
        try:
            prediction = solver.predict(train_examples, test_input)
            print(f"prediction: \n{prediction}")
            
            correct = is_correct(prediction, ground_truth)
            shape_acc = shape_accuracy(prediction, ground_truth)
            cell_acc = cell_accuracy(prediction, ground_truth)
            
            accruacies.append(correct)
            shape_accuracies.append(shape_acc)
            cell_accuracies.append(cell_acc)
            
            result = {
                "task_id": task_id,
                "correct": correct,
                "shape_accuracy": shape_acc,
                "cell_accuracy": cell_acc,
                "input_shape": np.array(test_input).shape,
                "ground_truth_shape": np.array(ground_truth).shape,
                "prediction_shape": np.array(prediction).shape,
            }
            
            results.append(result)
            
            
            print(f"Correct: {correct}")
            print(f"Shape Accuracy: {shape_acc}")
            print(f"Cell Accuracy: {cell_acc}")
            print(f"Input Shape: {result['input_shape']}")
            print(f"Ground Truth Shape: {result['ground_truth_shape']}")
            print(f"Prediction Shape: {result['prediction_shape']}")
            
            if cfg.evaluate.visualize:
                visualize_example(task_id, test_input, ground_truth, prediction)
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            traceback.print_exc()
            continue
        
    if results:
        avg_accuracy = np.mean(accruacies)
        avg_shape_accuracy = np.mean(shape_accuracies)
        avg_cell_accuracy = np.mean(cell_accuracies)

        print("\n===== Evaluation Results =====")
        print(f"Number of evaluated examples: {len(results)}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Shape Accuracy: {avg_shape_accuracy:.4f}")
        print(f"Average Cell Accuracy: {avg_cell_accuracy:.4f}")
        
        os.makedirs(cfg.evaluation_dir, exist_ok=True)
        with open(f"{cfg.evaluation_dir}/result-{cfg.train_name}.json", "w") as f:
            json.dump({
                "model_checkpoint": cfg.predict.checkpoint_name,
                "num_examples": len(results),
                "average_accuracy": avg_accuracy,
                "average_shape_accuracy": avg_shape_accuracy,
                "average_cell_accuracy": avg_cell_accuracy,
                "results": results
            }, f, indent=4)
        
        print(f"Results saved to {cfg.evaluation_dir}/result-{cfg.train_name}.json")
    else:
        print("No valid results to save.")
        
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("--- Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))

    if not cfg.train_name:
        raise ValueError("train_name must be specified ot predict")

    evaluate(cfg)

if __name__ == "__main__":
    main()