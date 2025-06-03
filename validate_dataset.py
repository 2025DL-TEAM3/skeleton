import os
import json

def is_valid_grid(grid):
    return isinstance(grid, list) and all(
        isinstance(row, list) and all(isinstance(cell, int) for cell in row) for row in grid
    )

def is_valid_entry(entry):
    return (
        isinstance(entry, dict) and
        ("input" in entry and "output" in entry) and
        is_valid_grid(entry["input"]) and
        is_valid_grid(entry["output"])
    )

def validate_file(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return False, "Root is not a list"
        for entry in data:
            if not is_valid_entry(entry):
                return False, f"Invalid entry: {entry}"
        return True, ""
    except Exception as e:
        return False, str(e)

# Update this path if necessary
folder = "../dataset_dark_eval"
all_valid = True

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        filepath = os.path.join(folder, filename)
        valid, error = validate_file(filepath)
        if not valid:
            print(f"❌ {filename} is invalid: {error}")
            all_valid = False

if all_valid:
    print("✅ All files are valid.")
else:
    print("❌ Some files are invalid.")


eval_dataset = "../dataset_eval100"
eval_task_ids = []
for json_name in os.listdir(eval_dataset):
    if json_name.endswith(".json"):
        task_id = json_name.split(".")[0]
        eval_task_ids.append(task_id)

original_dataset_dir = "../dataset"
original_task_ids = []
for json_name in os.listdir(original_dataset_dir):
    if json_name.endswith(".json"):
        task_id = json_name.split(".")[0]
        original_task_ids.append(task_id)

additional_dataset_dir = "../dataset_40"
additional_task_ids = []
for json_name in os.listdir(additional_dataset_dir):
    if json_name.endswith(".json"):
        task_id = json_name.split(".")[0]
        additional_task_ids.append(task_id)
        
print(f"Total eval tasks: {len(eval_task_ids)}")
print(f"Total original tasks: {len(original_task_ids)}")
print(f"Total additional tasks: {len(additional_task_ids)}")

print(f"{'Original tasks in eval: ' + str(len(set(eval_task_ids) & set(original_task_ids)))}")
print(f"{'Additional tasks in eval: ' + str(len(set(eval_task_ids) & set(additional_task_ids)))}")

num_original_in_eval = 0
for task_id in eval_task_ids:
    if task_id in original_task_ids:
        num_original_in_eval += 1
print(f"Number of original tasks in eval: {num_original_in_eval}")