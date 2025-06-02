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
