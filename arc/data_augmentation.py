### Copied from https://github.com/ironbar/arc24/blob/main/scripts/arc24/data_augmentation.py
from pprint import pprint
from copy import deepcopy
import random
import numpy as np
from typing import Literal, Callable, Optional, List
from functools import partial

from .datatypes import *

MAX_GRID_SIZE = 30 # 10 is for original, 30 for augmented

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def get_max_grid_shape(datapoint: DataPointDict) -> tuple[int, int]:
    all_inputs = [ex['input'] for ex in datapoint['train']] + [ex['input'] for ex in datapoint['test']]
    max_rows = max(len(g) for g in all_inputs)
    max_cols = max(len(g[0]) for g in all_inputs)
    return max_rows, max_cols

def _get_grid_augmentation_map() -> dict[str,tuple[Callable[[Grid], Grid], Callable]]:
    return {
        "geometric": (geometric_augmentation, get_random_geometric_augmentation_params),
        "color": (color_permutation, get_random_color_permutation_params),
    }

def grid_augmentation(grid: Grid, params_map: dict, augmentations_names: list[str]) -> Grid:
    for aug_name in augmentations_names:
        func, kwargs = params_map[aug_name]
        grid = func(grid, **kwargs)
    return grid

def random_datapoint_augmentation(
    datapoint: DataPointDict, 
    swap_train_and_test: bool = False,
    augmentations_names: Optional[List[str]] = None
) -> tuple[DataPointDict, dict]:
    """Same augmentation for every grid"""
    if augmentations_names is None:
        augmentations_names = ["geometric", "color"]
    params_map = dict()
    for key in augmentations_names:
        func, kwarg_generator = _get_grid_augmentation_map()[key]
        kwargs = kwarg_generator()
        params_map[key] = (func, kwargs)
    
    _augment_fn = partial(grid_augmentation, params_map=params_map, augmentations_names=augmentations_names)
    
    new_train = [_augment_example_dict(example, _augment_fn) for example in datapoint['train']]
    new_test = [_augment_example_dict(example, _augment_fn) for example in datapoint['test']]

    if swap_train_and_test:
        augmented_datapoint = random_swap_train_and_test({
            'train': new_test,
            'test': new_train
        })
    else:
        augmented_datapoint = {
            'train': new_train,
            'test': new_test
        }
    return augmented_datapoint, params_map

def reverse_grid_augmentation(grid: Grid, params_map: dict, augmentations_names: Optional[list[str]] = None, skip_names: Optional[list[str]] = None) -> Grid:
    if augmentations_names is None:
        augmentations_names = list(params_map.keys())
    
    for aug_name in augmentations_names:
        if skip_names is not None and aug_name in skip_names:
            continue
        
        _, kwargs = params_map[aug_name]
        
        reverse_func = None
        if aug_name == "geometric":
            reverse_func = reverse_geometric_augmentation
        elif aug_name == "color":
            reverse_func = reverse_color_permutation
        else:
            raise ValueError(f"Unknown augmentation: {aug_name}")
        grid = reverse_func(grid, **kwargs)
    return grid

def revesre_datapoint_augmentation(datapoint: DataPointDict, params_map: dict) -> DataPointDict:
    augmentations_names = list(params_map.keys())
    
    _reverse_augment_fn = partial(reverse_grid_augmentation, params_map=params_map, augmentations_names=augmentations_names)

    new_train = [_augment_example_dict(example, _reverse_augment_fn) for example in datapoint['train']]
    new_test = [_augment_example_dict(example, _reverse_augment_fn) for example in datapoint['test']]
    
    reversed_datapoint = {
        'train': new_train,
        'test': new_test
    }
    return reversed_datapoint

def _augment_example_dict(example: ExampleDict, augment: Callable[[Grid], Grid], targets: List[Literal["input", "output"]] = None) -> ExampleDict:
    augmented_example = deepcopy(example)
    if targets is None:
        targets = ['input', 'output']
    
    for target in targets:
        if target not in example:
            raise ValueError(f"Target '{target}' not found in example.")
        if example.get(target) is None:
            continue
        
        augmented_example[target] = augment(example[target])

    return augmented_example

### Per-Datapoint Augmentation Functions ###
def random_swap_train_and_test(datapoint: DataPointDict) -> DataPointDict:
    augmented_datapoint = datapoint
    all_examples = augmented_datapoint['train'] + augmented_datapoint['test']
    random.shuffle(all_examples)
    train_size = len(augmented_datapoint['train'])
    new_train = all_examples[:train_size]
    new_test = all_examples[train_size:]
    
    augmented_datapoint = {
        'train': new_train,
        'test': new_test
    }
    return augmented_datapoint

### Per-Grid Augmentation Functions ###
def color_permutation(grid: Grid, color_map: dict[int, int]) -> Grid:
    grid_np = np.array(grid, dtype=np.int16)
    
    lookup_table = np.arange(10, dtype=np.int16)
    for old_color, new_color in color_map.items():
        lookup_table[old_color] = new_color
    grid_np = lookup_table[grid_np]
    return grid_np.tolist()

def reverse_color_permutation(grid: Grid, color_map: dict[int, int]) -> Grid:
    grid_np = np.array(grid, dtype=np.int16)
    
    lookup_table = np.arange(10, dtype=np.int16)
    for old_color, new_color in color_map.items():
        lookup_table[new_color] = old_color
    grid_np = lookup_table[grid_np]
    return grid_np.tolist()

def get_random_color_permutation_params(change_background_probability: float = 0.1):
    colors = list(range(10))
    if random.random() < change_background_probability:
        new_colors = list(range(10))
        random.shuffle(new_colors)
    else:
        new_colors = list(range(1, 10))
        random.shuffle(new_colors)
        new_colors = [0] + new_colors
    
    color_map = {old: new for old, new in zip(colors, new_colors)}
    return dict(color_map=color_map)

def geometric_augmentation(grid: Grid, hflip: bool = True, n_rotations_90: int = 0) -> Grid:
    grid_np = np.array(grid)
    if hflip:
        grid_np = np.flip(grid_np, axis=1)
    grid_np = np.rot90(grid_np, k=n_rotations_90)
    return grid_np.tolist()

def reverse_geometric_augmentation(grid: Grid, hflip: bool = True, n_rotations_90: int = 0) -> Grid:
    grid_np = np.array(grid)
    if n_rotations_90 > 0:
        grid_np = np.rot90(grid_np, k=-n_rotations_90)
    if hflip:
        grid_np = np.flip(grid_np, axis=1)
    return grid_np.tolist()

def get_random_geometric_augmentation_params():
    return dict(hflip=random.choice([True, False]), n_rotations_90=random.randint(0, 3))
