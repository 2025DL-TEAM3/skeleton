### Copied from https://github.com/ironbar/arc24/blob/main/scripts/arc24/data_augmentation.py
from pprint import pprint
import random
import numpy as np
from typing import Literal, Callable
from functools import partial

from .datatypes import *

MAX_GRID_SIZE = 30 # 10 is for original, 30 for augmented

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def _print_grid(grid: Grid):
    for row in grid:
        print(" ".join(str(cell) for cell in row))
    print()

def get_max_grid_shape(datapoint: DataPointDict) -> tuple[int, int]:
    max_rows = max(len(example['input']) for example in datapoint['train'])
    max_cols = max(len(example['input'][0]) for example in datapoint['train'])
    return max_rows, max_cols

def _get_grid_augmentation_map() -> dict[str,tuple[Callable[[Grid], Grid], Callable]]:
    return {
        "geometric": (geometric_augmentation, get_random_geometric_augmentation_params),
        "color": (color_permutation, get_random_color_permutation_params),
        "upscale": (upscale, get_random_upscale_params),
        "padding": (add_padding, get_random_padding_params),
        "mirror": (mirror, get_random_mirror_params),
    }

def random_datapoint_augmentation(datapoint: DataPointDict, swap_train_and_test: bool = True) -> DataPointDict:
    """Same augmentation for every grid"""
    augmentations_names = ["geometric", "color"]
    params_map = dict()
    for key in augmentations_names:
        func, kwarg_generator = _get_grid_augmentation_map()[key]
        kwargs = kwarg_generator()
        params_map[key] = (func, kwargs)
    
    def augment(grid: Grid) -> Grid:
        for aug_name in augmentations_names:
            func, kwargs = params_map[aug_name]
            grid = func(grid, **kwargs)
        return grid
    
    augmented_datapoint = datapoint.copy()
    augmented_datapoint['train'] = [_augment_example_dict(example, augment) for example in datapoint['train']]
    augmented_datapoint['test'] = [_augment_example_dict(example, augment) for example in datapoint['test']]

    if swap_train_and_test:
        augmented_datapoint = random_swap_train_and_test(augmented_datapoint)
    return augmented_datapoint

def random_task_augmentation(datapoint: DataPointDict) -> DataPointDict:
    """Augment only one of input/output grids, resulting in different task"""
    augmentations_names = _get_grid_augmentation_map().keys()
    max_grid_shape = get_max_grid_shape(datapoint)
    params_map = dict()
    for aug_name, (func, kwarg_generator) in _get_grid_augmentation_map().items():
        if aug_name in ["upscale", "padding", "mirror"]:
            kwargs = kwarg_generator(max_grid_shape=max_grid_shape)
        else:
            kwargs = kwarg_generator()
        params_map[aug_name] = (func, kwargs)
    
    target = random.choice(['input', 'output'])
    
    def augment(grid: Grid) -> Grid:
        for aug_name in augmentations_names:
            func, kwargs = params_map[aug_name]
            grid = func(grid, **kwargs)
        return grid
    
    augmented_datapoint = datapoint.copy()
    augmented_datapoint['train'] = [_augment_example_dict(example, augment, [target]) for example in datapoint['train']]
    augmented_datapoint['test'] = [_augment_example_dict(example, augment, [target]) for example in datapoint['test']]
    
    # TODO: is this generated task valid?
    return augmented_datapoint

def _augment_example_dict(example: ExampleDict, augment: Callable[[Grid], Grid], targets: List[Literal["input", "output"]] = None) -> ExampleDict:
    augmented_example = example.copy()
    if targets is None:
        targets = ['input', 'output']
    
    for target in targets:
        if target not in example:
            raise ValueError(f"Target '{target}' not found in example.")
        augmented_example[target] = augment(example[target])

    return augmented_example

### Per-Datapoint Augmentation Functions ###
def permute_train_examples(datapoint: DataPointDict) -> DataPointDict:
    train_order = np.arange(len(datapoint['train']))
    np.random.shuffle(train_order)
    augmented_datapoint = datapoint.copy()
    augmented_datapoint['train'] = [datapoint['train'][i] for i in train_order]
    return augmented_datapoint

def random_swap_train_and_test(datapoint: DataPointDict) -> DataPointDict:
    augmented_datapoint = datapoint.copy()
    all_examples = augmented_datapoint['train'] + augmented_datapoint['test']
    random.shuffle(all_examples)
    train_size = len(augmented_datapoint['train'])
    augmented_datapoint['train'] = all_examples[:train_size]
    augmented_datapoint['test'] = all_examples[train_size:]
    return augmented_datapoint

### Per-Grid Augmentation Functions ###
def color_permutation(grid: Grid, color_map: dict[int, int]) -> Grid:
    grid_np = np.array(grid, dtype=np.int16)
    
    lookup_table = np.arange(10, dtype=np.int16)
    for old_color, new_color in color_map.items():
        lookup_table[old_color] = new_color
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

def get_random_geometric_augmentation_params():
    return dict(hflip=random.choice([True, False]), n_rotations_90=random.randint(0, 3))

def upscale(grid: Grid, factor: tuple[int, int]) -> Grid:
    grid_np = np.array(grid, dtype=np.int16)
    upscaled = np.repeat(np.repeat(grid_np, factor[0], axis=0), factor[1], axis=1)
    return upscaled.tolist()

def get_random_upscale_params(
    max_grid_shape: tuple[int, int], 
    min_upscale: int = 2,
    max_upscale: int = 4,
    same_upscale_probability: float = 0.5,
    n_tries: int = 10
):
    safe_max_upscale = (
        min(MAX_GRID_SIZE // max_grid_shape[0], max_upscale),
        min(MAX_GRID_SIZE // max_grid_shape[1], max_upscale)
    )
    
    if random.random() < same_upscale_probability:
        min_safe_max_upscale = min(safe_max_upscale)
        if min_safe_max_upscale < min_upscale:
            print("Warning: Grid is too large to upscale.")
            return 1
        _factor = random.randint(min_upscale, min_safe_max_upscale)
        return dict(factor=(_factor, _factor))
    else:
        if min(safe_max_upscale) < min_upscale:
            print("Warning: Grid is too large to upscale.")
            return 1
        factor = (1, 1)
        for _ in range(n_tries):
            factor = (
                random.randint(min_upscale, safe_max_upscale[0]),
                random.randint(min_upscale, safe_max_upscale[1])
            )
            if factor[0] != factor[1]:
                break
        return dict(factor=factor)

def add_padding(grid: Grid, color: int = 0, size: tuple[int, int] = (0, 0)) -> Grid:
    grid_np = np.array(grid, dtype=np.int16)
    padded = np.pad(
        grid_np,
        pad_width=((size[0], size[0]), (size[1], size[1])),
        mode='constant',
        constant_values=color
    )
    return padded.tolist()

def get_random_padding_params(
    max_grid_shape: tuple[int, int], 
    same_size_probability: float = 0.5,
    max_padding: int = 5,
    n_tries: int = 10
):
    safe_max_padding = (
        min(MAX_GRID_SIZE - max_grid_shape[0], max_padding),
        min(MAX_GRID_SIZE - max_grid_shape[1], max_padding)
    )
    
    if random.random() < same_size_probability:
        safe_max_padding = min(safe_max_padding)
        if safe_max_padding < 1:
            print("Warning: Grid is too large to add padding.")
            return (0, 0), (0, 0)
        size = random.randint(1, safe_max_padding)
        size = (size, size)
    else:
        if min(safe_max_padding) < 1:
            print("Warning: Grid is too large to add padding.")
            return (0, 0), (0, 0)
        for _ in range(n_tries):
            size = (random.randint(1, safe_max_padding[0]), random.randint(1, safe_max_padding[1]))
            if size[0] != size[1]:
                break
    color = random.randint(1, MAX_GRID_SIZE - 1)
    return dict(color=color, size=size)

def mirror(grid: Grid, axis: Literal["horizontal", "vertical"] | None = None, position: int = 0) -> Grid:
    if axis is None:
        return grid
    
    grid_np = np.array(grid)
    if axis == 'horizontal':
        mirrored = np.flip(grid_np, axis=0)
        if position == 0:
            merged = np.concatenate([grid_np, mirrored], axis=0)
        else:
            merged = np.concatenate([mirrored, grid_np], axis=0)
    elif axis == 'vertical':
        mirrored = np.flip(grid_np, axis=1)
        if position == 0:
            merged = np.concatenate([grid_np, mirrored], axis=1)
        else:
            merged = np.concatenate([mirrored, grid_np], axis=1)
    else:
        raise ValueError("Invalid axis. Choose 'horizontal' or 'vertical'.")
    return merged.tolist()

def get_random_mirror_params(max_grid_shape: tuple[int, int]):
    if MAX_GRID_SIZE // max_grid_shape[0] < 2:
        if MAX_GRID_SIZE // max_grid_shape[1] < 2:
            print("Warning: Grid is too large to add padding.")
            axis = None
        else:
            axis = 'vertical'
    elif MAX_GRID_SIZE // max_grid_shape[1] < 2:
        axis = 'horizontal'
    else:
        axis = random.choice(['horizontal', 'vertical'])
    return dict(axis=axis, position=random.randint(0, 1))