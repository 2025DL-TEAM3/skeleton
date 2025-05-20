import pytest
import random
import numpy as np
from copy import deepcopy

import os
import sys
# add the parent directory to the path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arc.data_augmentation import (
    geometric_augmentation,
    reverse_geometric_augmentation,
    get_random_geometric_augmentation_params,

    color_permutation,
    reverse_color_permutation,
    get_random_color_permutation_params,

    upscale,
    get_random_upscale_params,

    add_padding,
    get_random_padding_params,

    mirror,
    get_random_mirror_params,

    get_max_grid_shape,

    random_datapoint_augmentation,
    revesre_datapoint_augmentation,
    random_task_augmentation
)

# --- Unit tests for core augmentations ---

def test_geometric_and_reverse():
    grid = [[1, 2, 3], [4, 5, 6]]
    params = {'hflip': True, 'n_rotations_90': 2}
    aug = geometric_augmentation(grid, **params)
    rec = reverse_geometric_augmentation(aug, **params)
    assert rec == grid

    # random params roundtrip
    random.seed(42)
    params2 = get_random_geometric_augmentation_params()
    aug2 = geometric_augmentation(grid, **params2)
    rec2 = reverse_geometric_augmentation(aug2, **params2)
    assert rec2 == grid

@pytest.mark.parametrize("color_map", [
    {0: 5, 5: 0},
    dict(zip(range(10), reversed(range(10))))
])
def test_color_and_reverse(color_map):
    grid = [[0, 1, 9], [5, 0, 5]]
    perm = color_permutation(grid, color_map)
    rec = reverse_color_permutation(perm, color_map)
    assert rec == grid

    params = get_random_color_permutation_params()
    cmap = params['color_map']
    assert isinstance(cmap, dict)
    assert set(cmap.keys()) == set(range(10)) and set(cmap.values()) <= set(range(10))

@pytest.mark.parametrize("factor,expected", [
    ((2, 3), [[1,1,1,2,2,2], [1,1,1,2,2,2], [3,3,3,4,4,4], [3,3,3,4,4,4]]),
])
def test_upscale(factor, expected):
    grid = [[1, 2], [3, 4]]
    result = upscale(grid, factor)
    assert result == expected

    params = get_random_upscale_params((2,2), min_upscale=1, max_upscale=2)
    assert 'factor' in params and isinstance(params['factor'], tuple)


def test_padding_and_random():
    grid = [[1]]
    params = {'color': 9, 'size': (1, 2)}
    padded = add_padding(grid, **params)
    assert len(padded) == 1 + 2*1 and len(padded[0]) == 1 + 2*2
    assert padded[1][2] == 1

    rand_params = get_random_padding_params((1,1), same_size_probability=1.0)
    assert 'color' in rand_params and 'size' in rand_params
    c, s = rand_params['color'], rand_params['size']
    assert 0 <= c <= 9 and isinstance(s, tuple) and len(s) == 2


def test_mirror_variants():
    grid = [[1,2],[3,4]]
    vert = mirror(grid, axis='vertical', position=0)
    assert vert == [[1,2,2,1],[3,4,4,3]]

    hor = mirror(grid, axis='horizontal', position=1)
    assert hor == [[3,4],[1,2],[1,2],[3,4]]

    cp = mirror(grid, axis=None)
    assert cp == grid and cp is not grid

# --- Utility tests ---

def test_max_grid_shape():
    dp = {'train': [{'input': [[1,2],[3,4]]}], 'test': [{'input': [[5,6]]}]}
    assert get_max_grid_shape(dp) == (2, 2)
    dp2 = {'train': [{'input': [[1]]}], 'test': [{'input': [[2,3,4]]}]}
    assert get_max_grid_shape(dp2) == (1, 3)

# --- Integrated random datapoint / reverse roundtrip ---

def test_random_datapoint_roundtrip_no_swap():
    dp = {
        'train': [{'input': [[0]], 'output': [[1]]}],
        'test':  [{'input': [[2]], 'output': [[3]]}]
    }
    aug, params = random_datapoint_augmentation(dp, swap_train_and_test=False)
    rec = revesre_datapoint_augmentation(aug, params)
    assert rec == dp

# --- reversed augmentation error on unknown key ---

def test_reverse_unknown_augmentation_raises():
    dp = {
        'train': [{'input': [[1]], 'output': [[1]]}],
        'test':  [{'input': [[2]], 'output': [[2]]}]
    }
    params = {'foo': (lambda g, **kwargs: g, {})}
    with pytest.raises(ValueError):
        revesre_datapoint_augmentation(dp, params)

# --- random task augmentation ---

def test_random_task_augmentation_changes_one_side():
    dp = {
        'train': [{'input': [[1]], 'output': [[2]]}],
        'test':  [{'input': [[3]], 'output': [[4]]}]
    }
    aug = random_task_augmentation(dp)
    for split in ('train', 'test'):
        orig = dp[split][0]
        a = aug[split][0]
        has_in_diff = orig['input'] != a['input']
        has_out_diff = orig['output'] != a['output']
        assert has_in_diff ^ has_out_diff

if __name__ == "__main__":
    pytest.main(["-v", __file__])