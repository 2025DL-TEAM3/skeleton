sample_datapoint = {
    'train': [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[1, 2], [3, 4]]
        },
        {
            'input': [[5, 6], [7, 8]],
            'output': [[5, 6], [7, 8]]
        },
        
    ],
    'test': [
        {
            'input': [[9, 8], [7, 6]],
            'output': [[9, 8], [7, 6]]
        }
    ]
}

from myarc import data_augmentation

def _print_grid(grid):
    for row in grid:
        print(" ".join(str(cell) for cell in row))
    print()

def _print_datapoint(datapoint):
    print("Train Examples:")
    for example in datapoint['train']:
        print("Input:")
        _print_grid(example['input'])
        print("Output:")
        _print_grid(example['output'])
    
    print("Test Examples:")
    for example in datapoint['test']:
        print("Input:")
        _print_grid(example['input'])
        print("Output:")
        _print_grid(example['output'])


from pprint import pprint
from myarc import data_transform
print("Original Datapoint:")
_print_datapoint(sample_datapoint)

random_augmentation_transform = data_transform.RandomAugmentationTransform()


import time

start_time = time.time() 
for i in range(1):

    # sample_grid = [[1, 2], [3, 4]]

    # print("Original grid:")
    # _print_grid(sample_grid)

    # print("Augmented grid (Color):")
    # augmented_grid_color = data_augmentation.color_permutation(sample_grid, **data_augmentation.get_random_color_permutation_params())
    # _print_grid(augmented_grid_color)

    # print("Augmented grid (Geometric):")
    # augmented_grid_geometric = data_augmentation.geometric_augmentation(sample_grid, **data_augmentation.get_random_geometric_augmentation_params())
    # _print_grid(augmented_grid_geometric)

    # print("Augmented grid (Upscale):")
    # augmented_grid_upscale = data_augmentation.upscale(sample_grid, **data_augmentation.get_random_upscale_params(data_augmentation.get_max_grid_shape(sample_datapoint)))
    # _print_grid(augmented_grid_upscale)

    # print("Augmented grid (Padding):")
    # augmented_grid_padding = data_augmentation.add_padding(sample_grid, **data_augmentation.get_random_padding_params(data_augmentation.get_max_grid_shape(sample_datapoint)))
    # _print_grid(augmented_grid_padding)

    # print("Augmented grid (Mirror):")
    # augmented_grid_mirror = data_augmentation.mirror(sample_grid, **data_augmentation.get_random_mirror_params(data_augmentation.get_max_grid_shape(sample_datapoint)))   
    # _print_grid(augmented_grid_mirror)
    
    aug = random_augmentation_transform(sample_datapoint)

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")