import torch
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import Dataset, DataLoader


def one_hot_encode(position: int, num_classes: int):
    """
    Returns a one hot encoded 1d row tensor: Tensor([1, num_classes])

    Usage:
        # this will one hot encode a 1d vector with 5 classes
        # the 'number one' will be placed at index 0
        
        > one_hot_encode(0, 5)
        [1, 0, 0, 0, 0]
    """
    zero_vector = torch.zeros(num_classes, dtype=torch.float32) 
    zero_vector[position] = 1.
    return zero_vector

def plot_image_dataset(dataset: Dataset, n_items: int=1, cmap='gray') -> None:
    """
    Plot 1 or more images
    dataset: torch Dataset object containing X and Y matrices
    n_items: Number of items to display in the plot. Default is 1
    """
    assert isinstance(dataset, Dataset), "Dataset provided is not a torch Dataset class"
    assert len(dataset) >= n_items, "number of items is greater than dataset length"
    fig = plt.figure(figsize=(8, 8))

    for i in range(n_items):
        img = dataset[i][0]
        # this is so we can display 1 channel images as well
        if len(img.shape) != 3:
            img = img[0]
        else:
            img = img.T
        fig.add_subplot(5, 4, i+1)
        plt.imshow(img)
    plt.show()

# def calculate_mean(data: DataLoader, axis = 1, n_batches: int = 1) -> torch.Tensor:
#     """
#     Calculates mean of a given dataloader.
#     Returns the average of the mean of n_batches on the defined axis
    
#     Example:
#     This will calculate the mean of 5 batches and average them.
#     In total it will return a 1d torch tensor with 3 values (because axis 1 is 3 - channels)
#     > data.shape # (60, 3, 224, 244)
#     > calculate_mean(data, 1, 5)

#     Input
#     -----

#     Output
#     ------
#     1d torch tensor with the number of values in the defined axis
#     """
#     assert isinstance(data, DataLoader), "data is not a DataLoader object"
#     iter_data = iter(data)

#     for i in range(n_batches):
#         for c in data.shape[axis]:
