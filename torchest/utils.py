import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


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
        img = dataset[i][0][0]
        fig.add_subplot(5, 4, i+1)
        plt.imshow(img, cmap=cmap)
    plt.show()
