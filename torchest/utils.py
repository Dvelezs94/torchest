import torch

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