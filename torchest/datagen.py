"""
Useful functions for generating data to test neural networks
"""
import numpy as np
import torch

# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def spiral_datagen(samples: int, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns X and y vectors with specified number of samples * classes
    
    Usage: 
        x, y = spiral_datagen(200, 2) # 400 samples with 2 classes

    To print with pyplot use:
        colors = ['blue', 'green']
        for i in zip(x, y):
            plt.plot(i[0][0].item(), i[0][1].item(), 'ro', color=colors[i[1].item()])
        plt.show()
    """
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    X = torch.from_numpy(np.float32(X))
    y = torch.from_numpy(y)
    return X, y