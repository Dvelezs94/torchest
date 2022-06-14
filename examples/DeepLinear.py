import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchest.trainer import SimpleTrainer
from torchest.datagen import spiral_datagen
from torch.utils.data import Dataset, DataLoader, random_split

def graph_spiral(x, y):
    for i in zip(x, y):
        plt.plot(i[0][0].item(), i[0][1].item(), 'ro', color=colors[i[1].item()])
    plt.show()

"""
Prepare the data
"""
class_num =  3
X, Y = spiral_datagen(450, class_num)

class SpiralData(Dataset):
    X: torch.Tensor
    y: torch.Tensor

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

total_data = SpiralData(X, Y)
train_data_size = int(len(total_data) * 0.8)
dev_data_size = int((len(total_data) - train_data_size) / 2)
test_data_size = int(len(total_data) - train_data_size - dev_data_size)
# split data into train, dev and test sets
train_data, dev_data, test_data = random_split(total_data, [train_data_size, dev_data_size, test_data_size])
# Dataloaders
train_dataloader = DataLoader(train_data, batch_size=100)
dev_dataloader = DataLoader(dev_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=100)

"""
Visualize data
"""
colors = ['blue', 'green', 'yellow'] # 1 color per class
# graph_spiral(X, Y)

"""
Build Neural Network model
"""
model = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, class_num)
)

"""
Prepare Trainer
"""
# oddly enough nn.CrossEntropyLoss also computes softmax
# which is why we don't need to specify softmax layer in the model
loss_fn = nn.CrossEntropyLoss()
learning_rate= 1e-2
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = SimpleTrainer(model, loss_fn, optimizer, name="Spiral", wandb=True)

"""
Train
"""
epochs = 500
trainer.train(data_train=train_dataloader, data_dev=dev_dataloader, data_test=test_dataloader, epochs=epochs)
# trainer.plot_accuracy()
