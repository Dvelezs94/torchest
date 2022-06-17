import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchest.trainer import SimpleTrainer
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchest.utils import plot_image_dataset


"""
Prepare Data
"""
## need to flatten because I get 28*28 vectors
train_data = datasets.FashionMNIST(
    root="../datasets/",
    train=True,
    download=True,
    transform=ToTensor(),
)

classes = train_data.classes.copy()

dev_test_data = datasets.FashionMNIST(
    root="../datasets/",
    train=False,
    download=True,
    transform=ToTensor(),
)

dev_test_data_size = len(dev_test_data)
dev_data_size = int(dev_test_data_size / 2)
test_data_size = dev_test_data_size - dev_data_size
dev_data, test_data = random_split(dev_test_data, [dev_data_size, test_data_size])
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size)
dev_dataloader = DataLoader(dev_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

"""
Visualiza data
"""
#plot_image_dataset(dev_data, 20)

"""
Create Network graph
"""
class DWSConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

cnnmodel = nn.Sequential(
    nn.Conv2d(1, 5, 3, padding=1),
    # first depthwise-separable conv
    DWSConv2d(5, 8),
    # second depthwise-separable conv
    DWSConv2d(8, 12),
    nn.MaxPool2d(2, 2),
    # flatten
    nn.Flatten(),
    # # linear layers
    nn.Linear(14 * 14 * 12, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

"""
Prepare trainer
"""
loss_function = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(cnnmodel.parameters(), lr=1e-2)
trainer = SimpleTrainer(cnnmodel, loss_function, optimizer, wandb_project_name="MNIST", wandb=True)

"""
Train
"""
trainer.train(data_train=train_dataloader, 
            data_dev=dev_dataloader, 
            data_test=test_dataloader, epochs=20)

trainer.plot_accuracy()
"""
Predict
"""
# pending