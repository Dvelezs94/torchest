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
train_data = datasets.CIFAR10(
    root="datasets/",
    train=True,
    download=True,
    transform=ToTensor(),
)

classes = train_data.classes.copy()

dev_test_data = datasets.CIFAR10(
    root="datasets/",
    train=False,
    download=True,
    transform=ToTensor(),
)

dev_test_data_size = len(dev_test_data)
dev_data_size = int(dev_test_data_size / 2)
test_data_size = dev_test_data_size - dev_data_size
dev_data, test_data = random_split(dev_test_data, [dev_data_size, test_data_size])
batch_size = 128

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
cnnmodel = nn.Sequential(
    # first convolution
    nn.Conv2d(3, 6, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # second convolution
    nn.Conv2d(6, 16, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # flatten
    nn.Flatten(1),
    # # linear layers
    nn.Linear(576, 256),
    nn.ReLU(),
    nn.Linear(256, len(classes)),
)

"""
Prepare trainer
"""
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnnmodel.parameters(), lr=1e-2)
trainer = SimpleTrainer(cnnmodel, loss_function, optimizer, wandb_project_name="CIFAR10", wandb=False)

"""
Train
"""
trainer.train(data_train=train_dataloader, 
            data_dev=dev_dataloader, 
            data_test=test_dataloader, epochs=20)

# trainer.plot_accuracy()
"""
Predict
"""
# pending