"""
ResNet 50/101/152 implementstion (bottleneck residual blocks) implementation

This sample uses stanford cars dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchest.trainer import SimpleTrainer
from torchvision import datasets, transforms
from torchest.utils import plot_image_dataset

"""
Prepare Data
"""
IMAGE_SIZE = 224

train_data = datasets.StanfordCars(
    root ="../datasets/",
    split ='train',
    download = True,
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.469, 0.529, 0.554],
            std=[0.247, 0.235, 0.236]
            )
    ])
)

classes = train_data.classes.copy()

dev_test_data = datasets.StanfordCars(
    root = "../datasets/",
    split = 'test',
    download = True,
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.469, 0.529, 0.554],
            std=[0.247, 0.235, 0.236]
            )
    ])
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
Prepare model
"""
class VGGConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_conv: int) -> None:
        """
        VGG block which contains 'n_conv'  same convolutions and a Max pool layer in the end
        """
        super().__init__()
        self.convs = self._make_layers(in_channels, out_channels, n_conv)
        self.pool = nn.MaxPool2d(2, 2)
    
    def _make_layers(self, in_channels: int, out_channels: int, n_conv: int):
        layers = list()
        # generate first convolution which might receive a different input channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
        layers.append(nn.ReLU())
        # all further layers have same input and same output channels 
        for _ in range(n_conv - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
            layers.append(nn.ReLU())

        # we return the sequential module with unpacked layers
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.pool(x)
        return x

class VGG16(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        """
        in_channels (required): number of channels of input
        num_classes (required): number of classes
        """
        super().__init__()
        self.model = nn.Sequential(
            VGGConvBlock(in_channels, 64, 2),
            VGGConvBlock(64, 128, 2),
            VGGConvBlock(128, 256, 3),
            VGGConvBlock(256, 512, 3),
            VGGConvBlock(512, 512, 3),
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x 

"""
Train model
"""
vgg16model = VGG16(3, len(classes))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16model.parameters(), lr=1e-2)

trainer = SimpleTrainer(vgg16model, loss_function, optimizer, name='VGG', wandb=False)

trainer.train(train_dataloader, dev_dataloader, test_dataloader)