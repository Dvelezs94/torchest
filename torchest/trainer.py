from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import wandb as wdb
from tqdm import tqdm

class Trainer(ABC):
    """
    Trainer superclass. All trainers should inherit from this
    and implement the train method.
    """
    def __init__(self,
        model: nn.Module, 
        loss_function: nn.modules.loss._Loss, 
        optimizer: optim.Optimizer, 
        auto_detect_device: bool = True, 
        device: str = "",
        name: str = "",
        wandb: bool = False) -> None:
        """
        Inputs
        ------
        model (required): NN model (nn.Module) object
        loss_function (required): Loss function to minimize
        optimizer (required): Optimizer

        auto_detect_device (optional): Wether or not to detect GPU
        device (optional): force device to use (useful when oyu have more than 1 GPU)
        name (optional): Trainer name used for wandb
        wandb (optional): Wether or not to send metrics to wandb.
        """
        assert isinstance(model, torch.nn.Module), "NN model does not inherit from base Module class"
        assert isinstance(loss_function, torch.nn.modules.loss._Loss), "Provided loss function does not inherit from base _Loss class."
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer does not inherit from base Optimizer class"
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss = {}
        self.train_accuracy = {}
        self.dev_accuracy = {}
        self.test_accuracy = {}
        self.name = name
        self.wandb = wandb
        if self.wandb:
            self.setup_wandb()
        
        if auto_detect_device and device != "":
            chosen_device = device
        else:
            chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Trainer {self.name} using '{chosen_device}' device")
        self.device = chosen_device

    @abstractmethod
    def train(self, 
        X: torch.Tensor, 
        Y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 1,
        early_stoping: float = 0.0,
        **kwargs
    ) -> None:
        pass

    def setup_wandb(self):
        assert self.name, "Trainer name should be set in order to use wandb"
        model_layers = dict(self.model.named_modules())
        model_layers.pop('')
        model_layers = list(model_layers.values())
        optimizer_params = self.optimizer.defaults
        optimizer_params['name'] = type(self.optimizer).__name__
        n_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        wdb.init(project=self.name, config = {
            'optimizer': optimizer_params,
            'loss_function': self.loss_function,
            'layers': model_layers,
            'n_layers': len(model_layers),
            'n_trainable_params': n_trainable_params
        })
        wdb.watch(self.model)

    def calculate_accuracy(self, dataloader: DataLoader) -> float:
        """
        Runs all batches of the data through the model
        and returns the percentage of correct answers by the model

        Inputs
        ------
        dataloader (required): DataLoader object to calculate accuracy
        """
        total = len(dataloader.dataset)
        correct = 0
        self.model.eval()
        for batch, (item, label) in enumerate(dataloader):
            # send tensors to device
            item, label = item.to(self.device), label.to(self.device)

            with torch.no_grad():
                x = self.model(item)
                pred = torch.argmax(x, 1)
                pred = pred.data.cpu()
                batch_correct = (pred == label).int().sum().item()
                correct += batch_correct
        res = correct * 100. / total
        return res

    def plot_accuracy(self) -> None:
        """Plot train, dev and test accuracy"""
        plt.plot(list(self.train_accuracy.keys()), list(self.train_accuracy.values()), color="blue", label="Train set")
        plt.plot(list(self.dev_accuracy.keys()), list(self.dev_accuracy.values()), color="green", label="Dev set")
        plt.plot(list(self.test_accuracy.keys()), list(self.test_accuracy.values()), color="red", label="Test set")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("accuracy")
        plt.show()

class SimpleTrainer(Trainer):
    """
    Simple Trainer. 
    This implements the basic 5-step training loop

    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
    """

    def train(self, 
        data_train: DataLoader,
        data_dev: DataLoader = None,
        data_test: DataLoader = None,
        epochs: int = 100,
        early_stoping: float = 0.0
    ) -> None:
        """
        Automated training for the 5-step process in Pytorch

        Inputs
        ------
        data_train (required): Training Dataloader object

        data_dev (optional): Dev Dataloader object
        data_test (optional): Test dataloader object
        epochs (optional): Number of training epochs
        early_stoping (optional): Real value between 0 and 100 (percent). 
          It triggers the early stoping based on model accurady

        Outputs
        -------
        None   
        """
        self.model.to(self.device)
        # Train loop
        with tqdm(range(epochs), unit="epoch") as t:
            for epoch in t:
                self.model.train()
                train_losses = list()
                for X, Y in data_train:
                    # send tensors to device
                    X, Y = X.to(self.device), Y.to(self.device)

                    # 2. clear gradients
                    self.model.zero_grad()

                    # 2. Forward pass
                    y_pred = self.model(X)

                    # 3. compute loss
                    loss = self.loss_function(y_pred, Y)

                    # 4. Compte gradients gradients
                    loss.backward()

                    # 5. Adjust learnable parameters
                    self.optimizer.step()
                
                    train_losses.append(loss.detach().item())

                self.loss[epoch] = torch.tensor(train_losses).mean()
                self.train_accuracy[epoch] = self.calculate_accuracy(data_train)

                # Calculate dev set accuracy
                if data_dev:
                    self.dev_accuracy[epoch] = self.calculate_accuracy(data_dev)
                else:
                    self.dev_accuracy[epoch] = 0.0
                
                # Calculate test set accuracy
                if data_test:
                    self.test_accuracy[epoch] = self.calculate_accuracy(data_test)
                else:
                    self.test_accuracy[epoch] = 0.0

                if early_stoping > 0 and self.train_accuracy[epoch] >= early_stoping:
                    print("Trainig stoped due to early stoping")
                    break

                # wandb reports
                if self.wandb:
                    wdb.log({'loss': self.loss[epoch], 
                            'train_accuracy': self.train_accuracy[epoch],
                            'dev_accuracy': self.dev_accuracy[epoch],
                            'test_accuracy': self.test_accuracy[epoch]})
                            
                # tqdm progress bar
                t.set_postfix(epoch_loss=f"{self.loss[epoch]:>5f}", 
                    train_acc=f"{self.train_accuracy[epoch]:.2f}%",
                    dev_acc=f"{self.dev_accuracy[epoch]:.2f}%",
                    test_acc=f"{self.test_accuracy[epoch]:.2f}%",
                )