from abc import ABC, abstractmethod
from turtle import color
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

class Trainer:
    """
    Trainer superclass. All trainers should inherit from this
    and implement the train method.
    """
    def __init__(self, model: nn.Module, 
        loss_function: nn.modules.loss._Loss, 
        optimizer: optim.Optimizer, 
        auto_detect_device: bool =True, device: str = "") -> None:
        assert isinstance(model, torch.nn.Module), "NN model does not inherit from base Module class"
        assert isinstance(loss_function, torch.nn.modules.loss._Loss), "Provided loss function does not inherit from base _Loss class."
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer does not inherit from base Optimizer class"
        self.model = model
        self.optimizer = optimizer
        self.train_costs = {} # {epoch_num: value}
        self.dev_costs = {}
        self.test_costs = {}
        self.loss_function = loss_function

        if auto_detect_device and device != "":
            chosen_device = device
        else:
            chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Trainer using '{chosen_device}' device")
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

    def plot_costs(self) -> None:
        plt.plot(list(self.train_costs.keys()), list(self.train_costs.values()), color="blue", label="Train set")
        if len(self.dev_costs):
            plt.plot(list(self.dev_costs.keys()), list(self.dev_costs.values()), color="green", label="Dev set")
        if len(self.test_costs):
            plt.plot(list(self.test_costs.keys()), list(self.test_costs.values()), color="red", label="Test set")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

class SimpleTrainer(Trainer):
    """
    Simple Trainer. 
    This implements the basic 5-step training loop

    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
    """
    def test(self, data: DataLoader) -> float:
        """
        Runs all batches of the data through the model
        and returns the average cost
        """
        self.model.eval()
        dev_losses = list()
        for X, Y in data:
            # send tensors to device
            X, Y = X.to(self.device), Y.to(self.device)

            with torch.no_grad():
                y_pred = self.model(X)
                loss = self.loss_function(y_pred, Y)
                dev_losses.append(loss.item())

        return torch.tensor(dev_losses).mean().item()

    def train(self, 
        data_train: DataLoader,
        data_dev: DataLoader = None,
        data_test: DataLoader = None,
        epochs: int = 100,
        early_stoping: float = 0.0
    ) -> None:
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
                self.train_costs[epoch] = torch.tensor(train_losses).mean()
                train_accuracy = round(100 - (self.train_costs[epoch].item() * 100), 3)

                # Calculate dev set accuracy
                if data_dev:
                    self.dev_costs[epoch] = self.test(data_dev)
                
                # Calculate test set accuracy
                if data_test:
                    self.test_costs[epoch] = self.test(data_test)

                if early_stoping > 0 and train_accuracy >= early_stoping:
                    print("Trainig stoped due to early stoping")
                    break
                
                t.set_postfix(train_cost=f"{self.train_costs[epoch]:>5f}", 
                    train_accuracy=train_accuracy
                )