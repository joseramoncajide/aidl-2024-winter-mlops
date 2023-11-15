from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import random_split

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler

from dataset import MyDataset
from model import MyModel
from main import (train_single_epoch, eval_single_epoch, get_data_loaders)





def train_single_epoch(model, optimizer, data_loader, device=None):
    device = device or torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        total_loss += loss.item()

def eval_single_epoch(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct_predictions = 0
    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    # y_pred = pd.Series(all_predictions, name='Predicted')
    # y_actu = pd.Series(all_targets, name='Actual')
    # cm = pd.crosstab(y_actu, y_pred)
    
    avg_loss = total_loss / len(data_loader)
    # print(f'* avg_loss={avg_loss}')

    # accuracy_val = accuracy_score(y_actu, y_pred)
    # print(f'* accuracy_score={accuracy_val}')

    return avg_loss

class TrainChineseMNIST(tune.Trainable):
    def setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        data_transform = transforms.Compose([transforms.ToTensor()])

        # Specify the path to the CSV file and the root directory of the images
        labels_path = '/workspaces/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
        images_path = '/workspaces/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path

        # Create an instance of your custom dataset
        dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)
        # Split the dataset into train, validation, and test sets
        split_sizes = [10000, 2500, 2500]
        train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)
        self.train_loader, self.val_loader, self.test_loader  = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = config["batch_size"], num_workers=4)
        # self.train_loader, self.test_loader = get_data_loaders()
        self.model = MyModel(num_classes=config["num_classes"], num_units=config["num_units"]).to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))
        

    def step(self):
        train_single_epoch(
            self.model, self.optimizer, self.train_loader, device=self.device)
        acc = eval_single_epoch(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}
    
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        # self.experiment.log_asset(checkpoint_path, step=self._iteration)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    args = parser.parse_args()
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(local_mode=True)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")
    analysis = tune.run(
        TrainChineseMNIST,
        # metric="mean_accuracy",
        scheduler=sched,
        mode="max",
        stop={
            "mean_accuracy": 0.98,
            "training_iteration": 5 if args.smoke_test else 100
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": int(args.cuda)
        },
        checkpoint_freq=1,
        num_samples=1 if args.smoke_test else 50,
        config={
            "lr": tune.grid_search([1e-4, 1e-2]),
            "momentum": tune.grid_search([0.1, 0.9]),
            "batch_size": 64,
            "epochs": 10,
            "num_classes": 15,
            "num_units": 500
        },
    )

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))