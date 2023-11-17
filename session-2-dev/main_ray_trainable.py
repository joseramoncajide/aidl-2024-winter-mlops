from __future__ import print_function

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler

from model import MyModel
from filelock import FileLock

from datasplit import DataSplit


class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.labels_df.loc[idx, :]
        path = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        sample = Image.open(path)
        if self.transform:
            sample = self.transform(sample)

        return sample, code-1

data_transform = transforms.Compose([transforms.ToTensor()])


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=2):

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):

        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True)

        val_loader = DataLoader(
            val_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True)

        test_loader = DataLoader(
            test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True)
    
    return train_loader, val_loader, test_loader

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
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    return correct / total

class TrainChineseMNIST(tune.Trainable):
    
    # https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable/trainable.py
    def setup(self, config):
        
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        use_mps = not args.no_mps and torch.backends.mps.is_available()

        if use_cuda:
            self.device = torch.device("cuda")
        elif use_mps:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(self.device)

        data_transform = transforms.Compose([transforms.ToTensor()])

        # Specify the path to the CSV file and the root directory of the images
        labels_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
        images_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path

        # Create an instance of your custom dataset
        dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)
        
        # Split the dataset into train, validation, and test sets
        split_sizes = [10000, 2500, 2500]
        train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)
        
        self.train_loader, self.val_loader, self.test_loader  = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = config["batch_size"], num_workers=4)
        
        self.model = MyModel(num_classes=config["num_classes"], num_units=config["num_units"]).to(self.device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))
        

    def step(self):
        train_single_epoch(self.model, self.optimizer, self.train_loader, device=self.device)
        acc = eval_single_epoch(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}
    
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    args = parser.parse_args()
    if args.ray_address:
        ray.init(address=args.ray_address, num_cpus=8, num_gpus=4)
    else:
        ray.init(local_mode=True, num_cpus=8, num_gpus=4)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")
    analysis = tune.run(
        TrainChineseMNIST,
        name="TrainChineseMNIST",
        # metric="mean_accuracy",
        scheduler=sched,
        mode="max",
        stop={
            "mean_accuracy": 0.98,
            "training_iteration": 5 if args.smoke_test else 100
        },
        resources_per_trial={
            "cpu": 4,
            "gpu": 2 #int(args.cuda)
        },
        checkpoint_freq=1,
        num_samples=1 if args.smoke_test else 50,
        config={
            "lr": tune.grid_search([1e-4, 1e-2]),
            "momentum": tune.grid_search([0.1, 0.9]),
            "batch_size": 64,
            "epochs": 8,
            "num_classes": 15,
            "num_units": 500
        },
        # Note that restore vs resume, where restore is for a specific checkpoint of a single run, while resume is for an entire experiment. tune.run(..., restore=trial_checkpoint_path)
        # resume=True,
        # local_dir = "/Users/jcajidefernandez/ray_results/TrainChineseMNIST_2023-11-16_22-41-06",
    )

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()

    # TODO https://docs.ray.io/en/latest/tune/getting-started.html#evaluating-your-model-after-tuning