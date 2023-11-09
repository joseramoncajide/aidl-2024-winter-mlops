import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, random_split
# from torch.utils.data.sampler import SubsetRandomSampler

from dataset import ChineseMNISTDataset
from model import ChineseMNISTCNN

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_samples", help="amount of samples to train with", type=int, default=1000)
# parser.add_argument("--n_features", help="amount of features per sample", type=int, default=20)
# parser.add_argument("--n_hidden", help="amount of hidden neurons", type=int, default=128)
# parser.add_argument("--n_outputs", help="amount of outputs", type=int, default=15)
# parser.add_argument("--epochs", help="number of epochs to train", type=int, default=5)
# parser.add_argument("--batch_size", help="batch size", type=int, default=100)
# parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
# args = parser.parse_args()

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform (you can customize it as needed)
data_transform = transforms.Compose([transforms.ToTensor()])

# Specify the path to the CSV file and the root directory of the images
csv_file = '/workspace/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
root_dir = '/workspace/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path
chinese_mnist_dataset = ChineseMNISTDataset(csv_file=csv_file, root_dir=root_dir, transform=data_transform)

# Split the dataset into train, validation, and test sets
split_sizes = [10000, 2500, 2500]
train_dataset, val_dataset, test_dataset  = random_split(chinese_mnist_dataset, split_sizes)

def get_data_loaders(batch_size):
    train_loader = DataLoader(
        train_dataset.dataset,
        batch_size=batch_size,
        shuffle=True)

    val_loader = DataLoader(
        val_dataset.dataset,
        batch_size=batch_size,
        shuffle=False)

    test_loader = DataLoader(
        test_dataset.dataset,
        batch_size=batch_size,
        shuffle=False)
    
    return train_loader, val_loader, test_loader


# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

def train_func(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # loss_history.append(loss.item())


def test_func(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    
    r_labels = np.array([])
    correct_preds = np.array([])
    preds  = np.array([])

    from sklearn.metrics import accuracy_score
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            r_labels = np.concatenate((r_labels, target.numpy()))
            for index, item in enumerate(outputs):
                if r_labels[index] == torch.argmax(item):
                    correct_preds += 1

                preds  = np.concatenate((preds, torch.argmax(item).unsqueeze(-1).detach().numpy()))

    #return correct / total
    return accuracy_score(r_labels, preds)

def train_chinese_mnist(config):
    should_checkpoint = config.get("should_checkpoint", False)
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, test_loader, val_loader = get_data_loaders(batch_size = args.batch_size)
    #model = ConvNet().to(device)
    model = ChineseMNISTCNN(num_classes=args.n_outputs, num_units=args.n_hidden).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    while True:
        train_func(model, optimizer, train_loader, device)
        acc = test_func(model, test_loader, device)
        metrics = {"mean_accuracy": acc}

        # Report metrics (and possibly a checkpoint)
        if should_checkpoint:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(model.state_dict(), os.path.join(tempdir, "model.pt"))
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)


import os
import argparse
from filelock import FileLock
import tempfile

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import AsyncHyperBandScheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )

    parser.add_argument("--n_hidden", help="amount of hidden neurons", type=int, default=500)
    parser.add_argument("--n_outputs", help="amount of outputs", type=int, default=15)
    # parser.add_argument("--epochs", help="number of epochs to train", type=int, default=5)
    parser.add_argument("--batch_size", help="batch size", type=int, default=100)
    # parser.add_argument("--lr", help="learning rate", type=float, default=0.1)

    args, _ = parser.parse_known_args()

    ray.init(num_cpus=2 if args.smoke_test else None)

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {"cpu": 2, "gpu": int(args.cuda)}  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train_chinese_mnist, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=1 if args.smoke_test else 1000,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={
                "mean_accuracy": 0.98,
                "training_iteration": 5 if args.smoke_test else 100,
            },
        ),
        param_space={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        },
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)

    assert not results.errors

# model = ChineseMNISTCNN(num_classes=args.n_outputs, num_units=args.n_hidden).to(device)

# criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=args.lr)

# loss_history = []

# for epoch in range(args.epochs):
#     losses = []
#     print(f"Epoch {epoch+1}/{args.epochs}")
#     for i, (x, y) in enumerate(train_loader):
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         y_ = model(x)
#         loss = criterion(y_, y)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
#         loss_history.append(loss.item())

#         if i%50 == 0:
#             print(f"Epoch {epoch} [{i}/{len(train_loader)}]: loss: {np.mean(losses):.2f}, lr={optimizer.param_groups[0]['lr']}")