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
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from filelock import FileLock



class MyModel(nn.Module):
    def __init__(self, num_classes, num_units):
        super(MyModel, self).__init__()
        # kernel_size: convolutional kernel or filter size: 3x3
        # stride: A stride of 1 means the kernel moves one pixel at a time
        # padding: A padding of 1 means that a 1-pixel border is added to all sides of the input.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # does not change the spatial dimensions, so the input remains 64x64x16.
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # This operation effectively reduces the spatial dimensions by a factor of 2
        # 32x32x16
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # the input size for the third nn.Conv2d layer after these operations will be 16x16x32.
        # Input size from the previous layer: 16x16x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input size from the previous layer: 8x8x64

        # OK self.fc1 = nn.Linear(64 * 4 * 4 * 4, num_units)
        self.fc1 = nn.Linear(64 * 8 * 8, num_units)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(num_units, num_classes)

        # self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        # output = self.log_softmax(x)
        return x

    
from dataset_v2 import MyDataset


def get_data_loaders(train_dataset, val_dataset, batch_size=64, num_workers=2):

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
    
    return train_loader, val_loader


def load_data():
    data_transform = transforms.Compose([transforms.ToTensor()])

    # Specify the path to the CSV file and the root directory of the images
    labels_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
    images_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path

    # Create an instance of your custom dataset
    dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)
    
    # Split the dataset into train, validation, and test sets
    split_sizes = [10000, 2500, 2500]
    train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)
    return train_dataset, val_dataset, test_dataset 

def train_chinese_mnist(config):
    # net = Net(config["l1"], config["l2"])
    net = MyModel(num_classes=config["num_classes"], num_units=config["num_units"])

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()

    # if use_cuda:
    #     device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    device = torch.device("mps")

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])

    # To restore a checkpoint, use `train.get_checkpoint()`.
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # data_dir = os.path.abspath("./data")
    # trainset, testset = load_data(data_dir)

    train_dataset, val_dataset, _ = load_data()

    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset, batch_size = config["batch_size"], num_workers=4)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                # _, predicted = torch.max(outputs.data, 1) # or torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `train.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        train.report({"loss": (val_loss / val_steps), "accuracy": correct / total}, checkpoint=checkpoint)

        # ????
        return {"loss": (val_loss / val_steps), "accuracy": correct / total}

    print("Finished Training")


def test_best_model(best_result, config):
    best_trained_model = MyModel(num_classes=config["num_classes"], num_units= best_result.config["num_units"])
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    # trainset, testset = load_data()
    _, _, test_dataset = load_data()

    test_loader = DataLoader(
            test_dataset,
            num_workers=4,
            batch_size=4,
            shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs, 1)
            #_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Best trial test set accuracy: {}".format(correct / total))



def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    # search space
    config = {
        # "num_units": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([2, 4, 8, 16]),
        # "lr": tune.grid_search([1e-4, 1e-2]),
        # "momentum": tune.grid_search([0.1, 0.9]),
        "lr": 0.1,
        "momentum": 0.1,
        "batch_size": 64,
        "epochs": 12,
        "num_classes": 15,
        "num_units": 500
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_chinese_mnist),
            resources={"cpu": 4, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result, config)


if __name__ == "__main__":
    main(num_samples=2, max_num_epochs=12, gpus_per_trial=0)
