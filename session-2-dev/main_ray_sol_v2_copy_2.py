import torch
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import MyModel
from utils import accuracy
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint

import os
from filelock import FileLock

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print('Using device:', device)
print()

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
            shuffle=False)
        
        test_loader = DataLoader(
            test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_data():
    # data_transform = transforms.Compose([transforms.ToTensor()])
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    # Specify the path to the CSV file and the root directory of the images
    labels_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
    images_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path

    # Create an instance of your custom dataset
    dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)
    
    # Split the dataset into train, validation, and test sets
    split_sizes = [10000, 2500, 2500]
    train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)
    return train_dataset, val_dataset, test_dataset 

def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = F.cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(y.cpu(), y_.cpu())
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            acc = accuracy(y.cpu(), y_.cpu())
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_model(config):

    # data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    # my_dataset = MyDataset("/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/data/data", 
    #                        "/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/chinese_mnist.csv", 
    #                        transform=data_transforms)
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [10000, 2500, 2500])
    # train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    # test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    train_dataset, val_dataset, test_dataset = load_data()

    train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = config["batch_size"], num_workers=4)

    my_model = MyModel(config["num_classes"], config["num_units"]).to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    # optimizer = optim.SGD(my_model.parameters(), lr=config["lr"], momentum=config["momentum"])

    # To restore a checkpoint, use `train.get_checkpoint()`.
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        my_model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        val_loss, val_acc = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch} loss={val_loss:.2f} acc={val_acc:.2f}")
        

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `train.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (my_model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")

        # Send the current training result back to Tune
        train.report({'val_loss': val_loss, 'val_acc': val_acc}, checkpoint=checkpoint)
    
    test_loss, test_acc = eval_single_epoch(my_model, test_loader)
    print(f"Test loss={test_loss:.2f} acc={test_acc:.2f}")
    # train.report({'test_loss': test_loss}) # !!!!!!!!!!!!!!!!!!!!!

    #return my_model
    # return {"val_loss": val_loss}



if __name__ == "__main__":

    analysis = tune.run(
        train_model,
        # metric="val_loss",
        # mode="min",
        num_samples=2, #5,
        scheduler=ASHAScheduler(metric="val_acc", mode="max"),
        config = {
            "lr": tune.loguniform(1e-4, 1e-2),
            # "momentum": tune.uniform(0.1, 0.9),
            "batch_size": 64,
            "epochs": 1,
            # "h1": tune.randint(20, 40),
            # "h2": tune.randint(40, 80),
            # "h3": tune.randint(100, 140),
            # "h4": tune.randint(100, 140),
            "num_classes": 15,
            "num_units": 500

        })

    # print("Best hyperparameters found were: ", analysis.best_config)
    print(analysis.get_best_config(metric="val_acc", mode="max"))
    print("-----")


    # TODO https://docs.ray.io/en/latest/tune/getting-started.html
    # import os

    # logdir = results.get_best_result("mean_accuracy", mode="max").path
    # state_dict = torch.load(os.path.join(logdir, "model.pth"))

    # model = ConvNet()
    # model.load_state_dict(state_dict)