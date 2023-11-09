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

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", help="amount of samples to train with", type=int, default=1000)
parser.add_argument("--n_features", help="amount of features per sample", type=int, default=20)
parser.add_argument("--n_hidden", help="amount of hidden neurons", type=int, default=128)
parser.add_argument("--n_outputs", help="amount of outputs", type=int, default=15)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=5)
parser.add_argument("--batch_size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform (you can customize it as needed)
data_transform = transforms.Compose([transforms.ToTensor()])

# Specify the path to the CSV file and the root directory of the images
csv_file = '/workspace/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
root_dir = '/workspace/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path
chinese_mnist_dataset = ChineseMNISTDataset(csv_file=csv_file, root_dir=root_dir, transform=data_transform)

# Split the dataset into train, validation, and test sets
split_sizes = [10000, 2500, 2500]
train_dataset, val_dataset, test_dataset  = random_split(chinese_mnist_dataset, split_sizes)

batch_size = args.batch_size

train_loader = DataLoader(
    train_dataset.dataset,
    batch_size=batch_size,
    shuffle=True)

val_loader = DataLoader(
    val_dataset.dataset,
    batch_size=batch_size,
    shuffle=True)

test_loader = DataLoader(
    test_dataset.dataset,
    batch_size=batch_size,
    shuffle=False)

model = ChineseMNISTCNN(num_classes=args.n_outputs, num_units=args.n_hidden).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr)

loss_history = []

for epoch in range(args.epochs):
    losses = []
    print(f"Epoch {epoch+1}/{args.epochs}")
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        loss_history.append(loss.item())

        if i%50 == 0:
            print(f"Epoch {epoch} [{i}/{len(train_loader)}]: loss: {np.mean(losses):.2f}, lr={optimizer.param_groups[0]['lr']}")

plt.plot(loss_history)
plt.title("Training loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()
plt.savefig('my_plot.png')
