import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import pandas as pd
import numpy as np

import os
from filelock import FileLock

from sklearn.metrics import accuracy_score

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print('Using device:', device)
print()

# Define a transform (you can customize it as needed)
data_transform = transforms.Compose([transforms.ToTensor()])

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):

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

    return total_loss / len(data_loader)

def eval_single_epoch(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    correct_predictions = 0
    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    conf_matrix = np.zeros((15,15))

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

            for i in range(target.size(0)):
                label = target.data[i]
                # Update confusion matrix
                conf_matrix[label][predicted.data[i]] += 1

    y_pred = pd.Series(all_predictions, name='Predicted')
    y_actu = pd.Series(all_targets, name='Actual')
    cm = pd.crosstab(y_actu, y_pred)
    
    #return correct / total
    #print(f'* correct/total={correct / total}')
    
    # accuracy_val = accuracy(labels=target, outputs=outputs.data)
    # print(f'* acc={accuracy_val}')
    avg_loss = total_loss / len(data_loader)
    # print(f'* avg_loss={avg_loss}')

    accuracy_val = accuracy_score(y_actu, y_pred)
    # print(f'* accuracy_score={accuracy_val}')

    return avg_loss, accuracy_val, all_targets, all_predictions, conf_matrix, cm


def plot_confusion_matrix(targets, predictions, num_classes):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(1, num_classes + 1),
                yticklabels=np.arange(1, num_classes + 1))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    # plt.show()


def __plot_confusion_matrix(targets, predictions, num_classes):
    cm = confusion_matrix(targets, predictions)
    # plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(1, num_classes + 1),
            yticklabels=np.arange(1, num_classes + 1))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Validation)")
    # plt.show()


def train_model(config):
    
    # Specify the path to the CSV file and the root directory of the images
    labels_path = '../datasets/chinese_mnist.csv'  # Replace with the actual path
    images_path = '../datasets/data/data'  # Replace with the actual path

    # Create an instance of your custom dataset
    dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)

    # Split the dataset into train, validation, and test sets
    split_sizes = [10000, 2500, 2500]
    train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)

    train_loader, val_loader, test_loader  = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = config["batch_size"], num_workers=4)

    model = MyModel(num_classes=config["num_classes"], num_units=config["num_units"]).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    train_losses = []
    val_losses = []
    accuracies = []
    all_targets = []
    all_predictions = []

    for epoch in range(config["epochs"]):
        
        train_loss = train_single_epoch(model = model, optimizer = optimizer, data_loader = train_loader, device=device)

        eval_loss, accuracy_val, targets, predictions, conf_matrix, cm = eval_single_epoch(model=model, data_loader=val_loader, device=device)

        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(" Confusion Matrix: \n \n",  cm )
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Loss: {eval_loss:.4f} | Accuracy: {accuracy_val:.2%}")
        print("\n")

        train_losses.append(train_loss)
        val_losses.append(eval_loss)
        accuracies.append(accuracy_val)
        all_targets.extend(targets)
        all_predictions.extend(predictions)

    # print(train_losses)
    # print(val_losses)

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, config["epochs"] + 1), train_losses, label="Train Loss")
    plt.plot(range(1, config["epochs"] + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, config["epochs"] + 1), accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig('metrics.png')

    # Plot confusion matrix
    # plt.subplot(1, 3, 3)
    # labels = list(range(0,15,1))  # Assuming 15 labels
    # plot_confusion_matrix(all_targets, all_predictions, labels)
    # plt.savefig('cm1.png')

    # # plt.savefig('cm1.png')

    # plt.subplots(figsize=(10,9))
    # ax = sns.heatmap(conf_matrix, annot=True, vmax=20)
    # ax.set_xlabel('Predicted');
    # ax.set_ylabel('True');
    # plt.savefig('cm2.png')

    # conf_matrix = cm.to_numpy()

    # fig, ax = plt.subplots(figsize=(10,5))
    # im = ax.imshow(conf_matrix)

    # ax.set_xticks(np.arange(15))
    # ax.set_yticks(np.arange(15))

    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         text = ax.text(j, i, conf_matrix[i, j],
    #                     ha="center", va="center", color="w")
            
    # ax.set_xlabel('Actual targets')
    # ax.set_ylabel('Predicted targets')
    # ax.set_title('Confusion Matrix')
    # plt.savefig('cm3.png')

    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(targets, predictions, num_classes=config["num_classes"])
    plt.savefig('confusion_matrix.png')

    print("Model evaluation on test data:")

    # predictions = np.empty((0, len(test_dataset)), np.int32)
    # actual_values = np.empty((0, len(test_dataset)), np.int32)
    # print(f'** num expected predictions={len(predictions)}')

    # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs, 1)

            # predictions = np.append(predictions, predicted)
            # actual_values = np.append(actual_values, target)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # print(f'*** num collected predictions={len(predictions)}')
    # print(f'*** num actual_values={len(actual_values)}')
    # accuracy_test = accuracy(labels=actual_values, outputs=predictions)
    # print(f'* accuracy_test={accuracy_test}')
    print(confusion_matrix(all_targets, all_predictions))
    # print(f"* accuracy_score={accuracy_score(actual_values, predictions)}")
    print(f"** accuracy_score={accuracy_score(all_targets, all_predictions)}")

    # plot_confusion_matrix(actual_values, predictions, num_classes=15)
    # plt.savefig('cm5.png')

    # accuracy_test = accuracy(labels=target, outputs=outputs.data)
    # print(f'* labels={len(target)}')
    # print(f'* outputs={len(outputs.data)}')
    # print(f'* accuracy_test={accuracy_test}')

    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(all_targets, all_predictions, num_classes=config["num_classes"])
    plt.savefig('confusion_matrix_test_dataset.png')

    return model

if __name__ == "__main__":

    config = {
        "batch_size": 64,
        "lr": 0.1,
        "momentum": 0.1,
        "epochs": 12,
        "num_classes": 15,
        "num_units": 500
    }
    train_model(config)
