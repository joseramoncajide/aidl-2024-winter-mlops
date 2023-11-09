import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform (you can customize it as needed)
from torchvision import transforms
data_transform = transforms.Compose([transforms.ToTensor()])


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
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


def train_single_epoch(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        #if batch_idx * len(data) > EPOCH_SIZE:
        #    return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # loss_history.append(loss.item())


def eval_single_epoch(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            #if batch_idx * len(data) > TEST_SIZE:
            #    break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    #return correct / total
    print(f'* correct/total={correct / total}')
    acc = accuracy(labels=target, outputs=outputs.data)
    print(f'* acc={acc}')

    return acc

def _accuracy(network, data_loader):
  """
  This function computes accuracy
  """
  #  setting model state
  network.eval()
  
  #  instantiating counters
  total_correct = 0
  total_instances = 0

  #  iterating through batches
  with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(data_loader):
      images, labels = images.to(device), labels.to(device)

      #-------------------------------------------------------------------------
      #  making classifications and deriving indices of maximum value via argmax
      #-------------------------------------------------------------------------
      classifications = torch.argmax(network(images), dim=1)

      #--------------------------------------------------
      #  comparing indicies of maximum values and labels
      #--------------------------------------------------
      correct_predictions = sum(classifications==labels).item()

      #------------------------
      #  incrementing counters
      #------------------------
      total_correct+=correct_predictions
      total_instances+=len(images)
      print(f'* acc={round(total_correct/total_instances, 3)}')
  return round(total_correct/total_instances, 3)

def train_model(config):
    
    # Specify the path to the CSV file and the root directory of the images
    labels_path = '/workspace/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
    images_path = '/workspace/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path

    # Create an instance of your custom dataset
    dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)

    # Split the dataset into train, validation, and test sets
    split_sizes = [10000, 2500, 2500]
    train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)

    train_loader, test_loader, val_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = 100)

    model = MyModel(num_classes=15, num_units=500).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_single_epoch(model = model, optimizer = optimizer, train_loader = train_loader)
        acc = _accuracy(model, test_loader)
        #acc = eval_single_epoch(model=model, data_loader=test_loader)

    return acc


if __name__ == "__main__":

    config = {
        "lr": 0.1,
        "momentum": 0.1,
        "epochs": 10
    }
    train_model(config)
