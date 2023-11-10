import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

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
    
# TEST
# Define a transform (you can customize it as needed)
from torchvision import transforms
data_transform = transforms.Compose([transforms.ToTensor()])

# Specify the path to the CSV file and the root directory of the images
labels_path = '/workspace/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
images_path = '/workspace/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path

# Create an instance of your custom dataset
dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)

print(dataset.labels_df)
print(dataset[14995])

# Split the dataset into train, validation, and test sets
from torch.utils.data import random_split
split_sizes = [10000, 2500, 2500]
train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)

print("=== DATASET ===")
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))
print("=== / DATASET ===")

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
print(len(train_dataloader.dataset))

# Display image and label.
import matplotlib.pyplot as plt
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
plt.savefig('sample_image.png')
print(f"Label: {label}")

# ok
# for batch_idx, (data, target) in enumerate(test_dataloader):
#     print(f"Labels batch shape: {target.size()}")
#     print(data)
#     print(f"Target= {target}")

import os
from filelock import FileLock

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):

        train_loader = DataLoader(
            train_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=True)

        val_loader = DataLoader(
            val_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=True)

        test_loader = DataLoader(
            test_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=True)
    
    return train_loader, val_loader, test_loader

train_dataloader, val_dataloader, test_dataloader  = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = 4)
print("===")
print(len(test_dataloader.dataset))
print(len(test_dataloader.dataset) / 4) # batch_size = 4
print(len(test_dataloader))

del test_dataloader

test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
print("===")
print(len(test_dataloader.dataset))
print(len(test_dataloader.dataset) / 4) # batch_size = 4
print(len(test_dataloader))


images = []
for batch_idx, (data, target) in enumerate(test_dataloader):
    # print(f"Labels batch shape (2): {target.size()}")
    # print(data)
    # print(f"Target (2)= {target}")
    images.append(batch_idx)

print("===")
print(len(images))