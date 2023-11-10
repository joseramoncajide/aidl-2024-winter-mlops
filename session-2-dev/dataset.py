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

print(dataset.labels_df.head(3))
print(dataset.labels_df.loc[0, :].values.tolist())
print(dataset[0])
sample, label = dataset[0]
print(f" * label={label}")
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.imshow(sample.squeeze(), cmap="gray")
plt.title(f'label = {label}')
plt.show()
plt.savefig(f'sample_image.png')

# Split the dataset into train, validation, and test sets
from torch.utils.data import random_split
split_sizes = [10000, 2500, 2500]
train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)

print("datasets lenghts:")
print(f"* train_dataset={len(train_dataset)}")
print(f"* val_dataset={len(val_dataset)}")
print(f"* test_dataset={len(test_dataset)}")


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
print("DataLoader lenghts:")
print(f"* train_dataset={len(train_dataloader.dataset)}")
print(f"* val_dataloader={len(val_dataloader.dataset)}")
print(f"* test_dataloader={len(test_dataloader.dataset)}")


# Display image and label.
import matplotlib.pyplot as plt
train_features, train_labels = next(iter(train_dataloader))
print("Train DataLoader")
print(f"* Feature batch shape: {train_features.size()}")
print(f"* Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray")
plt.title(f'label = {label}')
plt.show()
plt.savefig(f'sample_image_from_dataloader.png')
print(f"* DataLoader image Label: {label}")

# ok
# for batch_idx, (data, target) in enumerate(test_dataloader):
#     print(f"Labels batch shape: {target.size()}")
#     print(data)
#     print(f"Target= {target}")

import os
from filelock import FileLock

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

train_dataloader, val_dataloader, test_dataloader  = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = 4, num_workers=4)
print("[A] test_dataloader from get_data_loaders() lenghts:")
print(f"* test_dataloader.dataset={len(test_dataloader.dataset)}")
print(f"* test_dataloader.dataset/batch_size={len(test_dataloader.dataset) / 4}")
print(f"* test_dataloader={len(test_dataloader)}")
del test_dataloader


test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
print("[B] test_dataloader from DataLoader lenghts:")
print(f"* test_dataloader.dataset={len(test_dataloader.dataset)}")
print(f"* test_dataloader.dataset/batch_size={len(test_dataloader.dataset) / 4}")
print(f"* test_dataloader={len(test_dataloader)}")


print("test_dataloader exec:")
images = []
for batch_idx, (data, target) in enumerate(test_dataloader):
    # print(f"Labels batch shape (2): {target.size()}")
    # print(data)
    # print(f"Target (2)= {target}")
    images.append(target.data.numpy().tolist())

print(f"* num images proccesed={len(images)}")
print(images[:10])