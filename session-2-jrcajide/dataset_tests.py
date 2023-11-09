import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

class ChineseMNISTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing dataset information.
            root_dir (str): Directory with all the images.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data_frame = pd.read_csv(csv_file)     
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        tmp_labels = self.data_frame.loc[idx, :]#.values.tolist()
        img_name = os.path.join(self.root_dir, "input_" + str(tmp_labels[0]) + '_' + str(tmp_labels[1]) + '_' + str(tmp_labels[2]) + '.jpg')
        print(f"* Image file name = {img_name}")
        image = Image.open(img_name)
        label = tmp_labels[2] - 1 #self.data_frame.iloc[idx, 2]
        print(f"* Image label = {label}")

        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage:
# Define a transform (you can customize it as needed)
data_transform = transforms.Compose([transforms.ToTensor()])

# Specify the path to the CSV file and the root directory of the images
csv_file = '/workspace/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
root_dir = '/workspace/aidl-2024-winter-mlops/datasets/data/data'  # Replace with the actual path

# Create an instance of your custom dataset
chinese_mnist_dataset = ChineseMNISTDataset(csv_file=csv_file, root_dir=root_dir, transform=data_transform)

# Access data samples and labels
image, label = chinese_mnist_dataset[14999]
print(image)
print(label)
print(torch.Tensor(label))

# Split the dataset into train, validation, and test sets
split_sizes = [10000, 2500, 2500]
train_dataset, val_dataset, test_dataset = random_split(chinese_mnist_dataset, split_sizes)

# Access data samples and labels
train_sample, train_label = train_dataset[0]
val_sample, val_label = val_dataset[0]
test_sample, test_label = test_dataset[0]

