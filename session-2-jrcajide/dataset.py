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
        tmp_labels = self.data_frame.loc[idx, :].values.tolist()
        img_name = os.path.join(self.root_dir, "input_" + str(tmp_labels[0]) + '_' + str(tmp_labels[1]) + '_' + str(tmp_labels[2]) + '.jpg')
        image = Image.open(img_name)
        label = tmp_labels[2] - 1 

        if self.transform:
            image = self.transform(image)

        # return image, torch.Tensor(label)
        return image, label
