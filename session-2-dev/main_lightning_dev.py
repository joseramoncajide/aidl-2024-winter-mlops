from typing import Any
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch
from torch.utils.data import DataLoader

# from dataset import MyDataset
# from model import MyModel
from utils import accuracy
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# MyDataset
import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F


import lightning as L
from lightning import LightningModule, Trainer
import torchmetrics

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from lightning.pytorch.loggers import TensorBoardLogger
# TODO from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import random_split

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


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, labels_path: str = "./", images_path: str = "./"):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        self.labels_path = labels_path
        self.images_path = images_path

    def prepare_data(self):
        pass
        

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        data = MyDataset(labels_path=self.labels_path, images_path=self.images_path, transform=self.transform)
        split_sizes = [10000, 2500, 2500]
        self.train, self.val, self.test = random_split(
            data, split_sizes, generator=torch.Generator().manual_seed(42)
        )

        #if stage == "predict":
        #    self.predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        # return DataLoader(self.train, batch_size=32)
        return DataLoader(self.train, batch_size=config["batch_size"], shuffle=True, num_workers=9,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=config["batch_size"], shuffle=False, num_workers=9,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=config["batch_size"], shuffle=False, num_workers=9,persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=32)

# MyModel



logger = TensorBoardLogger(save_dir='./logs')


class MyModel(LightningModule):

    

    def __init__(self, h1=32, h2=64, h3=128, h4=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, h1, 3, padding=1)
        self.conv2 = nn.Conv2d(h1, h2, 3, padding=1)
        self.conv3 = nn.Conv2d(h2, h3, 3, padding=1)

        self.fc1 = nn.Linear(8*8*h3, h4)
        self.fc2 = nn.Linear(h4, 15)

        self.pool = nn.MaxPool2d(2)

        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=15)
        # self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=15)
        # self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=15)
        # self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=15)
        from torchmetrics.classification import Accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=15)
        self.val_acc = Accuracy(task="multiclass", num_classes=15)
        self.test_acc = Accuracy(task="multiclass", num_classes=15)
        #self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes = 10)
        #self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes = 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=9,persistent_workers=True)
    #     return train_loader
    
    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=9,persistent_workers=True)
    #     return val_loader
    
    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=9,persistent_workers=True)
    #     return test_loader
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), config["lr"])
        # return optimizer
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95), 'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        # return {'loss': loss}

        self.train_acc(output, target)
        tensorboard_logs = {"train_loss": loss, "train_acc": self.train_acc}
        self.log_dict(tensorboard_logs, on_step=False, on_epoch=True)
        return loss

        #acc = self.accuracy(output, target)
        #self.log("ptl/train_accuracy", acc)
        

        # self.train_acc(output, target)
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        # logs = {'train_loss': loss}
        # self.log("train_loss", loss)
        # return {'loss': loss, 'log': logs}
    
    def on_train_epoch_end(self):
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)

        self.val_acc(output, target)
        tensorboard_logs = {"val_loss": loss, "val_acc": self.val_acc}
        self.log_dict(tensorboard_logs, on_step=False, on_epoch=True)

        # acc = self.accuracy(output, target)
        #self.valid_acc(output, target)
        # self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        # self.validation_step_outputs.append(output)
        self.validation_step_outputs.append(loss)
        # logs = {'val_loss': loss}
        # self.log("val_loss", loss)
        # return {'val_loss': loss, "val_accuracy": acc, 'log': logs}


    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        acc = self.test_acc(output, target)
        tensorboard_logs = {"test_loss": loss, "test_acc": self.test_acc}
        self.log_dict(tensorboard_logs, on_step=False, on_epoch=True)
        #self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
        self.test_step_outputs.append(loss)
        # logs = {'test_loss': loss}
        # self.log("test_loss", loss)
        # return {'test_loss': loss, "test_accuracy": acc, 'log': logs}  
    
    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()  # free memory
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss, on_epoch=True) 
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def on_test_epoch_end(self) -> None:
        all_preds = torch.stack(self.test_step_outputs)
        avg_loss = all_preds.mean()
        self.test_step_outputs.clear()  # free memory
        tensorboard_logs = {'test_loss': avg_loss}
        self.log('test_loss', avg_loss, on_epoch=True) 
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

# Custom Callbacks
class _MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')

    
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_init_end(self, trainer, pl_module):
        print('trainer is init now')
        
    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        # https://github.com/Lightning-AI/pytorch-lightning/issues/12095
        # table = wandb.Table(data=bin_means, columns=['index', 'count'])
        # wandb.log({"Distribution Bar Plot": wandb.plot.bar(table, "index", 'count', title="Distribution of idx")})

class MetricTracker(Callback):
    
    def __init__(self):
        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []

    def on_validation_epoch_end(self, trainer, module):
        val_loss = trainer.logged_metrics['val_loss'].item()
        val_acc = trainer.logged_metrics['val_acc'].item()
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        
    def on_train_epoch_end(self, trainer, module):
        train_loss = trainer.logged_metrics['train_loss'].item()
        train_acc = trainer.logged_metrics['train_acc'].item()
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)



if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "h1": 32,
        "h2": 64,
        "h3": 128,
        "h4": 128,
    }
    #my_model = train_model(config)

    # def load_data():
    #     from torch.utils.data import random_split

    #     # data_transform = transforms.Compose([transforms.ToTensor()])
    #     data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    #     # Specify the path to the CSV file and the root directory of the images
    #     labels_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/chinese_mnist.csv'  # Replace with the actual path
    #     images_path = '/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/data/data/'  # Replace with the actual path

    #     # Create an instance of your custom dataset
    #     dataset = MyDataset(labels_path=labels_path, images_path=images_path, transform=data_transform)
        
    #     # Split the dataset into train, validation, and test sets
    #     split_sizes = [10000, 2500, 2500]
    #     train_dataset, val_dataset, test_dataset  = random_split(dataset, split_sizes)
    #     return train_dataset, val_dataset, test_dataset 

    # train_dataset, val_dataset, test_dataset = load_data()


    
    net = MyModel()

    # trainer = Trainer(max_epochs=2)
    # trainer.fit(net)

    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Set Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    # saves checkpoints to 'model_path' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(dirpath='./pytorch-lightning',
                                          filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                                          monitor='val_loss', mode='min', save_top_k=3)

    dm = MNISTDataModule(labels_path='/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/chinese_mnist.csv', images_path='/Users/jcajidefernandez/Documents/code/aidl-2024-winter-mlops/datasets/data/data/')


    metric_tracker = MetricTracker()

    trainer = Trainer(fast_dev_run=False, max_epochs=10, callbacks=[metric_tracker, lr_monitor, MyPrintingCallback(), early_stopping, checkpoint_callback], 
                        default_root_dir='./pytorch-lightning', logger=logger) #gpus=1
#    trainer = Trainer(callbacks=[MyPrintingCallback()])
#       trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])

    trainer.fit(net,train_dataloaders=dm)
    trainer.validate(net, datamodule=dm)
    trainer.test(net, datamodule=dm)

    print("Best model path: " + checkpoint_callback.best_model_path)
     
    # TODO: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    # REF: https://www.kaggle.com/code/mukit0/pytorch-lightning-on-mnist-with-references-98

    import seaborn as sns
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(len(metric_tracker.train_loss)), y=metric_tracker.train_loss, label="train_loss")
    sns.lineplot(x=range(len(metric_tracker.val_loss)), y=metric_tracker.val_loss, label="val_loss")
    plt.legend()
    plt.savefig('loss.png')

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(len(metric_tracker.train_acc)), y=metric_tracker.train_acc, label="train_acc")
    sns.lineplot(x=range(len(metric_tracker.val_acc)), y=metric_tracker.val_acc, label="val_acc")
    plt.legend()
    plt.savefig('acc.png')