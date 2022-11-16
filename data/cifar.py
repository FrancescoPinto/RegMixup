import pytorch_lightning as pl
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from data.train_val_only_data_module import ValOnlyDataModule, TrainValOnlyDataModule
from data.datashift_datamodule import CompositeDataShiftDataModule

from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset

import torch

class CIFARDataModule(TrainValOnlyDataModule):

    def __init__(self, data_dir: str = '../data', dataset_name: str = 'cifar10', augment: bool = True, batch_size: int = 128, num_workers:int = 8,
                 train_transforms:transforms.Compose = None, test_transforms:transforms.Compose = None, supersample_factor:int = 0, valid_size:float = 0., size=(32,32)):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.original_num_classes = 10 if dataset_name == "cifar10" else 100
        if supersample_factor > 0:
            self.num_classes = self.original_num_classes * supersample_factor
        else:
            self.num_classes = self.original_num_classes
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.supersample_factor = supersample_factor
        self.valid_size = valid_size
        self.size = size 
        self.dataset_class = datasets.__dict__[self.dataset_name.upper()]

        # 2. Load the data + preprocessing & data augmentation
        mean = {
            'cifar10': [0.4914, 0.4822, 0.4465],
            'cifar100': [0.5071, 0.4867, 0.4408],
        }

        std = {
            'cifar10': [0.2023, 0.1994, 0.2010],
            'cifar100': [0.2675, 0.2565, 0.2761],
        }

        assert (self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100')

        # Data loading code
        normalize = transforms.Normalize(mean=mean[self.dataset_name],
                                         std=std[self.dataset_name])

        if self.train_transforms is None:
            if self.augment:
                self.train_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

            else:
                self.train_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])

        if self.test_transforms is None:
            self.test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    def setup(self, stage: str):
        train_dataset = self.get_train_set() # dataset_class(self.data_dir, train=True, transform=self.train_transforms)

        self.dims = tuple(train_dataset[0][0].shape)

        self.test_dataset = self.get_val_set()#dataset_class(self.data_dir, train=False, transform=self.test_transforms)

        if self.valid_size > 0:
            self.train_idx, self.val_idx = get_train_val_idx(train_dataset, self.valid_size)

            # directly get the class balanced dataset, don't wait for the loader
            self.train_dataset = Subset(train_dataset, self.train_idx)
            self.valid_dataset = Subset(train_dataset, self.val_idx)
        else:
            self.train_dataset = train_dataset
            self.valid_dataset = self.get_val_set() # dataset_class(self.data_dir, train=False, transform=self.test_transforms)

    def prepare_data(self):
        """Saves CIFAR files to `data_dir`""" #only download
        datasets.__dict__[self.dataset_name.upper()](self.data_dir, train=True, download=True)
        datasets.__dict__[self.dataset_name.upper()](self.data_dir, train=False, download=True)


    def get_train_set(self):
        return self.dataset_class(self.data_dir, train=True, transform=self.train_transforms)
    def get_val_set(self):
        return self.dataset_class(self.data_dir, train=False, transform=self.test_transforms)

class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, transform=None):
        self.inputs = inputs
        self.outputs = outputs

        self.transform = transform

    def __getitem__(self, index):
        x = self.inputs[index]

        if self.transform:
            x = self.transform(x)

        y = self.outputs[index]

        return x, y

    def __len__(self):
        return self.inputs.shape[0]

class Cifar10v6TestDataset(ValOnlyDataModule):

    def __init__(self, data_dir: str = '../data', dataset_name = "cifar-10v6", batch_size: int = 128, num_workers:int = 8, test_transforms:transforms.Compose = None):
        super(Cifar10v6TestDataset,self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.original_num_classes = 10
        self.num_classes = self.original_num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_transforms = test_transforms

    def prepare_data(self):
        #the pytorch loader will check the presence and throw an error if needed
        if os.path.exists(self.data_dir+"/cifar10-v6/cifar10.1_v6_data.npy") and os.path.exists(self.data_dir+"/cifar10-v6/cifar10.1_v6_labels.npy"): #and os.path.exists(self.data_dir+f"/imagenet/val/"):
            print("The dataset folders are available, make sure they're correctly setup")
        else:
            print("The root of the training/val set is not available, download the folder at " + folder)
            sys.exit()


    def get_val_set(self):
        return CustomTensorDataset(np.load(self.data_dir+"/cifar10-v6/cifar10.1_v6_data.npy"),
                                   torch.tensor(np.load(self.data_dir+"/cifar10-v6/cifar10.1_v6_labels.npy")),
                                   transform=self.test_transforms)


class Cifar10_2TestDataset(ValOnlyDataModule):

    def __init__(self, data_dir: str = '../data', dataset_name = "cifar-10_2", batch_size: int = 128, num_workers:int = 8, test_transforms:transforms.Compose = None):
        super(Cifar10_2TestDataset,self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.original_num_classes = 10
        self.num_classes = self.original_num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_transforms = test_transforms

    def prepare_data(self):
        #the pytorch loader will check the presence and throw an error if needed
        if os.path.exists(self.data_dir+"/cifar-10_2/cifar102_test.npz"): #and os.path.exists(self.data_dir+f"/imagenet/val/"):
            print("The dataset folders are available, make sure they're correctly setup")
        else:
            print("The root of the training/val set is not available, download the folder at " + folder)
            sys.exit()


    def get_val_set(self):
        return CustomTensorDataset(np.load(self.data_dir+"/cifar-10_2/cifar102_test.npz")["images"],
                                   torch.tensor(np.load(self.data_dir+"/cifar-10_2/cifar102_test.npz")["labels"]),
                                   transform=self.test_transforms)

