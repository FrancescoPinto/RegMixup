import pytorch_lightning as pl
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

#Credit: Ameya for train/val uniform class splitting mechanism
import numpy as np
def get_train_val_idx(train_dataset, valid_size):

    train_class_labels_dict = classwise_split(
        targets=train_dataset.targets)
    train_idx = []
    val_idx = []
    for cls, idxs in train_class_labels_dict.items():
        num_class_training = len(idxs)  # number of elements of class
        partition_size = int(np.floor((1 - valid_size) * num_class_training))

        train_idx += train_class_labels_dict[cls][:partition_size]
        val_idx += train_class_labels_dict[cls][partition_size:]
    return train_idx, val_idx

def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        targets (sequence): a sequence of class labels
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict

#data module that takes a Train/Val only dataset and turns it in Train/Val/Test if required (using Val as Test, and splitting Train) otherwise uses Val as Test
class TrainValOnlyDataModule(pl.LightningDataModule):
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


    def dataloader(self, stage: str):
        """Train/validation loaders."""
        shuffle = False
        if stage == "train":
            shuffle = True
            _dataset = self.train_dataset
        elif stage == "val":
            _dataset = self.valid_dataset
        elif stage == "test":
            _dataset = self.test_dataset
        loader = DataLoader(dataset=_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                            pin_memory=True)
        return loader


    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("val")

    def test_dataloader(self):
        return self.dataloader("test")  # test and validation data are the same


#returns only validation set for datasets without training set (e.g. robustness evaluation datasets like ImageNetA/O etc.)
class ValOnlyDataModule(pl.LightningDataModule):
    def setup(self, stage: str):
        self.test_dataset = self.get_val_set()
        self.dims = tuple(self.test_dataset[0][0].shape)

        self.valid_dataset = self.get_val_set()

    def dataloader(self, stage: str):
        """Train/validation loaders."""
        shuffle = False
        if stage == "train":
            raise Exception(f"Train dataset not available for {self} dataset type")
        elif stage == "val":
            _dataset = self.valid_dataset
        elif stage == "test":
            _dataset = self.test_dataset

        loader = DataLoader(dataset=_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                            pin_memory=True)
        return loader

    def val_dataloader(self):
        return self.dataloader("val")

    def test_dataloader(self):
        return self.dataloader("test")  # test and validation data are the same
