import pytorch_lightning as pl
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from data.train_val_only_data_module import ValOnlyDataModule


class SVHNDataModule(ValOnlyDataModule):
    def __init__(self, data_dir: str = '../data', batch_size: int = 128, num_workers:int = 8,
                 test_transforms:transforms.Compose = None):
        super(SVHNDataModule,self).__init__()
        self.dataset_class = datasets.SVHN
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_transforms = test_transforms


    def get_train_set(self):
        return self.dataset_class(self.data_dir, split="train", transform=self.train_transforms)
    def get_val_set(self):
        return self.dataset_class(self.data_dir, split="test", transform=self.test_transforms)

    def prepare_data(self):
        """Saves CIFAR files to `data_dir`""" #only download
        datasets.SVHN(self.data_dir, split="test", download=True)

