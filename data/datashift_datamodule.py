import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from data.train_val_only_data_module import ValOnlyDataModule, TrainValOnlyDataModule


class CompositeDataShiftDataModule(ValOnlyDataModule):
    def __init__(self, subsets_datasets, batch_size, num_workers):
        super(CompositeDataShiftDataModule,self).__init__()
        self.subsets_datasets_list = []
        self.subsets_datasets_names = []
        for d in subsets_datasets:
            self.subsets_datasets_names.append(d[0])
            self.subsets_datasets_list.append(d[1])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        print("WARNING: Download the dataset manually")

    def setup(self, stage: str):
        # import pdb; pdb.set_trace()
        self.test_dataset = self.get_val_set()
        self.valid_dataset = self.test_dataset

        self.dims = tuple(self.test_dataset[0][0].shape)

        self.offsets = self.valid_dataset.cumulative_sizes  # list with offsets of start new dataset

    def val_dataloader(self):
        return self.dataloader("val")

    def test_dataloader(self):
        return self.dataloader("test")  # test and validation data are the same

    @classmethod
    def get_composite_data_shift_module(cls, dataset_directory, dataset_name, transform, batch_size, num_workers):
        raise Exception("Should implement the method get_composite_data_shift_module")


