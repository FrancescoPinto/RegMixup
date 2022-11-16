import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch

class OODDataModule(pl.LightningDataModule):

        def __init__(self, ind_datamodule, ind_datamodule_name, ood_datamodules_dict, return_train_in_val=False):
            super().__init__()
            self.ind_datamodule = ind_datamodule
            self.ind_datamodule_name = ind_datamodule_name
            self.ood_datamodules_names, self.ood_datamodules = zip(*ood_datamodules_dict.items())
            self.ood_datamodules_names, self.ood_datamodules = list(self.ood_datamodules_names), list(self.ood_datamodules)
            self.num_classes = self.ind_datamodule.num_classes
            self.batch_size = self.ind_datamodule.batch_size
            self.num_workers = self.ind_datamodule.num_workers
            self.return_train_in_val = return_train_in_val

        def prepare_data(self):
            self.ind_datamodule.prepare_data()
            for d in self.ood_datamodules:
                d.prepare_data()

        def setup(self, stage: str):
            # self.ind_datamodule.setup(stage)
            # self.train_dataset = self.ind_datamodule.train_dataset

            # self.valid_dataset = self.ind_datamodule.valid_dataset
            mixed_datasets = self.mix_ind_and_ood("val")
            self.valid_dataset = torch.utils.data.ConcatDataset(mixed_datasets)
            self.valid_offsets = self.valid_dataset.cumulative_sizes #list with offsets of start new dataset


            mixed_datasets = self.mix_ind_and_ood("test")
            self.test_dataset = torch.utils.data.ConcatDataset(mixed_datasets)
            self.test_offsets = self.test_dataset.cumulative_sizes #list with offsets of start new dataset



        def mix_ind_and_ood(self, stage):
            # if self.return_train_in_val: #needed for tsne evaluations
            #     test_datasets = [self.ind_datamodule.train_dataset, self.ind_datamodule.valid_dataset]
            #     self.is_ind_labels = torch.ones(len(test_datasets[0])+len(test_datasets[1]))
            # else:
            self.ind_datamodule.setup(None)
            if stage == "test":
                ind_set = self.ind_datamodule.test_dataset
            elif stage == "val":
                ind_set = self.ind_datamodule.valid_dataset

            all_datasets = [ind_set]

            if stage == "val":
                self.is_ind_val_labels = torch.ones(len(all_datasets[0]))
                self.is_ood_val_labels = []
            elif stage == "test":
                self.is_ind_test_labels = torch.ones(len(all_datasets[0]))
                self.is_ood_test_labels = []


            for d in self.ood_datamodules:
                d.setup(None)
                if stage == "test":
                    ood_set = d.test_dataset
                    all_datasets.append(ood_set)
                    self.is_ood_test_labels.append(torch.zeros(len(all_datasets[-1])))
                elif stage == "val":
                    ood_set = d.valid_dataset
                    all_datasets.append(ood_set)
                    self.is_ood_val_labels.append(torch.zeros(len(all_datasets[-1])))

            return all_datasets

        def _dataloader(self, stage: str):
            """Train/validation loaders."""
            shuffle = False
            if stage == "train":
                shuffle = True
                _dataset = self.train_dataset
            elif stage == "val":
                _dataset = self.valid_dataset
            elif stage == "test":
                _dataset = self.test_dataset

            loader = DataLoader(dataset=_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                shuffle=shuffle, pin_memory=True)

            return loader

        def train_dataloader(self):
            return self._dataloader("train")

        def val_dataloader(self):
            return self._dataloader("val")

        def test_dataloader(self):
            return self._dataloader("test")  # test and validation data are the same


