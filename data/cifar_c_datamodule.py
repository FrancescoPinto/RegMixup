import pytorch_lightning as pl
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch

from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from data.datashift_datamodule import CompositeDataShiftDataModule

import sys
import torch

class CIFAR10_C(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'CIFAR-10-C'
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    filename = "cifar_10_c.tar"
    tgz_md5 = '56bf5dcef84df0e2308c6dcbcbbd8499' #already updated

    BENCHMARK_CORRUPTIONS = [
        'elastic_transform',
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'glass_blur',
        'motion_blur',
        'zoom_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'pixelate',
        'jpeg_compression',
    ]

    EXTRA_CORRUPTIONS = [
        'gaussian_blur',
        'saturate',
        'spatter',
        'speckle_noise',
    ]
    severities = [1,2,3,4,5,6]  # 5 levels of severity
    def __init__(
            self,
            root: str,
            corruption: str,
            severity: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10_C, self).__init__(root, transform=transform,
                                      target_transform=target_transform)


        self.labels_file_name = "labels"
        self.corruption = corruption
        self.severity = severity

        #only meant for evaluation, no training
        if download:
            self.download()

        if not self._check_exist():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
        self.targets = np.load(os.path.join(self.root, self.base_folder, self.labels_file_name + ".npy"))
        self.num_images = self.targets.shape[0] // 5
        self.targets = self.targets[:self.num_images] #labels of 5 severities are stacked, just need to read one of them
        self.data = np.load(os.path.join(self.root, self.base_folder, self.corruption + ".npy"))
        self.data = self.data[(self.severity - 1) * self.num_images: self.severity * self.num_images, :,:,:]
        self._load_meta()

    def _load_meta(self) -> None:
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index,:,:,:], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return self.num_images

    def _check_exist(self):
        return os.path.exists(os.path.join(self.root,self.filename)) and os.path.exists(os.path.join(self.root,self.base_folder))
    def download(self) -> None:
        if self._check_exist():
            print(f'Files for corruption {self.corruption} and severity {self.severity} already downloaded and verified')
            return
        else:
            print(f"Dataset not found in {os.path.join(self.root,self.filename)}, please download it from https://zenodo.org/record/2535967#.Yg-2v1vP3mE and https://zenodo.org/record/3555552#.Yg-2v1vP3mE. Due to a bug in torchvision's download_and_extract_archive we cannot easily download it automatically")
            sys.exit()
            #download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



class CIFAR100_C(CIFAR10_C):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'CIFAR-100-C'
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
    filename = "cifar_100_c.tar"
    tgz_md5 = '11f0ed0f1191edbf9fa23466ae6021d3' #already updated

    def _load_meta(self) -> None:
        self.classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


class CIFAR_C_DataModule(CompositeDataShiftDataModule):

    def __init__(self, corrupted_datasets_list, dataset_name = "cifar10",
                 batch_size: int = 128, num_workers:int = 8, test_transforms:transforms.Compose = None):
        super(CIFAR_C_DataModule,self).__init__(corrupted_datasets_list, batch_size, num_workers)
        self.dataset_name = dataset_name
        self.original_num_classes = 10 if dataset_name == "cifar10" else 100
        self.num_classes = self.original_num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        print("WARNING: Download the dataset manually")


    def get_val_set(self):
        return torch.utils.data.ConcatDataset(self.subsets_datasets_list)


    @classmethod
    def get_composite_data_shift_module(cls, dataset_directory,dataset_name, transform, batch_size, num_workers):
        max_severity = 6
        kwargs = {'num_workers': num_workers, "batch_size": batch_size}
        corrupted_datasets_list = []
        for corruption in CIFAR10_C.BENCHMARK_CORRUPTIONS:
            for severity in range(1, max_severity):
                corrupted_datasets_list.append(
                    (f"{corruption}_{severity}", corrupted_datasets[dataset_name.upper() + "_C"](
                        dataset_directory, corruption=corruption, severity=severity,
                        transform=transform, download=True)))
                print(corrupted_datasets_list[-1][0], len(corrupted_datasets_list[-1][1]))

        return CIFAR_C_DataModule(corrupted_datasets_list, **kwargs)


corrupted_datasets = {
    "CIFAR10_C": CIFAR10_C,
    "CIFAR100_C": CIFAR100_C
}
