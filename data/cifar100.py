"""
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
"""

from PIL import Image
import os
import os.path
import numpy as np
import pickle
import random
from sklearn.utils import shuffle
from typing import Any, Callable, Optional, Tuple
from copy import copy

from torchvision.datasets import VisionDataset
from data.utils import check_integrity, download_and_extract_archive


class CIFAR100(VisionDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        
        random.seed(42)

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
      
    #########################################################
    
    def __get_target_images__(self, target):
      images = []
      for i in range(0, self.__len__()):
        if self.targets[i] == target:
          images.append(i)
      return images
    
    def __incremental_indexes__(self, train: bool):
      subset = []
      t = []
      n = 0
      for i in range(0, 10):
        if train:
          t = []
          for j in range(n, n+10):
            var = self.__get_target_images__(j)
            var = var[0 : len(self.__get_target_images__(j))]
            t.extend(var)
          n = n + 10
          subset.append(t)
        else: 
          for j in range(n, n+10):
            var = self.__get_target_images__(j)
            var = var[0 : len(self.__get_target_images__(j))]
            t.extend(var)
          n = n + 10
          copied = copy(t)
          subset.append(copied)
      return subset


    
    def __shuffle__(self):
      indexes = list(range(0, 100))
      random.shuffle(indexes)
      ix = []
      for i in range(0, len(indexes)):
        ix.append(self.__get_target_images__(i))
      for i in range(0, len(ix)):
        for j in ix[i]:
          self.targets[j] = indexes[i]
    
    
    
    
    def __shuffle_seed__(self, random_seed):
        random.seed(random_seed)
        data = self.data
        targets = self.targets

        data_s, targets_s = shuffle(data, targets)
        self.data = data_s
        self.targets = targets_s
