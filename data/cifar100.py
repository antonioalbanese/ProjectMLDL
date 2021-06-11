"""
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
"""

from PIL import Image
import numpy as np
from typing import Any, Callable, Optional, Tuple
from copy import copy
import torch
from torchvision import transforms
from torchvision import datasets


class CIFAR100(torch.utils.data.Dataset):
    
    def __init__(self, root, train, download, random_state, transform=None):
        self.train = train
        self.transform = transform
        self.transform_status = True
        
        self.dataset = datasets.cifar.CIFAR100(root, train, download, transform)
        self.targets = np.array(self.dataset.targets)
        self.splits = self.make_class_splits(random_state)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        true_index = self.index_map[index]
        img, target = self.data[true_index], self.targets[true_index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if (self.transform is not None) and (self.transform_status is True): img = self.transform(img)
            
        target = self.target_map[target]

        return true_index, img, target

    def __len__(self) -> int:
        return len(self.index_map)
    
    def set_transform_status(self, state: bool):
        self.transform_status = bool
        
    def shuffle_list(self, random_state: int, list_to_shuffle):
        rs = np.random.RandomState(random_state)
        rs.shuffle(list_to_shuffle)
        return list_to_shuffle
    
    def make_class_splits(self, random_state: int):
        dict_splits = dict.fromkeys(np.arange(0, 10))
        rand_targets = list(range(0,  100))
        rand_targets = self.shuffle_list(rand_targets)
        
        for g in range(10):
            dict_splits[g] = rand_targets[g*10 : (g+1)*10]
            
        self.target_map = {key:value for value,key in enumerate(rand_targets)}
        return dict_splits
    
    def set_index_map(self, index_list):
        mask = np.isin(self.targets, index_list)
        self.index_map = {alias:true for alias,true in enumerate(np.where(mask)[0])}
        
    def get_true_index(self, alias):
        return self.index_map[alias]
        
    def train_val_split(self, val_size: float, random_state: int):
        l = len(self.index_map)
        split = int(np.floor(val_size*l))
        index_list = self.shuffle_list(list(range(l)))
        return index_list[split:], index_list[:split]
    
    def set_exemplars(self, index_list):
        self.index_map.update({
            alias:true 
            for alias,true in zip(range(len(self.index_map), len(index_list)), index_list)
        })
