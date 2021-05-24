import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import Subset, DataLoader

from PIL import Image

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def check_cifar100_dataloader(dl): #dl: list[run_i][group_10]
  coarse_label = [
  'apple',
  'aquarium_fish',
  'baby',
  'bear',
  'beaver',
  'bed',
  'bee',
  'beetle',
  'bicycle',
  'bottle',
  'bowl',
  'boy',
  'bridge',
  'bus',
  'butterfly',
  'camel',
  'can',
  'castle',
  'caterpillar',
  'cattle',
  'chair',
  'chimpanzee',
  'clock',
  'cloud',
  'cockroach',
  'couch',
  'crab',
  'crocodile',
  'cup',
  'dinosaur',
  'dolphin',
  'elephant',
  'flatfish',
  'forest',
  'fox',
  'girl',
  'hamster',
  'house',
  'kangaroo',
  'computer_keyboard',
  'lamp',
  'lawn_mower',
  'leopard',
  'lion',
  'lizard',
  'lobster',
  'man',
  'maple_tree',
  'motorcycle',
  'mountain',
  'mouse',
  'mushroom',
  'oak_tree',
  'orange',
  'orchid',
  'otter',
  'palm_tree',
  'pear',
  'pickup_truck',
  'pine_tree',
  'plain',
  'plate',
  'poppy',
  'porcupine',
  'possum',
  'rabbit',
  'raccoon',
  'ray',
  'road',
  'rocket',
  'rose',
  'sea',
  'seal',
  'shark',
  'shrew',
  'skunk',
  'skyscraper',
  'snail',
  'snake',
  'spider',
  'squirrel',
  'streetcar',
  'sunflower',
  'sweet_pepper',
  'table',
  'tank',
  'telephone',
  'television',
  'tiger',
  'tractor',
  'train',
  'trout',
  'tulip',
  'turtle',
  'wardrobe',
  'whale',
  'willow_tree',
  'wolf',
  'woman',
  'worm',
  ]

  dataiter = iter(dl[0][0])
  images = dataiter.next()
  plt.imshow(np.transpose(images[1][10]))

  print(coarse_label[images[2][10].item()])

def plot_confusion_matrix(targets, predictions, RANDOM_SEED, title, cmap = 'viridis', save_directory = None):
  cm = confusion_matrix(targets, predictions)
  fig, ax = plt.subplots(figsize = (5, 5), dpi = 100)
  sns.heatmap(cm, cmap=cmap, ax=ax)
  plt.ylabel('True class')
  plt.xlabel('Predicted class')
  plt.title("{} - seed: {}".format(title, RANDOM_SEED))
  if save_directory != None:
    fig.savefig(save_directory)
  plt.show()
  
def plot_test_accuracies(stats, save_directory = None):
  mean = np.array(stats)[:, 0]
  std = np.array(stats)[:, 1]
  fig, ax = plt.subplots(figsize = (10, 5), dpi = 100)
  x = np.arange(10, 101, 10)
  ax.errorbar(x, mean, std)
  ax.set_title("Test accuracy")
  ax.set_xlabel("Number of classes")
  ax.set_ylabel("Accuracy")
  plt.tight_layout()
  ax.legend()
  if save_directory != None:
    fig.savefig(save_directory)
  plt.show()
  
def plot_train_val(train_stats, val_stats, loss = bool, save_directory = None):
  train_mean = np.array(train_stats)[:, 0]
  train_std = np.array(train_stats)[:, 1]
  val_mean = np.array(val_stats)[:, 0]
  val_std = np.array(val_stats)[:, 1]
  fig, ax = plt.subplots(figsize = (10, 5), dpi = 100)
  x = np.arange(10, 101, 10)
  ax.errorbar(x, train_mean, train_std, label = 'Training')
  ax.errorbar(x, val_mean, val_std, label = 'Validation')
  if loss:
    ax.set_title("Training and validation loss")
    ax.set_ylabel("Loss")
  else:
    ax.set_title("Training and validation accuracy")
    ax.set_ylabel("Accuracy")
  ax.set_xlabel("Number of classes")
  plt.tight_layout()
  ax.legend()
  if save_directory != None:
    fig.savefig(save_directory)
  plt.show()
