import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from math import floor
from copy import copy, deepcopy
from model.lwf import LearningWithoutForgetting
from data.exemplar import Exemplar
import random

class iCaRL(LearningWithoutForgetting):
  
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl)
    self.BATCH_SIZE = BATCH_SIZE
    self.VALIDATE = True

    self.train_set = train_subset
    
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.memory_size = 2000
    self.exemplar_set = []
    self.means = None
  
  def train_model(self, num_epochs, herding: bool, classify: bool):
    
    cudnn.benchmark
    
    logs = {'group_train_loss': [float for j in range(10)],
             'group_train_accuracies': [float for j in range(10)],
             'predictions': [int],
             'test_accuracies': [float for j in range(10)],
             'true_labels': [int],
             'val_accuracies': [float for j in range(10)],
             'val_losses': [float for j in range(10)]}
    
    for g in range(10):
      self.net.to(self.DEVICE)
      if self.old_net is not None: self.old_net = self.old_net.to(self.DEVICE)
      
      self.parameters_to_optimize = self.net.parameters()
      self.optimizer = optim.SGD(self.parameters_to_optimize, lr=self.START_LR, momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
      self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.MILESTONES, gamma=self.GAMMA)
      
      best_acc = 0
      self.best_net = deepcopy(self.net)
      
      # augment train_set with exemplars and define DataLoaders for the current group
      self.update_representation(g)

      for epoch in range(num_epochs):
        e_loss, e_acc = self.train_epoch(g)
        e_print = epoch + 1
        print(f"Epoch {e_print}/{num_epochs} LR: {self.scheduler.get_last_lr()}")
        
        validate_loss, validate_acc = self.validate(g)
        g_print = g + 1
        print(f"Validation accuracy on group {g_print}/10: {validate_acc:.2f}")
        self.scheduler.step()
        
        if self.VALIDATE and validate_acc > best_acc:
          best_acc = validate_acc
          self.best_net = deepcopy(self.net)
          best_epoch = epoch
          print("Best model updated")
        print("")
        
      print(f"Group {g_print} Finished!")
      be_print = best_epoch + 1
      print(f"Best accuracy found at epoch {be_print}: {best_acc:.2f}")
      
      m = self.reduce_exemplar_set()
      self.construct_exemplar_set(self.train_set[g], m, herding)
      
      if classify is True:
        test_accuracy, true_targets, predictions = self.test_classify(g, self.train_set[g])
      else:
        test_accuracy, true_targets, predictions = self.test(g)
      print(f"Testing classes seen so far, accuracy: {test_accuracy:.2f}")
      print("")
      print("=============================================")
      print("")
      logs['group_train_loss'][g] = e_loss
      logs['group_train_accuracies'][g] = e_acc
      logs['val_losses'][g] = validate_loss
      logs['val_accuracies'][g] = validate_acc
      logs['test_accuracies'][g] = test_accuracy

      if g < 9:
        self.add_output_nodes()
        self.old_net = deepcopy(self.best_net)

    logs['true_labels'] = true_targets
    logs['predictions'] = predictions
    return logs

########################################################################################################################
  
  def test_classify(self, classes_group_idx, train_set):
    self.best_net.train(False)
    if self.best_net is not None: self.best_net.train(False)
    if self.old_net is not None: self.old_net.train(False)
    running_corrects = 0
    total = 0

    all_preds = torch.tensor([])
    all_preds = all_preds.type(torch.LongTensor)
    all_targets = torch.tensor([])
    all_targets = all_targets.type(torch.LongTensor)
    
    self.means = None
    if train_set is not None: train_set.dataset.set_transform_status(False)
    
    for _, images, labels in self.test_dl[classes_group_idx]:
      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)
      total += labels.size(0)

      with torch.no_grad():
        preds = self.classify(images, train_set)
      
      running_corrects += torch.sum(preds == labels.data).data.item()

      all_targets = torch.cat((all_targets.to(self.DEVICE), labels.to(self.DEVICE)), dim=0)
      all_preds = torch.cat((all_preds.to(self.DEVICE), preds.to(self.DEVICE)), dim=0)

    else:
      if train_set is not None: train_set.dataset.set_transform_status(True)
      accuracy = running_corrects / float(total)  

    return accuracy, all_targets, all_preds
  
  def update_representation(self, classes_group_idx):
    print(f"Length of exemplars set: {sum([len(self.exemplar_set[i]) for i in range(len(self.exemplar_set))])}")
    exemplars = Exemplar(self.exemplar_set, self.train_transform)
    ex_train_set = ConcatDataset([exemplars, self.train_set[classes_group_idx]])
    
    tmp_dl = DataLoader(ex_train_set,
                        batch_size=self.BATCH_SIZE,
                        shuffle=True, 
                        num_workers=4,
                        drop_last=True)
    self.train_dl[classes_group_idx] = copy(tmp_dl)
    
  def reduce_exemplar_set(self):
    m = floor(self.memory_size / self.net.fc.out_features)      
    print(f"Target number of exemplars: {m}")

    # from the current exemplar set, keep only first m
    for i in range(len(self.exemplar_set)):
      current_exemplar_set = self.exemplar_set[i]
      self.exemplar_set[i] = current_exemplar_set[:m]
    
    return m
  
  def construct_exemplar_set(self, train_set, m, herding: bool):   
    train_set.dataset.set_transform_status(False)    
    samples = [[] for i in range(10)]
    new_exemplar_set = [[] for i in range(10)]
    for _, images, labels in train_set:
      labels = labels % 10
      samples[labels].append(images)
    train_set.dataset.set_transform_status(True)
    
    if herding is True:
      new_exemplar_set = self.prioritized_selection(samples, new_exemplar_set, m)
    else:
      new_exemplar_set = self.random_selection(samples, new_exemplar_set, m)
    
    self.exemplar_set.extend(new_exemplar_set)
      
  def prioritized_selection(self, samples, exemplars, m):
    for i in range(10):
      print(f"Randomly extracting exemplars from class {i} of current split... ", end="")
      exemplars[i] = random.sample(samples[i], m) # delete this and implement algorithm 4 from iCaRL paper
      print(f"Extracted {len(exemplars[i])} exemplars.")
    return exemplars
  
  def random_selection(self, samples, exemplars, m):
    for i in range(10):
      print(f"Randomly extracting exemplars from class {i} of current split... ", end="")
      exemplars[i] = random.sample(samples[i], m)
      print(f"Extracted {len(exemplars[i])} exemplars.")
    return exemplars

########## ALGORITHM 1 ################################################################## 

  def classify(self, images, train_set=None):
    feature_map = self.features_extractor(images)
    for i in range(feature_map.size(0)):
      feature_map[i] = feature_map[i] / feature_map[i].norm()
    feature_map = feature_map.to(self.DEVICE)

    if self.means is None:
      self.mean_of_exemplars(train_set)

    class_labels = []
    for i in range(feature_map.size(0)):
      nearest_prototype = torch.argmin(torch.norm(feature_map[i]-self.means, dim=1))
      class_labels.append(nearest_prototype)
    
    return torch.stack(class_labels)

  def features_extractor(self, images, batch=True, transform=None):
    assert not (batch is False and transform is None), "if a PIL image is passed to extract_features, a transform must be defined"
    self.net.train(False)
    if self.best_net is not None: self.best_net.train(False)
    if self.old_net is not None: self.old_net.train(False)
    
    if batch is False:
      images = transform(images)
      images = images.unsqueeze(0)
    images = images.to(self.DEVICE)
    
    if self.VALIDATE: features = self.best_net.features(images)
    else: features = self.net.features(images)
    if batch is False: features = features[0]
    
    return features
  
  def mean_of_exemplars(self, train_set=None):
    print("Computing mean of exemplars... ", end="")
    self.means = []
    if train_set is not None:
      train_features = [[] for i in range(10)]
      for _, img, labels in train_set:
        f = self.features_extractor(img, False, self.test_transform)
        f = f / f.norm()
        train_features[labels % 10].append(f)

    num_classes = len(self.exemplar_set)
    for i in range(num_classes):
      if (train_dataset is not None) and (i in range(num_classes-10, num_classes)):
        f_list = train_features[i % 10]
      else:
        f_list = []

      for img in self.exemplar_set[i]:
        f = self.features_extractor(img, False, self.test_transform)
        f = f / f.norm()
        f_list.append(f)

      f_list = torch.stack(f_list)
      class_means = f_list.mean(dim=0)
      class_means = class_means/class_means.norm()

      self.means.append(class_means)

    self.means = torch.stack(self.means).to(self.DEVICE)
    print("done")
