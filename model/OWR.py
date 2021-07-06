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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import ParameterGrid

class owrIncremental(LearningWithoutForgetting):
  
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform, test_mode):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl)
    self.BATCH_SIZE = BATCH_SIZE
    self.VALIDATE = True

    self.train_set = train_subset
    
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.memory_size = 2000
    self.exemplar_set = []
    self.means = None
    self.test_mode = test_mode
  
  def train_model(self, num_epochs):
    
    cudnn.benchmark
    
    logs = {'group_train_loss': [float for j in range(5)],
             'group_train_accuracies': [float for j in range(5)],
             'predictions': [int],
             'test_accuracies': [float for j in range(5)],
             'true_labels': [int],
             'val_accuracies': [float for j in range(5)],
             'val_losses': [float for j in range(5)],
             'open_values': [float for j in range(5)],
             'closed_values': [float for j in range(5)]}
    
    for g in range(5):
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
      self.construct_exemplar_set(self.train_set[g], m)
      
      if self.test_mode == "open":
        test_accuracy, true_targets, predictions, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values = self.test_openset(g)
        corrects_in_unknown = torch.sum(only_unknown_targets == only_unknown_preds).data.item()
        print(f"Testing openset: 50 clasess never seen, rejection accuracy: {test_accuracy}")
        print(f"{only_unknown_targets.size(0)} unkown found over {true_targets.size(0)} images unkown")
        logs['open_values'] = all_values
      elif self.test_mode == "closed"
        test_accuracy, true_targets, predictions, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values = self.test_rejection(g)
        corrects_in_unknown = torch.sum(only_unknown_targets == only_unknown_preds).data.item()
        print(f"Testing classes seen so far, accuracy: {test_accuracy:.2f}")
        print(f"{only_unknown_targets.size()[0]} unkown found, {corrects_in_unknown} predictions of them where correct")
        logs['closed_values'] = all_values
      elif self.test_mode == "harmonic":
             #harmonic test
      
      
      
      
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
  
  def construct_exemplar_set(self, train_set, m):   
    train_set.dataset.set_transform_status(False)    
    samples = [[] for i in range(10)]
    new_exemplar_set = [[] for i in range(10)]
    for _, images, labels in train_set:
      labels = labels % 10
      samples[labels].append(images)
    train_set.dataset.set_transform_status(True)
    
    new_exemplar_set = self.random_selection(samples, new_exemplar_set, m)
    
    self.exemplar_set.extend(new_exemplar_set)
      
  
  def random_selection(self, samples, exemplars, m):
    for i in range(10):
      print(f"Randomly extracting exemplars from class {i} of current split... ", end="")
      exemplars[i] = random.sample(samples[i], m)
      print(f"Extracted {len(exemplars[i])} exemplars.")
    return exemplars


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
  
#################################################################
  def test_openset(self,classes_group_idx):
    self.best_net.train(False)
    threshold = 0.5
    running_corrects = 0
    total = 0

    all_preds_with_unknown = torch.tensor([])
    all_preds_with_unknown = all_preds_with_unknown.type(torch.LongTensor)
    all_targets = torch.tensor([])
    all_targets = all_targets.type(torch.LongTensor)
    all_targets_as_unknown = torch.tensor([])
    all_targets_as_unknown = all_targets.type(torch.LongTensor)
    only_unknown_values = torch.tensor([])
    only_unknown_values = only_unknown_values.type(torch.DoubleTensor)
    only_unknown_targets = torch.tensor([])
    only_unknown_targets = only_unknown_targets.type(torch.LongTensor)
    only_unknown_preds = torch.tensor([])
    only_unknown_preds = only_unknown_preds.type(torch.LongTensor)
    
    for i in range(5,10):
      for _, images, labels in self.test_dl[i]:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)
        total += labels.size(0)

        outputs = self.best_net(images)

        values, preds = torch.max(outputs.data, 1)
        below_mask = values < threshold
        unknowkn_class = classes_group_idx*10+10 #Assign an index to unknown class, for instance at the first iteration we have class from 0 to 9, unkown class will be 10
        preds_with_unknown = torch.where(below_mask, torch.tensor(unknowkn_class).to(self.DEVICE), preds)
        only_unknown_preds_batch = preds[below_mask]
        only_unknown_targets_batch = labels[below_mask]
        only_unknown_values_batch = values[below_mask]
        
        label_unknow_tensor = torch.tensor([unknowkn_class for _ in range(labels.size(0))]).to(self.DEVICE)
        all_targets_as_unknown = torch.cat((all_targets_as_unknown.to(self.DEVICE), label_unknow_tensor.to(self.DEVICE)), dim=0) #unknown class will be the true targets for all the test set, since we aspect that the model reject all of them
        
        running_corrects += torch.sum(preds_with_unknown == label_unknow_tensor.data).data.item()
        all_targets = torch.cat((all_targets.to(self.DEVICE), labels.to(self.DEVICE)), dim=0)
        only_unknown_preds = torch.cat((only_unknown_preds.to(self.DEVICE), only_unknown_preds_batch.to(self.DEVICE)), dim=0)
        only_unknown_targets = torch.cat((only_unknown_targets.to(self.DEVICE), only_unknown_targets_batch.to(self.DEVICE)), dim=0)
        only_unknown_values = torch.cat((only_unknown_values.to(self.DEVICE), only_unknown_values_batch.to(self.DEVICE)), dim=0)
        all_preds_with_unknown = torch.cat((all_preds_with_unknown.to(self.DEVICE), preds_with_unknown.to(self.DEVICE)), dim=0)

    else:
      accuracy = running_corrects / float(total)  
    return accuracy, all_targets, all_preds_with_unknown, only_unknown_targets, only_unknown_preds, only_unknown_values, values


  def test_rejection(self, classes_group_idx):
      self.best_net.train(False)
      threshold = 0.5
      running_corrects = 0
      total = 0

      all_preds_with_unknown = torch.tensor([])
      all_preds_with_unknown = all_preds_with_unknown.type(torch.LongTensor)
      all_targets = torch.tensor([])
      all_targets = all_targets.type(torch.LongTensor)
      only_unknown_values = torch.tensor([])
      only_unknown_values = only_unknown_values.type(torch.DoubleTensor)
      only_unknown_targets = torch.tensor([])
      only_unknown_targets = only_unknown_targets.type(torch.LongTensor)
      only_unknown_preds = torch.tensor([])
      only_unknown_preds = only_unknown_preds.type(torch.LongTensor)

      for _, images, labels in self.test_dl[classes_group_idx]:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)
        total += labels.size(0)

        outputs = self.best_net(images)

        values, preds = torch.max(outputs.data, 1)
        below_mask = values < threshold
        unknowkn_class = classes_group_idx*10+10 #Assign an index to unknown class, for instance at the first iteration we have class from 0 to 9, unkown class will be 10
        preds_with_unknown = torch.where(below_mask, torch.tensor(unknowkn_class).to(self.DEVICE), preds)
        only_unknown_preds_batch = preds[below_mask]
        only_unknown_targets_batch = labels[below_mask]
        only_unknown_values_batch = values[below_mask]


        running_corrects += torch.sum(preds_with_unknown == labels.data).data.item()

        all_targets = torch.cat((all_targets.to(self.DEVICE), labels.to(self.DEVICE)), dim=0)
        only_unknown_preds = torch.cat((only_unknown_preds.to(self.DEVICE), only_unknown_preds_batch.to(self.DEVICE)), dim=0)
        only_unknown_targets = torch.cat((only_unknown_targets.to(self.DEVICE), only_unknown_targets_batch.to(self.DEVICE)), dim=0)
        only_unknown_values = torch.cat((only_unknown_values.to(self.DEVICE), only_unknown_values_batch.to(self.DEVICE)), dim=0)
        all_preds_with_unknown = torch.cat((all_preds_with_unknown.to(self.DEVICE), preds_with_unknown.to(self.DEVICE)), dim=0)

      else:
        accuracy = running_corrects / float(total)  

      return accuracy, all_targets, all_preds_with_unknown, only_unknown_targets, only_unknown_preds, only_unknown_values, values
