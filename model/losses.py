import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from math import floor
from copy import copy, deepcopy
from model.icarl import iCaRL
from data.exemplar import Exemplar
import random

class iCaRL_Loss(iCaRL):
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform)
  
  def train_model(self, num_epochs, loss):
    
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
        e_loss, e_acc = self.train_epoch(g, loss)
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
      self.construct_exemplar_set(self.train_set[g], m, False)
      
      test_accuracy, true_targets, predictions = self.test_classify(g, self.train_set[g])
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
  
  def train_epoch(self, classes_group_idx, dist_loss):
    self.net.train()
    if self.old_net is not None: self.old_net.train(False)
    if self.best_net is not None: self.best_net.train(False)
    running_loss = 0
    running_corrects = 0
    total = 0

    for _, images, labels in self.train_dl[classes_group_idx]:
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      num_classes = self.net.fc.out_features
      
      if dist_loss is not None:
        if dist_loss == 'cosine':
          dist_criterion = nn.CosineEmbeddingLoss()
        elif dist_loss == 'l2':
          dist_criterion = nn.MSELoss()
        elif dist_loss == 'l1':
          dist_criterion = nn.L1Loss()
        else:
          dist_criterion = None
        output, loss = self.compute_loss(images, labels, num_classes, dist_criterion)
      else:
        one_hot_labels = self.onehot_encoding(labels)[:, num_classes-10: num_classes]
        output, loss = self.distill_loss(images, one_hot_labels, num_classes)

      running_loss += loss.item()
      _, preds = torch.max(output.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
      total += labels.size(0)
      
      loss.backward()
      self.optimizer.step()
      
    else:
      epoch_loss = running_loss/len(self.train_dl[classes_group_idx])
      epoch_acc = running_corrects/float(total)
      print("traing_loss = {0}".format(epoch_loss))
      
    return epoch_loss, epoch_acc
  
###############################################################################################################
  
  def compute_loss(self, images, labels, num_classes, dist_criterion=None):
    if dist_criterion is not None:
      class_criterion = nn.CrossEntropyLoss()

      if self.old_net is not None:
        self.old_net.to(self.DEVICE)    
        sigmoid = nn.Sigmoid()
        old_net_output = sigmoid(self.old_net(images))[:, :num_classes-10]
        output = self.net(images)
        dist_loss = dist_criterion(output[:,:num_classes-10], old_net_output,torch.ones(images.shape[0]).to(self.DEVICE))
        class_loss = class_criterion(output,labels)
        loss = dist_loss + class_loss

      else:
        output = self.net(images)
        loss = class_criterion(output,labels)      
    else:
      one_hot_labels = self.onehot_encoding(labels)[:, num_classes-10: num_classes]
      output, loss = self.distill_loss(images, one_hot_labels, num_classes)   
    return output, loss
