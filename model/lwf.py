import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.backends import cudnn
import numpy as np
from copy import copy, deepcopy
from model.trainer import Trainer

class LearningWithoutForgetting(Trainer):
  
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl)
    self.old_net = None
  
  def train_model(self, num_epochs):
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
      
      self.parameters_to_optimize = self.net.parameters()
      self.optimizer = optim.SGD(self.parameters_to_optimize, lr=self.START_LR, momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
      self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.MILESTONES, gamma=self.GAMMA)

      best_acc = 0
      self.best_net = deepcopy(self.net)

      for epoch in range(num_epochs):
        e_loss, e_acc = self.train_epoch(g)
        e_print = epoch + 1
        print(f"Epoch {e_print}/{num_epochs} LR: {self.scheduler.get_last_lr()}")
        
        validate_loss, validate_acc = self.validate(g)
        g_print = g + 1
        print(f"Validation accuracy on group {g_print}/10: {validate_acc:.2f}")
        self.scheduler.step()
        
        if validate_acc > best_acc:
          best_acc = validate_acc
          self.best_net = deepcopy(self.net)
          best_epoch = epoch
          print("Best model updated")
        print("")
        
      print(f"Group {g_print} Finished!")
      be_print = best_epoch + 1
      print(f"Best accuracy found at epoch {be_print}: {best_acc:.2f}")
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

  def train_epoch(self, classes_group_idx):
    self.net.train()
    running_loss = 0
    running_corrects = 0
    total = 0

    for _, images, labels in self.train_dl[classes_group_idx]:
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      one_hot_labels = self.onehot_encoding(labels) 
      
      if classes_group_idx == 0:
        output = self.net(images)    
        loss = self.criterion(output, one_hot_labels)
      else:
        loss = self.lwf_loss(images, one_hot_labels, classes_group_idx)

      running_loss += loss.item()
      _, preds = torch.max(output.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
      total += labels.size(0)
      
      loss.backward()
      self.optimizer.step()
      
    else:
      epoch_loss = running_loss/len(self.train_dl[classes_group_idx])
      epoch_acc = running_corrects/float(total)
      
    return epoch_loss, epoch_acc

  ##############################################################################
  def lwf_loss(self, images, one_hot_labels, classes_group_idx):
    # One hot encoding of new task labels
    num_classes = (classes_group_idx + 1) * 10
    new_classes = (np.arange(100)[range(num_classes - 10, num_classes)]).astype(np.int32)
    one_hot_labels = torch.stack([one_hot_labels[:, i] for i in new_classes], axis=1)
    
    # Old net forward pass
    sigmoid = nn.Sigmoid()
    old_outputs = sigmoid(self.old_net(images))
    old_classes = (np.arange(100)[range(num_classes - 10)]).astype(np.int32)
    old_outputs = torch.stack([old_outputs[:, i] for i in old_classes], axis=1)
    
    # Combine new and old class targets
    targets = torch.cat((old_outputs, one_hot_labels), 1)
    
    # New net forward pass
    output = self.net(images)
    all_classes = (np.arange(100)[range(num_classes)]).astype(np.int32)
    output = torch.stack([output[:, i] for i in all_classes], axis=1)
    
    # BCE Loss with sigmoids over outputs (over targets must be done manually)
    loss = self.criterion(output, targets)
    
    return loss
  ##############################################################################
  
  def validate(self, classes_group_idx):
    self.net.eval()
    running_loss = 0
    running_corrects = 0
    total = 0

    for _, images, labels in self.validation_dl[classes_group_idx]:
      total += labels.size(0)
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      one_hot_labels = self.onehot_encoding(labels) 
      if classes_group_idx == 0:
        output = self.net(images)    
        loss = self.criterion(output, one_hot_labels)
      else:
        loss = self.lwf_loss(images, one_hot_labels, classes_group_idx)

      running_loss += loss.item()
      _, preds = torch.max(output.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
      
    else:
      val_loss = running_loss/len(self.validation_dl[classes_group_idx])
      val_accuracy = running_corrects / float(total)

    return val_loss, val_accuracy
