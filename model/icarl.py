import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from copy import copy, deepcopy
from model.lwf import LearningWithoutForgetting
from data.exemplar import Exemplar

class iCaRL(LearningWithoutForgetting):
  
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_set, validation_set, test_set, train_transform, test_transform):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl)
   
    self.BATCH_SIZE = BATCH_SIZE

    self.train_set = train_set
    self.validation_set = validation_set
    self.test_set = test_set
    
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.memory_size = 2000
    self.exemplar_set = []
  
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
      
      # augment train_set with exemplars and define DataLoaders for the current group
      self.update_representation(g, self.train_set[g], self.validation_set[g])

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
      
      m = self.reduce_exemplar_set()
      self.construct_exemplar_set(self.train_set[g], m)
      
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
        output, loss = self.lwf_loss(images, one_hot_labels, classes_group_idx)

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

  def lwf_loss(self, images, one_hot_labels, classes_group_idx):
    self.old_net.to(self.DEVICE)
    self.old_net.eval()
    
    sigmoid = nn.Sigmoid()
    old_outputs = sigmoid(self.old_net(images))    
    one_hot_labels = torch.cat((old_outputs[:, 0:classes_group_idx*10], 
                                one_hot_labels[:, classes_group_idx*10:classes_group_idx*10+10]), 1)   
    output = self.net(images)   
    loss = self.criterion(output, one_hot_labels)
    
    return output, loss
  
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
        output, loss = self.lwf_loss(images, one_hot_labels, classes_group_idx)

      running_loss += loss.item()
      _, preds = torch.max(output.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
      
    else:
      val_loss = running_loss/len(self.validation_dl[classes_group_idx])
      val_accuracy = running_corrects / float(total)

    return val_loss, val_accuracy
  
  def test(self, classes_group_idx):
    self.best_net.train(False)
    running_corrects = 0
    total = 0

    all_preds = torch.tensor([])
    all_preds = all_preds.type(torch.LongTensor)
    all_targets = torch.tensor([])
    all_targets = all_targets.type(torch.LongTensor)
    
    for _, images, labels in self.test_dl[classes_group_idx]:
      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)
      total += labels.size(0)

      outputs = self.best_net(images)
      
      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()

      all_targets = torch.cat((all_targets.to(self.DEVICE), labels.to(self.DEVICE)), dim=0)
      all_preds = torch.cat((all_preds.to(self.DEVICE), preds.to(self.DEVICE)), dim=0)

    else:
      accuracy = running_corrects / float(total)  

    return accuracy, all_targets, all_preds
  
  def update_representation(self, classes_group_idx, train_set, validation_set):
    print(f"Length of exemplars set: {sum([len(self.exemplar_set[i]) for i in range(len(self.exemplar_set))])}")
    exemplars = Exemplar(self.exemplar_set, self.train_transform)
    ex_train_set = ConcatDataset([exemplars, train_set])
    
    tmp_dl = DataLoader(ex_train_set,
                        batch_size=self.BATCH_SIZE,
                        shuffle=True, 
                        num_workers=4,
                        drop_last=True)
    train_dl[classes_group_idx] = copy(tmp_dl)
    
    tmp_dl = DataLoader(validation_set,
                        batch_size=self.BATCH_SIZE,
                        shuffle=True, 
                        num_workers=4,
                        drop_last=True)
    validation_dl[classes_group_idx] = copy(tmp_dl)
    
  def reduce_exemplar_set(self):
    m = floor(self.memory_size / self.net.fc.out_features)      
    print(f"Target number of exemplars: {m}")

    # from the current exemplar set, keep only first m
    for i in range(len(self.exemplar_set)):
      current_exemplar_set = self.exemplar_set[i]
      self.exemplar_set[i] = current_exemplar_set[:m]
    
    return m
  
  def construct_exemplar_set(self, train_set, m):
    
    self.exemplar_set.extend(new_exemplar_set)
      
