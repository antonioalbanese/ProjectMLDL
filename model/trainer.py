import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.backends import cudnn
from copy import copy, deepcopy

#(self, device, net, param_opt, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, val_dl, test_dl)
#(self, device, net, criterion, optimizer, scheduler, train_dl, validation_dl, test_dl):

class Trainer(torch.nn.Module):
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl):
    super().__init__()
    self.DEVICE = device
    self.MILESTONES = MILESTONES
    self.MOMENTUM = MOMENTUM
    self.START_LR = LR
    self.WEIGHT_DECAY = WEIGHT_DECAY
    self.GAMMA = GAMMA

    self.net = net
    self.best_net = self.net

    self.criterion = nn.BCEWithLogitsLoss().to(self.DEVICE)
    self.parameters_to_optimize = self.net.parameters()
    self.optimizer = optim.SGD(self.parameters_to_optimize, lr=self.START_LR, momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
    
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.MILESTONES, gamma=self.GAMMA)

    self.train_dl = train_dl
    self.validation_dl = validation_dl
    self.test_dl = test_dl
    
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
      output = self.net(images)    
      loss = self.criterion(output, one_hot_labels)

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
  
  def validate(self, classes_group_idx):
    self.net.train(False)
    running_loss = 0
    running_corrects = 0
    total = 0

    for _, images, labels in self.validation_dl[classes_group_idx]:
      total += labels.size(0)
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      one_hot_labels = self.onehot_encoding(labels) 
      output = self.net(images)     
      loss = self.criterion(output, one_hot_labels)

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

  def add_output_nodes(self):
    self.net.fc = nn.Linear(self.net.fc.in_features, self.net.fc.out_features + 10, bias=False)
    self.net.fc.weight.data[:self.net.fc.out_features] = self.net.fc.weight.data

  def onehot_encoding(self, labels):   
    enc_labels = torch.eye(self.net.fc.out_features)[labels]
    return enc_labels.to(self.DEVICE)
