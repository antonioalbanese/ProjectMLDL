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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import ParameterGrid

class owrIncremental(iCaRL):
  
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform, test_mode, p_threshold):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform)
    
    self.test_mode = test_mode
    self.threshold = p_threshold
  
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
        print(f"Validation accuracy on group {g_print}/5: {validate_acc:.2f}")
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
      
      if self.test_mode == "open":
        test_accuracy, true_targets, predictions, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values = self.test_openset(g)
        corrects_in_unknown = torch.sum(only_unknown_targets == only_unknown_preds).data.item()
        print(f"Testing openset: 50 clasess never seen, rejection accuracy: {test_accuracy}")
        print(f"{only_unknown_targets.size(0)} unkown found over {true_targets.size(0)} images unkown")
        logs['open_values'] = all_values
      else:
        if self.test_mode == "closed":
          test_accuracy, true_targets, predictions, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values = self.test_rejection(g)
          corrects_in_unknown = torch.sum(only_unknown_targets == only_unknown_preds).data.item()
          print(f"Testing classes seen so far, accuracy: {test_accuracy:.2f}")
          print(f"{only_unknown_targets.size()[0]} unkown found, {corrects_in_unknown} predictions of them where correct")
          logs['closed_values'] = all_values
        else:
            if self.test_mode == "harmonic":
              #test_accuracy is the mean_acc
              test_accuracy,open_test_accuracy,closed_test_accuracy,open_true_targets,closed_true_targets,open_predictions,closed_predictions,open_unknown_targets,closed_unknown_targets,open_unknown_preds,closed_unknown_preds,open_unknown_values,closed_unknown_values,open_all_values,closed_all_values = self.harmonic_test(g)
              logs['open_values'][g] = open_all_values
              logs['closed_values'][g] = closed_all_values
              true_targets = torch.cat((open_true_targets.to(self.DEVICE), closed_true_targets.to(self.DEVICE))) 
              predictions = torch.cat((open_predictions.to(self.DEVICE), closed_predictions.to(self.DEVICE)))
              print(f"Testing on both open and closed world, harmonic mean acc: {test_accuracy:.2f}")
      
      
      print("")
      print("=============================================")
      print("")
      logs['group_train_loss'][g] = e_loss
      logs['group_train_accuracies'][g] = e_acc
      logs['val_losses'][g] = validate_loss
      logs['val_accuracies'][g] = validate_acc
      logs['test_accuracies'][g] = test_accuracy

      if g < 4:
        self.add_output_nodes()
        self.old_net = deepcopy(self.best_net)

    logs['true_labels'] = true_targets
    logs['predictions'] = predictions
    return logs

################################################################
  def harmonic_test(self, classes_group_idx):
    open_test_accuracy, open_true_targets, open_predictions, open_unknown_targets, open_unknown_preds, open_unknown_values, open_all_values = self.test_openset(classes_group_idx)
    closed_test_accuracy, closed_true_targets, closed_predictions, closed_unknown_targets, closed_unknown_preds, closed_unknown_values, closed_all_values = self.test_rejection(classes_group_idx)
    mean_acc = 1/((1/(open_test_accuracy+0.0001) + 1/(closed_test_accuracy+0.0001))/2)
    return mean_acc, open_test_accuracy, closed_test_accuracy, open_true_targets, closed_true_targets, open_predictions, closed_predictions, open_unknown_targets, closed_unknown_targets, open_unknown_preds, closed_unknown_preds, open_unknown_values, closed_unknown_values, open_all_values, closed_all_values       

  def test_openset(self,classes_group_idx):
    self.best_net.train(False)
    softmax = nn.Softmax(dim=1)
    threshold = self.threshold
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
    all_values = torch.tensor([])
    all_values = all_values.type(torch.LongTensor)
    
    for i in range(5,10):
      for _, images, labels in self.test_dl[i]:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)
        total += labels.size(0)

        outputs = self.best_net(images)

        values, preds = torch.max(softmax(outputs).data, 1)
        all_values = torch.cat((all_values.to(self.DEVICE),values.to(self.DEVICE)))
        below_mask = values < threshold
        #unknowkn_class = classes_group_idx*10+10 #Assign an index to unknown class, for instance at the first iteration we have class from 0 to 9, unkown class will be 10
        unknowkn_class = 100
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
    return accuracy, all_targets, all_preds_with_unknown, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values


  def test_rejection(self, classes_group_idx):
      self.best_net.train(False)
      softmax = nn.Softmax(dim=1)
      threshold = self.threshold
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
      all_values = torch.tensor([])
      all_values = all_values.type(torch.LongTensor)

      for _, images, labels in self.test_dl[classes_group_idx]:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)
        total += labels.size(0)

        outputs = self.best_net(images)

        values, preds = torch.max(softmax(outputs).data, 1)
        all_values = torch.cat((all_values.to(self.DEVICE),values.to(self.DEVICE)))
        below_mask = values < threshold
        #unknowkn_class = classes_group_idx*10+10 #Assign an index to unknown class, for instance at the first iteration we have class from 0 to 9, unkown class will be 10
        unknowkn_class = 101
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

      return accuracy, all_targets, all_preds_with_unknown, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values

    
###################################################################################################################################

class owrCosine(owrIncremental):
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform, test_mode, p_threshold):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform, test_mode, p_threshold)
  
  def train_epoch(self, classes_group_idx):
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
      #one_hot_labels = self.onehot_encoding(labels)[:, num_classes-10: num_classes]
      
      output, loss = self.compute_loss(images, labels, num_classes)

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
  
  def compute_loss(self, images, labels, num_classes):
    dist_criterion = nn.CosineEmbeddingLoss()
    class_criterion = nn.CrossEntropyLoss()

    if self.old_net is not None:
      self.old_net.to(self.DEVICE)    
      sigmoid = nn.Sigmoid()
      old_net_output = sigmoid(self.old_net(images))[:, :num_classes-10]
      output = self.net(images)
      dist_loss = dist_criterion(output[:,:num_classes-10], old_net_output, torch.ones(images.shape[0]).to(self.DEVICE))
      class_loss = class_criterion(output, labels)
      loss = dist_loss + class_loss

    else:
      output = self.net(images)
      loss = class_criterion(output, labels)      
    return output, loss
