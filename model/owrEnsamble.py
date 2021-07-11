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

#SnapshotEnsembleOWRClassifier
from snapshot_ensables.snapshot_owr import SnapshotEnsembleOWRClassifier
from snapshot_ensables.utils.logging import set_logger

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import ParameterGrid

class owrEnsemble(iCaRL):
  
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform, test_mode, p_threshold, n_estimators):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform)
    
    self.test_mode = test_mode
    self.threshold = p_threshold
    self.n_estimators = n_estimators
  
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

    ensemble = SnapshotEnsembleOWRClassifier(estimator=self.net, n_estimators=self.n_estimators, estimator_args=None, cuda=True)
    ensemble.set_optimizer('SGD',             # parameter optimizer
                    lr=self.START_LR,            # learning rate of the optimizer
                    weight_decay=self.WEIGHT_DECAY,
                    momentum=self.MOMENTUM)  # weight decay of the optimizer
    logger = set_logger('classification_mnist_mlp')
    
    for g in range(5):
      self.net.to(self.DEVICE)
      
      self.parameters_to_optimize = self.net.parameters()
      self.optimizer = optim.SGD(self.parameters_to_optimize, lr=self.START_LR, momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
      self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.MILESTONES, gamma=self.GAMMA)
      
      

      
      
      # augment train_set with exemplars and define DataLoaders for the current group
      self.update_representation(g)

      ensemble.fit(self.train_dl[g],
                   lr_clip=None,
                   epochs=num_epochs,
                   log_interval=num_epochs,
                   test_loader=self.validation_dl[g],
                   save_model=False,
                   save_dir=None,
                   classes_group_idx = g)

        
      print(f"Group {g+1} Finished!")
      
      
      m = self.reduce_exemplar_set()
      self.construct_exemplar_set(self.train_set[g], m, False)
      
      
      




      #test_accuracy is the mean_acc
      test_accuracy,open_test_accuracy,closed_test_accuracy,open_true_targets,closed_true_targets,open_predictions,closed_predictions,open_unknown_targets,closed_unknown_targets,open_unknown_preds,closed_unknown_preds,open_unknown_values,closed_unknown_values,open_all_values,closed_all_values = self.harmonic_test(g, ensemble)
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


  def harmonic_test(self, classes_group_idx, ensemble):
    open_test_accuracy, open_true_targets, open_predictions, open_unknown_targets, open_unknown_preds, open_unknown_values, open_all_values = self.test_openset(classes_group_idx, ensemble)
    closed_test_accuracy, closed_true_targets, closed_predictions, closed_unknown_targets, closed_unknown_preds, closed_unknown_values, closed_all_values = self.test_rejection(classes_group_idx, ensemble)
    mean_acc = 1/((1/open_test_accuracy + 1/closed_test_accuracy)/2)
    return mean_acc, open_test_accuracy, closed_test_accuracy, open_true_targets, closed_true_targets, open_predictions, closed_predictions, open_unknown_targets, closed_unknown_targets, open_unknown_preds, closed_unknown_preds, open_unknown_values, closed_unknown_values, open_all_values, closed_all_values       

  def test_openset(self,classes_group_idx, ensemble):
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

        outputs = ensemble.predict(images)

        values, preds = torch.max(outputs.data, 1)
        all_values = torch.cat((all_values.to(self.DEVICE),values.to(self.DEVICE)))
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
    return accuracy, all_targets, all_preds_with_unknown, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values


  def test_rejection(self, classes_group_idx, ensemble):
      ensemble.train(False)
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

        outputs = ensemble.predict(images)

        values, preds = torch.max(outputs.data, 1)
        all_values = torch.cat((all_values.to(self.DEVICE),values.to(self.DEVICE)))
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

      return accuracy, all_targets, all_preds_with_unknown, only_unknown_targets, only_unknown_preds, only_unknown_values, all_values
