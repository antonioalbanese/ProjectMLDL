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
from math import sqrt

#SnapshotEnsembleOWRClassifier
from snapshot_ensables.snapshot_owr import SnapshotEnsembleOWRClassifier
from snapshot_ensables.utils.logging import set_logger

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import ParameterGrid

class owrEnsemble(iCaRL):
  
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform, test_mode, p_threshold, n_estimators, confidence):
    super().__init__(device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl, BATCH_SIZE, train_subset, train_transform, test_transform)
    t_dict = {
      '0.6827' : 1,
      '0.900' : 1.47,
      '0.950' : 1.94,
      '0.975' : 2.45,
      '0.990' : 3.36
    }

    self.test_mode = test_mode
    self.threshold_list = p_threshold
    self.n_estimators = n_estimators
    self.confidence = t_dict[confidence]

  
  def train_model(self, num_epochs):
    
    cudnn.benchmark
    
    logs = {
             'predictions': [],
             'test_accuracies': [[] for j in range(5)],
             'true_labels': [],
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
      

      #test_accuracies contain mean_acc for different thresholds
      test_accuracies,open_test_accuracies,closed_test_accuracies,open_true_targets,closed_true_targets,open_predictions,closed_predictions,open_all_values,closed_all_values = self.harmonic_test(g, ensemble)
      true_targets = torch.cat((closed_true_targets.to(self.DEVICE), open_true_targets.to(self.DEVICE)))
      logs['open_values'][g] = open_all_values
      logs['closed_values'][g] = closed_all_values
      predictions = []
      for k in range(len(open_predictions)):
        predictions.append(torch.cat((open_predictions[k].to(self.DEVICE), closed_predictions[k].to(self.DEVICE))))
      
      print(f"Testing on both open and closed world")
      for en,acc in enumerate(test_accuracies):
        print(f"owr harmonic mean (threshold:{self.threshold_list[en]}) = {acc}")
      
      
      print("")
      print("=============================================")
      print("")
      #logs['group_train_loss'][g] = e_loss
      #logs['group_train_accuracies'][g] = e_acc
      #logs['val_losses'][g] = validate_loss
      #logs['val_accuracies'][g] = validate_acc
      logs['test_accuracies'][g] = test_accuracies

      if g < 4:
        self.add_output_nodes()

    logs['true_labels'] = true_targets
    logs['predictions'] = predictions
    return logs

################################################################


  def harmonic_test(self, classes_group_idx, ensemble):
    with torch.no_grad():
      ensemble.train(False)
      mean_accs = []
      open_test_accuracy, open_true_targets, open_predictions_list, open_all_values= self.test_openset(classes_group_idx, ensemble)
      closed_test_accuracy, closed_true_targets, closed_predictions_list, closed_all_values = self.test_rejection(classes_group_idx, ensemble)
      print(f"{open_test_accuracy},{closed_test_accuracy}")
      for n in range(len(open_test_accuracy)):
        mean_acc = 1/((1/open_test_accuracy[n] + 1/closed_test_accuracy[n])/2)
        mean_accs.append(mean_acc)

    return mean_accs, open_test_accuracy, closed_test_accuracy, open_true_targets, closed_true_targets, open_predictions_list, closed_predictions_list, open_all_values, closed_all_values       



  def test_openset(self,classes_group_idx, ensemble):
    softmax = nn.Softmax(dim=1).to(self.DEVICE)
    threshold_list = self.threshold_list
    running_corrects_list = [0 for _ in range(len(threshold_list))]
    total = 0
    accuracies = []
    unknowkn_class = classes_group_idx*10+10 #Assign an index to unknown class, for instance at the first iteration we have class from 0 to 9, unkown class will be 10

    
    all_targets = torch.tensor([])
    all_targets = all_targets.type(torch.LongTensor)
    all_values = torch.tensor([])
    all_values = all_values.type(torch.LongTensor)
    preds_with_unknown_list = [torch.tensor([]) for _ in range(len(threshold_list))]

    for _, images, labels in self.test_dl[classes_group_idx]:
      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)
      total += labels.size(0)

      outputs, variances = ensemble.predict_with_variance(images)

      values, preds = torch.max(outputs.data, 1)
      pred_vars = torch.tensor([])
      pred_vars = pred_vars.type(torch.DoubleTensor)
      pred_vars = pred_vars.to(self.DEVICE)
      for en,pred in enumerate(preds):
        pred_vars = torch.cat((pred_vars.to(self.DEVICE), torch.tensor([variances[en,pred]]).to(self.DEVICE)))

      all_values = torch.cat((all_values.to(self.DEVICE),values.to(self.DEVICE)))
      all_targets = torch.cat((all_targets.to(self.DEVICE), labels.to(self.DEVICE)), dim=0)
      label_unknow_tensor = torch.tensor([unknowkn_class for _ in range(labels.size(0))]).to(self.DEVICE)
      for k,threshold in enumerate(threshold_list):
        stats = (values - threshold)/(torch.sqrt(pred_vars)/sqrt(self.n_estimators))
        below_mask = stats < self.confidence
        preds_with_unknown = torch.where(below_mask.to(self.DEVICE), torch.tensor(unknowkn_class).to(self.DEVICE), preds.to(self.DEVICE))
        running_corrects_list[k] += torch.sum(preds_with_unknown == label_unknow_tensor.data).data.item()
        preds_with_unknown_list[k] = torch.cat((preds_with_unknown_list[k].to(self.DEVICE), preds_with_unknown.to(self.DEVICE)), dim=0)
        
    for corr in running_corrects_list:
      accuracies.append(corr/float(total))


    return accuracies, all_targets, preds_with_unknown_list, all_values

  
  def test_rejection(self, classes_group_idx, ensemble):
    softmax = nn.Softmax(dim=1).to(self.DEVICE)
    threshold_list = self.threshold_list
    running_corrects_list = [0 for _ in range(len(threshold_list))]
    total = 0
    accuracies = []
    unknowkn_class = classes_group_idx*10+10 #Assign an index to unknown class, for instance at the first iteration we have class from 0 to 9, unkown class will be 10

    
    all_targets = torch.tensor([])
    all_targets = all_targets.type(torch.LongTensor)
    all_values = torch.tensor([])
    all_values = all_values.type(torch.LongTensor)
    preds_with_unknown_list = [torch.tensor([]) for _ in range(len(threshold_list))]

    for _, images, labels in self.test_dl[classes_group_idx]:
      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)
      total += labels.size(0)

      outputs, variances = ensemble.predict_with_variance(images)

      values, preds = torch.max(outputs.data, 1)
      pred_vars = torch.tensor([])
      pred_vars = pred_vars.type(torch.DoubleTensor)
      pred_vars = pred_vars.to(self.DEVICE)
      for en,pred in enumerate(preds):
        pred_vars = torch.cat((pred_vars.to(self.DEVICE), torch.tensor([variances[en,pred]]).to(self.DEVICE)))

      all_values = torch.cat((all_values.to(self.DEVICE),values.to(self.DEVICE)))
      all_targets = torch.cat((all_targets.to(self.DEVICE), labels.to(self.DEVICE)), dim=0)
      for k,threshold in enumerate(threshold_list):
        stats = (values - threshold)/(torch.sqrt(pred_vars)/sqrt(self.n_estimators))
        below_mask = stats < self.confidence
        preds_with_unknown = torch.where(below_mask.to(self.DEVICE), torch.tensor(unknowkn_class).to(self.DEVICE), preds.to(self.DEVICE))
        running_corrects_list[k] += torch.sum(preds_with_unknown == labels.data).data.item()
        preds_with_unknown_list[k] = torch.cat((preds_with_unknown_list[k].to(self.DEVICE), preds_with_unknown.to(self.DEVICE)), dim=0)
        
        
    for corr in running_corrects_list:
      accuracies.append(corr/float(total))


    return accuracies, all_targets, preds_with_unknown_list, all_values
    
