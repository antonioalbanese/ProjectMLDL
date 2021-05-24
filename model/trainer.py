import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from copy import deepcopy
from copy import copy

#(self, device, net, param_opt, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, val_dl, test_dl)
#(self, device, net, criterion, optimizer, scheduler, train_dl, validation_dl, test_dl):

class Trainer():
  def __init__(self, device, net, LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, train_dl, validation_dl, test_dl):

    self.DEVICE = device
    self.MILESTONES = MILESTONES
    self.MOMENTUM = MOMENTUM
    self.START_LR = LR
    self.WEIGHT_DECAY = WEIGHT_DECAY
    self.GAMMA = GAMMA

    self.net = net
    self.best_net = self.net

    self.criterion = nn.BCEWithLogitsLoss()
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
      #inizialize LR and STEP
      self.parameters_to_optimize = self.net.parameters()
      self.optimizer = optim.SGD(self.parameters_to_optimize, lr=self.START_LR, momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
      self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.MILESTONES, gamma=self.GAMMA)

      best_acc = 0
      self.best_net = deepcopy(self.net)

      # ad ogni epoca chiamo il train, salvo le loss e trovo il modello migliore con l'evaluation
      for epoch in range(num_epochs):
        e_loss, e_acc = self.train_epoch(g)
        print(f"Epoch[{epoch}] loss: {e_loss} LR: {self.scheduler.get_last_lr()}")
        
        validate_loss, validate_acc = self.validate(g)
        print(f"Validation on group[{g}] of 10 classes")
        print(f"val loss: {validate_loss}")
        print(f"val acc: {validate_acc}")
        self.scheduler.step()
        
        if validate_acc > best_acc:
          best_acc = validate_acc
          self.best_net = deepcopy(self.net)
          best_epoch = epoch
          print("Best model updated")
        print("")
        

      print(f"Group[{g}]Finished!")
      print(f"Best model at epoch {best_epoch}, best accuracy: {best_acc:.2f}")
      print("")
      test_accuracy, true_targets, predictions = self.test(g)
      print(f"Testing classes seen so far, accuracy: {test_accuracy}")
      print("")
      print("=============================================")
      print("")
      logs['group_train_loss'][g] = e_loss
      logs['group_train_accuracies'][g] = e_acc
      logs['val_losses'][g] = validate_loss
      logs['val_accuracies'][g] = validate_acc
      logs['test_accuracies'][g] = test_accuracy

      if g < 9:
        self.increment_classes()

    logs['true_labels'] = true_targets
    logs['predictions'] = predictions
    return logs

  def train_epoch(self, classes_group_idx):
    self.net.train()

    running_loss = 0
    running_corrects = 0
    total = 0
    # per ogni gruppo di classi in train mi prendo le labels e le immagini
    # azzero i gradienti, mi sposto sulla gpu e faccio onehot delle labels
    # calcolo output, loss, etc.
    # alla fine di tutti i batches aggiorno la epoch_loss
    for _, images, labels in self.train_dl[classes_group_idx]:
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      one_hot_labels = self.to_onehot(labels) 

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
    # come train(), ma prendo anche le predictions per fare qualche statistica
    for _, images, labels in self.validation_dl[classes_group_idx]:
      total += labels.size(0)
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      #one_hot_labels = self.to_onehot(labels) 

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
    """Test the model.
    Returns:
        accuracy (float): accuracy of the model on the test set
    """

    self.best_net.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    total = 0

    all_preds = torch.tensor([]) # to store all predictions
    all_preds = all_preds.type(torch.LongTensor)
    all_targets = torch.tensor([])
    all_targets = all_targets.type(torch.LongTensor)
    # solito ciclo
    for _, images, labels in self.test_dl[classes_group_idx]:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)
        total += labels.size(0)

        # Forward Pass
        outputs = self.best_net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

        # Append batch predictions and labels
        all_targets = torch.cat(
            (all_targets.to(self.DEVICE), labels.to(self.DEVICE)), dim=0
        )
        all_preds = torch.cat(
            (all_preds.to(self.DEVICE), preds.to(self.DEVICE)), dim=0
        )

    # Calculate accuracy
    accuracy = running_corrects / float(total)  


    return accuracy, all_targets, all_preds

  def increment_classes(self, n=10):
    """Add n classes in the final fully connected layer."""

    in_features = self.net.fc.in_features  # size of each input sample
    out_features = self.net.fc.out_features  # size of each output sample
    weight = self.net.fc.weight.data

    self.net.fc = nn.Linear(in_features, out_features+n)
    self.net.fc.weight.data[:out_features] = weight

  def to_onehot(self, targets): 
    '''
    Args:
    targets : dataloader.dataset.targets of the new task images
    '''
    num_classes = self.net.fc.out_features
    one_hot_targets = torch.eye(num_classes)[targets]

    return one_hot_targets.to(self.DEVICE)
