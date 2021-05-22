import torch
import torch.nn as nn
from torch.backends import cudnn
from copy import deepcopy
from copy import copy

class Trainer():
  def __init__(self, device, net, criterion, optimizer, scheduler, train_dl, validation_dl, test_dl):

    self.DEVICE = device

    self.net = net
    self.best = self.net

    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler

    self.train_dl = train_dl
    self.validation_dl = validation_dl
    self.test_dl = test_dl

    
  def train_model(self, num_epochs):
    cudnn.benchmark
    epoch_losses = [[float for k in range(num_epochs)] for j in range(10)]
    test_acc_list = [float for k in range(10)]
    for g in range(10):
      self.net.to(self.DEVICE)

      best_acc = 0
      self.best_net = deepcopy(self.net)

      # ad ogni epoca chiamo il train, salvo le loss e trovo il modello migliore con l'evaluation
      for epoch in range(num_epochs):
        e_loss = self.train_epoch(g)
        epoch_losses[g][epoch]=e_loss
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
      test_acc_list[g] = test_accuracy

      if g < 9:
        self.increment_classes()

    return epoch_losses, test_acc_list, true_targets, predictions

  def train_epoch(self, classes_group_idx):
    self.net.train()

    running_loss = 0
    # per ogni gruppo di classi in train mi prendo le labels e le immagini
    # azzero i gradienti, mi sposto sulla gpu e faccio onehot delle labels
    # calcolo output, loss, etc.
    # alla fine di tutti i batches aggiorno la epoch_loss
    for _,images, labels in self.train_dl[classes_group_idx]:
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      one_hot_labels = self.to_onehot(labels) 

      output = self.net(images)
      loss = self.criterion(output, one_hot_labels)
      loss.backward()
      self.optimizer.step()

      running_loss += loss.item()
    else:
      epoch_loss = running_loss/len(self.train_dl[classes_group_idx])
      
    return epoch_loss
  
  def validate(self, classes_group_idx):
    self.net.train(False)
    running_loss=0
    running_corrects = 0
    total = 0
    # come train(), ma prendo anche le predictions per fare qualche statistica
    for _, images, labels in self.validation_dl[classes_group_idx]:
      total += labels.size(0)
      self.optimizer.zero_grad()

      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      one_hot_labels = self.to_onehot(labels) 

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
