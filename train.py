import os
import numpy as np
import pretrainedmodels
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from seResNeXt50 import SEResNeXt50
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/Training2')

batchsize = 32


def dataloader():
    print(os.getcwd())
    number = input("choose dataset 1 for synthetic,2 for real world or 3 for medical")
    dataset = "SCDBB"
    if number == "1":
        dataset = "SCDBB"
    elif number == "2":
        dataset = "real world"
    elif number == "3":
        dataset = "medical"
    print(f'here directory is {os.getcwd()}')
    data_dir = f"..//..//tuk//Project//{dataset}/"
    print(f'datadir is {data_dir}')
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(124),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
            transforms.Resize(124),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    images = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    print(images)
    dataloaders = {x: torch.utils.data.DataLoader(images[x], batch_size=batchsize,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders


def train_model(model, batch_size, patience, n_epochs, optimizer, criterion, scheduler):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    dataloaders = dataloader()
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(1, n_epochs + 1):
        # prep model for training
        print("entered Training Loop")
        model.train()
        print("set model to train mode")
        for batch, (data, target) in enumerate(dataloaders['train'], 1):
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            #print("passing data to models")
            output = model(data.to(torch.device("cuda:0")))
            #print("got output")
            # calculate the loss
            loss = criterion(output, target.to(torch.device("cuda:0")))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            writer.add_scalar('training loss', loss)
            train_losses.append(loss.item())

        # validate the model #
        Y_pred = []
        correct = 0
        total = 0
        epoch_loss = 0.0
        acc = 0.0
        test_size = 0
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, labels in dataloaders['val']:
                # images, labels = data
                outputs = model(data.to(torch.device("cuda:0")))
                # calculate the loss
                loss = criterion(output, target.to(torch.device("cuda:0")))
                # record validation loss
                valid_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(torch.device("cuda:0"))).sum().item()
        valid_acc = (
                100 * correct / total)
        print(f'Accuracy of the network on the {total} validation images: %d %%' % valid_acc)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_acc, model)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',valid_loss , epoch)
        writer.add_scalar('ACC/val', valid_acc, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step(valid_acc)
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('..//masters//save_model//checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses


def main():
    model = SEResNeXt50()
    model = model.to(torch.device("cuda:0"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    n_epochs = 100
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    print("training model bro")
    model, train_loss, valid_loss = train_model(model, batchsize, patience, n_epochs, optimizer, criterion, scheduler)



if __name__ == "__main__":
    main()
