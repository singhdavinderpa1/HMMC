import torch
import torch.nn as nn
import os
import pretrainedmodels
import torchvision.models as models
from torchvision import datasets, models, transforms
import torchvision
import torch.optim as optim
from pytorchtools import EarlyStopping
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from seResNeXt50 import SEResNeXt50

batchsize = 4
criterion = nn.CrossEntropyLoss()

print(os.getcwd())


def load_test():
    print('entered')
    data_path = '..//..//tuk//Project//SCDBB//test'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize,
        num_workers=0,
        shuffle=False
    )
    return train_loader


def evaluate_model(model, test_loader, criterion):
    model.eval()  # prep model for evaluation
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    for data, target in test_loader:
        if len(target.data) != batchsize:
            print('not same batch')
            break
        target = target.to('cuda')
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.to(torch.device("cuda:0")))
        # calculate the loss
        #loss = criterion(output, target.to(torch.device("cuda:0")))
        # update test loss
        #test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batchsize):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    #test_loss = test_loss / len(test_loader.dataset)
    #print('Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        '''else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))'''

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    test_loader = load_test()
    model = SEResNeXt50()
    model = model.to(torch.device("cuda:0"))
    model.load_state_dict(torch.load('..//masters//save_model//checkpoint (18).pt'))
    evaluate_model(model, test_loader, criterion)