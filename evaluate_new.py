import torch
import torch.nn as nn
import os
import pretrainedmodels
import torchvision.models as models
from torchvision import datasets, models, transforms
import csv
import cv2
import torchvision
import torch.optim as optim
from pytorchtools import EarlyStopping
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from seResNeXt50 import SEResNeXt50

batchsize = 4
criterion = nn.CrossEntropyLoss()
datasetpath = "..//..//tuk//Project"

print(os.getcwd())


class GenericDataSet(torch.utils.data.Dataset):
    def __init__(self, datafile, class_names, transforms=None, dataset='SCDB'):
        print(datafile)
        print(f'class names are {class_names}')
        self.datafile = datafile
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transforms = transforms
        self.dataset = dataset
        self._get_filenames_and_labels()
        # print(self.filepaths)

    def __getitem__(self, index):
        # TODO: Make this less ugly
        filename = self.filepaths[index].split('/')[-1].split('.')[0]
        image = cv2.imread(self.filepaths[index])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(type(image))
        if self.transforms:
            image = self.transforms(image)
            # image = augmented

        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)

    def _get_filenames_and_labels(self):

        if isinstance(self.datafile, list):
            self.filepaths = self.datafile
            self.labels = np.zeros_like(self.filepaths)
        else:
            # If not list, then it should be csv file
            with open(self.datafile, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter='|')

                self.filepaths = []
                self.labels = []

                for row in reader:
                    self.filepaths.append(os.path.join(datasetpath, self.dataset,
                                                       row[0]))  # f'..//..//tuk//Project//{dataset}//{row[0]}')
                    self.labels.append(int(row[1]))


def dataloader():
    print(os.getcwd())
    # number = input("choose dataset 1 for synthetic,2 for real world or 3 for medical")
    number = 1
    dataset = "SCDB"
    if number == "1":
        dataset = "SCDB"
    elif number == "2":
        dataset = "real world"
    elif number == "3":
        dataset = "medical"

    data_dir = f"..//tuk//Project//{dataset}/"
    print(f'datadir is {data_dir}')
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(124),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    print(f'directory is {os.getcwd()}')
    images = GenericDataSet(os.path.join(datasetpath, dataset, f'test.csv'), os.path.join(datasetpath), data_transforms['test'],dataset)
     # f"..//..//tuk//Project//{dataset}//{x}.csv", f"..//..//tuk//Project//{dataset}",
    #     data_transforms[x], dataset) for x in ['train', 'val']}
    print(images)
    dataloaders = torch.utils.data.DataLoader(images, batch_size=batchsize,shuffle=False, num_workers=4)

    return dataloaders


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
        # loss = criterion(output, target.to(torch.device("cuda:0")))
        # update test loss
        # test_loss += loss.item() * data.size(0)
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
    # test_loss = test_loss / len(test_loader.dataset)
    # print('Test Loss: {:.6f}\n'.format(test_loss))
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
    test_loader = dataloader()
    model = SEResNeXt50()
    model = model.to(torch.device("cuda:0"))
    model.load_state_dict(torch.load('..//masters//save_model//checkpoint (17).pt'))
    evaluate_model(model, test_loader, criterion)
