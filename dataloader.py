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

    dataloaders = {x: torch.utils.data.DataLoader(images[x], batch_size=batchsize,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    print("retuyrniing the stuff")
    return dataloaders


def main():
    data = dataloader()
    for i, (j, k) in enumerate(data['train']):
        print("entered")
        break

if __name__ == "__main__":
    main()