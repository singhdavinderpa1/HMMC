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

class SEResNeXt50(nn.Module):
    def __init__(self):
        super(SEResNeXt50, self).__init__()
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        self.classifier_layer = nn.Sequential(
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        batch_size, _, _, _ = x.shape  # taking out batch_size from input image
        x = self.model.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)  # then reshaping the batch_size
        x = self.classifier_layer(x)
        return x
