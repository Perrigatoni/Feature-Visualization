from __future__ import division, print_function

import time
import os
import copy

import math
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

print("PyTorch Version: ",torch.__version__)
print("TorchVision Version: ",torchvision.__version__)

data_dir = r"C:\Users\Noel\Documents\THESIS STUFF\Data"
download = True
input_size = 224
data_transforms = {
    'train': transforms.Compose([transforms.RandomResizedCrop(input_size),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ]) , 
    'test': transforms.Compose([transforms.Resize(input_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
    'valid': transforms.Compose([transforms.Resize(input_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),}


image_datasets = {x: datasets.CelebA(root=data_dir, split=str(x), target_type='attr', transform=data_transforms[x], download=download) for x in ['train', 'valid', 'test']}
inputs, labels = image_datasets['train'].__getitem__(163)
print(labels.data)
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'valid']}

batch_size = 64
print(f" Lngth of dataset in training phase: {len(dataloaders_dict['train'].dataset)}")

def batch_num_extractor(dataloaders_dict, batch_size):
    
    total_samples = {}
    batches_present = {}
    for phase in ['train', 'valid']:
        total_samples[phase] = len(dataloaders_dict[phase].dataset)
        batches_present[phase] = math.ceil(total_samples[phase] / batch_size)
    
    return batches_present


batches_dict = batch_num_extractor(dataloaders_dict, batch_size)
print(batches_dict['train'], batches_dict['valid'])