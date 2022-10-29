# training file for pytorch

import os
import argparse
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
from torch.utils.data import ImageFolder, DataLoader
from pathlib2 import Path
import wget

# get dataset
def download(source, target, force_clear=False):
    if source.startwith('http'):
        wget.download(source,target)
        print('done and dusted')

def tf(path):
    img = tv.io.read_image(path)
    img = tv.io.decode_jpg(img, 3) # here this represents 3 channels ( RGB )
    return img

def load_dataset(source,target, force_clear=False):
    download(source, target, force_clear)

    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img = transformation(img)

    dataset_train = ImageFolder(target, transform=tf)
    dataset_test = ImageFolder(target, transform=tf)
    return dataset_train, dataset_test

# load dataloader
def load_dataloader(source, target, force_clear=False):
    dataset_train, dataset_test = load_dataset(source,target, force_clear=False)
    train_loader = DataLoader(dataset_train, num_workers=4, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset_test, num_workers=4, batch_size=10, shuffle=True)

    return train_loader, test_loader

def load_model(source, target, force_clear=False):
    model = tv.models.resnet18(weights=None)
    model.fc = nn.Linear(512,2)
    return model

def train_model(source, target, path, forc_clear=False):
    train_loader, test_loader = load_dataloader(source, target, forc_clear=False)
    model = load_model(source, target, forc_clear=False)

    epochs = 1000
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameter(), lr=0.01)

    for epoch in epochs:
        for (image, label) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted = model(image)

            loss = criterion(predicted, label.long())
            loss.backward()

            optimizer.step()

    for epoch in epochs:
        for i, (image, label) in enumerate(test_loader):
            optimizer.zero_grad()
            predicted = model(image)

            loss = criterion(predicted, label.long())
            loss.backward()

            optimizer.step()

    torch.save(model.state_dict(), path)
    

    return model

    
