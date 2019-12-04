from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy
import pandas as pd
import os

from utils import *

def simple_data_transformer():
    transform = transforms.ToTensor()

def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def cifar10_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
       ])

def mnist_transformer():
    return torchvision.transforms.Compose([
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])

class Ring(Dataset):
    def __init__(self, path, transform=None, return_idx = True, testset=False):
        if not testset:
            self.ring = pd.read_csv(os.path.join(path, "simple_data/ring.csv"))
        else:
            self.ring = pd.read_csv(os.path.join(path, "simple_data/ring_test.csv"))
        self.transform = transform
        self.return_idx = return_idx
        self.testset = testset

    def __len__(self):
        return len(self.ring)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data = self.ring.iloc[index, 1:].to_numpy()
        target = self.ring.iloc[index, 0]

        data = data.astype(numpy.float32)
        target = int(target)

        if self.transform:
            data = self.transform(data)

        if self.return_idx:
            return data, target, index
        else:
            return data, target

class MNIST(Dataset):
    def __init__(self, path):
        self.mnist = datasets.MNIST(root=path,
                                        download=True,
                                        train=True,
                                        transform=mnist_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.mnist[index]

        return data, target, index

    def __len__(self):
        return len(self.mnist)


class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        # Your transformations here (or set it in CIFAR10)

        return data, target, index

    def __len__(self):
        return len(self.cifar100)


class ImageNet(Dataset):
    def __init__(self, path):
        self.imagenet = datasets.ImageFolder(root=path, transform=imagenet_transformer)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.imagenet[index]

        return data, target, index

    def __len__(self):
        return len(self.imagenet)
