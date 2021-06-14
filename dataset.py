from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
import os
import torch
import sys
import numpy as np

MNIST_mean = 0.1307
MNIST_std = 0.3081

class MNISTDataset(Dataset):
    def __init__(self, path, train=True, normalize=False):
        if train:
            data = torch.load(os.path.join(path, 'training.pt'))
            self.dataset, self.labels = data
        else:
            data = torch.load(os.path.join(path, 'test.pt'))
            self.dataset, self.labels = data
        # self.normalize = normalize
        self.dataset = torch.tensor(self.dataset, dtype=torch.float32)
        self.dataset /= 255.0
        if normalize:
            self.dataset = (self.dataset - MNIST_mean) / MNIST_std
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx].unsqueeze(dim=0)


class MNISTSupervisedDataset(Dataset):
    def __init__(self, path, train=True, normalize=False):
        if train:
            data = torch.load(os.path.join(path, 'training.pt'))
            self.dataset, self.labels = data
        else:
            data = torch.load(os.path.join(path, 'test.pt'))
            self.dataset, self.labels = data
        # self.normalize = normalize
        self.dataset = torch.tensor(self.dataset, dtype=torch.float32)
        self.dataset /= 255.0
        if normalize:
            self.dataset = (self.dataset - MNIST_mean) / MNIST_std
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx].unsqueeze(dim=0), self.labels[idx]


class GMM:
    def __init__(self, radius=7, num_classes=10):
        self.means = []
        angles = [2*np.pi/10*i for i in range(num_classes)]
        for theta in angles:
            self.means.append([radius*np.cos(theta), radius*np.sin(theta)])
        
    def __getitem__(self, indices: torch.Tensor):
        mapping = map(self.means.__getitem__, indices)
        accessed_list = torch.tensor(list(mapping), dtype=torch.float32)
        return accessed_list

class OneHot:
    def __init__(self, num_classes=10):
        self.encoding_vecs = np.eye(num_classes)
        
    def __getitem__(self, indices: torch.Tensor):
        mapping = map(self.encoding_vecs.__getitem__, indices)
        accessed_list = torch.tensor(list(mapping), dtype=torch.float32)
        return accessed_list

class MNISTFlowDataset(Dataset):
    def __init__(self, path, train=True):
        if train:
            data = torch.load(os.path.join(path, 'training.pt'))
            self.dataset, self.labels = data
        else:
            data = torch.load(os.path.join(path, 'test.pt'))
            self.dataset, self.labels = data
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class MNISTFlowSupervisedDataset(Dataset):
    def __init__(self, path, train=True):
        if train:
            data = torch.load(os.path.join(path, 'training.pt'))
            self.dataset, self.labels = data
        else:
            data = torch.load(os.path.join(path, 'test.pt'))
            self.dataset, self.labels = data
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

class MNISTAEDataset(Dataset):
    def __init__(self, path, model, train=True, normalize=False):
        if train:
            data = torch.load(os.path.join(path, 'training.pt'))
            self.dataset, self.labels = data
        else:
            data = torch.load(os.path.join(path, 'test.pt'))
            self.dataset, self.labels = data
        # self.normalize = normalize
        self.dataset = torch.tensor(self.dataset, dtype=torch.float32)
        self.dataset /= 255.0
        if normalize:
            self.dataset = (self.dataset - MNIST_mean) / MNIST_std
        try:
            model = model.to('cpu')
            model.eval()
        except:
            pass
        self.model = model

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        inp = self.dataset[idx].unsqueeze(dim=0)
        inp = inp.unsqueeze(dim=0)
        out = self.model.encoder(inp)
        out = out.detach().squeeze(dim=0)
        return out
    