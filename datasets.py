import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import pandas as pd



class CelebaDataset(Dataset):

    def __init__(self, X_Train, Y_Train, transform=None):
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.transform = transform

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X_Train[idx]
        y = self.Y_Train[idx]

        if self.transform:
            x = self.transform(x)
            y = torch.tensor(y, dtype=torch.float32)
            
        return x, y
    

def get_celeba(augment, dataroot):
    
    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)])
    
    train_transform = transforms.Compose(transformations)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)])
    
    
    file = open(os.path.join(dataroot, 'train_64x64.pkl'), 'rb')
    X_train = pickle.load(file)
    file.close()
    
    y_train = pd.read_csv(os.path.join(dataroot, 'train.csv')).to_numpy().astype(int)


    file = open(os.path.join(dataroot, 'test_64x64.pkl'), 'rb')
    X_test = pickle.load(file)
    file.close()
    
    y_test = pd.read_csv(os.path.join(dataroot, 'test.csv')).to_numpy().astype(int)
    
    train_dataset = CelebaDataset(X_train, y_train, transform=train_transform)
    test_dataset = CelebaDataset(X_test, y_test, transform=test_transform)
    
    return train_dataset, test_dataset