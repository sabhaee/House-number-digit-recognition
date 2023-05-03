import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import copy 
import h5py


class SVHN_custom_dataset(Dataset):
    def __init__(self, path, transform=None,target_transform=None):
        # super().__init__(path, transform=transform, target_transform=target_transform)
        h5_file = h5py.File(path, "r")
        self.data = copy.deepcopy(h5_file['data'][...])
        
        self.labels = copy.deepcopy(h5_file['label'][...].astype(np.int64).squeeze())
        
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        

        self.transform = transform
        self.target_transform = target_transform
        h5_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        if self.transform is not None:
            img = self.transform(img)
        target = int(self.labels[index])
        if self.target_transform is not None:
            
            target = self.target_transform(target)
        
        return img, target

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    def __init__(self, dataLoader, device):
        self.dl = dataLoader
        self.device = device
        
    def __iter__(self):
        for batch in self.dl: 
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


class SVHN_classifier(nn.Module):
    def __init__(self):
        super(SVHN_classifier, self).__init__()
        
        self.Mymodel = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,stride=2), 
            nn.Dropout(p=0.05), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,stride=2),
            nn.Dropout(p=0.05),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,stride=2),
            nn.Dropout(p=0.05) , 
            nn.Flatten(),
            nn.Linear(128*4*4,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,11))

    def forward(self, x):

        return self.Mymodel(x)