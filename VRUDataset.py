from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image

class VRUDataset(Dataset):
    def __init__(self, json_path="new.json", data_path="done", transform=None):
        with open(json_path) as f:
            self.json = json.load(f)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.json['obj'])
    
    def __getitem__(self, idx):
        img_name = self.json['obj'][idx]['f_name']
        img = Image.open(os.path.join(self.data_path, img_name))
        sample = {
            'image': self.transform(img),
            'label': self.json['obj'][idx]['label']
        }
        
        return sample


if __name__ == "__main__":
    V = VRUDataset()