from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from PIL import Image
import numpy as np

from VRUDataset import VRUDataset
from models.cnn import CNN

EPOCH = 1


def train():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = VRUDataset(transform=transform,data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Im")
    train_set = DataLoader(train_set, batch_size=5, shuffle=5, num_workers=1)

    cnn = CNN()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    # enc = OneHotEncoder()
    enc = LabelEncoder()
    # train
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs = data.get("image")
            labels = data.get("label")
            labels = np.array(labels).reshape(-1, 1)
            #print(labels)
            enc.fit(labels)
            encodedLabels = enc.transform(labels)
            encodedLabels = torch.Tensor(encodedLabels).unsqueeze(0)
            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()
            # forward + backward + optimize
            #print(inputs.shape)
            outputs = cnn(inputs)
            #print(outputs.shape)
            # import ipdb; ipdb.set_trace()
            #print(outputs.shape, encodedLabels.shape)
            loss = criterion(outputs, encodedLabels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    print('Finished Training')
    





if __name__ == "__main__":
    train()