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
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import numpy as np

from VRUDataset import VRUDataset
from models.cnn import CNN

EPOCH = 2


def train():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = VRUDataset(transform=transform)
    train_set = DataLoader(train_set, batch_size=4, shuffle=4, num_workers=4)

    cnn = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    enc = OneHotEncoder()
    # train
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data.get("image")
            labels = data.get("label")
            labels = np.array(labels).reshape(-1, 1)
            print(labels)
            enc.fit(labels)
            encodedLabels = enc.transform(labels).toarray()
            encodedLabels = torch.Tensor(encodedLabels)
            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = cnn(inputs)
            # import ipdb; ipdb.set_trace()
            loss = criterion(outputs, encodedLabels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    





if __name__ == "__main__":
    train()