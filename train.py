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
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from VRUDataset import VRUDataset
from models.cnn import CNN

EPOCH = 2


def train():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = VRUDataset(transform=transform,json_path="train.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Train")
    val_set = VRUDataset(transform=transform,json_path="val.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Val")
    train_set = DataLoader(train_set,shuffle=True)
    val_set = DataLoader(val_set,shuffle=True)

    cnn = CNN()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    # enc = OneHotEncoder()
    enc = LabelEncoder()
    classes=['wheelchair','walking_frame','crutches','person','push_wheelchair']
    enc.fit(classes)
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    #cuda = torch.device('cuda')
    # train
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs = data.get("image")
            labels = data.get("label")
            labels = np.array(labels).reshape(-1, 1)
            #if torch.cuda.is_available():
                #inputs = torch.tensor(inputs).cuda()
                #labels = torch.tensor(labels).cuda()
            
            encodedLabels = enc.transform(labels)

            encodedLabels = torch.Tensor(encodedLabels).unsqueeze(0)
            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()
            #if torch.cuda.is_available():
                #cnn=cnn.cuda()
                #criterion=criterion.cuda()
            # forward + backward + optimize
            #print(inputs.shape)
            train_outputs = cnn(inputs)
            
            #print(outputs.shape)
            # import ipdb; ipdb.set_trace()
            loss = criterion(train_outputs, encodedLabels)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] train loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
                #softmax=torch.exp(train_outputs)
                #prob=list(softmax.numpy())
                #train_pred=np.argmax(prob,axis=1)
                #train_accuracy=accuracy_score(labels,train_pred)
                #print("Training Accuracy = ",train_accuracy)
                plt.plot(train_losses, label='Training loss')
                plt.legend()
                plt.show()


        running_loss=0.0
        for i, data in enumerate(val_set,0):
            inputs = data.get("image")
            labels = data.get("label")
            labels = np.array(labels).reshape(-1, 1)
            #if torch.cuda.is_available():
                #inputs = torch.tensor(inputs).cuda()
                #labels = torch.tensor(labels).cuda()
                #cnn=cnn.cuda()
                #criterion=criterion.cuda()

            #enc.fit(labels)
            encodedLabels = enc.transform(labels)
            val_encodedLabels = torch.Tensor(encodedLabels).unsqueeze(0)
            val_outputs=cnn(inputs)
            val_loss=criterion(val_outputs,val_encodedLabels)
            running_loss+= val_loss.item()
            val_losses.append(val_loss)
            if i % 1000 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] validation loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
                #softmax=torch.exp(val_outputs)
                #prob=list(softmax.numpy())
                #val_pred=np.argmax(prob,axis=1)
                #val_accuracy=accuracy_score(labels,val_pred)
                #print("Validation Accuracy = ",val_accuracy)
                plt.plot(val_losses, label='Validation loss')
                plt.legend()
                plt.show()
            

    print('Finished Training')
    





if __name__ == "__main__":
    train()