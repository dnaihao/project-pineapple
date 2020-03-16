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

EPOCH = 30


def train():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = VRUDataset(transform=transform,json_path="train.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Train")
    val_set = VRUDataset(transform=transform,json_path="val.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Val")
    train_set = DataLoader(train_set,shuffle=True,batch_size=32)
    val_set = DataLoader(val_set,shuffle=True,batch_size=32)

    cnn = CNN()
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    if torch.cuda.is_available():
        cnn=cnn.cuda()
        criterion=criterion.cuda()

    # enc = OneHotEncoder()
    enc = LabelEncoder()
    classes=['wheelchair','not_wheelchair']
    enc.fit(classes)
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    #cuda = torch.device('cuda')
    count=0
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
            encodedLabels=np.transpose(encodedLabels)
            if torch.cuda.is_available():
                inputs = torch.tensor(inputs).cuda()
                encodedLabels = torch.tensor(encodedLabels).cuda()

            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()
            #if torch.cuda.is_available():
                #cnn=cnn.cuda()
                #criterion=criterion.cuda()
            # forward + backward + optimize
            #print(inputs.shape)

            train_outputs = cnn(inputs)

            #print(train_outputs.shape)

            #print(encodedLabels.shape)
            #print(outputs.shape)
            # import ipdb; ipdb.set_trace()
            loss = criterion(train_outputs, encodedLabels)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            encodedLabels=encodedLabels.cpu()
            inputs=inputs.cpu()
            train_outputs=train_outputs.cpu()
            train_pred=np.argmax(train_outputs.detach().numpy(),axis=1)
            #train_acc.append(train_pred)\
            #print(train_pred)
            #print(encodedLabels[:,0])
            encodedLabels = encodedLabels[:, 0]
            for a, b in zip(train_pred, encodedLabels):
                if a == b:
                    count =count+1
            #if (train_pred==encodedLabels[0]):
             #   count+=1
            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] train loss: %.3f' %
                    (epoch + 1, (i + 1)*3200, running_loss / 100))
                running_loss = 0.0
                #softmax=torch.exp(train_outputs)
                #prob=list(softmax.numpy())
                #train_pred=np.argmax(prob,axis=1)
                #train_pred=np.argmax(train_outputs.detach().numpy(),axis=1)
                #print(train_pred)
                #train_accuracy=accuracy_score(labels,train_pred)
                #print("Training Accuracy = ",train_accuracy)
                '''plt.plot(train_losses, label='Training loss')
                plt.legend()
                plt.show()'''
                print("Training Accuracy = ",count/3200.0)
                count=0
                if (i%18000)==17999:
                    plt.plot(train_losses, label='Training loss')
                    plt.legend()
                    plt.show()


        running_loss=0.0
        count=0
        torch.cuda.empty_cache()
        '''for i, data in enumerate(val_set,0):
            inputs = data.get("image")
            labels = data.get("label")
            labels = np.array(labels).reshape(-1, 1)
            encodedLabels = enc.transform(labels)
            val_encodedLabels = torch.Tensor(encodedLabels).unsqueeze(0)
            val_encodedLabels=np.transpose(val_encodedLabels)
            if torch.cuda.is_available():
                inputs = torch.tensor(inputs).cuda()
                val_encodedLabels = torch.tensor(val_encodedLabels).cuda()
            
            #enc.fit(labels)
            
            
            val_outputs=cnn(inputs)
            val_loss=criterion(val_outputs,val_encodedLabels)
            running_loss+= val_loss.item()
            val_losses.append(val_loss)
            val_encodedLabels=val_encodedLabels.cpu()
            inputs=inputs.cpu()
            val_pred=np.argmax(train_outputs.detach().numpy(),axis=1)
            val_encodedLabels = val_encodedLabels[:, 0]
            for a, b in zip(train_pred, val_encodedLabels):
                if a == b:
                    count += 1
            
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] validation loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                #softmax=torch.exp(val_outputs)
                #prob=list(softmax.numpy())
                #val_pred=np.argmax(prob,axis=1)
                print("Validation Accuracy = ", count/6400.0)
                count=0
                #val_accuracy=accuracy_score(labels,val_pred)
                #print("Validation Accuracy = ",val_accuracy)
                #.plot(val_losses, label='Validation loss')
                #plt.legend()
                #plt.show()
        '''

    print('Finished Training')
    





if __name__ == "__main__":
    train()