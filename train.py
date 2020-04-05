from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
import copy
import pickle

EPOCH = 25

def encode(labels):
    encodedLabels = []
    for l in labels:
        if l == "wheelchair":
            encodedLabels.append([1])
        elif l == "not_wheelchair":
            encodedLabels.append([0])
    encodedLabels= torch.tensor(encodedLabels)
    return encodedLabels


def train():
    print("Starting Train")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = VRUDataset(transform=transform,json_path="train.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Train")
    val_set = VRUDataset(transform=transform,json_path="val.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Val")
    train_set = DataLoader(train_set,batch_size=32,shuffle=True)
    val_set = DataLoader(val_set,shuffle=True,batch_size=48)

    cnn = CNN()
    weight = torch.FloatTensor([0.8, 4.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.00001)
    if torch.cuda.is_available():
        print("Using GPU")
        cnn=cnn.cuda()
        criterion=criterion.cuda()
        weight=weight.cuda()
    # enc = OneHotEncoder()
    enc = LabelEncoder()
    #classes=['wheelchair','not_wheelchair']
    #enc.fit(classes)
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    #cuda = torch.device('cuda')
    count=0
    val_count=0
    positive_count=0
    negative_count=0
    false_positive=0
    false_negative=0
    true_positive=0
    true_negative=0
    # train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        #for (i, data), (iv, val_data) in zip(enumerate(train_set, 0), enumerate(val_set, 0)):
        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs = data.get("image")
            labels = data.get("label")
            #print(labels)
            labels = np.array(labels).reshape(-1, 1)
            encodedLabels = encode(labels)

            #print(encodedLabels)
            if torch.cuda.is_available():
                inputs = torch.tensor(inputs).cuda()
                encodedLabels = torch.tensor(encodedLabels).cuda()


            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()

            train_outputs = cnn(inputs)
            #val_outputs= cnn(val_inputs)
            # import ipdb; ipdb.set_trace()
            #print(train_outputs.cpu())
            #print(torch.max(encodedLabels, 1)[0])
            #print(torch.max(train_outputs.cpu(), 1).indices.reshape(-1,1))
            
            loss = criterion(train_outputs,torch.max(encodedLabels, 1)[0])
            #loss= loss*weight
            #loss=loss.mean()
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            encodedLabels=encodedLabels.cpu()
            inputs=inputs.cpu()
            train_outputs=train_outputs.cpu()
            train_pred=np.argmax(train_outputs.detach().numpy(),axis=1)
            encodedLabels = encodedLabels[:, 0]
            for a, b in zip(train_pred, encodedLabels):
                if a == b:
                    if a==0: 
                        true_negative= true_negative+1
                    else: 
                        true_positive = true_positive+1
                    count= count+1
                if b==1:
                    positive_count = positive_count+1 
                    if a!=1:
                        false_negative = false_negative+1
                else:
                    negative_count = negative_count+1
                if a==1:
                    if b!=1:
                        false_positive = false_positive+1
            running_loss += loss.item()

            if i % 100 == 99:    # print every 2000 mini-batches
                print()
                print('[%d, %5d] train loss: %.3f' %
                    (epoch + 1, (i + 1), running_loss / 100))
                running_loss = 0.0
                #train_accuracy=accuracy_score(labels,train_pred)
                #print("Training Accuracy = ",train_accuracy)
                '''plt.plot(train_losses, label='Training loss')
                plt.legend()
                plt.show()'''

                print("Percentage of Wheelchairs = ", positive_count/3200.0)
                print("Training Accuracy = ",count/3200.0)
                print("True Positive % = ",true_positive/3200.0)
                print("True Negative % = ",true_negative/3200.0)
                print("False Positive % = ", false_positive/3200.0)
                print("False Negative % = ",false_negative/3200.0)
                
                count=0
                positive_count=0
                negative_count=0
                false_positive=0
                false_negative=0
                true_negative=0
                true_positive=0
                ## Validation Below
                torch.cuda.empty_cache()
                torch.no_grad()
                val_count=0
                if (epoch%5==4):
                    for it, val_data in enumerate(val_set,0):
                        val_inputs = val_data.get("image")
                        val_labels = val_data.get("label")
                        val_labels = np.array(val_labels).reshape(-1, 1)
                        val_encodedLabels=encode(val_labels)
                        if torch.cuda.is_available():
                            val_inputs = torch.tensor(val_inputs).cuda()
                            val_encodedLabels = torch.tensor(val_encodedLabels).cuda()
                        val_outputs=cnn(val_inputs)
                        val_encodedLabels=val_encodedLabels.cpu()
                        val_pred=np.argmax(train_outputs.detach().numpy(),axis=1)
                        val_encodedLabels = val_encodedLabels[:, 0]
                        for a, b in zip(val_pred, val_encodedLabels):
                            if a == b:
                                val_count += 1
                        
                    print('Validation Accuracy = %.3f'  % (val_count/(32*len(val_set))))
                    print()
                torch.save(cnn, './/Saved Models//model'+str(epoch+1)+'.pt')

            
        running_loss=0.0
        count=0
        torch.cuda.empty_cache()
        torch.no_grad()
        '''for i, data in enumerate(val_set,0):
            inputs = data.get("image")
            labels = data.get("label")
            labels = np.array(labels).reshape(-1, 1)

            val_encodedLabels=encode(labels)
            if torch.cuda.is_available():
                inputs = torch.tensor(inputs).cuda()
                val_encodedLabels = torch.tensor(val_encodedLabels).cuda()
            
            #enc.fit(labels)
            
            
            val_outputs=cnn(inputs)
            val_loss=criterion(val_outputs,torch.max(val_encodedLabels, 1)[1])
            running_loss+= val_loss.item()
            val_losses.append(val_loss)
            val_encodedLabels=val_encodedLabels.cpu()
            inputs=inputs.cpu()
            val_pred=np.argmax(train_outputs.detach().numpy(),axis=1)
            val_encodedLabels = val_encodedLabels[:, 0]
            for a, b in zip(train_pred, val_encodedLabels):
                if a == b:
                    count += 1
            torch.cuda.empty_cache()
            torch.no_grad()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] validation loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                #softmax=torch.exp(val_outputs)
                #prob=list(softmax.numpy())
                #val_pred=np.argmax(prob,axis=1)
                print("Validation Accuracy = ", count/400.0)
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