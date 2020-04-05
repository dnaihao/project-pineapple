import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from autoencoder import Autoencoder
from VRUDataset import VRUDataset

NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
net = Autoencoder()
if torch.cuda.is_available():
		net=net.cuda()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_set = VRUDataset(transform=transform,json_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Code\\project-pineapple\\train.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Train")
val_set = VRUDataset(transform=transform,json_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Code\\project-pineapple\\val.json",data_path="C:\\Users\\shubh\\OneDrive\\Desktop\\ENTR 390\\Dataset\\Val")
train_set = DataLoader(train_set,shuffle=True,batch_size=BATCH_SIZE)
val_set = DataLoader(val_set,shuffle=True,batch_size=BATCH_SIZE)

def train(net, train_set, NUM_EPOCHS):
	print("Started training")
	net = Autoencoder()
	criterion = nn.MSELoss()
	train_loss=[]
	optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
	if torch.cuda.is_available():
		net=net.cuda()
		criterion=criterion.cuda()

	
	for epoch in range(NUM_EPOCHS):
		running_loss = 0.0
		for i, data in enumerate(train_set, 0):
			img = data.get("image")
			img = img.cuda()
			img = img.view(img.size(0), -1)
			optimizer.zero_grad()
			outputs = net(img)
			loss = criterion(outputs, img)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		loss =running_loss/len(train_set)
		train_loss.append(loss)
		train_loss.append(loss)
		print('Epoch {} of {}, Train Loss: {:.3f}'.format(
			epoch+1, NUM_EPOCHS, loss))

		#if epoch % 5 == 0:
			#save_decoded_image(outputs.cpu().data, epoch)
	print("Finished training")
	return train_loss

def save_decoded_image(img, epoch):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, './Decoded_train/linear_ae_image{}.png'.format(epoch))


def test_image_reconstruction(net, testloader):
	if torch.cuda.is_available():
		net=net.cuda()
	for batch in testloader:
		img = batch.get("image")
		img = img.cuda()
		img = img.view(img.size(0), -1)
		outputs = net(img)
		outputs = outputs.view(outputs.size(0), 3, 100, 100).cpu().data
		save_image(outputs, '.\\Results\\img.png')


def train_autoencoder():
	train_loss = train(net, train_set, NUM_EPOCHS)
	plt.figure()
	plt.plot(train_loss)
	plt.title('Train Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig('deep_ae_wheechair.png')
	test_image_reconstruction(net, val_set)
 

if __name__ == "__main__":
	train_autoencoder()