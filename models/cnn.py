import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16,8,5)
        self.fc0 = nn.Linear(648,256)
        self.fc1 = nn.Linear(256, 64)
        #self.fc1_5= nn.Linear(512,128)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16,2)

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1, 648)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc1_5(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
