import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.conv3 = nn.Conv2d(8,3,3)
        #self.fc0 = nn.Linear(19200,7744)
        self.fc1 = nn.Linear(7744, 512)
        self.fc1_5= nn.Linear(512,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,2)

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1, 7744)
        #x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_5(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
