import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=30000, out_features=10000)
        self.enc2 = nn.Linear(in_features=10000, out_features=2048)
        self.enc3 = nn.Linear(in_features=2048, out_features=256)
        self.enc4 = nn.Linear(in_features=256, out_features=64)
        self.enc5 = nn.Linear(in_features=64, out_features=16)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=16, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=256)
        self.dec3 = nn.Linear(in_features=256, out_features=2048)
        self.dec4 = nn.Linear(in_features=2048, out_features=10000)
        self.dec5 = nn.Linear(in_features=10000, out_features=30000)
 
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
 
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x
 
net = Autoencoder()
