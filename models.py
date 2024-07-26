import torch.nn as nn
import torch
import torch.nn.functional as F
import ot

class SinkhornLayer(nn.Module):
    def __init__(self, epsilon=0.1, max_iter=100, tau=1e-9):
        super(SinkhornLayer, self).__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tau = tau

    def forward(self, x, y):
        # x and y should be probability distributions
        C = torch.cdist(x, y, p=2)  # Compute the cost matrix
        C = C.cpu().detach().numpy()  # Convert to numpy for POT compatibility
        
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        
        # Perform Sinkhorn algorithm
        pi = ot.sinkhorn(x, y, C, self.epsilon, numItermax=self.max_iter, stopThr=self.tau)
        
        pi = torch.tensor(pi).to(x.device)  # Convert back to tensor
        return pi

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
    


class MLPSinkhorn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, output_dim)
        
        self.sinkhorn = SinkhornLayer()

    def forward(self, x, y):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        
        # Apply the Sinkhorn layer
        pi = self.sinkhorn(x, y)
        
        return pi