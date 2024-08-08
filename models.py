import torch.nn as nn
import torch
import torch.nn.functional as F
import ot
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
    

class Sinkhorn(nn.Module):
    def __init__(self, n_iters=50, epsilon=0.1):
        super(Sinkhorn, self).__init__()
        self.n_iters = n_iters
        self.epsilon = epsilon

    def forward(self, cost_matrix):
        n = cost_matrix.size(0)
        K = torch.exp(-cost_matrix / self.epsilon)
        u = torch.ones(n).to(cost_matrix.device) / n
        v = torch.ones(n).to(cost_matrix.device) / n
        
        for _ in range(self.n_iters):
            u = 1.0 / torch.matmul(K, v)
            v = 1.0 / torch.matmul(K.t(), u)
        
        transport_plan = torch.matmul(torch.diag(u), torch.matmul(K, torch.diag(v)))
        return transport_plan

def cost_matrix(x, y):
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    C = torch.sum((x_col - y_lin) ** 2, 2)
    return C
    
class MLPWithSinkhorn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(MLPWithSinkhorn, self).__init__()
        hidden_one = hidden_size
        hidden_two = hidden_size // 2
        hidden_three = hidden_size // 4
        
        self.fc1 = nn.Linear(input_dim, hidden_one)  
        self.bn1 = nn.BatchNorm1d(hidden_one)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_one, hidden_two)
        self.bn2 = nn.BatchNorm1d(hidden_two)
        self.dropout2 = nn.Dropout(0.5)
        self.sinkhorn = Sinkhorn()
        self.fc3 = nn.Linear(hidden_two, hidden_three)
        self.bn3 = nn.BatchNorm1d(hidden_three)
        self.fc4 = nn.Linear(hidden_three, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Apply Sinkhorn layer
        cost = cost_matrix(x, x)
        transport_plan = self.sinkhorn(cost)
        x = torch.matmul(transport_plan, x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
