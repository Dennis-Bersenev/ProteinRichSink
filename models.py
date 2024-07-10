import torch.nn as nn
import torch
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4_mean = nn.Linear(256, latent_dim)
        self.fc4_log_var = nn.Linear(256, latent_dim)
        self.fc5 = nn.Linear(latent_dim, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 1024)
        self.fc8 = nn.Linear(1024, input_dim)
        
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.fc4_mean(h), self.fc4_log_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.fc5(z))
        h = torch.relu(self.fc6(h))
        h = torch.relu(self.fc7(h))
        return torch.sigmoid(self.fc8(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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
        x = self.dropout3(x)

        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x
    


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim, hidden_dims):
        super(CVAE, self).__init__()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        in_dim = input_dim + cond_dim
        for h_dim in hidden_dims:
            self.encoder_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.fc21 = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc22 = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        in_dim = latent_dim + cond_dim
        for h_dim in reversed(hidden_dims):
            self.decoder_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.fc4 = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, x, c):
        xc = torch.cat((x, c), dim=1)
        for layer in self.encoder_layers:
            xc = F.relu(layer(xc))
        return self.fc21(xc), self.fc22(xc)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        zc = torch.cat((z, c), dim=1)
        for layer in self.decoder_layers:
            zc = F.relu(layer(zc))
        return torch.sigmoid(self.fc4(zc))
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

