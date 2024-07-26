import anndata as ad
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from torchmetrics.functional import mean_squared_error, pearson_corrcoef, spearman_corrcoef
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Convert the counts etc to PyTorch tensors
def counts_to_tensor(data: ad.AnnData):
    counts_matrix = data.layers['counts'].toarray()
    counts_tensor = torch.tensor(counts_matrix, dtype=torch.float32)
    return counts_tensor


# Adapted from: https://github.com/DanHanh/scLinear/blob/main/inst/python/preprocessing.py 
def zscore_normalization_and_svd(X: np.ndarray, n_components):
    
    
    X_sd = np.std(X, axis=1).reshape(-1, 1)
    X_sd[X_sd == 0] = 1
    X_normalized = (X - np.mean(X, axis=1).reshape(-1, 1)) / X_sd
    svd = TruncatedSVD(n_components=n_components)
    X_lowdim = svd.fit_transform(X_normalized)
    return X_lowdim

def min_max_normalize(X: np.ndarray):
    """
    Normalize a numpy array to the range [0, 1] using min-max normalization.

    Parameters:
    array (np.ndarray): The numpy array to normalize.

    Returns:
    np.ndarray: The normalized numpy array.
    """
    min_val = np.min(X)
    max_val = np.max(X)
    normalized_array = (X - min_val) / (max_val - min_val)
    return normalized_array


# Adapted from: https://github.com/DanHanh/scLinear/blob/main/inst/python/evaluate.py
def evaluate_correlations(y_pred, y_test, verbose=True):

    # Calculate RMSE
    rmse = mean_squared_error(y_pred, y_test, squared=False).item()
    
    # Initialize sums
    pearson_sum = 0
    spearman_sum = 0
    
    # Calculate Pearson and Spearman correlations
    for i in range(len(y_test)):
        pearson_sum += pearson_corrcoef(y_test[i], y_pred[i]).item()
        spearman_sum += spearman_corrcoef(y_test[i], y_pred[i]).item()
    
    pearson_corr = pearson_sum / len(y_test)
    spearman_corr = spearman_sum / len(y_test)
    
    if verbose:
        print("RMSE:", rmse)
        print("Pearson correlation:", pearson_corr)
        print("Spearman correlation:", spearman_corr)
        
    return rmse, pearson_corr, spearman_corr


####### VAE Helpers #######
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (e.g., MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss, recon_loss, kl_loss


def train_vae(model, data, epochs, optimizer):
    
    # Define the dataset and dataloader

    dataset = TensorDataset(data, data)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            optimizer.step()
        
        avg_train_loss = train_loss / len(dataloader.dataset)
        avg_recon_loss = recon_loss_total / len(dataloader.dataset)
        avg_kl_loss = kl_loss_total / len(dataloader.dataset)
        
        print(f'Epoch {epoch + 1}, Total Loss: {avg_train_loss:.4f}, Reconstruction Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')


def sample_from_latent(model, data, condition, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Move data to the appropriate device
        data, condition = data.to(device), condition.to(device)
        
        # Encode the data to get the latent distribution parameters
        mu, logvar = model.encode(data, condition)
        
        # Sample from the latent distribution
        z = model.reparameterize(mu, logvar)
        
        return z

def cvae_loss(recon_x, x, mu, logvar, input_dim):
    x = x.view(-1, input_dim)
    
    # Check tensor shapes
    assert recon_x.shape == x.shape, f"Shape mismatch: {recon_x.shape} vs {x.shape}"
    assert recon_x.min() >= 0.0 and recon_x.max() <= 1.0, "recon_batch values out of range [0, 1]"
    assert x.min() >= 0 and x.max() <= 1, "data values out of range [0, 1]"

    
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
