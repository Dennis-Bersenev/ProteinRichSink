import anndata as ad
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from torchmetrics.functional import mean_squared_error, pearson_corrcoef, spearman_corrcoef



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


# Adapted from: https://github.com/DanHanh/scLinear/blob/main/inst/python/evaluate.py
def evaluate(y_pred, y_test, verbose=True):

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
