import anndata as ad
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD



# Convert the counts etc to PyTorch tensors
def counts_to_tensor(data: ad.AnnData):
    counts_matrix = data.layers['counts'].toarray()
    counts_tensor = torch.tensor(counts_matrix, dtype=torch.float32)
    return counts_tensor


# Adapted from: https://github.com/DanHanh/scLinear/blob/main/inst/python/preprocessing.py 
def zscore_normalization(
        X: np.ndarray
) -> np.ndarray:
    """
    Row-wise Z-score normalization.
    Parameters
    ----------
    X
        Data matrix.
    """
    TruncatedSVD(n_components=n_components)
    X_sd = np.std(X, axis=1).reshape(-1, 1)
    X_sd[X_sd == 0] = 1
    X_normalized = (X - np.mean(X, axis=1).reshape(-1, 1)) / X_sd
    return X_normalized.astype(np.float32)

