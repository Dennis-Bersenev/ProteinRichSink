import anndata as ad
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



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
def evaluate_correlations(y_hat, y, verbose=True):

    y_true = y.detach().cpu().numpy()
    y_pred = y_hat.detach().cpu().numpy()
    # Calculate RMSE
    rmse = mean_squared_error(y_pred, y_true, squared=False)
    
    # Initialize sums
    pearson_sum = 0
    spearman_sum = 0
    
    # Calculate Pearson and Spearman correlations
    for i in range(len(y_true)):
        pearson_sum += pearson_corrcoef(y[i], y_hat[i]).item()
        spearman_sum += spearman_corrcoef(y[i], y_hat[i]).item()
    
    pearson_corr = pearson_sum / len(y)
    spearman_corr = spearman_sum / len(y)
    
    if verbose:
        print("RMSE:", rmse)
        print("Pearson correlation:", pearson_corr)
        print("Spearman correlation:", spearman_corr)
        
    return rmse, pearson_corr, spearman_corr

# Gets eval stats per category (where category is the protein abundance level being predicted)
def evals_by_category(y_hat, y, num_proteins, outpath, protein_names):

    # Convert PyTorch tensors to NumPy arrays
    y_true = y.detach().cpu().numpy()
    y_pred = y_hat.detach().cpu().numpy()

    # Split into individual components
    y_true_split = [y_true[:, i] for i in range(num_proteins)]
    y_pred_split = [y_pred[:, i] for i in range(num_proteins)]


    mae = [mean_absolute_error(y_true_split[i], y_pred_split[i]) for i in range(17)]
    mse = [mean_squared_error(y_true_split[i], y_pred_split[i]) for i in range(17)]
    r2 = [r2_score(y_true_split[i], y_pred_split[i]) for i in range(17)]

    # Display the results
    for i in range(num_proteins):
        s = f"Category {protein_names[i]}: MAE = {mae[i]}, MSE = {mse[i]}, R² = {r2[i]}"
        print(s)
        with open(outpath, "w") as file:
            file.write(s)
    
    palette = sns.color_palette("husl", 17)

    # Plot metrics
    plt.figure(figsize=(20, 8))

    # Plot MAE
    plt.subplot(1, 3, 1)
    bars = plt.bar(protein_names, mae, color=palette)
    plt.xlabel('protein_names', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Mean Absolute Error per protein', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=10)
    plt.savefig('results/mae_per_protein.png')

    # Plot MSE
    plt.subplot(1, 3, 2)
    bars = plt.bar(protein_names, mse, color=palette)
    plt.xlabel('protein_names', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Mean Squared Error per protein', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=10)
    plt.savefig('results/mse_per_protein.png')


    # Plot R²
    plt.subplot(1, 3, 3)
    bars = plt.bar(protein_names, r2, color=palette)
    plt.xlabel('protein_names', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.title('R² per protein', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('metrics_per_protein.png')
    plt.show()

    plt.figure(figsize=(22, 15))
    for i in range(num_proteins):
        plt.subplot(4, 5, i+1)
        sns.histplot(y_true_split[i] - y_pred_split[i], kde=True, color=palette[i])
        plt.title(protein_names[i], fontsize=12)
        # plt.xlabel('Error', fontsize=10)
        # plt.ylabel('Density', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('error_distribution_per_protein.png')
    plt.show()
    
    
    return
