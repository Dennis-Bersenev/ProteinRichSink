import anndata as ad
import torch

# Convert the counts etc to PyTorch tensors
def counts_to_tensor(data: ad.AnnData):
    counts_matrix = data.layers['counts'].toarray()
    counts_tensor = torch.tensor(counts_matrix, dtype=torch.float32)
    return counts_tensor