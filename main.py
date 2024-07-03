import scanpy as sc
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import * 
from models import FFNN, VAE, MLP
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import matplotlib.pyplot as plt

def main():
    ################################################### Data Prep #####################################################
    data = "data/pbmc_10k_protein_v3_raw_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(data, genome=None, gex_only=False, backup_url=None)

    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_counts=10) # number of times that RNA is present in the dataset
    sc.pp.filter_cells(adata, min_counts=100) # number of rna molecules in each cell

    protein = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    
    # 80/20 split rule
    split = math.ceil(adata.n_vars * 0.8)
    gex_train = rna[:split, :].copy()
    gex_test = rna[split:, :].copy()

    adx_train = protein[:split, :].copy()
    adx_test = protein[split:, :].copy()

    # Validate via barcodes & convert to tensors  
    if (gex_train.obs.index.tolist() != adx_train.obs.index.tolist()) or (gex_test.obs.index.tolist() != adx_test.obs.index.tolist()):
        raise RuntimeError("Train and Test datasets are mismatched.")

    
    ################################################### ML Training ###################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parsing model from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='ffnn, vae, todo')
    args = parser.parse_args()
    

    # Hyperparameters
    input_size = rna.n_vars         # Number of unique rna molecules
    hidden_size = 512               # Hyper-parameter
    output_size = protein.n_vars    # Number of unique proteins
    latent_size = 64                # For VAEs
    learning_rate = 0.001
    num_epochs = 100

    x_train = counts_to_tensor(gex_train).to(device)
    x_test = counts_to_tensor(gex_test).to(device)

    y_train = counts_to_tensor(adx_train).to(device)
    y_test = counts_to_tensor(adx_test).to(device)


    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # 1) FFNN
    if args.model == 'ffnn':
        model = FFNN()
    
    # 2 VAE
    elif args.model == 'vae': 
        model = VAE()
    
    # 3 MLP
    elif args.model == 'mlp': 
        model = MLP(input_size, output_size).to(device)
    
    else:
        print("Testing")
        # for batch_X, batch_y in train_loader:
        #     print(batch_X, batch_y)
        return
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    """
    TODO:
    1. Make the models better, manually
    3. Normalize the OG data via best-practices related to cite-seq!
    """

    # Training 
    train_mse_vals = []
    best_train_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        train_mse_vals.append(train_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'./models/{args.model}.pth')
    

    # Evals
    # Final evaluation on test set
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            test_loss += loss.item() * X_batch.size(0)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # Plotting the MSE over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_mse_vals, label='MSE Training Vals')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'{args.model} MSE During Training')
    plt.legend()
    plt.savefig(f'./results/{args.model}_mse_training_plot.png')  # Save the plot as a PNG file
    plt.show()
    


if __name__ == "__main__":
    main()