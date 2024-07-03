import scanpy as sc
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import * 
from models import FeedforwardNN, CVAE
import torch.nn as nn
import torch.optim as optim
import argparse
import math

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

    adx_train = protein[:60000, :].copy()
    adx_test = protein[60000:, :].copy()

    # Validate via barcodes & convert to tensors  
    if (gex_train.obs.index.tolist() != adx_train.obs.index.tolist()) or (gex_test.obs.index.tolist() != adx_test.obs.index.tolist()):
        raise RuntimeError("Train and Test datasets are mismatched.")

    x_train = counts_to_tensor(gex_train)
    x_test = counts_to_tensor(gex_test)

    y_train = counts_to_tensor(adx_train)
    y_test = counts_to_tensor(adx_test)


    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    ################################################### ML Training ###################################################

    # Hyperparameters
    input_size = rna.n_vars         # Number of unique rna molecules
    hidden_size = 512               # Hyper-parameter
    output_size = protein.n_vars    # Number of unique proteins
    latent_size = 64                # For VAEs
    learning_rate = 0.001
    num_epochs = 100


    """
    TODO:
    1. Make the models better
    2. Plot the train & test accuracy to visualize performance.
    3. Normalize the OG data via best-practices related to cite-seq!
    """

    # Parsing model from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='ffnn, vae, todo')
    args = parser.parse_args()


    # 1) FFNN
    if args.model == 'ffnn':
        model = FeedforwardNN(input_size, hidden_size, output_size)

        criterion = nn.MSELoss()  
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 1  # Adjust as needed

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            
            print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    # 2 VAE
    elif args.model == 'vae': 
        model = CVAE(input_size, hidden_size, latent_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                recon_y, mu, logvar = model(batch_x)
                loss = model.loss(recon_y, batch_y, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}')
    else:
        # TODO
        print("unsupported")




if __name__ == "__main__":
    main()