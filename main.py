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
import muon 


def main():
    ################################################### Data Prep #####################################################
    data = "data/pbmc_10k_protein_v3_raw_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(data, genome=None, gex_only=False, backup_url=None)

    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_counts=100) # number of times that RNA is present in the dataset
    sc.pp.filter_cells(adata, min_counts=500) # number of biomolecules in each cell

    protein = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    # Filtering cells not expressing both types of biomolecules
    sc.pp.filter_cells(rna, min_counts=1)
    sc.pp.filter_cells(protein, min_counts=1)
    common_cells = protein.obs_names.intersection(rna.obs_names)
    protein = protein[common_cells, :]
    rna = rna[common_cells, :]
    
    # Doing normalization and SVD steps
    # TODO: this is really the area where to change things up!
    sc.pp.log1p(rna)
    rna_norm = zscore_normalization_and_svd(rna.X.toarray(), n_components=300) # Same as ScLinear authors
    muon.prot.pp.clr(protein)
    protein_norm = protein.X.toarray()
    
    # 80/20 split rule
    split = math.ceil(rna_norm.shape[0] * 0.8)
    gex_train = rna_norm[:split, :]
    gex_test = rna_norm[split:, :]

    adx_train = protein_norm[:split, :]
    adx_test = protein_norm[split:, :]
    # print(rna_norm.shape)
    # print(protein_norm.shape)
    # print(gex_train.shape)
    # print(adx_train.shape)
    
    ################################################### ML Training ###################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Parsing model from command line
    parser = argparse.ArgumentParser()
    # NOTE: all will probably use the model, but with modified RNA dimensionality reduction
    parser.add_argument('--model', type=str, required=True, help='todo')
    parser.add_argument('--desc', type=str, required=True, help='describe the experiment')
    args = parser.parse_args()
    

    # Hyperparameters
    input_size = rna_norm.shape[1]          # Number of unique rna molecules
    hidden_size = 512                       # Hyper-parameter
    output_size = protein_norm.shape[1]     # Number of unique proteins
    latent_size = 64                        # For VAEs
    learning_rate = 0.001
    num_epochs = 100

    x_train = torch.from_numpy(gex_train).to(device)
    x_test = torch.from_numpy(gex_test).to(device)

    y_train = torch.from_numpy(adx_train).to(device)
    y_test = torch.from_numpy(adx_test).to(device)


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
    0. Make another file for re-running existing models! And add some code to save the stats
    1. Add the Sinkhorn layers!
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

    # NOTE: author eval metric
    y_pred = model(x_test)
    evaluate(y_pred, y_test, verbose=True)
    
    
    # Plotting the MSE over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_mse_vals, label='MSE Training Vals')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'{args.desc}: MSE During Training')
    plt.legend()
    plt.savefig(f'./results/{args.model}_mse_training_plot.png')  # Save the plot as a PNG file
    plt.show()
    


if __name__ == "__main__":
    main()