import scanpy as sc
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from utils import * 
from models import VAE, MLP, CVAE
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import matplotlib.pyplot as plt
import muon 


def main():

    ################################################### Arg Parsing #####################################################
    # Parsing model from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='ffnn or cvae')
    parser.add_argument('--desc', type=str, required=True, help='describe the experiment for bookkeeping')
    parser.add_argument('--epochs', type=int, required=True, help='how many epochs do you want all neural nets to use?')
    args = parser.parse_args()
    

    ################################################### Data Prep #####################################################
    data = "data/pbmc_10k_protein_v3_raw_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(data, genome=None, gex_only=False, backup_url=None)

    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_counts=50) # number of times that RNA is present in the dataset
    sc.pp.filter_cells(adata, min_counts=100) # number of biomolecules in each cell

    protein = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    # Filtering cells not expressing both types of biomolecules
    sc.pp.filter_cells(rna, min_counts=1)
    sc.pp.filter_cells(protein, min_counts=1)
    common_cells = protein.obs_names.intersection(rna.obs_names)
    protein = protein[common_cells, :]
    rna = rna[common_cells, :]
    
    # RNA Normalization
    sc.pp.log1p(rna)
    # rna_norm = zscore_normalization_and_svd(rna.X.toarray(), n_components=300) # Same as ScLinear authors
    rna_norm = min_max_normalize(rna.X.toarray()) # If skipping dim reduction step
    
    
    # Protein Normalization 
    muon.prot.pp.clr(protein)
    protein_norm = protein.X.toarray()
    



    # 80/20 split rule
    split = math.ceil(rna_norm.shape[0] * 0.8)
    validation_split = math.ceil(rna_norm.shape[0] * 0.95)
    gex_train = rna_norm[:split, :]
    gex_test = rna_norm[split:validation_split, :]
    gex_valid =  rna_norm[validation_split:, :]

    adx_train = protein_norm[:split, :]
    adx_test = protein_norm[split:validation_split, :]
    adx_valid = protein_norm[validation_split:, :]
    print(f'Normalized RNA array shape: {rna_norm.shape}')
    print(f'Normalized Protein array shape: {protein_norm.shape}')
    print(f'Original RNA shape: {rna.X.shape}')
    print(f'Original Protein shape: {protein.X.shape}')
    
    ################################################### ML Training ###################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    # Hyperparameters
    input_size = rna_norm.shape[1]          # Number of unique rna molecules
    output_size = protein_norm.shape[1]     # Number of unique proteins
    latent_size = output_size               # For VAEs: you choose, doesn't theoretically matter
    conditional_size = output_size          # For CVAEs: based on protein dataset shape
    learning_rate = 0.001
    c = 3 # scaling factor
    hidden_dims = [c*1024, c*512, c*256, c*128]  

    # print(f'IN: {input_size}, latent: {output_size}, conditional: {conditional_size}')

    x_train = torch.from_numpy(gex_train).to(device)
    x_test = torch.from_numpy(gex_test).to(device)
    x_valid = torch.from_numpy(gex_valid).to(device)

    y_train = torch.from_numpy(adx_train).to(device)
    y_test = torch.from_numpy(adx_test).to(device)
    y_valid = torch.from_numpy(adx_valid).to(device)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # Some debugging checks
    print(f'train GEX shape: {x_train.shape}')
    print(f'test GEX  shape: {x_test.shape}')
    
    print(f'train ADX shape: {y_train.shape}')
    print(f'test ADX shape: {y_test.shape}')
    

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = CVAE(input_dim=input_size, latent_dim=latent_size, cond_dim=conditional_size, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss_arr = []
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):  
            data, target = data.to(device), target.to(device)
        
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data, target)
            
            # Debugging checks
            if not (recon_batch.min() >= 0 and recon_batch.max() <= 1):
                print(f"recon_batch values out of range at batch {batch_idx}")
            
            loss = cvae_loss(recon_batch, data, mu, logvar, input_size)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')

        train_loss_arr.append(train_loss / len(train_loader.dataset))
    
    ################################################### Evals ###################################################
    # Final evaluation on test set
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    
    with torch.no_grad():  
        for batch_idx, (data, target) in enumerate(test_loader):  
            data, target = data.to(device), target.to(device)
            
            recon_batch, mu, logvar = model(data, target)

            # Debugging checks
            if not (recon_batch.min() >= 0 and recon_batch.max() <= 1):
                print(f"recon_batch values out of range at batch {batch_idx}")
            
            loss = cvae_loss(recon_batch, data, mu, logvar, input_size)
            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss}')


    # NOTE: original author eval metric
    y_pred = sample_from_latent(model, x_valid, y_valid, device=device)
    rmse, pearson_corr, spearman_corr = evaluate(y_pred, y_valid, verbose=True)
    
    
    # Plotting the loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_loss_arr, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.desc}: Loss During Training')
    plt.legend()
    plt.savefig(f'./results/{args.model}_loss_training_plot.png')  # Save the plot as a PNG file
    # plt.show()

    # Saving statistics to text file
    stat1 = f'Train Loss: {train_loss:.4f}'
    stat2 = f'Test Loss: {test_loss:.4f}'
    stat3 = f"RMSE: {rmse}"
    stat4 = f"Pearson correlation: {pearson_corr}"
    stat5 = f"Spearman correlation: {spearman_corr}"
    stats = "\n".join([stat1, stat2, stat3, stat4, stat5])

    # Specify the filename
    filename = "./results/stats.txt"

    # Write the strings to the file
    with open(filename, "w") as file:
        file.write(stats)
    


"""
TODO:
1. Tune the VAE approach.
2. Add evals across protein types to see which proteins have the best/worst scores 
3. Add the Sinkhorn layers!
"""

if __name__ == "__main__":
    main()