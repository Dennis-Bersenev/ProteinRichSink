import scanpy as sc
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import * 
from models import VAE, MLP
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
    
    # RNA Normalization (NOTE: the dimensionality reduction here is an important choice! This version uses a VAE to get reduced GEX data)
    sc.pp.log1p(rna)
    rna_model = VAE(rna.n_vars, 300)
    rna_optimizer = optim.Adam(rna_model.parameters(), lr=1e-3)

    # Train the model
    expression_tensor = torch.from_numpy(rna.X.toarray())
    train_vae(model=rna_model, data=expression_tensor, epochs=args.epochs, optimizer=rna_optimizer)

    # Get the compressed representation of the data
    rna_model.eval()
    with torch.no_grad():
        mu, _ = rna_model.encode(expression_tensor)
        rna_norm = mu.numpy()  
         
    
    # Protein Normalization Step
    muon.prot.pp.clr(protein)
    protein_norm = protein.X.toarray()
    



    # 80/20 split rule
    split = math.ceil(rna_norm.shape[0] * 0.8)
    gex_train = rna_norm[:split, :]
    gex_test = rna_norm[split:, :]

    adx_train = protein_norm[:split, :]
    adx_test = protein_norm[split:, :]
    print(f'Normalized RNA array shape: {rna_norm.shape}')
    print(f'Normalized Protein array shape: {protein_norm.shape}')
    print(f'Original RNA shape: {rna.X.shape}')
    print(f'Original Protein shape: {protein.X.shape}')
    print(f'Gex train shape: {gex_train.shape}')
    print(f'Gex test shape: {gex_test.shape}')
    
    
    ################################################### ML Training ###################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    # Hyperparameters
    input_size = rna_norm.shape[1]          # Number of unique rna molecules
    hidden_size = 512                       # Hyper-parameter
    output_size = protein_norm.shape[1]     # Number of unique proteins
    latent_size = 64                        # For VAEs
    learning_rate = 0.001

    x_train = torch.from_numpy(gex_train).to(device)
    x_test = torch.from_numpy(gex_test).to(device)

    y_train = torch.from_numpy(adx_train).to(device)
    y_test = torch.from_numpy(adx_test).to(device)


    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MLP(input_size, output_size).to(device)
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    """
    TODO:
    1. Tune the VAE approach.
    2. Add the Sinkhorn layers!
    """

    # Training 
    train_mse_vals = []
    best_train_loss = float("inf")
    for epoch in range(args.epochs):
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
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}')
        
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

    # NOTE: original author eval metric
    y_pred = model(x_test)
    rmse, pearson_corr, spearman_corr = evaluate(y_pred, y_test, verbose=True)
    
    
    # Plotting the MSE over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_mse_vals, label='MSE Training Vals')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'{args.desc}: MSE During Training')
    plt.legend()
    plt.savefig(f'./results/{args.model}_mse_training_plot.png')  # Save the plot as a PNG file
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
    


if __name__ == "__main__":
    main()