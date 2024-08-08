import scanpy as sc
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from utils import * 

from models import MLP, MLPWithSinkhorn
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import matplotlib.pyplot as plt
import muon 



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels_max).sum().item()
    
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy





def main():
     ################################################### Arg Parsing #####################################################
    # Parsing model from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='MLP or SH')
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

    # Saving mapping of 0-16 to protein names via array.
    protein_names = [item.split('_')[0] for item in protein.var_names]
    

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
    hidden_size = 1024
    learning_rate = 0.001
    num_epochs = args.epochs

    x_train = torch.from_numpy(gex_train).to(device)
    x_test = torch.from_numpy(gex_test).to(device)
    x_valid = torch.from_numpy(gex_valid).to(device)

    y_train = torch.from_numpy(adx_train).to(device)
    y_test = torch.from_numpy(adx_test).to(device)
    y_valid = torch.from_numpy(adx_valid).to(device)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)


    if args.model == 'SH':
        model = MLPWithSinkhorn(input_size, output_size, hidden_size).to(device)
    else:
        model = MLP(input_size, output_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Initialize model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training/Eval loop
    train_mse_vals = []
    best_train_loss = float("inf")
    train_loss_arr = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        train_loss_arr.append(train_loss)
        train_mse_vals.append(train_loss)
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'./models/{args.model}.pth')
    
    
    ################################################### Evals ###################################################
    
    # Final evaluation on the validation set
    valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, device)
    print(f"Test Loss: {valid_loss:.4f}, Test Accuracy: {valid_accuracy:.4f}")
    
    # Plotting the loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_loss_arr, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.desc}: Loss During Training')
    plt.legend()
    plt.savefig(f'./results/{args.model}_mse_training_plot.png')  # Save the plot as a PNG file
    plt.show()

    
    y_pred = model(x_valid)
    evals_by_category(y_pred, y_valid, output_size, 'results/stats_by_protein.txt', protein_names)
    
    # NOTE: original author eval metric
    rmse, pearson_corr, spearman_corr = evaluate_correlations(y_pred, y_valid, verbose=True)
    

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