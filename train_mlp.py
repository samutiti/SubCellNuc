import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data import ESMDataset
from mlp import MLP
import argparse
import yaml

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion # will use contrastive loss
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        pass

    def forward(self, subcell, esm): # NOTE: i hope this will work
        # scaled pairwise cosine similarities [n, n]
        logits = (subcell @ esm.transpose(-2,-1)) * torch.exp(self.temperature)

        # symmetric loss function from: https://medium.com/correll-lab/building-clip-from-scratch-68f6e42d35f4, Nguyen 2024
        labels = torch.arange(logits.shape[0]).to(self.device)

        loss_s = nn.functional.cross_entropy(logits.transpose(-2,-1), labels) # row-wise loss (subcell)
        loss_e = nn.functional.cross_entropy(logits, labels) # column-wise loss (esm)

        loss = (loss_s + loss_e) / 2
        return loss
    

def main():
    parser = argparse.ArgumentParser(description='Train MLP on ESM embeddings')
    parser.add_argument('config', type=str, help='Path to configuration file, .yaml file')
    args = parser.parse_args()

    # Load hyperparameters from YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Hyperparameters
    input_dim = 1536 # this should match the dimension of the Subcell embeddings
    hidden_dim = config.get('hidden_dim', 1360)
    output_dim = 1280 # this should match the dimension of the ESM-like embeddings you want to generate
    data_path = config.get('datapath', 'data.npy')
    learning_rate = config.get('learning_rate', 0.001)
    num_epochs = config.get('num_epochs', 20)
    batch_size = config.get('batch_size', 32)
    shuffle = config.get('shuffle', True)
    mlp_weights_path = config.get('mlp_weights_path', 'mlp_weights.pth')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    dataset = ESMDataset(filepath=data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=ESMDataset.collate_fn)

    # Model, Loss, Optimizer
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Trainer
    trainer = Trainer(model, dataloader, optimizer, criterion, device)
    trainer.train(num_epochs)
    trainer.save_model(mlp_weights_path)

if __name__ == "__main__":
    main()
    