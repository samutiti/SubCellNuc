import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import argparse

class ESMDataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_list = self._load_data()

    def _load_data(self):
        # data will be stored as a numpy array of shape (subcell_embeddings, [esm_embeddings])
        # take the mean of the esm embeddings across the sequence dimension to get a fixed-size vector (potentially, need to make a determination)
        # load the data from the .npy file
        if self.filepath.endswith('.npy'):
            data = np.load(self.filepath, allow_pickle=True)
        elif self.filepath.endswith('.h5ad'):
            with h5py.File(self.filepath, 'r') as f:
                data = f['data'][:]  # Assuming the dataset is stored under the key 'data'
        elif self.filepath.endswith('.pth'):
            data = torch.load(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {self.filepath}")
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Assuming each item in data_list is a tuple of (input_tensor, label)
        return self.data_list[idx]
    
    def collate_fn(batch):
        # This function will be used to collate the data into batches
        inputs = []
        labels = []
        for item in batch:
            input_tensor, label = item
            inputs.append(input_tensor)
            labels.append(label)
        return torch.stack(inputs), torch.tensor(labels)
    
