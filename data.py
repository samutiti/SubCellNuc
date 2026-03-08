# data.py
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset


class EmbeddingPairDataset(Dataset):
    """
    Expects data items stored as:
      (image_emb: Tensor[1536], protein_embs: Tensor[N,1280])
    where N can vary per sample.
    """
    def __init__(self, filepath):
        """
        items: list of tuples (img_emb, prot_embs)
               img_emb:  torch.Tensor [1536]
               prot_embs: torch.Tensor [N,1280]
        """
        self.items = self.load_data(filepath)

    def load_data(self, path):
        """
        Supports .pt (torch), .pkl (pickle), and .npy (numpy) formats.
        Expects the loaded data to be a list of tuples: (img_emb, prot_embs)"""
        if path.endswith(".pt"):
            return torch.load(path)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".npy"):
            data = np.load(path, allow_pickle=True)
            return data.tolist()  # convert to list of tuples
        else:
            raise ValueError(f"Unsupported file format: {path}")
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, prots = self.items[idx]
        img = torch.as_tensor(img, dtype=torch.float32)
        prots = torch.as_tensor(prots, dtype=torch.float32)
        return img, prots


def collate_variable_proteins(batch):
    """
    batch: list of (img_emb[1536], prot_embs[N,1280])
    Returns:
      imgs:  [B,1536]
      prots: [B,Nmax,1280]
      mask:  [B,Nmax] True where valid
    """
    imgs, prots_list = zip(*batch)

    imgs = torch.stack(imgs, dim=0)  # [B,1536]
    B = len(prots_list)
    Nmax = max(p.size(0) for p in prots_list) # max number of proteins in batch
    D = prots_list[0].size(1) # should be 1280

    prots = torch.zeros((B, Nmax, D), dtype=torch.float32)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)

    for i, p in enumerate(prots_list):
        n = p.size(0)
        prots[i, :n] = p # copy valid protein embeddings
        mask[i, :n] = True # mark valid positions

    return imgs, prots, mask