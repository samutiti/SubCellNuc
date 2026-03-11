# data.py
import torch
import os
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset


class EmbeddingPairDataset(Dataset):
    """
    Data is stored in matched files (esm_0_20000 -- subcell_0_20000)
    Expecting ~ 15GB of data to be loaded
    """
    def __init__(self, filedir):
        """
        Parameters
        filedir: (str) = a string representing the directory where the esm and subcell file(s) are stored
            expects the embeddings to be stored in different files
        items: list of tuples (img_emb, prot_embs)
               img_emb:  torch.Tensor [1536]
               prot_embs: torch.Tensor [N,1280]
        """
        self.items = self.load_data(Path(filedir))

    def load_data(self, path):
        """
        Returns the loaded data to be a list of tuples: (img_emb, prot_embs)"""
        file_list = os.listdir(path)
        esm_files = []
        subcell_files = []

        for file in file_list:
            if file[0:3] == 'esm':
                esm_files.append(file)
            else: subcell_files.append(file)
        
        # directory should ONLY contain esm and subcell numpy files that are matched by name
        assert len(subcell_files) == len(esm_files) 
        assert subcell_files[0].endswith('npy') or subcell_files[0].endswith('npz')
        assert esm_files[0].endswith('npy') or esm_files[0].endswith('npz')

        esm_files.sort()
        subcell_files.sort()

        esm_data = []
        sub_data = []

        for esm_f, subcell_f in zip(esm_files, subcell_files):
            assert esm_f[-12:] == subcell_f[-12:] # make sure the files match
            esm_data.extend(np.load(path / esm_f, allow_pickle=True))
            sub_data.extend(np.load(path / subcell_f, allow_pickle=True))

        return list(zip(sub_data, esm_data))
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, prots = self.items[idx]
        prots = np.array(prots)
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