# data_v2.py
import torch
import os
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset


class EmbeddingPairDatasetV2(Dataset):
    """
    Loads paired (subcell, ESM) embeddings together with per-row metadata
    (gene_names, locations, atlas_name) produced by make_subcell_esm_data_3.py.

    Directory layout expected (all files share the same numeric suffix):
        subcell_<start>_<end>.npy   – float32 [N, 1536]
        esm_<start>_<end>.npy       – object array of variable-length lists
        meta_<start>_<end>.pkl      – DataFrame with columns: gene_names, locations, atlas_name

    Args:
        filedir:        Path to the directory described above.
        min_gene_count: Minimum number of occurrences for a gene_names key to
                        receive a class index.  Rows whose key falls below this
                        threshold still contribute to the CLIP loss; their
                        gene_idx is set to -1 and they are skipped in the
                        identity cross-entropy.
    """

    def __init__(self, filedir, min_gene_count=50, atlas_filter=None):
        self.atlas_filter = atlas_filter
        self.items, self.gene_vocab, self.loc_vocab = self._load_data(
            Path(filedir), min_gene_count
        )

    # ------------------------------------------------------------------
    def _load_data(self, path, min_gene_count):
        file_list = os.listdir(path)

        subcell_files = sorted(f for f in file_list if f.startswith("subcell_"))
        esm_files     = sorted(f for f in file_list if f.startswith("esm_"))
        meta_files    = sorted(f for f in file_list if f.startswith("meta_"))

        assert len(subcell_files) == len(esm_files) == len(meta_files), (
            "Mismatch in number of subcell / esm / meta files"
        )

        raw_items = []  # (subcell_arr, esm_list, gene_key, loc_str, atlas_str)

        for sub_f, esm_f, meta_f in zip(subcell_files, esm_files, meta_files):
            # Verify the numeric range suffix matches across all three
            sub_suffix  = sub_f[len("subcell_"):]          # e.g. "0_20000.npy"
            esm_suffix  = esm_f[len("esm_"):]
            meta_suffix = meta_f[len("meta_"):].replace(".pkl", ".npy")
            assert sub_suffix == esm_suffix == meta_suffix, (
                f"File suffix mismatch: {sub_f}, {esm_f}, {meta_f}"
            )

            sub_data  = np.load(path / sub_f,  allow_pickle=True)
            esm_data  = np.load(path / esm_f,  allow_pickle=True)
            meta_df   = pd.read_pickle(path / meta_f)

            assert len(sub_data) == len(esm_data) == len(meta_df)

            for i in range(len(sub_data)):
                atlas_str = meta_df.iloc[i].get("atlas_name", "")
                if self.atlas_filter and atlas_str != self.atlas_filter:
                    continue
                raw_items.append((
                    sub_data[i],
                    esm_data[i],
                    meta_df.iloc[i]["gene_names"],
                    meta_df.iloc[i]["locations"],
                    atlas_str,
                ))

        # ------ Build gene vocabulary (with frequency filter) ----------
        gene_counts = Counter(item[2] for item in raw_items if item[2])
        gene_vocab  = {
            gene: idx
            for idx, gene in enumerate(
                k for k, v in sorted(gene_counts.items()) if v >= min_gene_count
            )
        }

        # ------ Build location vocabulary (multi-label) ----------------
        all_locs = set()
        for _, _, _, loc_str, _ in raw_items:
            if loc_str:
                for loc in str(loc_str).split(","):
                    loc = loc.strip()
                    if loc:
                        all_locs.add(loc)
        loc_vocab = {loc: idx for idx, loc in enumerate(sorted(all_locs))}

        # ------ Resolve gene_idx per row --------------------------------
        items = []
        for sub, esm, gene_key, loc_str, atlas_str in raw_items:
            gene_idx = gene_vocab.get(gene_key, -1)
            items.append((sub, esm, gene_idx, loc_str, atlas_str))

        return items, gene_vocab, loc_vocab

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_arr, prots_list, gene_idx, loc_str, _atlas = self.items[idx]

        img = torch.as_tensor(img_arr, dtype=torch.float32)

        if prots_list is None or len(prots_list) == 0:
            prots = torch.zeros((0, 1280), dtype=torch.float32)
        else:
            prots = torch.from_numpy(
                np.stack(prots_list).astype(np.float32)
            )  # [N, 1280]

        gene_id_t = torch.tensor(gene_idx, dtype=torch.long)

        # Multi-hot location vector
        n_locs = len(self.loc_vocab)
        loc_vec = torch.zeros(n_locs, dtype=torch.float32)
        if loc_str:
            for loc in str(loc_str).split(","):
                loc = loc.strip()
                if loc in self.loc_vocab:
                    loc_vec[self.loc_vocab[loc]] = 1.0

        return img, prots, gene_id_t, loc_vec


def collate_variable_proteins_v2(batch):
    """
    Collate function for EmbeddingPairDatasetV2.

    Returns:
        imgs:       [B, 1536]
        prots:      [B, Nmax, 1280]  (zero-padded)
        mask:       [B, Nmax] bool, True = valid position
        gene_ids:   [B]  long  (-1 means no class assigned)
        loc_labels: [B, n_locs] float32 multi-hot
    """
    imgs, prots_list, gene_ids, loc_labels = zip(*batch)

    imgs       = torch.stack(imgs,       dim=0)   # [B, 1536]
    gene_ids   = torch.stack(gene_ids,   dim=0)   # [B]
    loc_labels = torch.stack(loc_labels, dim=0)   # [B, n_locs]

    B    = len(prots_list)
    Nmax = max(p.size(0) for p in prots_list) or 1
    D    = 1280

    prots = torch.zeros((B, Nmax, D),  dtype=torch.float32)
    mask  = torch.zeros((B, Nmax),     dtype=torch.bool)

    for i, p in enumerate(prots_list):
        n = p.size(0)
        if n > 0:
            prots[i, :n] = p
            mask[i, :n]  = True

    return imgs, prots, mask, gene_ids, loc_labels
