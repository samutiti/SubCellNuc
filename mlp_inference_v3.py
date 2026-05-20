# mlp_inference_v3.py
import json
import yaml
import torch
import numpy as np
import pandas as pd
import anndata
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from mlp_models_v3 import (
    ImageEncoderV3, ProteinEncoderV3, ProteinPool,
    ProteinIdentityHead, LocalizationHead,
)
from data_v2 import EmbeddingPairDatasetV2, collate_variable_proteins_v2

versions = [10]
FILTER = 'U2OS'
for VERSION in versions:
    print(
        f'running v3 inference on version {VERSION} of training -- {FILTER}'
    )

    ########## FIT PARAMS HERE ################
    config_filepath = f"/scratch/users/samutiti/U54/SubCellNuc/configs/train_v{VERSION}.yml"
    model_dir       = Path(f"/scratch/users/samutiti/U54/SubCellNuc/training_V{VERSION}")
    outpath         = model_dir / f"inference_{FILTER}.h5ad" if FILTER is not None else model_dir / f"inference.h5ad"
    ###########################################


    def run(img_enc, prot_enc, pool, dataloader: DataLoader, device: str):
        img_enc.eval()
        prot_enc.eval()
        pool.eval()

        all_h  = []
        all_z  = []
        all_pz = []

        with torch.inference_mode():
            for imgs, prots, mask, gene_ids, loc_labels in tqdm(dataloader, desc="Inference"):
                imgs  = imgs.to(device, non_blocking=True)
                prots = prots.to(device, non_blocking=True)
                mask  = mask.to(device, non_blocking=True)

                h, z   = img_enc(imgs)
                pooled = pool(prots, mask=mask)
                _, pz  = prot_enc(pooled)

                all_h.append(h.cpu().numpy())
                all_z.append(z.cpu().numpy())
                all_pz.append(pz.cpu().numpy())

        return (
            np.concatenate(all_h,  axis=0),
            np.concatenate(all_z,  axis=0),
            np.concatenate(all_pz, axis=0),
        )


    # -------------------------
    # Setup
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

    dataset = EmbeddingPairDatasetV2(
        config["filedir"],
        min_gene_count=config.get("inference_min_gene_count", 50),
        atlas_filter=FILTER,
    )

    print(f"Dataset size : {len(dataset)}")
    print(f"Gene classes : {len(dataset.gene_vocab)}")
    print(f"Loc classes  : {len(dataset.loc_vocab)}")

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_variable_proteins_v2,
    )

    # -------------------------
    # Load model
    # -------------------------
    model_filepath = model_dir / "checkpoint.pt"
    vocab_path     = model_dir / "vocab.json"

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    gene_vocab = vocab["gene_vocab"]
    loc_vocab  = vocab["loc_vocab"]

    n_genes = len(gene_vocab)
    n_locs  = len(loc_vocab)

    hidden_dim      = config["hidden_dim"]
    shared_dim      = config.get("shared_dim", 512)
    prot_hidden_dim = config.get("prot_hidden_dim", hidden_dim)

    img_enc       = ImageEncoderV3(
        in_dim=1536, hidden_dim=hidden_dim, shared_dim=shared_dim,
    ).to(device)
    prot_enc      = ProteinEncoderV3(
        in_dim=1280, hidden_dim=prot_hidden_dim, shared_dim=shared_dim,
    ).to(device)
    pool          = ProteinPool(dim=1280).to(device)
    identity_head = ProteinIdentityHead(in_dim=hidden_dim, n_classes=n_genes).to(device)
    loc_head      = LocalizationHead(in_dim=hidden_dim, n_locs=n_locs).to(device)

    checkpoint = torch.load(model_filepath, map_location=device, weights_only=False)
    img_enc.load_state_dict(checkpoint["img_enc_state"])
    prot_enc.load_state_dict(checkpoint["prot_enc_state"])
    pool.load_state_dict(checkpoint["pool_state"])
    identity_head.load_state_dict(checkpoint["identity_head_state"])
    loc_head.load_state_dict(checkpoint["loc_head_state"])

    # -------------------------
    # Run inference
    # -------------------------
    h_embed, z_embed, pz_embed = run(img_enc, prot_enc, pool, dataloader, device)

    # Build metadata from dataset items (order matches DataLoader with shuffle=False)
    inv_gene_vocab = {v: k for k, v in gene_vocab.items()}

    gene_names  = []
    locations   = []
    gene_idxs   = []
    atlas_names = []

    for _sub, _esm, gene_idx, loc_str, atlas_str in dataset.items:
        gene_idxs.append(gene_idx)
        gene_names.append(inv_gene_vocab.get(gene_idx, ""))
        locations.append(loc_str if loc_str else "")
        atlas_names.append(atlas_str if atlas_str else "")

    obs_df = pd.DataFrame({
        "gene_name":  gene_names,
        "gene_idx":   gene_idxs,
        "locations":  locations,
        "atlas_name": atlas_names,
    })

    # X        = image projection in the shared latent space (z_img)
    # obsm[h]  = image hidden representation
    # obsm[pz] = ESM projection in the shared latent space (z_prot)
    adata = anndata.AnnData(X=z_embed, obs=obs_df)
    adata.obsm["h"]      = h_embed
    adata.obsm["prot_z"] = pz_embed

    adata.write(outpath)

    print(f"Saved AnnData to: {outpath}")
    print(f"Shape: {adata.shape}")
