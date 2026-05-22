# mlp_inference_v3.py
"""
v3 inference entrypoint.

Two modes:

1. Training-data mode (default): iterates an EmbeddingPairDatasetV2 from the
   configured filedir and runs the image + ESM encoders. Reproduces the
   previous behavior of this script.

2. External-h5ad mode (--external-h5ad PATH): reads an AnnData h5ad whose .X
   is a matrix of [N, 1536] image embeddings (typically produced by
   build_inference_anndata.py). Runs only the image encoder, plus optionally
   the localization and protein-identity heads. The output AnnData inherits
   every obs column from the input AnnData.
"""

import argparse
import json
import yaml
import torch
import numpy as np
import pandas as pd
import anndata
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from mlp_models_v3 import (
    ImageEncoderV3, ProteinEncoderV3, ProteinPool,
    ProteinIdentityHead, LocalizationHead,
)
from data_v2 import EmbeddingPairDatasetV2, collate_variable_proteins_v2


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_paired(img_enc, prot_enc, pool, dataloader, device):
    img_enc.eval()
    prot_enc.eval()
    pool.eval()

    all_h, all_z, all_pz = [], [], []

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


def run_image_only(img_enc, identity_head, loc_head, dataloader, device,
                   include_loc=False, include_identity=False):
    img_enc.eval()
    if include_identity:
        identity_head.eval()
    if include_loc:
        loc_head.eval()

    all_h, all_z = [], []
    all_loc = [] if include_loc else None
    all_id  = [] if include_identity else None

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference"):
            imgs = batch[0].to(device, non_blocking=True)
            h, z = img_enc(imgs)

            all_h.append(h.cpu().numpy())
            all_z.append(z.cpu().numpy())

            if include_loc:
                all_loc.append(loc_head(h).cpu().numpy())
            if include_identity:
                all_id.append(identity_head(h).cpu().numpy())

    h_arr = np.concatenate(all_h, axis=0)
    z_arr = np.concatenate(all_z, axis=0)
    loc_arr = np.concatenate(all_loc, axis=0) if include_loc else None
    id_arr  = np.concatenate(all_id,  axis=0) if include_identity else None
    return h_arr, z_arr, loc_arr, id_arr


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config, vocab, model_filepath, device,
               build_prot_branch=True):
    n_genes = len(vocab["gene_vocab"])
    n_locs  = len(vocab["loc_vocab"])

    hidden_dim      = config["hidden_dim"]
    shared_dim      = config.get("shared_dim", 512)
    prot_hidden_dim = config.get("prot_hidden_dim", hidden_dim)

    img_enc       = ImageEncoderV3(
        in_dim=1536, hidden_dim=hidden_dim, shared_dim=shared_dim,
    ).to(device)
    identity_head = ProteinIdentityHead(in_dim=hidden_dim, n_classes=n_genes).to(device)
    loc_head      = LocalizationHead(in_dim=hidden_dim, n_locs=n_locs).to(device)

    if build_prot_branch:
        prot_enc = ProteinEncoderV3(
            in_dim=1280, hidden_dim=prot_hidden_dim, shared_dim=shared_dim,
        ).to(device)
        pool = ProteinPool(dim=1280).to(device)
    else:
        prot_enc = None
        pool = None

    checkpoint = torch.load(model_filepath, map_location=device, weights_only=False)
    img_enc.load_state_dict(checkpoint["img_enc_state"])
    identity_head.load_state_dict(checkpoint["identity_head_state"])
    loc_head.load_state_dict(checkpoint["loc_head_state"])
    if build_prot_branch:
        prot_enc.load_state_dict(checkpoint["prot_enc_state"])
        pool.load_state_dict(checkpoint["pool_state"])

    return img_enc, prot_enc, pool, identity_head, loc_head


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_paired_mode(args):
    versions = args.versions
    FILTER = args.filter

    for VERSION in versions:
        print(f"running v3 inference on version {VERSION} of training -- {FILTER}")

        config_filepath = (
            args.config if args.config
            else f"/scratch/users/samutiti/U54/SubCellNuc/configs/train_v{VERSION}.yml"
        )

        model_dir = Path(
            args.model_dir if args.model_dir
            else f"/scratch/users/samutiti/U54/SubCellNuc/training_V{VERSION}"
        )
        outpath = Path(args.output) if args.output else (
            model_dir / (f"inference_{FILTER}.h5ad" if FILTER is not None else "inference.h5ad")
        )

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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
            collate_fn=collate_variable_proteins_v2,
        )

        vocab_path = model_dir / "vocab.json"
        model_filepath = model_dir / "checkpoint.pt"
        with open(vocab_path, "r") as f:
            vocab = json.load(f)

        img_enc, prot_enc, pool, identity_head, loc_head = load_model(
            config, vocab, model_filepath, device, build_prot_branch=True
        )

        h_embed, z_embed, pz_embed = run_paired(
            img_enc, prot_enc, pool, dataloader, device
        )

        inv_gene_vocab = {v: k for k, v in vocab["gene_vocab"].items()}
        gene_names, locations, gene_idxs, atlas_names = [], [], [], []
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

        adata = anndata.AnnData(X=z_embed, obs=obs_df)
        adata.obsm["h"]      = h_embed
        adata.obsm["prot_z"] = pz_embed

        outpath.parent.mkdir(parents=True, exist_ok=True)
        adata.write(outpath)

        print(f"Saved AnnData to: {outpath}")
        print(f"Shape: {adata.shape}")


def run_external_mode(args):
    if not args.output:
        raise ValueError("--output is required in --external-h5ad mode")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve config + model dir (defaults follow the same layout as paired mode)
    config_filepath = (
        args.config if args.config
        else f"/scratch/users/samutiti/U54/SubCellNuc/configs/train_v{args.versions[0]}.yml"
    )
    model_dir = Path(
        args.model_dir if args.model_dir
        else f"/scratch/users/samutiti/U54/SubCellNuc/training_V{args.versions[0]}"
    )

    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

    vocab_path = model_dir / "vocab.json"
    model_filepath = model_dir / "checkpoint.pt"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    gene_vocab = vocab["gene_vocab"]
    loc_vocab  = vocab["loc_vocab"]

    img_enc, _prot_enc, _pool, identity_head, loc_head = load_model(
        config, vocab, model_filepath, device, build_prot_branch=False
    )

    print(f"Loading external embeddings from {args.external_h5ad}")
    in_adata = anndata.read_h5ad(args.external_h5ad)
    X = in_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    n_obs, n_var = X.shape
    expected_dim = 1536
    if n_var != expected_dim:
        raise ValueError(
            f"External h5ad has {n_var} features; expected {expected_dim} "
            f"(ImageEncoderV3 in_dim)."
        )
    print(f"External shape: {X.shape}")

    dataset = TensorDataset(torch.from_numpy(X))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    h_arr, z_arr, loc_logits, id_logits = run_image_only(
        img_enc, identity_head, loc_head, dataloader, device,
        include_loc=args.include_loc_preds,
        include_identity=args.include_identity_preds,
    )

    # Build output obs from the input AnnData's obs verbatim, then attach
    # head predictions as additional columns / obsm entries.
    obs = in_adata.obs.copy().reset_index(drop=True)
    obs.index = obs.index.astype(str)

    out = anndata.AnnData(X=z_arr, obs=obs)
    out.obsm["h"] = h_arr

    if args.include_loc_preds:
        out.obsm["loc_logits"] = loc_logits
        loc_probs = 1.0 / (1.0 + np.exp(-loc_logits))  # sigmoid (multi-label)
        out.obsm["loc_probs"] = loc_probs.astype(np.float32)
        inv_loc_vocab = {v: k for k, v in loc_vocab.items()}
        top1_loc_idx = loc_logits.argmax(axis=1)
        out.obs["loc_pred_top1"] = [
            inv_loc_vocab.get(int(i), "") for i in top1_loc_idx
        ]
        out.uns["loc_vocab"] = list(
            sorted(loc_vocab.keys(), key=lambda k: loc_vocab[k])
        )

    if args.include_identity_preds:
        out.obsm["identity_logits"] = id_logits
        # softmax for stable probs
        shifted = id_logits - id_logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        identity_probs = exp / exp.sum(axis=1, keepdims=True)
        out.obsm["identity_probs"] = identity_probs.astype(np.float32)
        inv_gene_vocab = {v: k for k, v in gene_vocab.items()}
        top1_gene_idx = id_logits.argmax(axis=1)
        out.obs["identity_pred_top1"] = [
            inv_gene_vocab.get(int(i), "") for i in top1_gene_idx
        ]
        out.uns["gene_vocab"] = list(
            sorted(gene_vocab.keys(), key=lambda k: gene_vocab[k])
        )

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(outpath)

    print(f"Saved AnnData to: {outpath}")
    print(f"Shape: {out.shape}")
    print(f"obs columns: {list(out.obs.columns)}")
    print(f"obsm keys  : {list(out.obsm.keys())}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-h5ad", default=None,
        help=(
            "Path to an AnnData h5ad with image embeddings in .X. When set, "
            "runs the image encoder only against this file; otherwise the "
            "script runs paired training-data inference."
        ),
    )
    parser.add_argument(
        "--versions", type=int, nargs="+", default=[10],
        help="Training version(s). In external mode only the first is used.",
    )
    parser.add_argument(
        "--filter", default="U2OS",
        help="Atlas filter (paired mode only).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional override for the training YAML config path.",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Optional override for the training output directory containing "
             "checkpoint.pt and vocab.json.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output h5ad path. Required for --external-h5ad mode; optional in "
             "paired mode (defaults to <model-dir>/inference[_<filter>].h5ad).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--include-loc-preds", action="store_true",
        help="External mode: also run loc_head and store logits/probs/top-1.",
    )
    parser.add_argument(
        "--include-identity-preds", action="store_true",
        help="External mode: also run identity_head and store logits/probs/top-1.",
    )
    args = parser.parse_args()

    if args.external_h5ad is None:
        if args.include_loc_preds or args.include_identity_preds:
            print(
                "[warn] --include-loc-preds / --include-identity-preds are "
                "ignored in paired training-data mode."
            )
        run_paired_mode(args)
    else:
        run_external_mode(args)


if __name__ == "__main__":
    main()
