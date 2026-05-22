# build_inference_anndata.py
"""
Combine three inputs into a single AnnData h5ad ready for `mlp_inference_v3.py
--external-h5ad`:

  1. An AnnData h5ad of new image embeddings (n_obs x 1536) in `.X`.
  2. A per-crop metadata table (csv / pickle / parquet) with one row per
     embedding, in the same order, containing at least a gene-name column
     (default `Gene_Name_HPA`).
  3. The Human Protein Atlas `subcellular_location.csv`, used to look up
     `Main location` per gene.

The output AnnData has:
  - X        : the input embeddings (float32)
  - obs      : every metadata column except `Baselink` and `Unnamed: 9`,
               plus a new `Main location` column populated by joining
               `Gene_Name_HPA` against `subcellular_location.csv`. Genes
               with no match get "".
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


def load_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported metadata file extension: {suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", required=True,
        help="Path to AnnData h5ad with embeddings in .X (shape [N, 1536]).",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to per-crop metadata table; row order must match --embeddings.",
    )
    parser.add_argument(
        "--sc-locs", required=True,
        help="Path to subcellular_location.csv (HPA).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output h5ad path.",
    )
    parser.add_argument(
        "--drop-cols", nargs="*", default=["Baselink", "Unnamed: 9"],
        help="Metadata columns to drop before writing obs.",
    )
    parser.add_argument(
        "--gene-col", default="Gene_Name_HPA",
        help="Metadata column with gene symbols used to look up Main location.",
    )
    parser.add_argument(
        "--sc-locs-gene-col", default="Gene name",
        help="Column in sc_locs that holds the gene symbol.",
    )
    parser.add_argument(
        "--sc-locs-target-col", default="Main location",
        help="Column in sc_locs to copy onto obs.",
    )
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    metadata_path = Path(args.metadata)
    sc_locs_path = Path(args.sc_locs)
    output_path = Path(args.output)

    print(f"Loading embeddings from {embeddings_path}")
    emb_ad = anndata.read_h5ad(embeddings_path)
    X = emb_ad.X
    # densify if sparse, ensure float32
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    n_obs, n_var = X.shape
    print(f"Embeddings shape: {X.shape}")

    print(f"Loading metadata from {metadata_path}")
    meta = load_metadata(metadata_path)
    if len(meta) != n_obs:
        raise ValueError(
            f"Metadata has {len(meta)} rows but embeddings has {n_obs}; "
            f"row order must match one-to-one."
        )

    keep_cols = [c for c in meta.columns if c not in set(args.drop_cols)]
    obs = meta[keep_cols].copy().reset_index(drop=True)

    if args.gene_col not in obs.columns:
        raise KeyError(
            f"--gene-col {args.gene_col!r} not found in metadata columns: "
            f"{list(obs.columns)}"
        )

    print(f"Loading sc_locs from {sc_locs_path}")
    sc_locs = pd.read_csv(sc_locs_path)
    for col in (args.sc_locs_gene_col, args.sc_locs_target_col):
        if col not in sc_locs.columns:
            raise KeyError(
                f"sc_locs missing column {col!r}; available: {list(sc_locs.columns)}"
            )

    sc_locs_first = sc_locs.drop_duplicates(
        subset=[args.sc_locs_gene_col], keep="first"
    )
    loc_map = dict(
        zip(
            sc_locs_first[args.sc_locs_gene_col].astype(str),
            sc_locs_first[args.sc_locs_target_col].astype(str),
        )
    )

    gene_names = obs[args.gene_col].astype(str)
    main_loc = gene_names.map(loc_map)
    # pandas .map yields NaN for misses; convert to "" for consistent dtype
    main_loc = main_loc.where(main_loc.notna(), "").astype(str)
    # Also normalize the literal string "nan" that can sneak in from astype(str)
    main_loc = main_loc.replace({"nan": ""})
    obs[args.sc_locs_target_col] = main_loc.values

    n_found = int((main_loc != "").sum())
    print(
        f"Resolved {args.sc_locs_target_col} for {n_found}/{n_obs} rows "
        f"({100.0 * n_found / max(n_obs, 1):.1f}%)"
    )

    # Carry over the input h5ad's `index` column (per the question's schema)
    # under a non-colliding name, so we never lose the original ordering key.
    if "index" in emb_ad.obs.columns and "embedding_index" not in obs.columns:
        obs["embedding_index"] = emb_ad.obs["index"].to_numpy()

    obs.index = obs.index.astype(str)

    out = anndata.AnnData(X=X, obs=obs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output_path)

    print(f"Wrote {output_path}")
    print(f"Output shape: {out.shape}")
    print(f"obs columns : {list(out.obs.columns)}")


if __name__ == "__main__":
    main()
