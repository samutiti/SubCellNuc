import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -------------------------------
# Paths
# -------------------------------
input_path = Path("/scratch/users/samutiti/U54/SubCellNuc/training_V04/inference_analyzed.h5ad")   # <-- update
output_path = Path("/scratch/users/samutiti/U54/SubCellNuc/training_V04/mean_u2os_inference_analyzed.h5ad")

# -------------------------------
# Load data
# -------------------------------
print("Loading AnnData...")
adata = sc.read_h5ad(input_path)
adata = adata[adata.obs['atlas_name'] == 'U2OS']

# -------------------------------
# Ensure gene_names exists
# -------------------------------
if "gene_names" not in adata.obs:
    raise ValueError("gene_names column not found in adata.obs")

# Drop NaNs if present
adata = adata[~adata.obs["gene_names"].isna()].copy()

# -------------------------------
# Compute mean embeddings per gene_names
# -------------------------------
print("Computing mean embeddings per gene_names...")

# Convert to DataFrame for grouping
X_df = pd.DataFrame(
    adata.X,
    index=adata.obs["gene_names"].values
)

# Group by gene_names and average
mean_embeddings = X_df.groupby(level=0).mean()

print(f"Reduced from {adata.n_obs} cells → {mean_embeddings.shape[0]} atlas groups")

# -------------------------------
# Create new AnnData object
# -------------------------------
print("Constructing new AnnData...")

adata_mean = ad.AnnData(
    X=mean_embeddings.values
)

# Store gene_names as obs
adata_mean.obs["gene_names"] = mean_embeddings.index.astype(str)
adata_mean.obs_names = adata_mean.obs["gene_names"]

# -------------------------------
# Compute neighbors + UMAP
# -------------------------------
print("Running PCA (optional but recommended)...")
sc.pp.pca(adata_mean, n_comps=50)

print("Computing neighbors...")
sc.pp.neighbors(adata_mean, n_neighbors=15, n_pcs=50)

print("Computing UMAP...")
sc.tl.umap(adata_mean)

# -------------------------------
# Store UMAP in obs
# -------------------------------
adata_mean.obs["mean_umap_x"] = adata_mean.obsm["X_umap"][:, 0]
adata_mean.obs["mean_umap_y"] = adata_mean.obsm["X_umap"][:, 1]

# -------------------------------
# Save result
# -------------------------------
print("Saving...")
adata_mean.write(output_path)

print(f"Done! Saved to: {output_path}")