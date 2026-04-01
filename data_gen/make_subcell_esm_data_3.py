import numpy as np
import pandas as pd
import torch
from unipressed import IdMappingClient
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os, time, pickle

SAVE_DIR = "/scratch/users/samutiti/U54/paired_embeddings_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

############################################
# Load subcell embeddings
############################################

hpa_df, embeddings = torch.load(
    "/scratch/users/samutiti/U54/embeddings/all_harmonized_features_microscope_vit.pth",
    weights_only=False
)

embeddings = embeddings.cpu().numpy()

############################################
# Load ESM embeddings and build lookup dict
############################################

print("Loading ESM embeddings...")

esm_df = pd.read_csv(
    "/scratch/users/samutiti/U54/embeddings/esm_mean_emb_df.csv"
)

gene_col = esm_df.columns[1]
vec_cols = esm_df.columns[2:1282]
print('vec_cols shape ', vec_cols)

esm_lookup = {
    row[gene_col]: row[vec_cols].values.astype(np.float32)
    for _, row in esm_df.iterrows()
}

del esm_df

############################################
# Query UniProt once for all genes
############################################

print("Collecting unique gene names...")

all_genes = set()
for g in hpa_df["gene_names"]:
    for name in g.split(","):
        all_genes.add(name.strip())

client = IdMappingClient()

print("Querying UniProt mapping...")

response = client.submit(
    source="GeneCards",
    dest="UniProtKB",
    ids=list(all_genes)
)
time.sleep(3600.0)

gene_to_uniprot = {}
for entry in response.each_result():
    gene_to_uniprot[entry["from"]] = entry["to"]

############################################
# Worker function
############################################

def process_row(args):
    idx, emb, gene_str, loc_str, atlas_str = args

    genes = [g.strip() for g in gene_str.split(",")]

    esm_vectors = []

    for g in genes:
        uid = gene_to_uniprot.get(g)

        if uid is None:
            continue

        vec = esm_lookup.get(uid)

        if vec is not None:
            esm_vectors.append(vec)

    if len(esm_vectors) == 0:
        return None

    # Sort gene names so ordering doesn't create spurious distinct classes
    canonical_gene_key = ",".join(sorted(genes))

    # Normalize loc/atlas to empty string if missing
    loc_out = str(loc_str) if loc_str is not None and not (isinstance(loc_str, float) and np.isnan(loc_str)) else ""
    atlas_out = str(atlas_str) if atlas_str is not None and not (isinstance(atlas_str, float) and np.isnan(atlas_str)) else ""

    return (emb.astype(np.float32), esm_vectors, canonical_gene_key, loc_out, atlas_out)


############################################
# Parallel chunk processing
############################################

CHUNK = 20000
N = len(embeddings)

print("Processing dataset...")

for start in range(0, N, CHUNK):

    end = min(start + CHUNK, N)

    print(f"Chunk {start} : {end}")

    args = [
        (
            i,
            embeddings[i],
            hpa_df["gene_names"].iloc[i],
            hpa_df["locations"].iloc[i],
            hpa_df["atlas_name"].iloc[i],
        )
        for i in range(start, end)
    ]

    with Pool(cpu_count()) as p:
        results = list(
            tqdm(
                p.imap(process_row, args),
                total=len(args)
            )
        )

    results = [r for r in results if r is not None]

    subcell = np.stack([r[0] for r in results])
    esm = np.array([r[1] for r in results], dtype=object)

    meta = pd.DataFrame({
        "gene_names": [r[2] for r in results],
        "locations":  [r[3] for r in results],
        "atlas_name": [r[4] for r in results],
    })

    np.save(f"{SAVE_DIR}/subcell_{start}_{end}.npy", subcell)
    np.save(f"{SAVE_DIR}/esm_{start}_{end}.npy", esm)
    meta.to_pickle(f"{SAVE_DIR}/meta_{start}_{end}.pkl")

print("Done.")
