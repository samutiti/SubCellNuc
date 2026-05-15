import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
import random
import argparse
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument('emb_data', help='filepath to the torch file (.pth) containing a tuple of embeddings and associated data_frame')
parser.add_argument('disease', help='disease cluster to analyze')
args = parser.parse_args()
DISEASE = args.disease

print(f'### Assess {DISEASE} in Embeddings')

# -----------------------------
# 1. Define XAP list
# -----------------------------
# Normalize function
def normalize_gene(g):
    return g.replace("-", "").replace("_", "").strip().upper()
    

with open('/scratch/users/samutiti/U54/data/autoantigen_disease_mapping_v2.json', 'r') as f:
    disease_dict = json.load(f)

def sample_pairwise_means(A, B=None, n_samples=1_000_000, batch_size=10000):
    """
    Estimate mean cosine similarity via random sampling.
    If B is None: sample within A (excluding diagonal)
    """
    nA = A.shape[0]
    if B is None:
        idx1 = np.random.randint(0, nA, size=n_samples)
        idx2 = np.random.randint(0, nA, size=n_samples)
        mask = idx1 != idx2
        sims = (A[idx1[mask]] * A[idx2[mask]]).sum(axis=1)
    else:
        nB = B.shape[0]
        idx1 = np.random.randint(0, nA, size=n_samples)
        idx2 = np.random.randint(0, nB, size=n_samples)
        sims = (A[idx1] * B[idx2]).sum(axis=1)

    return sims.mean(), sims

# -----------------------------
# 2. Identify U2OS rows
# -----------------------------
embeddings, data_df = torch.load(args.emb_data, weights_only=False)
# -----------------------------
# 3. Determine XAP membership
# -----------------------------
def is_disease(gene_string):
    genes = [normalize_gene(g) for g in gene_string.split(",")]
    for g in genes:
        try:
            if disease_dict[g] == DISEASE:
                return True
        except: continue
    return False

try: genes = data_df["gene_name"].values
except:
    try: genes = data_df["gene"].values
    except:
        print(f'genes not findable in dataframe with columns: {data_df}')
xap_mask = np.array([is_disease(g) for g in genes])

print("Total samples:", len(genes))
print("Disease samples:", xap_mask.sum())
print("Non-Disease samples:", (~xap_mask).sum())

# -----------------------------
# 4. Normalize embeddings once
# -----------------------------
emb = embeddings
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

xap_emb = emb[xap_mask]
non_emb = emb[~xap_mask]

n_x = len(xap_emb)
n_n = len(non_emb)

print(f"\nComputing similarities without full matrix...")

# -----------------------------
# 5. Compute group similarities efficiently
# -----------------------------

# # XAP–XAP
# xap_dot = xap_emb @ xap_emb.T
# sum_xap_xap = xap_dot.sum() - np.trace(xap_dot)
# mean_xap_xap = sum_xap_xap / (n_x * (n_x - 1))

# # Non–Non
# non_dot = non_emb @ non_emb.T
# sum_non_non = non_dot.sum() - np.trace(non_dot)
# mean_non_non = sum_non_non / (n_n * (n_n - 1))

# # XAP–Non
# xap_non_dot = xap_emb @ non_emb.T
# mean_xap_non = xap_non_dot.mean()

mean_xap_xap, xap_vals = sample_pairwise_means(xap_emb, n_samples=1_000_000)
mean_non_non, non_vals = sample_pairwise_means(non_emb, n_samples=1_000_000)
mean_xap_non, xap_non_vals = sample_pairwise_means(xap_emb, non_emb, n_samples=1_000_000)

print("\nMean cosine similarity:")
print("dis–dis:", mean_xap_xap)
print("Non–Non:", mean_non_non)
print("dis–Non:", mean_xap_non)

# -----------------------------
# 6. Effect size (Cohen's d)
# -----------------------------
def fast_cohens_d(mean1, mean2, mat1, mat2):
    var1 = mat1.var()
    var2 = mat2.var()
    return (mean1 - mean2) / np.sqrt((var1 + var2) / 2)

def cohens_d_from_samples(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)

print("\nEffect size (Cohen's d):")
print("dis vs Non–Non:", cohens_d_from_samples(xap_vals, non_vals))
print("dis vs Between:", cohens_d_from_samples(xap_vals, xap_non_vals))

# -----------------------------
# 7. t-test
# -----------------------------
# XAP–XAP (exclude diagonal)
t_stat, p_val = ttest_ind(xap_vals, xap_non_vals, equal_var=False)

print("\nT-test dis–dis vs dis–Non:")
print("t =", t_stat, "p =", p_val)

# -----------------------------
# 8. Permutation test
# -----------------------------
def fast_permutation_test(emb, labels, n_perm=100, sample_size=200_000):
    observed, _ = sample_pairwise_means(emb[labels], n_samples=sample_size)

    diffs = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(labels)
        val, _ = sample_pairwise_means(emb[shuffled], n_samples=sample_size)
        diffs.append(val)

    diffs = np.array(diffs)
    return observed, (np.sum(diffs >= observed) + 1) / (n_perm + 1)

obs, perm_p = fast_permutation_test(emb, xap_mask, n_perm=200)
print("\nPermutation test p-value:", perm_p)

# -----------------------------
# 9. kNN enrichment (local clustering metric)
# -----------------------------
import faiss

d = emb.shape[1]
index = faiss.IndexFlatIP(d)  # cosine if normalized
index.add(emb.astype(np.float32))

k = 10
distances, indices = index.search(emb.astype(np.float32), k+1)

# exclude self (first neighbor)
neighbor_labels = xap_mask[indices[:, 1:]]

# fraction of neighbors that are XAP
xap_neighbor_fraction = neighbor_labels.mean(axis=1)

print("\nMean fraction of Disease neighbors:")
print("For disease samples:", np.mean(xap_neighbor_fraction[xap_mask]))
print("For Non-dis samples:", np.mean(xap_neighbor_fraction[~xap_mask]))
