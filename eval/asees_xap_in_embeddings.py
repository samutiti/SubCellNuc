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
parser.add_argument('--use_acc', help='use accession keys instead of gene names to query XAPness', default=False, action='store_true')
args = parser.parse_args()

use_acc = args.use_acc # useful for some data storage methods
if use_acc: print('### Assess XAP in Embeddings is using accession labels in place of gene names')

# -----------------------------
# 1. Define XAP list
# -----------------------------
# Normalize function
def normalize_gene(g):
    return g.replace("-", "").replace("_", "").strip().upper()
    
xap_set_normalized = {normalize_gene(g) for g in {
"HMGB1", "ILF2", "XRN2", "POLDIP3", "RNF20", "FUBP3", "HNRPLL", "ILF3",
"SSB", "SRRT", "EIF4A3", "RNMT", "PTBP2", "SAP18", "WTAP", "KHDRBS1",
"ERH", "CIZ1", "HNRNPH3", "SYNCRIP", "NONO", "FUS", "DHX9", "RALY",
"DDX17", "HNRPDL", "KHSRP", "RNF2", "HNRNPA3", "HNRNPUL2", "HNRNPD",
"DDX5", "DDX39B", "PTBP1", "HNRNPR", "ZFR", "ELAVL1", "SRSF2",
"HNRNPM", "RBMXL1", "TARDBP", "MYEF2", "HNRNPU", "HNRNPC",
"HNRNPL", "HNRNPAB", "HNRNPK", "HNRNPA2B1", "HNRNPA1",
"HNRNPA0", "RBM14", "TRA2B", "SFPQ", "IGF2BP1", "SARNP",
"SRSF3", "MATR3", "SRSF5", "SPEN", "RBM15", "RBM3",
"SRSF7", "RBFOX2", "SRSF9", "SAFB", "THOC4", "DDX39A",
"CELF1", "PCGF5", "RYBP", "TRIM71", "SRSF10", "RBM4",
"YTHDC1", "TRIM6", "LIN28A", "SLTM", "SAFB2", "L1TD1",
"MYBBP1A", "IGF2BP3"
}}

with open('xap_to_accession_mapping.json', 'r') as f:
    xap_acc_list = json.load(f)
xap_set_acc = list(xap_acc_list.values())

# -----------------------------
# 2. Identify U2OS rows
# -----------------------------
embeddings, data_df = torch.load(args.emb_data, weights_only=False)
# -----------------------------
# 3. Determine XAP membership
# -----------------------------
def is_xap(gene_string):
    if use_acc:
        return gene_string in xap_set_acc # assuming gene string is singular
    genes = [normalize_gene(g) for g in gene_string.split(",")]
    return any(g in xap_set_normalized for g in genes)

try: genes = data_df["gene"].values
except:
    print(f'gene not available in dataframe with columns: {data_df}')
xap_mask = np.array([is_xap(g) for g in genes])

print("Total samples:", len(genes))
print("XAP samples:", xap_mask.sum())
print("Non-XAP samples:", (~xap_mask).sum())

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

# XAP–XAP
xap_dot = xap_emb @ xap_emb.T
sum_xap_xap = xap_dot.sum() - np.trace(xap_dot)
mean_xap_xap = sum_xap_xap / (n_x * (n_x - 1))

# Non–Non
non_dot = non_emb @ non_emb.T
sum_non_non = non_dot.sum() - np.trace(non_dot)
mean_non_non = sum_non_non / (n_n * (n_n - 1))

# XAP–Non
xap_non_dot = xap_emb @ non_emb.T
mean_xap_non = xap_non_dot.mean()

print("\nMean cosine similarity:")
print("XAP–XAP:", mean_xap_xap)
print("Non–Non:", mean_non_non)
print("XAP–Non:", mean_xap_non)

# -----------------------------
# 6. Effect size (Cohen's d)
# -----------------------------
def fast_cohens_d(mean1, mean2, mat1, mat2):
    var1 = mat1.var()
    var2 = mat2.var()
    return (mean1 - mean2) / np.sqrt((var1 + var2) / 2)

print("\nEffect size (Cohen's d):")
print("XAP vs Non–Non:", fast_cohens_d(mean_xap_xap, mean_non_non, xap_dot, non_dot))
print("XAP vs Between:", fast_cohens_d(mean_xap_xap, mean_xap_non, xap_dot, xap_non_dot))

# -----------------------------
# 7. t-test
# -----------------------------
# XAP–XAP (exclude diagonal)
xap_vals = xap_dot[~np.eye(n_x, dtype=bool)]

# XAP–Non
xap_non_vals = xap_non_dot.flatten()

t_stat, p_val = ttest_ind(xap_vals, xap_non_vals, equal_var=False)

print("\nT-test XAP–XAP vs XAP–Non:")
print("t =", t_stat, "p =", p_val)

# -----------------------------
# 8. Permutation test
# -----------------------------
def fast_permutation_test(emb, labels, n_perm=200):
    n_x = labels.sum()
    observed = (emb[labels] @ emb[labels].T).mean()

    diffs = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(labels)
        val = (emb[shuffled] @ emb[shuffled].T).mean()
        diffs.append(val)

    return observed, (np.sum(np.array(diffs) >= observed) + 1) / (n_perm + 1)

obs, perm_p = fast_permutation_test(emb, xap_mask, n_perm=200)
print("\nPermutation test p-value:", perm_p)

# -----------------------------
# 9. kNN enrichment (local clustering metric)
# -----------------------------
k = 10
nbrs = NearestNeighbors(
    n_neighbors=k+1,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

nbrs.fit(emb)  # use normalized embeddings
distances, indices = nbrs.kneighbors(emb)

# exclude self (first neighbor)
neighbor_labels = xap_mask[indices[:, 1:]]

# fraction of neighbors that are XAP
xap_neighbor_fraction = neighbor_labels.mean(axis=1)

print("\nMean fraction of XAP neighbors:")
print("For XAP samples:", np.mean(xap_neighbor_fraction[xap_mask]))
print("For Non-XAP samples:", np.mean(xap_neighbor_fraction[~xap_mask]))
