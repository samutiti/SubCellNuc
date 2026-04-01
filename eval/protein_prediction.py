import numpy as np
import pandas as pd
import anndata as ad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Running on {device}')

log_file = 'protein_prediction_results.txt'

files = [
    "/scratch/users/samutiti/U54/SubCellNuc/training_V04/inference_analyzed.h5ad",
    "/scratch/users/samutiti/U54/embeddings/all_harmonized_features_microscope_vit.h5ad"
]

titles = ['MLP features', 'subcell features']

for file, title in zip(files, titles):

    with open(log_file, 'a') as f:
        print(f'Protein prediction on {title}', file=f)
    print(f'Protein prediction on {title}')

    # -------------------------------
    # Load data (memory efficient)
    # -------------------------------
    adata = ad.read_h5ad(file, backed=None)  # keep in memory for speed

    X = adata.X.astype(np.float32)  # HUGE speedup vs float64

    # faster categorical encoding
    y_cat = pd.Categorical(adata.obs["gene_names"])
    y_encoded = y_cat.codes
    y = y_cat.astype(str)

    locations = adata.obs["locations"].values

    # -------------------------------
    # Filter rare proteins
    # -------------------------------
    counts = pd.value_counts(y)
    keep = counts[counts >= 50].index

    mask = np.isin(y, keep)

    X = X[mask]
    y_encoded = y_encoded[mask]
    y = y[mask]
    locations = locations[mask]

    # -------------------------------
    # Train/test split
    # -------------------------------
    X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
        X, y_encoded, locations,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    # -------------------------------
    # Normalize
    # -------------------------------
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================================================
    # Logistic Regression
    # =========================================================
    # tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = nn.Linear(X_train.shape[1], len(np.unique(y_train))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(10):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    # inference
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        y_prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    top5 = top_k_accuracy_score(y_test, y_prob, k=5)

    with open(log_file, 'a') as f:
        print(f"\nTop-1 Accuracy: {acc:.4f}", file=f)
        print(f"Top-5 Accuracy: {top5:.4f}", file=f)
    print(f"\nTop-1 Accuracy: {acc:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")

    # =========================================================
    # Nuclear mask (vectorized, faster)
    # =========================================================
    loc_test_lower = pd.Series(loc_test).str.lower()
    nuclear_mask = loc_test_lower.str.contains("nuc", na=False).values

    # =========================================================
    # kNN
    # =========================================================
    knn = KNeighborsClassifier(
        n_neighbors=10,
        metric="cosine",
        n_jobs=-1
    )
    knn.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)

    with open(log_file, 'a') as f:
        print(f"kNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}", file=f)
    print(f"kNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

    print('pipeline complete')