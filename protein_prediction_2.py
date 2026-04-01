import numpy as np
import pandas as pd
import anndata as ad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
import traceback

# -------------------------------
# Config
# -------------------------------
LOG_FILE = "protein_prediction_results.txt"
SUBSAMPLE_SIZE = 200_000   # set None to disable
BATCH_SIZE = 1024
EPOCHS = 10
USE_GPU = torch.cuda.is_available()

device = "cuda" if USE_GPU else "cpu"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        print(msg, file=f, flush=True)


def main():
    log(f"Running on {device}")

    files = [
        "/scratch/users/samutiti/U54/SubCellNuc/training_V04/inference_analyzed.h5ad",
        "/scratch/users/samutiti/U54/embeddings/all_harmonized_features_microscope_vit.h5ad"
    ]

    titles = ['MLP features', 'subcell features']

    for file, title in zip(files, titles):

        log(f"\n=== Protein prediction on {title} ===")

        # -------------------------------
        # Load data
        # -------------------------------
        log("Loading AnnData into memory...")
        adata = ad.read_h5ad(file, backed=None)

        log("Converting X to float32...")
        X = np.asarray(adata.X, dtype=np.float32)

        y_raw = adata.obs["gene_names"].astype(str).values
        locations = adata.obs["locations"].values

        log(f"Initial shape: {X.shape}")

        # -------------------------------
        # Remove NaNs
        # -------------------------------
        log("Removing NaNs...")
        valid_mask = pd.notnull(y_raw)

        X = X[valid_mask]
        y_raw = y_raw[valid_mask]
        locations = locations[valid_mask]

        # -------------------------------
        # Filter rare proteins FIRST
        # -------------------------------
        log("Filtering rare proteins (>=50)...")
        counts = pd.Series(y_raw).value_counts()
        keep = counts[counts >= 50].index

        mask = np.isin(y_raw, keep)

        X = X[mask]
        y_raw = y_raw[mask]
        locations = locations[mask]

        log(f"Post-filter shape: {X.shape}")

        # -------------------------------
        # Subsample AFTER filtering
        # -------------------------------
        if SUBSAMPLE_SIZE is not None and len(X) > SUBSAMPLE_SIZE:
            log(f"Subsampling to {SUBSAMPLE_SIZE} cells...")
            idx = np.random.choice(len(X), size=SUBSAMPLE_SIZE, replace=False)
            X = X[idx]
            y_raw = y_raw[idx]
            locations = locations[idx]

        # -------------------------------
        # FINAL label encoding (CRITICAL FIX)
        # -------------------------------
        log("Encoding labels...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)

        n_classes = len(le.classes_)
        log(f"Number of classes: {n_classes}")

        # -------------------------------
        # Train/test split
        # -------------------------------
        log("Splitting train/test...")
        X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
            X, y_encoded, locations,
            test_size=0.2,
            stratify=y_encoded,
            random_state=42
        )

        # Debug sanity check
        log(f"y_train min: {y_train.min()}, max: {y_train.max()}")
        assert y_train.min() == 0
        assert y_train.max() == n_classes - 1

        # -------------------------------
        # Normalize
        # -------------------------------
        log("Scaling features...")
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -------------------------------
        # Convert to tensors
        # -------------------------------
        log("Converting to tensors...")
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # -------------------------------
        # Model
        # -------------------------------
        log("Initializing model...")
        model = nn.Linear(X_train.shape[1], n_classes)

        if USE_GPU:
            model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # -------------------------------
        # Training
        # -------------------------------
        log("Starting training...")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for xb, yb in loader:
                if USE_GPU:
                    xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            log(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

        # -------------------------------
        # Inference
        # -------------------------------
        log("Running inference...")
        model.eval()

        if USE_GPU:
            X_test_t = X_test_t.to(device)

        with torch.no_grad():
            logits = model(X_test_t)
            y_prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = y_prob.argmax(axis=1)

        acc = accuracy_score(y_test, y_pred)

        # FIXED top-k accuracy
        top5 = top_k_accuracy_score(
            y_test,
            y_prob,
            k=5,
            labels=np.arange(n_classes)
        )

        log(f"Top-1 Accuracy: {acc:.4f}")
        log(f"Top-5 Accuracy: {top5:.4f}")

        # -------------------------------
        # Nuclear mask
        # -------------------------------
        log("Computing nuclear mask...")
        loc_test_lower = pd.Series(loc_test).str.lower()
        nuclear_mask = loc_test_lower.str.contains("nuc", na=False).values

        # -------------------------------
        # kNN
        # -------------------------------
        log("Running kNN...")
        knn = KNeighborsClassifier(
            n_neighbors=10,
            metric="cosine",
            n_jobs=-1
        )
        knn.fit(X_train, y_train)

        y_pred_knn = knn.predict(X_test)
        knn_acc = accuracy_score(y_test, y_pred_knn)

        log(f"kNN Accuracy: {knn_acc:.4f}")

        log("Pipeline complete for this dataset.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err_msg = "".join(traceback.format_exception(*sys.exc_info()))
        print(err_msg, flush=True)
        with open(LOG_FILE, "a") as f:
            print("\n=== ERROR ===", file=f)
            print(err_msg, file=f)
        raise