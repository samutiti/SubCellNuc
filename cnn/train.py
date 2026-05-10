"""Training script: joint reconstruction + protein classification.

Usage:
    python -m cnn.train --train_csv path/to/train.csv --val_csv path/to/val.csv \
                        --in_channels 2 --epochs 50 --batch_size 64

================================================================================
EDIT POINTS (search for `# EDIT:` to find every spot you must touch)
================================================================================
1. CLI defaults below           — set your default train/val CSV paths
2. cnn/dataset.py CSV_COLUMNS    — column names in your CSV
3. cnn/dataset.py PROTEIN_LABEL_COLUMN — protein-identity column name
4. cnn/dataset.py build_label_encoder  — if you have a canonical proteins list
5. RECON_WEIGHT / CLS_WEIGHT     — loss balance (defaults below)
================================================================================
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cnn.dataset import build_dataloaders
from cnn.model import ProteinCNN


# EDIT: tune these to balance the two objectives. With normalized [0, 1] images,
# MSE is small (~0.01-0.1) and CE is order ~ln(num_classes), so we usually weight
# reconstruction up. Watch the per-loss curves on the first epoch and adjust.
RECON_WEIGHT = 1.0
CLS_WEIGHT = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # EDIT: replace the `required=True` defaults with your real paths if you want
    # to launch with no arguments.
    p.add_argument("--train_csv", type=str, required=True, help="CSV with image paths + protein label")
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--in_channels", type=int, default=2, choices=[2, 3],
                   help="2 = protein+nucleus, 3 = protein+MT+nucleus")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints/cnn")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "cls": 0.0, "correct": 0, "n": 0}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            out = model(images)
            recon_loss = F.mse_loss(out["recon"], images)
            cls_loss = F.cross_entropy(out["logits"], labels)
            loss = RECON_WEIGHT * recon_loss + CLS_WEIGHT * cls_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            totals["loss"] += loss.item() * bs
            totals["recon"] += recon_loss.item() * bs
            totals["cls"] += cls_loss.item() * bs
            totals["correct"] += (out["logits"].argmax(dim=1) == labels).sum().item()
            totals["n"] += bs

    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "recon": totals["recon"] / n,
        "cls": totals["cls"] / n,
        "acc": totals["correct"] / n,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, label_encoder = build_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        in_channels=args.in_channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(label_encoder)
    print(f"num_classes={num_classes}  train_batches={len(train_loader)}"
          f"  val_batches={len(val_loader) if val_loader else 0}")

    # Persist label encoder so inference can use the same mapping
    with open(out_dir / "label_encoder.json", "w") as f:
        json.dump(label_encoder, f, indent=2)

    model = ProteinCNN(
        in_channels=args.in_channels,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, args.device, train=True)
        log = (f"epoch {epoch:03d}  "
               f"train loss={train_stats['loss']:.4f} (recon={train_stats['recon']:.4f}, "
               f"cls={train_stats['cls']:.4f}, acc={train_stats['acc']:.3f})")

        if val_loader is not None:
            val_stats = run_epoch(model, val_loader, optimizer, args.device, train=False)
            log += (f"  |  val loss={val_stats['loss']:.4f} "
                    f"(recon={val_stats['recon']:.4f}, cls={val_stats['cls']:.4f}, "
                    f"acc={val_stats['acc']:.3f})")
            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "label_encoder": label_encoder,
                    "epoch": epoch,
                }, out_dir / "best.pt")

        print(log)
        scheduler.step()

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "label_encoder": label_encoder,
        "epoch": args.epochs,
    }, out_dir / "last.pt")
    print(f"saved checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
