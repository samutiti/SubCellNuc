"""Training: joint reconstruction + protein classification on single-cell crops.

Two data backends are supported:

(1) Directory mode (preferred for this dataset):
    python -m cnn.train --data_dir /path/to/crops \
        --mask_mode none --use_mt 0 --epochs 50

(2) CSV mode (legacy, see cnn/dataset.py):
    python -m cnn.train --train_csv path/to/train.csv --val_csv path/to/val.csv \
        --in_channels 2 --epochs 50

Mask modes (directory mode):
    --mask_mode none    no masking
    --mask_mode cell    multiply each channel by the cell mask
    --mask_mode nuclei  multiply each channel by the nuclei mask

MT toggle (directory mode):
    --use_mt 0  drop microtubule channel (2-channel input: protein + nucleus)
    --use_mt 1  include MT             (3-channel input: protein + MT + nucleus)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cnn.model import ProteinCNN


RECON_WEIGHT = 1.0
CLS_WEIGHT = 1.0


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Directory-mode args
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory containing per-cell PNG/TIFF crops. "
                        "If set, uses cnn.dataset_v2 (recommended).")
    p.add_argument("--mask_mode", type=str, default="none",
                   choices=["none", "cell", "nuclei"],
                   help="On-the-fly masking applied to all channels.")
    p.add_argument("--use_mt", type=str2bool, default=False,
                   help="Include microtubule (red) channel.")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=0)

    # CSV-mode args (legacy)
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--in_channels", type=int, default=None, choices=[None, 2, 3],
                   help="CSV mode only. Directory mode infers from --use_mt.")

    # Common
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints/cnn")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", type=str2bool, default=True, help="Mixed precision (CUDA only).")
    p.add_argument("--compile", type=str2bool, default=False, help="torch.compile the model.")
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def build_data(args):
    """Return (train_loader, val_loader, label_encoder, in_channels, info)."""
    if args.data_dir is not None:
        from cnn.dataset_v2 import build_dataloaders_from_dir
        train_loader, val_loader, label_encoder, stats = build_dataloaders_from_dir(
            data_dir=args.data_dir,
            mask_mode=args.mask_mode,
            use_mt=args.use_mt,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_frac=args.val_frac,
            seed=args.split_seed,
        )
        in_channels = 3 if args.use_mt else 2
        return train_loader, val_loader, label_encoder, in_channels, stats

    if args.train_csv is None:
        raise ValueError("Provide --data_dir (directory mode) or --train_csv (CSV mode).")
    from cnn.dataset import build_dataloaders
    in_channels = args.in_channels or 2
    train_loader, val_loader, label_encoder = build_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        in_channels=in_channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    stats = {"mode": "csv"}
    return train_loader, val_loader, label_encoder, in_channels, stats


def run_epoch(model, loader, optimizer, scaler, device, train: bool, use_amp: bool, log_every: int = 20):
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "cls": 0.0, "correct": 0, "n": 0}

    ctx = torch.enable_grad() if train else torch.no_grad()
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp and device.startswith("cuda"))

    t_start = time.perf_counter()
    with ctx:
        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with amp_ctx:
                out = model(images)
                recon_loss = F.mse_loss(out["recon"], images)
                cls_loss = F.cross_entropy(out["logits"], labels)
                loss = RECON_WEIGHT * recon_loss + CLS_WEIGHT * cls_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = images.size(0)
            totals["loss"] += loss.item() * bs
            totals["recon"] += recon_loss.item() * bs
            totals["cls"] += cls_loss.item() * bs
            totals["correct"] += (out["logits"].argmax(dim=1) == labels).sum().item()
            totals["n"] += bs

            if train and log_every and (step + 1) % log_every == 0:
                elapsed = time.perf_counter() - t_start
                ips = totals["n"] / max(elapsed, 1e-6)
                print(f"    step {step+1}/{len(loader)}  "
                      f"loss={loss.item():.4f}  acc_run={totals['correct']/totals['n']:.3f}  "
                      f"{ips:.0f} img/s")

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

    train_loader, val_loader, label_encoder, in_channels, info = build_data(args)
    num_classes = len(label_encoder)
    print(f"data: {info}")
    print(f"in_channels={in_channels}  num_classes={num_classes}  "
          f"train_batches={len(train_loader)}  val_batches={len(val_loader) if val_loader else 0}")

    with open(out_dir / "label_encoder.json", "w") as f:
        json.dump(label_encoder, f, indent=2)
    with open(out_dir / "run_config.json", "w") as f:
        cfg = {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()}
        cfg["in_channels"] = in_channels
        cfg["data_info"] = info
        json.dump(cfg, f, indent=2)

    model = ProteinCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
    ).to(args.device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_stats = run_epoch(model, train_loader, optimizer, scaler, args.device,
                                train=True, use_amp=use_amp, log_every=args.log_every)
        log = (f"epoch {epoch:03d}  "
               f"train loss={train_stats['loss']:.4f} (recon={train_stats['recon']:.4f}, "
               f"cls={train_stats['cls']:.4f}, acc={train_stats['acc']:.3f})")

        if val_loader is not None and len(val_loader) > 0:
            val_stats = run_epoch(model, val_loader, optimizer, None, args.device,
                                  train=False, use_amp=use_amp, log_every=0)
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

        log += f"  [{time.perf_counter() - t0:.1f}s]"
        print(log, flush=True)
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
