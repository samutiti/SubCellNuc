# train_mlp_v2.py
import os, argparse, json, datetime
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mlp_models_v2 import (
    ImageProjectorV2, ProteinPool,
    ProteinIdentityHead, LocalizationHead,
    combined_loss,
)
from data_v2 import EmbeddingPairDatasetV2, collate_variable_proteins_v2


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, projector, pool, identity_head, loc_head,
                    optimizer, epoch, n_genes, n_locs):
    ckpt = {
        "epoch":               epoch,
        "projector_state":     projector.state_dict(),
        "pool_state":          pool.state_dict(),
        "identity_head_state": identity_head.state_dict(),
        "loc_head_state":      loc_head.state_dict(),
        "optimizer_state":     optimizer.state_dict(),
        "n_genes":             n_genes,
        "n_locs":              n_locs,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, projector, pool, identity_head, loc_head,
                    optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    projector.load_state_dict(ckpt["projector_state"])
    pool.load_state_dict(ckpt["pool_state"])
    identity_head.load_state_dict(ckpt["identity_head_state"])
    loc_head.load_state_dict(ckpt["loc_head_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    projector, pool, identity_head, loc_head,
    loader, optimizer, device,
    temperature=0.07, lambda_id=1.0, lambda_loc=1.0,
    tqdm_obj=None,
):
    projector.train()
    pool.train()
    identity_head.train()
    loc_head.train()

    total_loss = total_clip = total_id = total_loc = 0.0
    n_samples = 0

    for imgs, prots, mask, gene_ids, loc_labels in loader:
        imgs       = imgs.to(device)        # [B, 1536]
        prots      = prots.to(device)       # [B, Nmax, 1280]
        mask       = mask.to(device)        # [B, Nmax]
        gene_ids   = gene_ids.to(device)    # [B]
        loc_labels = loc_labels.to(device)  # [B, n_locs]

        img_h, img_z = projector(imgs)               # [B, hidden], [B, 1280]
        prot_z       = pool(prots, mask=mask)        # [B, 1280]

        loss, l_clip, l_id, l_loc = combined_loss(
            img_h, img_z, prot_z,
            gene_ids, loc_labels,
            identity_head, loc_head,
            temperature=temperature,
            lambda_id=lambda_id,
            lambda_loc=lambda_loc,
        )

        if tqdm_obj is not None:
            tqdm_obj.set_postfix(loss=f"{loss.item():.4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        B = imgs.size(0)
        total_loss += loss.item()  * B
        total_clip += l_clip.item() * B
        total_id   += l_id.item()  * B
        total_loc  += l_loc.item() * B
        n_samples  += B

    n = max(n_samples, 1)
    return total_loss/n, total_clip/n, total_id/n, total_loc/n


@torch.no_grad()
def evaluate(
    projector, pool, identity_head, loc_head,
    loader, device,
    temperature=0.07, lambda_id=1.0, lambda_loc=1.0,
):
    projector.eval()
    pool.eval()
    identity_head.eval()
    loc_head.eval()

    total_loss = total_clip = total_id = total_loc = 0.0
    n_samples = 0

    for imgs, prots, mask, gene_ids, loc_labels in tqdm(loader, leave=False):
        imgs       = imgs.to(device)
        prots      = prots.to(device)
        mask       = mask.to(device)
        gene_ids   = gene_ids.to(device)
        loc_labels = loc_labels.to(device)

        img_h, img_z = projector(imgs)
        prot_z       = pool(prots, mask=mask)

        loss, l_clip, l_id, l_loc = combined_loss(
            img_h, img_z, prot_z,
            gene_ids, loc_labels,
            identity_head, loc_head,
            temperature=temperature,
            lambda_id=lambda_id,
            lambda_loc=lambda_loc,
        )

        B = imgs.size(0)
        total_loss += loss.item()  * B
        total_clip += l_clip.item() * B
        total_id   += l_id.item()  * B
        total_loc  += l_loc.item() * B
        n_samples  += B

    n = max(n_samples, 1)
    return total_loss/n, total_clip/n, total_id/n, total_loc/n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to training config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    outdir = Path(config["output_dir"])
    os.makedirs(outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = EmbeddingPairDatasetV2(
        config["filedir"],
        min_gene_count=config.get("min_gene_count", 50),
    )
    print(f"Dataset size : {len(dataset)}")
    print(f"Gene classes : {len(dataset.gene_vocab)}")
    print(f"Loc classes  : {len(dataset.loc_vocab)}")

    n_genes = len(dataset.gene_vocab)
    n_locs  = len(dataset.loc_vocab)

    # Save vocab for downstream inference / evaluation
    vocab_path = outdir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(
            {"gene_vocab": dataset.gene_vocab, "loc_vocab": dataset.loc_vocab},
            f, indent=2,
        )
    print(f"Vocabulary saved to {vocab_path}")

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size   = total_size - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=0,
        collate_fn=collate_variable_proteins_v2,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_proteins_v2,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    projector     = ImageProjectorV2(
        in_dim=1536, out_dim=1280, hidden_dim=config["hidden_dim"]
    ).to(device)

    pool          = ProteinPool(dim=1280).to(device)

    identity_head = ProteinIdentityHead(
        in_dim=config["hidden_dim"], n_classes=n_genes
    ).to(device)

    loc_head      = LocalizationHead(
        in_dim=config["hidden_dim"], n_locs=n_locs
    ).to(device)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(projector.parameters())
        + list(pool.parameters())
        + list(identity_head.parameters())
        + list(loc_head.parameters()),
        lr=config["learning_rate"],
        weight_decay=1e-2,
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    ckpt_path   = outdir / "checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        start_epoch = load_checkpoint(
            ckpt_path, projector, pool, identity_head, loc_head,
            optimizer, map_location=device,
        )
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    temperature = config.get("temperature", 0.07)
    lambda_id   = config.get("lambda_id",   1.0)
    lambda_loc  = config.get("lambda_loc",  1.0)
    num_epochs  = config["num_epochs"]

    ct     = datetime.datetime.now()
    writer = SummaryWriter(outdir / f"logs/{str(ct)}")

    loss_log = {"train": [], "val": []}

    for epoch in range(start_epoch, num_epochs):
        loop = tqdm(train_loader, leave=False)
        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")

        tr_loss, tr_clip, tr_id, tr_loc = train_one_epoch(
            projector, pool, identity_head, loc_head,
            train_loader, optimizer, device,
            temperature=temperature, lambda_id=lambda_id, lambda_loc=lambda_loc,
            tqdm_obj=loop,
        )

        vl_loss, vl_clip, vl_id, vl_loc = evaluate(
            projector, pool, identity_head, loc_head,
            val_loader, device,
            temperature=temperature, lambda_id=lambda_id, lambda_loc=lambda_loc,
        )

        # TensorBoard
        writer.add_scalars("loss/total", {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("loss/clip",  {"train": tr_clip, "val": vl_clip}, epoch)
        writer.add_scalars("loss/id",    {"train": tr_id,   "val": vl_id},   epoch)
        writer.add_scalars("loss/loc",   {"train": tr_loc,  "val": vl_loc},  epoch)

        print(
            f"epoch={epoch+1:>4}  "
            f"train_loss={tr_loss:.4f} (clip={tr_clip:.4f} id={tr_id:.4f} loc={tr_loc:.4f})  "
            f"val_loss={vl_loss:.4f} (clip={vl_clip:.4f} id={vl_id:.4f} loc={vl_loc:.4f})"
        )

        loss_log["train"].append(
            {"total": tr_loss, "clip": tr_clip, "id": tr_id, "loc": tr_loc}
        )
        loss_log["val"].append(
            {"total": vl_loss, "clip": vl_clip, "id": vl_id, "loc": vl_loc}
        )

        if epoch % 5 == 0:
            save_checkpoint(
                outdir / f"ckpt_epoch_{epoch}.pt",
                projector, pool, identity_head, loc_head,
                optimizer, epoch + 1, n_genes, n_locs,
            )

        save_checkpoint(
            ckpt_path,
            projector, pool, identity_head, loc_head,
            optimizer, epoch + 1, n_genes, n_locs,
        )

    # ------------------------------------------------------------------
    # Persist loss log
    # ------------------------------------------------------------------
    loss_filepath = outdir / "train_val_loss.json"
    data = {}
    if loss_filepath.exists():
        with open(loss_filepath, "r") as f:
            data = json.load(f)
    data[str(ct)] = loss_log
    with open(loss_filepath, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
