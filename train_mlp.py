# train_mpl.py
import os
import torch
from torch.utils.data import DataLoader

from models import ImageProjector, ProteinPool, clip_loss
from data import EmbeddingPairDataset, collate_variable_proteins


def save_checkpoint(path, projector, pool, optimizer, epoch):
    ckpt = {
        "epoch": epoch,
        "projector_state": projector.state_dict(),
        "pool_state": pool.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, projector, pool, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    projector.load_state_dict(ckpt["projector_state"])
    pool.load_state_dict(ckpt["pool_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0)


def train_one_epoch(projector, pool, loader, optimizer, device, temperature=0.07):
    projector.train()
    pool.train()

    total_loss = 0.0
    for imgs, prots, mask in loader:
        imgs = imgs.to(device)       # [B,1536]
        prots = prots.to(device)     # [B,N,1280]
        mask = mask.to(device)       # [B,N]

        img_z = projector(imgs)                  # [B,1280]
        prot_z = pool(prots, mask=mask)          # [B,1280]
        loss = clip_loss(img_z, prot_z, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(projector, pool, loader, device, temperature=0.07):
    projector.eval()
    pool.eval()

    total_loss = 0.0
    for imgs, prots, mask in loader:
        imgs = imgs.to(device)
        prots = prots.to(device)
        mask = mask.to(device)

        img_z = projector(imgs)
        prot_z = pool(prots, mask=mask)
        loss = clip_loss(img_z, prot_z, temperature=temperature)
        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Replace this with your real loaded tensors ----
    # items is a list of (img_emb[1536], prot_embs[N,1280])
    # prot_embs can have variable N across items.
    items = []
    for _ in range(200):
        img = torch.randn(1536)
        N = torch.randint(low=1, high=6, size=()).item()
        prots = torch.randn(N, 1280)
        items.append((img, prots))

    train_items = items[:160]
    val_items = items[160:]

    train_ds = EmbeddingPairDataset(train_items)
    val_ds = EmbeddingPairDataset(val_items)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_variable_proteins,
        drop_last=True,  # CLIP loss expects square logits; drop_last avoids tiny last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_proteins,
        drop_last=False,
    )

    projector = ImageProjector(in_dim=1536, out_dim=1280).to(device)
    pool = ProteinPool(dim=1280).to(device)

    # Optimizer includes parameters from BOTH modules
    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(pool.parameters()),
        lr=1e-4,
        weight_decay=1e-2,
    )

    ckpt_path = "checkpoint.pt"
    start_epoch = 0
    if os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(ckpt_path, projector, pool, optimizer, map_location=device)

    for epoch in range(start_epoch, start_epoch + 10):
        train_loss = train_one_epoch(projector, pool, train_loader, optimizer, device)
        val_loss = evaluate(projector, pool, val_loader, device)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if epoch % 5 == 0:
            save_checkpoint(f'ckpt_epoch_{epoch}.pt', projector, pool, optimizer, epoch + 1)
        save_checkpoint('checkpoint.pt', projector, pool, optimizer, epoch + 1)

if __name__ == "__main__":
    main()