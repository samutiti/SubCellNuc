# train_mpl.py
import os, argparse, json, datetime
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mlp_models import ImageProjector, ProteinPool, clip_loss
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


def train_one_epoch(projector, pool, loader, optimizer, device, temperature=0.07, tqdm_obj=None):
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
        if tqdm_obj is not None: tqdm_obj.set_postfix(loss=loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0) # NOTE: I don't know what the point is here?

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(projector, pool, loader, device, temperature=0.07):
    projector.eval()
    pool.eval()

    total_loss = 0.0
    for imgs, prots, mask in tqdm(loader):
        imgs = imgs.to(device)
        prots = prots.to(device)
        mask = mask.to(device)

        img_z = projector(imgs)
        prot_z = pool(prots, mask=mask)
        loss = clip_loss(img_z, prot_z, temperature=temperature)
        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='provide the filepath to the training config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # configure data storage location
    outdir = Path(config['output_dir'])
    os.makedirs(outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EmbeddingPairDataset(config['filedir'])
    print('whole dataset: ', len(dataset))

    # compute test and train split
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    print('train_size: ', train_size)
    val_size = total_size - train_size

    train_ds, val_ds = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        num_workers=0,
        collate_fn=collate_variable_proteins,
        drop_last=True,  # CLIP loss expects square logits; drop_last avoids tiny last batch
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_proteins,
        drop_last=True,
    )

    projector = ImageProjector(in_dim=1536, out_dim=1280, hidden_dim=config['hidden_dim']).to(device)
    pool = ProteinPool(dim=1280).to(device)

    # Optimizer includes parameters from BOTH modules
    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(pool.parameters()),
        lr=config['learning_rate'],
        weight_decay=1e-2,
    )

    ckpt_path = "checkpoint.pt"
    start_epoch = 0
    if os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(ckpt_path, projector, pool, optimizer, map_location=device)

    epoch_train_losses = []
    epoch_val_losses = []

    ct = datetime.datetime.now()
    writer = SummaryWriter(outdir / f"logs/{str(ct)}")
    num_epochs = config['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        loop = tqdm(train_loader, leave=False)
        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(projector, pool, train_loader, optimizer, device, config.get("temperature", 0.07), loop)
        val_loss = evaluate(projector, pool, val_loader, device)

        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)

        writer.add_scalar("train", train_loss, epoch)
        writer.add_scalar("val", val_loss, epoch)

        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if epoch % 5 == 0:
            save_checkpoint(outdir / f'ckpt_epoch_{epoch}.pt', projector, pool, optimizer, epoch + 1)

        save_checkpoint(outdir / 'checkpoint.pt', projector, pool, optimizer, epoch + 1)
    
    # Save loss data
    loss_filepath = str(outdir / 'train_val_loss.json')
    ct = datetime.datetime.now()
    data = {}
    # if data log exists, load it
    if os.path.exists(loss_filepath):
        with open(loss_filepath, 'r') as f:
            data = json.load(f)
    # add timestamp + losses to data
    data[str(ct)] = {
        'train': epoch_train_losses,
        'val': epoch_val_losses
    }
    # write back to json
    with open(loss_filepath, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()