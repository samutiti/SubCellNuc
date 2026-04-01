import anndata
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import SubcellDataset, subcell_collate
from mlp_models import ImageProjector


########## FIT PARAMS HERE ################
config_filepath = "/scratch/users/samutiti/U54/SubCellNuc/configs/train_v04.yml"
model_filepath = "/scratch/users/samutiti/U54/SubCellNuc/training_V04/checkpoint.pt"
data_filepath = "/scratch/users/samutiti/U54/embeddings/all_harmonized_features_microscope_vit.pth"
outpath = "/scratch/users/samutiti/U54/SubCellNuc/training_V04/inference.h5ad"
###########################################


def run(model, dataloader: DataLoader, device: str):
    model.eval()

    all_embeds = []
    data_lists = {}

    with torch.inference_mode():
        for embeds, info_dicts in tqdm(dataloader, desc="Inference"):
            
            # Move to device
            embeds = embeds.to(device, non_blocking=True)

            # Forward pass
            out = model(embeds)

            # Move to CPU numpy
            out = out.cpu().numpy()
            all_embeds.append(out)

            # Collect metadata
            for key, values in info_dicts.items():
                if key not in data_lists:
                    data_lists[key] = []
                data_lists[key].extend(values)

    # Stack once (FAST)
    mlp_embed = np.concatenate(all_embeds, axis=0)

    # Build AnnData
    obs_df = pd.DataFrame(data_lists)
    adata = anndata.AnnData(X=mlp_embed, obs=obs_df)

    return adata


# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SubcellDataset(data_filepath)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
    collate_fn=subcell_collate
)

# -------------------------
# Load model
# -------------------------
with open(config_filepath, "r") as f:
    config = yaml.safe_load(f)

model = ImageProjector(
    in_dim=1536,
    out_dim=1280,
    hidden_dim=config["hidden_dim"],
).to(device)

checkpoint = torch.load(model_filepath, map_location=device)
model.load_state_dict(checkpoint["projector_state"])


# -------------------------
# Run inference
# -------------------------
adata = run(model, dataloader, device)

# Save
adata.write(outpath)

print(f"Saved AnnData to: {outpath}")
print(f"Shape: {adata.shape}")