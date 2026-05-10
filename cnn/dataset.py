"""Dataset for protein image embedding training.

Loads multi-channel protein images (2ch: protein+nucleus, or 3ch: protein+MT+nucleus)
and a protein-identity label for joint reconstruction + classification training.

================================================================================
EDIT POINTS (search for `# EDIT:` to find every spot you must touch)
================================================================================
1. CSV_COLUMNS — column names in your CSV
2. PROTEIN_LABEL_COLUMN — column holding protein identity
3. ProteinDataset.__init__ — point to your CSV, set channel mode, image size
4. ProteinDataset._load_image — replace with your image loader if not TIFF/PNG
5. build_label_encoder — confirm your label list / mapping
================================================================================
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image_utils


# EDIT: rename these to match your CSV column headers ---------------------------
CSV_COLUMNS = {
    "protein": "protein_image",   # path to protein channel image
    "nucleus": "nucleus_image",   # path to nucleus (DAPI) image
    "mt": "mt_image",             # path to microtubule image (only used if in_channels=3)
}
PROTEIN_LABEL_COLUMN = "protein_id"   # column holding the protein identity (string or int)
# --------------------------------------------------------------------------------


def min_max_norm(x: np.ndarray) -> np.ndarray:
    """Per-channel min-max normalization to [0, 1]."""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        ch = x[c]
        lo, hi = ch.min(), ch.max()
        out[c] = (ch - lo) / (hi - lo + 1e-8)
    return out


def build_label_encoder(csv_path: str) -> Dict[str, int]:
    """Build a {protein_name: idx} mapping from the unique values in the CSV.

    EDIT: if you have a fixed canonical label list (e.g., from a proteins.txt file),
    load it here instead so train / val / test all use the same mapping.
    """
    df = pd.read_csv(csv_path)
    proteins = sorted(df[PROTEIN_LABEL_COLUMN].astype(str).unique().tolist())
    return {name: i for i, name in enumerate(proteins)}


class ProteinDataset(Dataset):
    """Returns (image_tensor, label_idx) for joint reconstruction + classification."""

    def __init__(
        self,
        csv_path: str,
        label_encoder: Dict[str, int],
        in_channels: int = 2,        # EDIT: 2 (protein+nucleus) or 3 (protein+MT+nucleus)
        image_size: int = 128,
        augment: bool = False,
    ):
        assert in_channels in (2, 3), "in_channels must be 2 or 3"
        self.in_channels = in_channels
        self.image_size = image_size
        self.label_encoder = label_encoder

        # EDIT: confirm CSV path / filtering logic for your splits
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.lstrip("#")
        # Drop rows whose protein label isn't in the encoder (e.g., held-out proteins)
        self.df = self.df[self.df[PROTEIN_LABEL_COLUMN].astype(str).isin(label_encoder)].reset_index(drop=True)

        self.records: List[Dict] = self.df.to_dict("records")

        # Augmentations: simple flips/rotations work well for microscopy
        if augment:
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            ])
        else:
            self.aug = None

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> np.ndarray:
        """Load a single grayscale channel as float32 HxW.

        EDIT: replace with your loader if you don't use the project's image_utils
        (e.g., for OME-TIFF, multi-page TIFFs, or HPA URLs).
        """
        return image_utils.read_grayscale_image(path).astype(np.float32)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize HxW to (image_size, image_size) using torch interpolation."""
        if img.shape[-2:] == (self.image_size, self.image_size):
            return img
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return t.squeeze(0).squeeze(0).numpy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        # Channel order is fixed: [protein, nucleus] or [protein, MT, nucleus]
        # (protein first so the model gets a consistent "primary" channel)
        channels = []
        channels.append(self._resize(self._load_image(rec[CSV_COLUMNS["protein"]])))
        if self.in_channels == 3:
            channels.append(self._resize(self._load_image(rec[CSV_COLUMNS["mt"]])))
        channels.append(self._resize(self._load_image(rec[CSV_COLUMNS["nucleus"]])))

        img = np.stack(channels, axis=0)            # (C, H, W)
        img = min_max_norm(img)                      # [0, 1]
        img_t = torch.from_numpy(img)

        if self.aug is not None:
            img_t = self.aug(img_t)

        label_idx = self.label_encoder[str(rec[PROTEIN_LABEL_COLUMN])]
        return {
            "image": img_t,                          # (C, H, W) float32 in [0, 1]
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


def build_dataloaders(
    train_csv: str,
    val_csv: Optional[str],
    in_channels: int = 2,
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    label_encoder: Optional[Dict[str, int]] = None,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], Dict[str, int]]:
    """Convenience builder. Pass `label_encoder=None` to derive it from train_csv."""
    if label_encoder is None:
        label_encoder = build_label_encoder(train_csv)

    train_ds = ProteinDataset(train_csv, label_encoder, in_channels, image_size, augment=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if val_csv is not None:
        val_ds = ProteinDataset(val_csv, label_encoder, in_channels, image_size, augment=False)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader, label_encoder
