"""Filename-based dataset for single-cell crop training.

Files in the data directory follow the convention:
    {PROTEIN}_{PLATE}_{WELL}_{SITE}_cell{N}_{kind}.png

where `kind` is one of:
    crop_blue        crop_green        crop_red        crop_yellow
    crop_masked_blue crop_masked_green crop_masked_red crop_masked_yellow
    cellmask         nucleimask

Channel mapping (HPA convention):
    blue   -> nucleus (DAPI)
    green  -> protein of interest (label)
    red    -> microtubules (MT)
    yellow -> ER (unused here)

Training regimens are parameterised by two switches:
    mask_mode in {"none", "cell", "nuclei"}   -> apply mask on the raw crop
    use_mt    in {False, True}                -> include the red (MT) channel

For each regimen the dataset filters to cells that have *all* required files;
incomplete cells (e.g. missing MT, missing cellmask) are silently skipped.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Channel name -> role
COLOR_TO_ROLE = {
    "blue":   "nucleus",
    "green":  "protein",
    "red":    "mt",
    "yellow": "er",
}
ROLE_TO_COLOR = {v: k for k, v in COLOR_TO_ROLE.items()}

# Output channel order produced by ProteinCropDataset.
# - 2-channel: [protein, nucleus]
# - 3-channel: [protein, mt, nucleus]
CHANNEL_ORDER_2CH = ["protein", "nucleus"]
CHANNEL_ORDER_3CH = ["protein", "mt", "nucleus"]

VALID_MASK_MODES = ("none", "cell", "nuclei")

FILENAME_RE = re.compile(
    r"^(?P<protein>[A-Za-z0-9]+)"
    r"_(?P<plate>\d+)"
    r"_(?P<well>[A-Z]+\d+)"
    r"_(?P<site>\d+)"
    r"_cell(?P<cell>\d+)"
    r"_(?P<kind>.+)$"
)


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def parse_filename(stem: str) -> Optional[Tuple[str, str, str]]:
    """Return (protein, cell_key, kind) or None if the stem does not parse."""
    m = FILENAME_RE.match(stem)
    if m is None:
        return None
    d = m.groupdict()
    cell_key = f"{d['protein']}_{d['plate']}_{d['well']}_{d['site']}_cell{d['cell']}"
    return d["protein"], cell_key, d["kind"]


def scan_directory(data_dir: str | Path, extensions: Sequence[str] = (".png", ".tif", ".tiff")) -> Dict[str, Dict]:
    """Walk `data_dir` and group files by (protein, plate, well, site, cell).

    Returns: {cell_key: {"protein": str, "files": {kind: Path, ...}}}
    """
    data_dir = Path(data_dir)
    cells: Dict[str, Dict] = defaultdict(lambda: {"protein": None, "files": {}})

    ext_set = {e.lower() for e in extensions}
    for path in data_dir.iterdir():
        if path.suffix.lower() not in ext_set or not path.is_file():
            continue
        parsed = parse_filename(path.stem)
        if parsed is None:
            continue
        protein, cell_key, kind = parsed
        cells[cell_key]["protein"] = protein
        cells[cell_key]["files"][kind] = path

    return dict(cells)


def required_kinds(mask_mode: str, use_mt: bool) -> List[str]:
    """Files each cell must have for the given regimen."""
    if mask_mode not in VALID_MASK_MODES:
        raise ValueError(f"mask_mode must be one of {VALID_MASK_MODES}, got {mask_mode!r}")
    needed = [f"crop_{ROLE_TO_COLOR['protein']}", f"crop_{ROLE_TO_COLOR['nucleus']}"]
    if use_mt:
        needed.append(f"crop_{ROLE_TO_COLOR['mt']}")
    if mask_mode == "cell":
        needed.append("cellmask")
    elif mask_mode == "nuclei":
        needed.append("nucleimask")
    return needed


def filter_complete(
    cells: Dict[str, Dict],
    mask_mode: str,
    use_mt: bool,
) -> List[Dict]:
    """Keep only cells with every required file. Returns a list of records."""
    needed = required_kinds(mask_mode, use_mt)
    records: List[Dict] = []
    for cell_key, info in cells.items():
        files = info["files"]
        if all(k in files for k in needed):
            records.append({
                "cell_key": cell_key,
                "protein": info["protein"],
                "files": {k: files[k] for k in files},  # keep all, we may use masks
            })
    return records


def build_label_encoder_from_records(records: Sequence[Dict]) -> Dict[str, int]:
    proteins = sorted({r["protein"] for r in records})
    return {p: i for i, p in enumerate(proteins)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _read_grayscale(path: Path) -> np.ndarray:
    """Read PNG/TIFF as 2-D float32, taking max projection if multi-channel."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if img.ndim == 3:
        img = img.max(axis=2)
    return img.astype(np.float32)


def _min_max_norm_chw(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        ch = x[c]
        lo, hi = float(ch.min()), float(ch.max())
        out[c] = (ch - lo) / (hi - lo + 1e-8)
    return out


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[-2:] == (size, size):
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class ProteinCropDataset(Dataset):
    """Per-cell crop dataset driven by directory scanning.

    Args:
        records: list from `filter_complete(...)`.
        label_encoder: {protein_name: idx} mapping.
        mask_mode: "none" | "cell" | "nuclei". Applied to every channel.
        use_mt: include the MT (red) channel.
        image_size: square size to resize crops to.
        augment: random flips + 90-degree rotations.
    """

    def __init__(
        self,
        records: Sequence[Dict],
        label_encoder: Dict[str, int],
        mask_mode: str = "none",
        use_mt: bool = False,
        image_size: int = 128,
        augment: bool = False,
    ):
        if mask_mode not in VALID_MASK_MODES:
            raise ValueError(f"mask_mode must be one of {VALID_MASK_MODES}")
        self.records = [r for r in records if r["protein"] in label_encoder]
        self.label_encoder = label_encoder
        self.mask_mode = mask_mode
        self.use_mt = use_mt
        self.image_size = image_size
        self.channel_order = CHANNEL_ORDER_3CH if use_mt else CHANNEL_ORDER_2CH

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

    @property
    def in_channels(self) -> int:
        return len(self.channel_order)

    def _load_channel(self, files: Dict[str, Path], role: str) -> np.ndarray:
        color = ROLE_TO_COLOR[role]
        img = _read_grayscale(files[f"crop_{color}"])
        return _resize(img, self.image_size)

    def _load_mask(self, files: Dict[str, Path]) -> Optional[np.ndarray]:
        if self.mask_mode == "none":
            return None
        key = "cellmask" if self.mask_mode == "cell" else "nucleimask"
        mask = _read_grayscale(files[key])
        mask = _resize(mask, self.image_size)
        # Treat any nonzero pixel as foreground.
        return (mask > 0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        files = rec["files"]

        channels = [self._load_channel(files, role) for role in self.channel_order]
        img = np.stack(channels, axis=0)  # (C, H, W)

        mask = self._load_mask(files)
        if mask is not None:
            img = img * mask[None, :, :]

        img = _min_max_norm_chw(img)
        img_t = torch.from_numpy(img)

        if self.aug is not None:
            img_t = self.aug(img_t)

        label_idx = self.label_encoder[rec["protein"]]
        return {
            "image": img_t,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "cell_key": rec["cell_key"],
        }


# ---------------------------------------------------------------------------
# Train/val split + dataloaders
# ---------------------------------------------------------------------------

def stratified_split(
    records: Sequence[Dict],
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """Per-protein hold-out split. Cells from the same protein appear in both
    splits; cell_keys (FOV+cell) are kept disjoint."""
    rng = np.random.default_rng(seed)
    by_protein: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_protein[r["protein"]].append(r)

    train, val = [], []
    for protein, items in by_protein.items():
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(items) * val_frac))) if len(items) > 1 else 0
        val_idx = set(idx[:n_val].tolist())
        for i, item in enumerate(items):
            (val if i in val_idx else train).append(item)
    return train, val


def build_dataloaders_from_dir(
    data_dir: str,
    mask_mode: str = "none",
    use_mt: bool = False,
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
    val_frac: float = 0.1,
    seed: int = 0,
    label_encoder: Optional[Dict[str, int]] = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int], Dict[str, int]]:
    """Scan `data_dir`, filter to cells complete for the regimen, split, build loaders.

    Returns:
        train_loader, val_loader, label_encoder, stats
        stats: counts and per-protein totals for logging.
    """
    cells = scan_directory(data_dir)
    records = filter_complete(cells, mask_mode=mask_mode, use_mt=use_mt)
    if not records:
        raise RuntimeError(
            f"No complete cells found in {data_dir} for mask_mode={mask_mode}, "
            f"use_mt={use_mt}. Required files: {required_kinds(mask_mode, use_mt)}"
        )

    if label_encoder is None:
        label_encoder = build_label_encoder_from_records(records)
    records = [r for r in records if r["protein"] in label_encoder]

    train_recs, val_recs = stratified_split(records, val_frac=val_frac, seed=seed)

    train_ds = ProteinCropDataset(
        train_recs, label_encoder, mask_mode=mask_mode, use_mt=use_mt,
        image_size=image_size, augment=True,
    )
    val_ds = ProteinCropDataset(
        val_recs, label_encoder, mask_mode=mask_mode, use_mt=use_mt,
        image_size=image_size, augment=False,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, **loader_kwargs,
    )

    stats = {
        "total_scanned": len(cells),
        "total_complete": len(records),
        "train": len(train_recs),
        "val": len(val_recs),
        "num_proteins": len(label_encoder),
    }
    return train_loader, val_loader, label_encoder, stats
