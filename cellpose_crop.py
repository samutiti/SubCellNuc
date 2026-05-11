#!/usr/bin/env python3
"""
Segment cells in microscope FOV images with Cellpose and save per-cell crops.

Two input modes
---------------
Single-file mode  (default)
    One image per FOV, any channel layout.

    python cellpose_crop.py /data/fovs/
    python cellpose_crop.py FOV001.tif --seg-channel 1

Split-channel mode  (--multichannel)
    Each channel is a separate file: {NAME}_{color}.jpg
    Channels are stacked in --channel-order order.

    python cellpose_crop.py /data/fovs/ --multichannel --seg-color blue

Output format
-------------
By default, crops in --multichannel mode are saved as separate per-channel
files:  {NAME}{suffix}{cell_id}_{color}{out_ext}
e.g.    FOV001_cell0001_blue.tif

Use --no-split-output to save a single stacked TIFF instead.
In single-file mode the default is stacked; use --split-output to split.

    --out-ext   extension for split-output files: .tif (default) | .png | .jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

DEFAULT_COLORS = ["blue", "green", "red", "yellow"]
SUPPORTED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Return image as numpy array. TIFFs via tifffile; others via PIL."""
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(str(path))
    from PIL import Image
    return np.array(Image.open(path))


def canonical_hwc(img: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) → (H, W, C); leave (H, W) and (H, W, C) unchanged."""
    if img.ndim == 3 and img.shape[0] <= 8 and img.shape[0] < img.shape[1]:
        return np.moveaxis(img, 0, -1)
    return img


def save_array(arr: np.ndarray, path: Path) -> None:
    """Save a 2-D array; format inferred from path extension."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        tifffile.imwrite(str(path), arr, compression="zlib")
        return
    from PIL import Image
    if arr.dtype != np.uint8:
        max_val = (
            np.iinfo(arr.dtype).max
            if np.issubdtype(arr.dtype, np.integer)
            else float(arr.max()) or 1.0
        )
        arr = (arr.astype(np.float32) / max_val * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))


# ---------------------------------------------------------------------------
# Split-channel helpers
# ---------------------------------------------------------------------------

def group_by_color(paths: list[Path], colors: list[str]) -> dict[str, dict[str, Path]]:
    """
    Group files whose stems end with _{color} by their base name.
    Returns {base_name: {color: path}}.
    """
    color_set = set(colors)
    groups: dict[str, dict[str, Path]] = {}
    for path in paths:
        for color in color_set:
            if path.stem.endswith(f"_{color}"):
                base = path.stem[: -len(f"_{color}")]
                groups.setdefault(base, {})[color] = path
                break
    return groups


def load_multichannel(
    color_paths: dict[str, Path], channel_order: list[str]
) -> tuple[np.ndarray, list[str]]:
    """
    Stack available per-color images into (H, W, C).
    Returns (stacked_array, actual_color_order).
    """
    channels: list[np.ndarray] = []
    actual_order: list[str] = []

    for color in channel_order:
        if color not in color_paths:
            continue
        img = canonical_hwc(load_image(color_paths[color]))
        if img.ndim == 3:        # JPEG decoded as RGB — take first plane
            img = img[:, :, 0]
        channels.append(img)
        actual_order.append(color)

    if not channels:
        raise ValueError("No channels found to stack.")

    return np.stack(channels, axis=-1), actual_order


# ---------------------------------------------------------------------------
# Cellpose
# ---------------------------------------------------------------------------

def build_cellpose(model_type: str, use_gpu: bool) -> tuple[object, dict]:
    """
    Construct a Cellpose model once. Returns (model, eval_kwargs).

    Cellpose >= 4 only ships Cellpose-SAM; `model_type` and `channels` are
    ignored. Cellpose 3.x keeps the cyto/nuclei zoo and uses channels=[0,0]
    for single-plane inputs.
    """
    from cellpose import models
    try:
        from cellpose import version as cp_version
    except ImportError:
        cp_version = "0.0.0"

    major = int(cp_version.split(".")[0]) if cp_version[:1].isdigit() else 0

    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print(
                    "Warning: --gpu set but torch.cuda.is_available() is False; "
                    "Cellpose will run on CPU (expect hours/FOV).",
                    file=sys.stderr,
                )
        except ImportError:
            pass

    if major >= 4:
        print(f"Cellpose {cp_version} detected: using Cellpose-SAM "
              f"(--model {model_type!r} ignored).", file=sys.stderr)
        return models.CellposeModel(gpu=use_gpu), {}

    print(f"Cellpose {cp_version} detected: using model_type={model_type!r}.",
          file=sys.stderr)
    return models.Cellpose(model_type=model_type, gpu=use_gpu), {"channels": [0, 0]}


def run_cellpose(model, eval_kwargs: dict, img_2d: np.ndarray, diameter) -> np.ndarray:
    """Run Cellpose on a 2-D (H, W) image; return integer label mask."""
    out = model.eval(
        img_2d,
        diameter=diameter,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        **eval_kwargs,
    )
    return out[0].astype(np.int32)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def pad_crop(img: np.ndarray, cy: int, cx: int, crop_size: int) -> np.ndarray:
    """Return a crop_size × crop_size patch centred on (cy, cx), zero-padded at borders."""
    half = crop_size // 2
    h, w = img.shape[:2]

    src_y0, src_y1 = cy - half, cy + half
    src_x0, src_x1 = cx - half, cx + half

    dst_y0 = max(0, -src_y0)
    dst_x0 = max(0, -src_x0)

    cy0, cy1 = max(0, src_y0), min(h, src_y1)
    cx0, cx1 = max(0, src_x0), min(w, src_x1)
    ph, pw = cy1 - cy0, cx1 - cx0

    if img.ndim == 2:
        out = np.zeros((crop_size, crop_size), dtype=img.dtype)
        out[dst_y0:dst_y0 + ph, dst_x0:dst_x0 + pw] = img[cy0:cy1, cx0:cx1]
    else:
        out = np.zeros((crop_size, crop_size, img.shape[2]), dtype=img.dtype)
        out[dst_y0:dst_y0 + ph, dst_x0:dst_x0 + pw, :] = img[cy0:cy1, cx0:cx1, :]

    return out


# ---------------------------------------------------------------------------
# Core segment-and-crop  (shared by both modes)
# ---------------------------------------------------------------------------

def segment_and_crop(
    img: np.ndarray,
    base_name: str,
    seg_channel: int,
    out_dir: Path,
    args,
    color_names: list[str] | None = None,
) -> int:
    """
    Segment img with Cellpose and write one crop per cell.

    color_names controls output layout:
      None          → single stacked TIFF per cell
      list[str]     → one file per channel named {cell_stem}_{color}{out_ext}
    """
    if img.ndim == 3:
        n_ch = img.shape[2]
        if seg_channel >= n_ch:
            raise ValueError(
                f"{base_name}: seg channel {seg_channel} out of range "
                f"(image has {n_ch} channels)"
            )
        seg_img = img[:, :, seg_channel]
    else:
        seg_img = img

    masks = run_cellpose(args.cellpose_model, args.cellpose_eval_kwargs, seg_img, args.diameter)

    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0]

    if len(cell_ids) == 0:
        tqdm.write(f"  [warn] {base_name}: no cells detected", file=sys.stderr)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    for cell_id in cell_ids:
        ys, xs = np.where(masks == cell_id)
        cy = int(round(ys.mean()))
        cx = int(round(xs.mean()))
        crop = pad_crop(img, cy, cx, args.crop_size)
        cell_stem = f"{base_name}{args.suffix}{cell_id:04d}"

        if color_names is not None and crop.ndim == 3:
            for i, color in enumerate(color_names):
                save_array(crop[:, :, i], out_dir / f"{cell_stem}_{color}{args.out_ext}")
        else:
            tifffile.imwrite(
                str(out_dir / f"{cell_stem}.tif"), crop, compression="zlib"
            )

    return len(cell_ids)


# ---------------------------------------------------------------------------
# Per-FOV entry points
# ---------------------------------------------------------------------------

def process_single(image_path: Path, split_output: bool, channel_order: list[str], args) -> int:
    img = canonical_hwc(load_image(image_path))
    out_dir = Path(args.output) if args.output else image_path.parent

    color_names = None
    if split_output and img.ndim == 3:
        n_ch = img.shape[2]
        color_names = (
            channel_order if len(channel_order) == n_ch
            else [f"ch{i}" for i in range(n_ch)]
        )

    return segment_and_crop(img, image_path.stem, args.seg_channel, out_dir, args, color_names)


def process_multichannel(
    base_name: str,
    color_paths: dict[str, Path],
    channel_order: list[str],
    split_output: bool,
    args,
) -> int:
    img, actual_order = load_multichannel(color_paths, channel_order)

    if args.seg_color:
        if args.seg_color not in actual_order:
            raise ValueError(
                f"{base_name}: --seg-color '{args.seg_color}' not available "
                f"(found: {actual_order})"
            )
        seg_channel = actual_order.index(args.seg_color)
    else:
        seg_channel = args.seg_channel

    color_names = actual_order if split_output else None
    ref_path = next(iter(color_paths.values()))
    out_dir = Path(args.output) if args.output else ref_path.parent
    return segment_and_crop(img, base_name, seg_channel, out_dir, args, color_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_paths(inputs: list[str], extensions: set[str]) -> list[Path]:
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in extensions:
                paths.extend(sorted(p.glob(f"*{ext}")))
                paths.extend(sorted(p.glob(f"*{ext.upper()}")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Warning: {inp} not found — skipping.", file=sys.stderr)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Segment cells with Cellpose and save per-cell crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="+",
                        help="Input image file(s) or directory of FOVs")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: same dir as input image)")
    parser.add_argument("--model", default="cyto2",
                        choices=["cyto", "cyto2", "cyto3", "nuclei"])
    parser.add_argument("--diameter", type=float, default=None,
                        help="Expected cell diameter in pixels (None = auto)")
    parser.add_argument("--suffix", default="_cell",
                        help="Inserted between image stem and cell index in output filename")
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--ext", default=".tif,.tiff,.png,.jpg,.jpeg",
                        help="Extensions scanned when input is a directory")

    # Single-file mode
    parser.add_argument("--seg-channel", type=int, default=0,
                        help="0-indexed channel for segmentation (single-file mode)")

    # Split-channel input mode
    parser.add_argument("--multichannel", action="store_true",
                        help="Group {NAME}_{color}.jpg files into per-FOV stacks")
    parser.add_argument("--channel-order", default=",".join(DEFAULT_COLORS),
                        help="Comma-separated colour names defining stack order")
    parser.add_argument("--seg-color", default=None,
                        help="Colour to segment on in --multichannel mode (e.g. blue)")

    # Output layout
    parser.add_argument("--split-output", dest="split_output", action="store_true",
                        help="Save each channel as a separate file (default in --multichannel)")
    parser.add_argument("--no-split-output", dest="split_output", action="store_false",
                        help="Save all channels as a single stacked TIFF")
    parser.add_argument("--out-ext", default=".tif",
                        choices=[".tif", ".tiff", ".png", ".jpg"],
                        help="File extension for split-output channel files")
    parser.set_defaults(split_output=None)   # None → resolved per-mode below

    args = parser.parse_args()

    extensions = {e.strip().lower() for e in args.ext.split(",")}
    channel_order = [c.strip().lower() for c in args.channel_order.split(",")]

    # Default: split in multichannel mode, stacked in single-file mode
    split_output = args.split_output if args.split_output is not None else args.multichannel

    all_paths = collect_paths(args.input, extensions)
    if not all_paths:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    args.cellpose_model, args.cellpose_eval_kwargs = build_cellpose(args.model, args.gpu)

    total_cells = 0

    if args.multichannel:
        groups = group_by_color(all_paths, channel_order)
        if not groups:
            print(
                "No files matched the {NAME}_{color}.ext pattern. "
                "Check --channel-order and --ext.",
                file=sys.stderr,
            )
            sys.exit(1)

        out_label = f"split ({args.out_ext}/channel)" if split_output else "stacked TIFF"
        print(
            f"Found {len(groups)} FOV(s) across {len(all_paths)} channel file(s). "
            f"Output: {out_label}. Running Cellpose ({args.model})..."
        )

        for base_name in tqdm(sorted(groups), unit="FOV"):
            color_paths = groups[base_name]
            try:
                n = process_multichannel(base_name, color_paths, channel_order, split_output, args)
                tqdm.write(f"  {base_name}: {n} cell(s)  [channels: {list(color_paths)}]")
                total_cells += n
            except Exception as exc:
                tqdm.write(f"  [error] {base_name}: {exc}", file=sys.stderr)

    else:
        print(f"Found {len(all_paths)} image(s). Running Cellpose ({args.model})...")
        for path in tqdm(all_paths, unit="FOV"):
            try:
                n = process_single(path, split_output, channel_order, args)
                tqdm.write(f"  {path.name}: {n} cell(s)")
                total_cells += n
            except Exception as exc:
                tqdm.write(f"  [error] {path.name}: {exc}", file=sys.stderr)

    print(f"\nDone. {total_cells} crops saved.")


if __name__ == "__main__":
    main()
