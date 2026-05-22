#!/usr/bin/env python3
"""
Segment cells in microscope FOV images with Cellpose and save per-cell crops.

Speedups vs. the original
-------------------------
* Prefetched I/O: a background ThreadPool loads the next FOV(s) while the GPU
  is busy running Cellpose. This is usually the biggest practical win, since
  the GPU otherwise sits idle during disk reads.
* Vectorised centroid extraction via scipy.ndimage.center_of_mass (replaces
  the per-cell np.where, which was O(N_cells * H * W)).
* Parallel crop writes through a worker thread pool.
* Optional FOV-batched Cellpose inference: model.eval() can take a list of
  FOVs and amortise the per-call Python overhead. Tune with --fov-batch.

Resume safety
-------------
Each finished FOV is appended to `_cellpose_crop_manifest.csv` in the output
directory. Re-running the same command skips FOVs already recorded with
matching (crop_size, suffix, seg_channel, seg_color, split_output, diameter,
model, source mtime). Use --no-resume to force re-processing everything.

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
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from tqdm import tqdm

DEFAULT_COLORS = ["blue", "green", "red", "yellow"]
SUPPORTED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

MANIFEST_NAME = "_cellpose_crop_manifest.csv"
MANIFEST_FIELDS = [
    "base_name", "n_cells", "crop_size", "suffix",
    "seg_channel", "seg_color", "split_output",
    "diameter", "model", "src_mtime", "timestamp",
]


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
        print('using gpu')
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


# ---------------------------------------------------------------------------
# Cropping & centroids
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


def fast_centroids(masks: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Vectorised centroid extraction.

    Returns a list of (label_id, cy, cx) for every present label in `masks`.
    Replaces the original per-label `np.where(masks == k)` loop which was
    O(N_labels * H * W); this is O(H * W) regardless of label count.
    """
    max_lbl = int(masks.max())
    if max_lbl == 0:
        return []
    from scipy import ndimage
    centers = ndimage.center_of_mass(
        masks > 0, masks, np.arange(1, max_lbl + 1)
    )
    out: list[tuple[int, int, int]] = []
    for label_id, c in enumerate(centers, start=1):
        cy, cx = c
        if np.isnan(cy):       # label_id absent (gap in labelling)
            continue
        out.append((label_id, int(round(cy)), int(round(cx))))
    return out


# ---------------------------------------------------------------------------
# Manifest (resume support)
# ---------------------------------------------------------------------------

def _resume_key(base_name: str, crop_size: int, suffix: str, seg_channel: int,
                seg_color: Optional[str], split_output: bool,
                diameter, model: str, src_mtime: float) -> tuple:
    return (
        base_name,
        int(crop_size),
        str(suffix),
        int(seg_channel),
        str(seg_color or ""),
        bool(split_output),
        "" if diameter is None else f"{float(diameter):.4f}",
        str(model or ""),
        round(float(src_mtime), 3),
    )


class Manifest:
    """Append-only CSV of completed FOVs. Survives process restarts."""

    def __init__(self, path: Path):
        self.path = path
        self._done: set[tuple] = set()
        self._lock = threading.Lock()
        self._fp = None
        self._writer = None

    def load(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                try:
                    key = _resume_key(
                        base_name=row["base_name"],
                        crop_size=int(row["crop_size"]),
                        suffix=row["suffix"],
                        seg_channel=int(row["seg_channel"]),
                        seg_color=row.get("seg_color") or "",
                        split_output=str(row["split_output"]).lower() == "true",
                        diameter=(float(row["diameter"]) if row.get("diameter") else None),
                        model=row.get("model") or "",
                        src_mtime=float(row["src_mtime"]),
                    )
                    self._done.add(key)
                except Exception:
                    # Bad/old row — ignore so a clean run can still proceed.
                    continue
        return len(self._done)

    def already_done(self, key: tuple) -> bool:
        return key in self._done

    def open(self) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fp = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=MANIFEST_FIELDS)
        if new_file:
            self._writer.writeheader()
            self._fp.flush()

    def append(self, **row) -> None:
        with self._lock:
            self._writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
            self._fp.flush()
            try:
                os.fsync(self._fp.fileno())
            except (OSError, AttributeError):
                pass

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


# ---------------------------------------------------------------------------
# Work items + prefetch loader
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    base_name: str
    single_path: Optional[Path]
    color_paths: Optional[dict[str, Path]]
    out_dir: Path
    src_mtime: float


@dataclass
class LoadedItem:
    item: WorkItem
    img: Optional[np.ndarray]
    color_order: Optional[list[str]]   # multichannel: actual color order
    seg_channel_idx: int
    error: Optional[Exception] = None


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


def collect_work_items(args, channel_order: list[str]) -> list[WorkItem]:
    extensions = {e.strip().lower() for e in args.ext.split(",")}
    all_paths = collect_paths(args.input, extensions)
    if not all_paths:
        return []

    items: list[WorkItem] = []
    if args.multichannel:
        groups = group_by_color(all_paths, channel_order)
        for base_name in sorted(groups):
            color_paths = groups[base_name]
            ref_path = next(iter(color_paths.values()))
            out_dir = Path(args.output) if args.output else ref_path.parent
            src_mtime = max(p.stat().st_mtime for p in color_paths.values())
            items.append(WorkItem(
                base_name=base_name,
                single_path=None,
                color_paths=color_paths,
                out_dir=out_dir,
                src_mtime=src_mtime,
            ))
    else:
        for path in all_paths:
            out_dir = Path(args.output) if args.output else path.parent
            items.append(WorkItem(
                base_name=path.stem,
                single_path=path,
                color_paths=None,
                out_dir=out_dir,
                src_mtime=path.stat().st_mtime,
            ))
        items.sort(key=lambda w: w.base_name)
    return items


def load_one(item: WorkItem, channel_order: list[str], seg_color: Optional[str],
             seg_channel: int) -> LoadedItem:
    """Load one FOV from disk. Errors are captured on the LoadedItem."""
    try:
        if item.color_paths is not None:
            img, actual_order = load_multichannel(item.color_paths, channel_order)
            if seg_color:
                if seg_color not in actual_order:
                    raise ValueError(
                        f"{item.base_name}: --seg-color '{seg_color}' not "
                        f"available (found: {actual_order})"
                    )
                idx = actual_order.index(seg_color)
            else:
                idx = seg_channel
            return LoadedItem(item, img, actual_order, idx)
        img = canonical_hwc(load_image(item.single_path))
        idx = seg_channel
        if img.ndim == 3:
            n_ch = img.shape[2]
            if seg_channel >= n_ch:
                raise ValueError(
                    f"{item.base_name}: seg channel {seg_channel} out of "
                    f"range (image has {n_ch} channels)"
                )
        return LoadedItem(item, img, None, idx)
    except Exception as exc:
        return LoadedItem(item, None, None, 0, error=exc)


def prefetch_loader(
    work_items: list[WorkItem],
    channel_order: list[str],
    seg_color: Optional[str],
    seg_channel: int,
    io_workers: int,
    prefetch: int,
):
    """Yield LoadedItem in input order, prefetching ahead with a thread pool."""
    if io_workers <= 0 or len(work_items) <= 1:
        for w in work_items:
            yield load_one(w, channel_order, seg_color, seg_channel)
        return

    prefetch = max(1, prefetch)
    executor = ThreadPoolExecutor(max_workers=io_workers,
                                  thread_name_prefix="fov-loader")
    try:
        pending = []
        idx_next = 0
        while idx_next < min(prefetch, len(work_items)):
            pending.append(executor.submit(
                load_one, work_items[idx_next],
                channel_order, seg_color, seg_channel,
            ))
            idx_next += 1

        for _ in range(len(work_items)):
            loaded = pending.pop(0).result()
            if idx_next < len(work_items):
                pending.append(executor.submit(
                    load_one, work_items[idx_next],
                    channel_order, seg_color, seg_channel,
                ))
                idx_next += 1
            yield loaded
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Save dispatch
# ---------------------------------------------------------------------------

def dispatch_crop_writes(loaded: LoadedItem, mask: np.ndarray,
                         args, split_output: bool, save_pool: ThreadPoolExecutor):
    """Compute centroids, generate crops, submit per-crop writes. Returns (futures, n_cells)."""
    centroids = fast_centroids(mask)
    if not centroids:
        return [], 0

    loaded.item.out_dir.mkdir(parents=True, exist_ok=True)
    img = loaded.img

    color_order: Optional[list[str]] = None
    if split_output and img.ndim == 3:
        n_ch = img.shape[2]
        if loaded.color_order is not None and len(loaded.color_order) == n_ch:
            color_order = loaded.color_order
        else:
            # Single-file mode (or mismatch): synthesize channel names so
            # downstream filenames stay deterministic.
            color_order = [f"ch{i}" for i in range(n_ch)]

    futures = []
    for label_id, cy, cx in centroids:
        crop = pad_crop(img, cy, cx, args.crop_size)
        cell_stem = f"{loaded.item.base_name}{args.suffix}{label_id:04d}"
        if color_order is not None:
            for i, color in enumerate(color_order):
                target = loaded.item.out_dir / f"{cell_stem}_{color}{args.out_ext}"
                futures.append(save_pool.submit(
                    save_array, crop[:, :, i].copy(), target
                ))
        else:
            target = loaded.item.out_dir / f"{cell_stem}.tif"
            futures.append(save_pool.submit(
                tifffile.imwrite, str(target), crop, compression="zlib"
            ))
    return futures, len(centroids)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def _split_masks_output(masks_out, n: int) -> list[np.ndarray]:
    """Normalise model.eval() output to a list of N 2-D label masks."""
    if n == 1:
        return [np.asarray(masks_out)]
    if isinstance(masks_out, np.ndarray) and masks_out.ndim == 3:
        return [masks_out[i] for i in range(masks_out.shape[0])]
    return [np.asarray(m) for m in masks_out]


def process_batch(batch: list[LoadedItem], args, split_output: bool,
                  save_pool: ThreadPoolExecutor, manifest: Manifest,
                  pbar: tqdm) -> int:
    """Run Cellpose on a batch of FOVs, save crops, record in manifest."""
    seg_inputs = []
    for loaded in batch:
        img = loaded.img
        seg_inputs.append(
            img[:, :, loaded.seg_channel_idx] if img.ndim == 3 else img
        )

    t0 = time.perf_counter()
    out = args.cellpose_model.eval(
        seg_inputs if len(seg_inputs) > 1 else seg_inputs[0],
        diameter=args.diameter,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        **args.cellpose_eval_kwargs,
    )
    tqdm.write(f"  cellpose eval (FOVs={len(seg_inputs)}): "
               f"{time.perf_counter() - t0:.1f}s")

    all_masks = _split_masks_output(out[0], len(seg_inputs))

    total_cells = 0
    for loaded, mask in zip(batch, all_masks):
        mask = np.asarray(mask).astype(np.int32, copy=False)
        try:
            futures, n_cells = dispatch_crop_writes(
                loaded, mask, args, split_output, save_pool
            )
            # Wait for THIS FOV's writes to finish before manifesting it,
            # so a mid-run crash never leaves a "done" entry with missing files.
            for f in futures:
                f.result()
            if n_cells == 0:
                tqdm.write(f"  [warn] {loaded.item.base_name}: no cells detected",
                           file=sys.stderr)
            manifest.append(
                base_name=loaded.item.base_name,
                n_cells=n_cells,
                crop_size=args.crop_size,
                suffix=args.suffix,
                seg_channel=args.seg_channel,
                seg_color=args.seg_color or "",
                split_output=str(bool(split_output)),
                diameter=("" if args.diameter is None else f"{float(args.diameter):.4f}"),
                model=args.model,
                src_mtime=f"{loaded.item.src_mtime:.3f}",
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
            extra = ""
            if loaded.color_order is not None:
                extra = f"  [channels: {loaded.color_order}]"
            tqdm.write(f"  {loaded.item.base_name}: {n_cells} cell(s){extra}")
            total_cells += n_cells
        except Exception as exc:
            tqdm.write(f"  [error] {loaded.item.base_name}: {exc}", file=sys.stderr)
        finally:
            pbar.update(1)
    return total_cells


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
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
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Cellpose tile batch size (per-FOV internal batching).")
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
    parser.set_defaults(split_output=None)   # resolved per-mode below

    # New: pipeline / resume controls
    parser.add_argument("--fov-batch", type=int, default=4,
                        help="How many FOVs to send to model.eval() at once.")
    parser.add_argument("--io-workers", type=int, default=4,
                        help="Threads for prefetching FOV images from disk.")
    parser.add_argument("--save-workers", type=int, default=8,
                        help="Threads for writing per-cell crops to disk.")
    parser.add_argument("--prefetch", type=int, default=None,
                        help="How many FOVs to keep loaded ahead of the GPU "
                             "(default: fov_batch + 2).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore any existing manifest and re-process all FOVs.")
    parser.add_argument("--manifest", default=None,
                        help="Override manifest CSV path "
                             f"(default: <output>/{MANIFEST_NAME}).")
    parser.set_defaults(resume=True)
    return parser


def run(args) -> None:
    channel_order = [c.strip().lower() for c in args.channel_order.split(",")]
    split_output = args.split_output if args.split_output is not None else args.multichannel

    work_items = collect_work_items(args, channel_order)
    if not work_items:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    manifest_dir = Path(args.output) if args.output else work_items[0].out_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else manifest_dir / MANIFEST_NAME
    manifest = Manifest(manifest_path)

    if args.resume:
        n_known = manifest.load()
        if n_known:
            print(f"Manifest: {n_known} prior FOV(s) on record at {manifest_path}.")
        keep: list[WorkItem] = []
        for w in work_items:
            key = _resume_key(
                base_name=w.base_name,
                crop_size=args.crop_size,
                suffix=args.suffix,
                seg_channel=args.seg_channel,
                seg_color=args.seg_color,
                split_output=split_output,
                diameter=args.diameter,
                model=args.model,
                src_mtime=w.src_mtime,
            )
            if not manifest.already_done(key):
                keep.append(w)
        skipped = len(work_items) - len(keep)
        if skipped:
            print(f"Resume: skipping {skipped} FOV(s) already complete.")
        work_items = keep

    if not work_items:
        print("Nothing to do.")
        return

    out_label = f"split ({args.out_ext}/channel)" if split_output else "stacked TIFF"
    print(f"Processing {len(work_items)} FOV(s). Output: {out_label}. "
          f"Cellpose model: {args.model}. "
          f"fov_batch={args.fov_batch}, io_workers={args.io_workers}, "
          f"save_workers={args.save_workers}.")

    args.cellpose_model, args.cellpose_eval_kwargs = build_cellpose(args.model, args.gpu)
    args.cellpose_eval_kwargs["batch_size"] = args.batch_size

    prefetch = args.prefetch if args.prefetch is not None else args.fov_batch + 2

    manifest.open()
    save_pool = ThreadPoolExecutor(max_workers=max(1, args.save_workers),
                                   thread_name_prefix="crop-writer")
    pbar = tqdm(total=len(work_items), unit="FOV")
    total_cells = 0
    try:
        loader = prefetch_loader(
            work_items, channel_order, args.seg_color, args.seg_channel,
            io_workers=max(0, args.io_workers), prefetch=prefetch,
        )
        batch: list[LoadedItem] = []
        for loaded in loader:
            if loaded.error is not None:
                tqdm.write(
                    f"  [load-error] {loaded.item.base_name}: {loaded.error}",
                    file=sys.stderr,
                )
                pbar.update(1)
                continue
            batch.append(loaded)
            if len(batch) >= max(1, args.fov_batch):
                total_cells += process_batch(batch, args, split_output,
                                             save_pool, manifest, pbar)
                batch = []
        if batch:
            total_cells += process_batch(batch, args, split_output,
                                         save_pool, manifest, pbar)
    finally:
        pbar.close()
        save_pool.shutdown(wait=True)
        manifest.close()

    print(f"\nDone. {total_cells} crops saved. Manifest: {manifest_path}")


def main():
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
