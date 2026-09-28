"""
combine_input_files_reben.py — one-time preprocessing pass that combines each reBEN optical
patch's 3 separate single-band GeoTIFFs (B04/B03/B02) into one physical 3-band GeoTIFF.

Why: ReBENRawDataset.__getitem__ was doing 3 separate rasterio.open() calls per patch (one per
band), each carrying its own GDAL driver-initialization overhead, plus a network round-trip on
this beegfs-backed dataset. Combining them means a single rasterio.open()+read() pulls all 3
bands at once. Confirmed via nvidia-smi dmon that this pipeline is I/O-bound (GPU idle ~70% of
sampled time), so cutting per-item open overhead is the right lever to test.

Originally built these as zero-copy GDAL VRTs (metadata-only, no data duplication) via the
gdalbuildvrt CLI tool, but that binary isn't installed in this environment (pip-installed
rasterio bundles its own private GDAL library for read/write, not the separate GDAL command-line
utilities). Rather than add a new GDAL package dependency, this writes a real physical combined
GeoTIFF instead, using only rasterio (already proven to work everywhere else in this project).
Costs a bit of extra disk space (~86KB/patch for a real copy vs near-zero for a true VRT -
trivial at this dataset's scale) but achieves the identical runtime goal: one file open instead
of three.

Uses the exact same patch-selection logic as utils.dataset_e2e.load_reben_splits (same filtering,
same train_ids[:max_patches] prefix-slice) so that a training job launched with the same
--max_patches value reads exactly the patches this script built combined files for - no separate
patch-list file needed, just matching the number.

Combined files are saved alongside the source files as "{patch_id}_stacked.tif" in the same
patch directory - ReBENRawDataset checks for this file first and only falls back to opening the
3 raw band files individually if it's missing, so partial coverage (e.g. only the reduced-scale
train/val/test split) is safe and doesn't break anything for patches that haven't been built yet.

Parallelized across --num_workers processes: each patch's combine is fully independent (reads 3
files, writes 1, no shared state, no ordering dependency), and the num_workers DataLoader finding
this project already confirmed (this same beegfs filesystem tolerates concurrent I/O well) is the
motivation for not running the full-dataset ~45h estimate serially.

Usage:
    python combine_input_files_reben.py \
        --s2_root /beegfs/scratch/callumdempsey/data/reben/BigEarthNet-S2 \
        --metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
        --num_workers 16
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
import rasterio

from utils.misc import log_msg

BANDS = ["B04", "B03", "B02"]


def _patch_ids_for_split(df, split_name, max_patches, is_train):
    ids = df[df.split == split_name].patch_id.tolist()
    if max_patches:
        # Mirrors load_reben_splits exactly: train gets the full max_patches, val/test get //5.
        return ids[:max_patches] if is_train else ids[:max_patches // 5]
    return ids


def _stacked_path(s2_root: Path, patch_id: str) -> Path:
    tile_id = "_".join(patch_id.split("_")[:-2])
    return s2_root / tile_id / patch_id / f"{patch_id}_stacked.tif"


def build_stacked_tif_for_patch(s2_root: Path, patch_id: str, overwrite: bool) -> str:
    """Returns "built", "skipped_existing", or "skipped_missing" - classified here rather than by
    the caller re-checking file existence afterward, so parallel dispatch doesn't need a second
    stat call per patch just to categorize the result."""
    tile_id = "_".join(patch_id.split("_")[:-2])
    patch_dir = s2_root / tile_id / patch_id
    out_tif = _stacked_path(s2_root, patch_id)

    if out_tif.exists() and not overwrite:
        return "skipped_existing"

    band_files = [patch_dir / f"{patch_id}_{band}.tif" for band in BANDS]
    missing = [f for f in band_files if not f.exists()]
    if missing:
        log_msg(f"WARNING: skipping {patch_id}, missing band file(s): {missing}")
        return "skipped_missing"

    arrays = []
    profile = None
    for f in band_files:
        with rasterio.open(f) as src:
            arrays.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    profile.update(count=len(arrays))
    with rasterio.open(out_tif, "w", **profile) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr, i)

    return "built"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s2_root", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--max_patches", type=int, default=None,
                        help="Must match the --max_patches value the training job will use, so the "
                             "same patches get combined files built. Omit to build for the full dataset.")
    parser.add_argument("--include_snow", action="store_true")
    parser.add_argument("--include_cloud", action="store_true")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test"],
                        help="Which splits to build combined files for - e.g. pass just 'train' for "
                             "a quick timing test without waiting on val/test too.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild files that already exist (default: skip patches that already have one).")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Each patch's combine is fully independent, so this parallelizes across "
                             "processes rather than running the full-dataset job serially. This is a "
                             "write-heavy workload, unlike the num_workers DataLoader finding this is "
                             "motivated by (which was read-only) - scaling behavior on this filesystem "
                             "hasn't been separately confirmed for writes, so treat higher values as "
                             "worth testing rather than assumed safe.")
    args = parser.parse_args()

    s2_root = Path(args.s2_root)
    df = pd.read_parquet(args.metadata_path)
    if not args.include_snow:
        df = df[~df.contains_seasonal_snow]
    if not args.include_cloud:
        df = df[~df.contains_cloud_or_shadow]

    for split_name in args.splits:
        is_train = split_name == "train"
        patch_ids = _patch_ids_for_split(df, split_name, args.max_patches, is_train)
        log_msg(f"Building combined files for {len(patch_ids)} '{split_name}' patches "
                f"({args.num_workers} workers)...")

        built = skipped_existing = skipped_missing = 0
        worker_fn = partial(build_stacked_tif_for_patch, s2_root, overwrite=args.overwrite)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for i, status in enumerate(executor.map(worker_fn, patch_ids)):
                if status == "built":
                    built += 1
                elif status == "skipped_existing":
                    skipped_existing += 1
                else:
                    skipped_missing += 1
                if (i + 1) % 1000 == 0:
                    log_msg(f"  {split_name}: {i + 1}/{len(patch_ids)} processed "
                            f"(built={built}, skipped_existing={skipped_existing}, skipped_missing={skipped_missing})")

        log_msg(f"Done with '{split_name}'. Built {built} new combined files, skipped {skipped_existing} "
                f"already-existing, {skipped_missing} skipped due to missing band files.")


if __name__ == "__main__":
    main()
