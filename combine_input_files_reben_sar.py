"""
combine_input_files_reben_sar.py - one-time preprocessing pass that combines each reBEN SAR
patch's 2 separate single-band GeoTIFFs (VV/VH) into one physical 2-band GeoTIFF.

SAR counterpart of combine_input_files_reben.py (see that script's docstring for the full
motivation). ReBENSARRawDataset.__getitem__ was opening 3 files per patch (VV, VH, reference
map); combining VV+VH brings that down to 2, the same as the optical pipeline after its combine
pass. Values are copied through unchanged (raw dB, float32) - the per-band z-scoring and the
optional despeckle/Lee filtering stay in the loader, applied after reading, exactly as before.

Band order in the combined file is ReBENSARRawDataset.BANDS (VV, VH), which the loader relies on.

Patch selection mirrors utils.dataset_e2e.load_reben_sar_splits (same snow/cloud filtering, same
split, same train_pairs[:max_patches] / val,test[:max_patches // 5] prefix-slice), so a training
job launched with the same --max_patches reads exactly the patches built here.

Combined files are saved alongside the source files as "{s1_name}_stacked.tif" in the same patch
directory. The loader checks for this file first and falls back to the 2 raw band files if it's
missing, so partial coverage never breaks anything.

Usage:
    python combine_input_files_reben_sar.py \
        --s1_root /beegfs/scratch/callumdempsey/data/reben_SAR/BigEarthNet-S1 \
        --metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
        --num_workers 16
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
import rasterio

from utils.misc import log_msg
from utils.dataset_e2e import ReBENSARRawDataset

BANDS = ReBENSARRawDataset.BANDS  # ["VV", "VH"] - single source of truth for the band order


def _s1_names_for_split(df, split_name, max_patches, is_train):
    names = df[df.split == split_name].s1_name.tolist()
    if max_patches:
        # Mirrors load_reben_sar_splits exactly: train gets the full max_patches, val/test get //5.
        return names[:max_patches] if is_train else names[:max_patches // 5]
    return names


def _patch_dir(s1_root: Path, s1_name: str) -> Path:
    # S1 names carry the MGRS tile as a 3rd trailing token ({scene}_{tile}_{row}_{col}), so strip
    # three tokens to get the scene folder - same as ReBENSARRawDataset.__getitem__.
    tile_id = "_".join(s1_name.split("_")[:-3])
    return s1_root / tile_id / s1_name


def stacked_path(s1_root: Path, s1_name: str) -> Path:
    return _patch_dir(s1_root, s1_name) / f"{s1_name}_stacked.tif"


def build_stacked_tif_for_patch(s1_root: Path, s1_name: str, overwrite: bool) -> str:
    """Returns "built", "skipped_existing", or "skipped_missing"."""
    patch_dir = _patch_dir(s1_root, s1_name)
    out_tif = stacked_path(s1_root, s1_name)

    if out_tif.exists() and not overwrite:
        return "skipped_existing"

    band_files = [patch_dir / f"{s1_name}_{band}.tif" for band in BANDS]
    missing = [f for f in band_files if not f.exists()]
    if missing:
        log_msg(f"WARNING: skipping {s1_name}, missing band file(s): {missing}")
        return "skipped_missing"

    arrays = []
    profile = None
    for f in band_files:
        with rasterio.open(f) as src:
            arrays.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    # Write to a temp name and rename into place, so a job killed mid-write (or two workers
    # racing on the same patch) can never leave a truncated file that the loader would trust.
    # PID in the temp name so two workers can't write the same temp file (possible if the metadata
    # maps two optical patches to one s1_name).
    tmp_tif = out_tif.with_name(f"{out_tif.name}.{os.getpid()}.tmp")
    profile.update(count=len(arrays))
    with rasterio.open(tmp_tif, "w", **profile) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr, i)
    tmp_tif.replace(out_tif)

    return "built"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s1_root", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--max_patches", type=int, default=None,
                        help="Must match the --max_patches value the training job will use, so the "
                             "same patches get combined files built. Omit to build for the full dataset.")
    parser.add_argument("--include_snow", action="store_true")
    parser.add_argument("--include_cloud", action="store_true")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild files that already exist (default: skip patches that already have one).")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Each patch's combine is fully independent, so this parallelizes across processes.")
    args = parser.parse_args()

    s1_root = Path(args.s1_root)
    df = pd.read_parquet(args.metadata_path)
    if not args.include_snow:
        df = df[~df.contains_seasonal_snow]
    if not args.include_cloud:
        df = df[~df.contains_cloud_or_shadow]

    for split_name in args.splits:
        is_train = split_name == "train"
        s1_names = _s1_names_for_split(df, split_name, args.max_patches, is_train)
        log_msg(f"Building combined SAR files for {len(s1_names)} '{split_name}' patches "
                f"({args.num_workers} workers)...")

        built = skipped_existing = skipped_missing = 0
        worker_fn = partial(build_stacked_tif_for_patch, s1_root, overwrite=args.overwrite)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for i, status in enumerate(executor.map(worker_fn, s1_names, chunksize=64)):
                if status == "built":
                    built += 1
                elif status == "skipped_existing":
                    skipped_existing += 1
                else:
                    skipped_missing += 1
                if (i + 1) % 10000 == 0:
                    log_msg(f"  {split_name}: {i + 1}/{len(s1_names)} processed "
                            f"(built={built}, skipped_existing={skipped_existing}, skipped_missing={skipped_missing})")

        log_msg(f"Done with '{split_name}'. Built {built} new combined files, skipped {skipped_existing} "
                f"already-existing, {skipped_missing} skipped due to missing band files.")


if __name__ == "__main__":
    main()
