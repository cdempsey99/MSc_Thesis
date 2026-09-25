"""
build_reben_vrts.py — one-time preprocessing pass that combines each reBEN optical patch's
3 separate single-band GeoTIFFs (B04/B03/B02) into one 3-band GDAL VRT file.

Why: ReBENRawDataset.__getitem__ was doing 3 separate rasterio.open() calls per patch (one per
band), each carrying its own GDAL driver-initialization overhead, plus a network round-trip on
this beegfs-backed dataset. A VRT lets a single rasterio.open()+read() pull all 3 bands at once.
This does NOT reduce actual bytes read from disk - the saving is fewer separate file-open calls,
not less data transferred. Confirmed via nvidia-smi dmon that this pipeline is I/O-bound (GPU idle
~70% of sampled time), so cutting per-item open overhead is the right lever to test.

Uses the exact same patch-selection logic as utils.dataset_e2e.load_reben_splits (same filtering,
same train_ids[:max_patches] prefix-slice) so that a training job launched with the same
--max_patches value reads exactly the patches this script built VRTs for - no separate
patch-list file needed, just matching the number.

VRTs are saved alongside the source files as "{patch_id}_stacked.vrt" in the same patch
directory - ReBENRawDataset checks for this file first and only falls back to opening the 3
raw band files individually if it's missing, so partial coverage (e.g. only the reduced-scale
train/val/test split) is safe and doesn't break anything for patches that haven't been built yet.

Usage:
    python build_reben_vrts.py \
        --s2_root /beegfs/scratch/callumdempsey/data/reben/BigEarthNet-S2 \
        --metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
        --max_patches 25000
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

from utils.misc import log_msg

BANDS = ["B04", "B03", "B02"]


def _patch_ids_for_split(df, split_name, max_patches, is_train):
    ids = df[df.split == split_name].patch_id.tolist()
    if max_patches:
        # Mirrors load_reben_splits exactly: train gets the full max_patches, val/test get //5.
        return ids[:max_patches] if is_train else ids[:max_patches // 5]
    return ids


def build_vrt_for_patch(s2_root: Path, patch_id: str, overwrite: bool) -> bool:
    tile_id = "_".join(patch_id.split("_")[:-2])
    patch_dir = s2_root / tile_id / patch_id
    out_vrt = patch_dir / f"{patch_id}_stacked.vrt"

    if out_vrt.exists() and not overwrite:
        return False

    band_files = [patch_dir / f"{patch_id}_{band}.tif" for band in BANDS]
    missing = [f for f in band_files if not f.exists()]
    if missing:
        log_msg(f"WARNING: skipping {patch_id}, missing band file(s): {missing}")
        return False

    subprocess.run(
        ["gdalbuildvrt", "-separate", "-overwrite", str(out_vrt)] + [str(f) for f in band_files],
        check=True, capture_output=True, text=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s2_root", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--max_patches", type=int, default=None,
                        help="Must match the --max_patches value the training job will use, so the "
                             "same patches get VRTs built. Omit to build for the full dataset.")
    parser.add_argument("--include_snow", action="store_true")
    parser.add_argument("--include_cloud", action="store_true")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test"],
                        help="Which splits to build VRTs for - e.g. pass just 'train' for a quick "
                             "timing test without waiting on val/test too.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild VRTs that already exist (default: skip patches that already have one).")
    args = parser.parse_args()

    s2_root = Path(args.s2_root)
    df = pd.read_parquet(args.metadata_path)
    if not args.include_snow:
        df = df[~df.contains_seasonal_snow]
    if not args.include_cloud:
        df = df[~df.contains_cloud_or_shadow]

    built = 0
    skipped_existing = 0
    skipped_missing = 0
    for split_name in args.splits:
        is_train = split_name == "train"
        patch_ids = _patch_ids_for_split(df, split_name, args.max_patches, is_train)
        log_msg(f"Building VRTs for {len(patch_ids)} '{split_name}' patches...")
        for i, patch_id in enumerate(patch_ids):
            result = build_vrt_for_patch(s2_root, patch_id, args.overwrite)
            if result:
                built += 1
            else:
                out_vrt = s2_root / "_".join(patch_id.split("_")[:-2]) / patch_id / f"{patch_id}_stacked.vrt"
                if out_vrt.exists():
                    skipped_existing += 1
                else:
                    skipped_missing += 1
            if (i + 1) % 1000 == 0:
                log_msg(f"  {split_name}: {i + 1}/{len(patch_ids)} processed "
                        f"(built={built}, skipped_existing={skipped_existing}, skipped_missing={skipped_missing})")

    log_msg(f"Done. Built {built} new VRTs, skipped {skipped_existing} already-existing, "
            f"{skipped_missing} skipped due to missing band files.")


if __name__ == "__main__":
    main()
