import bisect
import rasterio
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from rasterio.windows import Window
from scipy.ndimage import median_filter
from torch.utils.data import Dataset
from pathlib import Path
from utils.misc import log_msg
from configs.config import CORINE_TO_REBEN

_CORINE_LUT = np.zeros(65536, dtype=np.int64)
for corine_code, class_idx in CORINE_TO_REBEN.items():
    _CORINE_LUT[corine_code] = class_idx


class FBPRawDataset(Dataset):
    """
    Dataset for end-to-end training — returns raw unnormalised image patches
    and corresponding masks. No pre-baking.
    """
    def __init__(self, img_paths, mask_paths, patch_size=224, stride=224,
                 augment=False, max_samples=None):
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []

        for img_p, mask_p in zip(img_paths, mask_paths):
            with rasterio.open(img_p) as src:
                h, w = src.height, src.width

            for y in range(0, h - patch_size, stride):
                for x in range(0, w - patch_size, stride):
                    self.samples.append((img_p, mask_p, x, y))
                    if max_samples and len(self.samples) >= max_samples:
                        break
                if max_samples and len(self.samples) >= max_samples:
                    break
            if max_samples and len(self.samples) >= max_samples:
                break

        log_msg(f"FBPRawDataset: {len(self.samples)} patches from {len(img_paths)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, x, y = self.samples[idx]
        win = Window(x, y, self.patch_size, self.patch_size)

        with rasterio.open(img_p) as src:
            # Raw pixel values, no normalisation — Clay handles this internally
            img = src.read([1, 2, 3], window=win).astype(np.float32)

        with rasterio.open(mask_p) as src:
            mask = src.read(1, window=win).astype(np.int64)

        img_tensor = torch.from_numpy(img)   # [3, 224, 224]
        mask_tensor = torch.from_numpy(mask) # [224, 224]

        if self.augment:
            img_tensor, mask_tensor = self._augment(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

    def _augment(self, img, mask):
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])
        k = torch.randint(0, 4, (1,)).item()
        img = torch.rot90(img, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])
        return img, mask

    def get_patch_info(self, idx):
        img_p, mask_p, x, y = self.samples[idx]
        return img_p, mask_p, x, y


class ReBENRawDataset(Dataset):
    """
    Dataset for reBEN e2e training.
    Loads pre-cut 120x120 S2 patches (B04/B03/B02) and resizes to 224x224.
    Use load_reben_splits() to construct train/val/test instances from metadata.parquet.
    """
    BANDS = ["B04", "B03", "B02"]  # Red, Green, Blue — all 10m native res (120x120)
    WAVELENGTHS = torch.tensor([0.6646, 0.5598, 0.4924], dtype=torch.float32)  # μm

    def __init__(self, patch_ids, s2_root, ref_root, augment=False):
        self.patch_ids = patch_ids
        self.s2_root = Path(s2_root)
        self.ref_root = Path(ref_root)
        self.augment = augment
        log_msg(f"ReBENRawDataset: {len(patch_ids)} patches")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        tile_id = "_".join(patch_id.split("_")[:-2])

        bands = []
        for band in self.BANDS:
            tif = self.s2_root / tile_id / patch_id / f"{patch_id}_{band}.tif"
            with rasterio.open(tif) as src:
                bands.append(src.read(1).astype(np.float32))

        img_tensor = torch.from_numpy(np.stack(bands, axis=0)) / 10000.0  # [3, 120, 120] — S2 DN → reflectance
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224, 224),
                                   mode='bilinear', align_corners=False).squeeze(0)

        ref_dir = self.ref_root / tile_id / patch_id
        ref_tif = next(ref_dir.glob("*.tif"))
        with rasterio.open(ref_tif) as src:
            mask = src.read(1).astype(np.uint16)
        mask = _CORINE_LUT[mask]  # CORINE codes → 0-indexed class labels (0 = ignore)

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(224, 224),
                                    mode='nearest').squeeze().long()

        if self.augment:
            img_tensor, mask_tensor = self._augment(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

    def _augment(self, img, mask):
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])
        k = torch.randint(0, 4, (1,)).item()
        img = torch.rot90(img, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])
        return img, mask

    def get_patch_info(self, idx):
        patch_id = self.patch_ids[idx]
        tile_id = "_".join(patch_id.split("_")[:-2])
        return str(self.s2_root / tile_id / patch_id), 0, 0


def load_reben_splits(metadata_path, s2_root, ref_root,
                      exclude_snow=True, exclude_cloud=True, max_patches=None,
                      max_val_patches=None):
    """
    Reads metadata.parquet and returns train/val/test ReBENRawDataset objects
    using the official reBEN splits. Filters snowy/cloudy patches by default.
    max_patches limits each split independently (useful for QA runs).
    """
    df = pd.read_parquet(metadata_path)
    if exclude_snow:
        df = df[~df.contains_seasonal_snow]
    if exclude_cloud:
        df = df[~df.contains_cloud_or_shadow]

    train_ids = df[df.split == 'train'].patch_id.tolist()
    val_ids   = df[df.split == 'validation'].patch_id.tolist()
    test_ids  = df[df.split == 'test'].patch_id.tolist()

    if max_patches:
        train_ids = train_ids[:max_patches]
        val_ids   = val_ids[:max_patches // 5]
        test_ids  = test_ids[:max_patches // 5]
    if max_val_patches:
        val_ids = val_ids[:max_val_patches]

    log_msg(f"reBEN splits (exclude_snow={exclude_snow}, exclude_cloud={exclude_cloud}): "
            f"{len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test")

    return (
        ReBENRawDataset(train_ids, s2_root, ref_root, augment=True),
        ReBENRawDataset(val_ids,   s2_root, ref_root, augment=False),
        ReBENRawDataset(test_ids,  s2_root, ref_root, augment=False),
    )


class ReBENSARRawDataset(Dataset):
    """
    Dataset for reBEN SAR e2e training.
    Loads pre-cut 120x120 Sentinel-1 RTC patches (VV/VH, already in dB) and
    resizes to 224x224. Masks are the same CORINE reference maps used for the
    optical pipeline, keyed by the paired S2 patch_id (labels are sensor-independent).
    Use load_reben_sar_splits() to construct train/val/test instances from metadata.parquet.
    """
    BANDS = ["VV", "VH"]
    WAVELENGTHS = torch.tensor([3.5, 4.0], dtype=torch.float32)  # μm — Clay metadata.yaml nominal SAR values
    # Clay metadata.yaml sentinel-1-rtc band stats (z-score normalisation; data is already dB)
    BAND_MEAN = {"VV": -12.113, "VH": -18.673}
    BAND_STD = {"VV": 8.314, "VH": 8.017}

    def __init__(self, pairs, s1_root, ref_root, augment=False, despeckle=False):
        """pairs: list of (patch_id, s1_name) tuples — patch_id keys the mask, s1_name keys the SAR image."""
        self.pairs = pairs
        self.s1_root = Path(s1_root)
        self.ref_root = Path(ref_root)
        self.augment = augment
        self.despeckle = despeckle
        log_msg(f"ReBENSARRawDataset: {len(pairs)} patches" + (" (despeckled, 3x3 median)" if despeckle else ""))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        patch_id, s1_name = self.pairs[idx]
        # S1 product IDs don't embed the MGRS tile (unlike S2's ..._T33UUP), so reBEN
        # appends it as a 3rd trailing token: {scene}_{tile}_{row}_{col}. Strip all three
        # to get the on-disk scene folder, not just row/col as with the S2 patch_id.
        tile_id = "_".join(s1_name.split("_")[:-3])

        bands = []
        for band in self.BANDS:
            tif = self.s1_root / tile_id / s1_name / f"{s1_name}_{band}.tif"
            with rasterio.open(tif) as src:
                arr = src.read(1).astype(np.float32)
            if self.despeckle:
                arr = median_filter(arr, size=3)  # light speckle reduction, applied in dB before z-scoring
            arr = (arr - self.BAND_MEAN[band]) / self.BAND_STD[band]
            bands.append(arr)

        img_tensor = torch.from_numpy(np.stack(bands, axis=0))  # [2, 120, 120] — already dB, z-scored
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224, 224),
                                   mode='bilinear', align_corners=False).squeeze(0)

        ref_dir = self.ref_root / "_".join(patch_id.split("_")[:-2]) / patch_id
        ref_tif = next(ref_dir.glob("*.tif"))
        with rasterio.open(ref_tif) as src:
            mask = src.read(1).astype(np.uint16)
        mask = _CORINE_LUT[mask]  # CORINE codes → 0-indexed class labels (0 = ignore)

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(224, 224),
                                    mode='nearest').squeeze().long()

        if self.augment:
            img_tensor, mask_tensor = self._augment(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

    def _augment(self, img, mask):
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])
        k = torch.randint(0, 4, (1,)).item()
        img = torch.rot90(img, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])
        return img, mask

    def get_patch_info(self, idx):
        patch_id, s1_name = self.pairs[idx]
        tile_id = "_".join(s1_name.split("_")[:-3])
        return str(self.s1_root / tile_id / s1_name), 0, 0


def load_reben_sar_splits(metadata_path, s1_root, ref_root,
                          exclude_snow=True, exclude_cloud=True, max_patches=None,
                          max_val_patches=None, despeckle=False):
    """
    Reads metadata.parquet and returns train/val/test ReBENSARRawDataset objects
    using the official reBEN splits and the s1_name column for S1<->S2 pairing.
    Filters snowy/cloudy patches by default (flags are derived from the optical
    scene, but apply to the same underlying land parcel/date so are still valid for SAR).
    max_patches limits each split independently (useful for QA runs).
    """
    df = pd.read_parquet(metadata_path)
    if exclude_snow:
        df = df[~df.contains_seasonal_snow]
    if exclude_cloud:
        df = df[~df.contains_cloud_or_shadow]

    def pairs_for(split_name):
        sub = df[df.split == split_name]
        return list(zip(sub.patch_id.tolist(), sub.s1_name.tolist()))

    train_pairs = pairs_for('train')
    val_pairs   = pairs_for('validation')
    test_pairs  = pairs_for('test')

    if max_patches:
        train_pairs = train_pairs[:max_patches]
        val_pairs   = val_pairs[:max_patches // 5]
        test_pairs  = test_pairs[:max_patches // 5]
    if max_val_patches:
        val_pairs = val_pairs[:max_val_patches]

    log_msg(f"reBEN SAR splits (exclude_snow={exclude_snow}, exclude_cloud={exclude_cloud}): "
            f"{len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    return (
        ReBENSARRawDataset(train_pairs, s1_root, ref_root, augment=True,  despeckle=despeckle),
        ReBENSARRawDataset(val_pairs,   s1_root, ref_root, augment=False, despeckle=despeckle),
        ReBENSARRawDataset(test_pairs,  s1_root, ref_root, augment=False, despeckle=despeckle),
    )


class BakedReBENDataset(Dataset):
    """
    Map-style dataset for pre-baked reBEN embeddings produced by extract_embeddings_reben.py.
    Shards are memory-mapped so only accessed pages are loaded into RAM.
    Variable shard sizes are handled by probing each shard at init (cheap with mmap=True).
    """
    def __init__(self, shard_dir, split, augment=False):
        from utils.dataset import random_augment
        self._augment_fn = random_augment
        self.augment = augment

        shard_dir = Path(shard_dir)
        self.shard_files = sorted(shard_dir.glob(f"{split}_shard_*.pt"))

        if not self.shard_files:
            raise FileNotFoundError(f"No shards found for split '{split}' in {shard_dir}")

        # Probe each shard for its actual size using mmap (reads only tensor metadata, not data)
        self.shard_sizes = []
        for sf in self.shard_files:
            data = torch.load(sf, map_location='cpu', mmap=True)
            self.shard_sizes.append(data['features'].shape[0])

        self.cumulative = []
        running = 0
        for s in self.shard_sizes:
            running += s
            self.cumulative.append(running)

        self._current_path = None
        self._current_data = None

        log_msg(f"BakedReBENDataset ({split}): {self.cumulative[-1]} patches, "
                f"{len(self.shard_files)} shards")

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, idx):
        shard_idx = bisect.bisect_right(self.cumulative, idx)
        local_idx = idx - (self.cumulative[shard_idx - 1] if shard_idx > 0 else 0)

        path = self.shard_files[shard_idx]
        if path != self._current_path:
            self._current_path = path
            self._current_data = torch.load(path, map_location='cpu', mmap=True)

        features = self._current_data['features'][local_idx]       # [1024, 28, 28] float32
        mask = self._current_data['masks'][local_idx].long()        # [224, 224]

        if self.augment:
            features, mask = self._augment_fn(features, mask)

        return features, mask