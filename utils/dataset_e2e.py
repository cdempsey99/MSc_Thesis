import rasterio
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from rasterio.windows import Window
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
                      exclude_snow=True, exclude_cloud=True, max_patches=None):
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

    log_msg(f"reBEN splits (exclude_snow={exclude_snow}, exclude_cloud={exclude_cloud}): "
            f"{len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test")

    return (
        ReBENRawDataset(train_ids, s2_root, ref_root, augment=True),
        ReBENRawDataset(val_ids,   s2_root, ref_root, augment=False),
        ReBENRawDataset(test_ids,  s2_root, ref_root, augment=False),
    )