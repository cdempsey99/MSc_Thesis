import rasterio
import numpy as np
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset
from pathlib import Path
from utils.misc import log_msg


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