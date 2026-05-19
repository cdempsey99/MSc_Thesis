import rasterio
import numpy as np
from pathlib import Path

data_dir = Path("/beegfs/scratch/callumdempsey/data")
all_tifs = sorted(data_dir.glob("*.tif"))[:20]  # Sample 20 images

print(f"Computing stats from {len(all_tifs)} images...")

all_means = []
all_stds = []

for i, tif in enumerate(all_tifs):
    with rasterio.open(tif) as src:
        img = src.read().astype(np.float32)  # [4, H, W] — NIR, R, G, B
        # Per band mean and std
        all_means.append(img.mean(axis=(1, 2)))
        all_stds.append(img.std(axis=(1, 2)))
    print(f"[{i+1}/{len(all_tifs)}] {tif.name}")

mean = np.mean(all_means, axis=0)
std = np.mean(all_stds, axis=0)

print(f"\nMean per band (NIR, R, G, B): {mean}")
print(f"Std per band (NIR, R, G, B): {std}")