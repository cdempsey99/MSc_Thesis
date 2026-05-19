import rasterio
import numpy as np

with rasterio.open("/beegfs/scratch/callumdempsey/data/GF2_PMS1__L1A0000564539-MSS1.tif") as src:
    print(f"Number of bands: {src.count}")
    print(f"Band descriptions: {src.descriptions}")
    for i in range(1, src.count + 1):
        band = src.read(i).astype(np.float32)
        print(f"Band {i}: min={band.min():.1f}, max={band.max():.1f}, mean={band.mean():.1f}, std={band.std():.1f}")