import torch
import psutil
import os

def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

path = "/beegfs/scratch/callumdempsey/results/embeddings/clay_v1/patch224_stride112/GF2_PMS1__L1A0000962382-MSS1_embeddings.pt"

print(f"Before load: {mem():.2f}GB")

# Test 1: normal load
data = torch.load(path, map_location="cpu")
print(f"After normal load: {mem():.2f}GB")
del data

import gc
gc.collect()
print(f"After del: {mem():.2f}GB")

# Test 2: mmap load
data_mmap = torch.load(path, map_location="cpu", mmap=True)
print(f"After mmap load: {mem():.2f}GB")

# Access just one patch
patch = data_mmap['features'][0]
print(f"After accessing one patch: {mem():.2f}GB")

# Access 32 patches
patches = data_mmap['features'][:32]
print(f"After accessing 32 patches: {mem():.2f}GB")