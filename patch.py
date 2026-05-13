import torch
import json
from pathlib import Path

embedding_dir = Path("/beegfs/scratch/callumdempsey/results/embeddings/clay_v1/patch224_stride112")
meta_path = embedding_dir / "metadata.json"

# Load existing metadata
with open(meta_path, "r") as f:
    meta = json.load(f)

# Count patches per file
patch_counts = {}
all_pt_files = sorted(embedding_dir.glob("*_embeddings.pt"))
print(f"Counting patches in {len(all_pt_files)} files...")

for i, pt_file in enumerate(all_pt_files):
    data = torch.load(pt_file, map_location="cpu")
    count = int(data['features'].size(0))
    patch_counts[pt_file.name] = count
    del data
    print(f"[{i+1}/{len(all_pt_files)}] {pt_file.name}: {count} patches")

# Add to existing metadata and save
meta["patch_counts"] = patch_counts
meta["total_patches"] = sum(patch_counts.values())

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)

print(f"\nDone. Total patches: {meta['total_patches']}")
print(f"Updated metadata saved to {meta_path}")