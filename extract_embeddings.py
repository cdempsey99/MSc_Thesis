import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse

from utils.dataset import FBPPatchDataset
from models.encoder import initialize_clay_encoder, get_encoder_representation
from utils.training import log_msg

# Set Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_extraction(args):
    """The core logic for moving through the dataset and saving .pt files."""

    # 1. Path Management
    version_str = f"patch{args.patch_size}_stride{args.stride}"
    output_base = Path(args.out_dir) / "embeddings" / "clay_v1" / version_str

    # 2. Data Discovery (Simplified Partition Logic)
    data_dir = Path(args.data_dir)
    all_tifs = sorted(list(data_dir.glob("*.tif")))

    if args.max_images:
        all_tifs = all_tifs[:args.max_images]

    # Initialize Encoder
    encoder_model = initialize_clay_encoder().to(DEVICE)
    encoder_model.eval()

    log_msg(f"Starting extraction for {len(all_tifs)} images into {output_base}")

    with torch.no_grad():
        for img_p in all_tifs:
            mask_p = data_dir / (img_p.stem + "_24label.png")
            target_file = output_base / f"{img_p.stem}_embeddings.pt"

            # Check if directory exists (Create on the fly)
            target_file.parent.mkdir(parents=True, exist_ok=True)

            if target_file.exists():
                log_msg(f"Skipping {img_p.stem} - already exists.")
                continue

            # Process single image
            ds = FBPPatchDataset([img_p], [mask_p],
                                 patch_size=args.patch_size,
                                 stride=args.stride)

            if len(ds) == 0:
                continue

            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

            all_feats = []
            all_masks = []

            for batch_imgs, batch_msks in loader:
                feats = get_encoder_representation(batch_imgs.to(DEVICE), encoder_model)
                all_feats.append(feats.cpu())
                all_masks.append(batch_msks.cpu())

            # Save the "Packed" image embeddings
            torch.save({
                'features': torch.cat(all_feats, dim=0),
                'masks': torch.cat(all_masks, dim=0)
            }, target_file)

            log_msg(f"Finished {img_p.stem}")

    # 3. Save Metadata Passport
    metadata = {
        "patch_size": args.patch_size,
        "stride": args.stride,
        "encoder": "clay_v1_pruned",
        "timestamp": time.ctime()
    }
    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extraction Script")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    # We call the function here
    run_extraction(args)