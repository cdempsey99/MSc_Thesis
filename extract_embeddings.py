import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse

from utils.dataset import FBPPatchDataset
from models.encoder import initialize_clay_encoder, get_encoder_representation, FBP_WAVES_LEGACY, FBP_WAVES_GF2_BGR, FBP_GSD
from utils.training import log_msg

# Set Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_extraction(args):
    """The core logic for moving through the dataset and saving .pt files."""

    # 1. Path Management
    version_str = f"patch{args.patch_size}_stride{args.stride}"
    output_base = Path(args.out_dir) / "embeddings" / "fbp" / "clay_v1" / version_str

    # 2. Data Discovery (Simplified Partition Logic)
    data_dir = Path(args.data_dir)
    all_tifs = sorted(list(data_dir.glob("*.tif")))

    if args.max_images:
        all_tifs = all_tifs[:args.max_images]

    # Initialize Encoder
    encoder_model = initialize_clay_encoder(single_norm=args.clay_single_norm, native_forward=args.clay_native_forward,
                                            gsd=FBP_GSD).to(DEVICE)
    waves = FBP_WAVES_GF2_BGR if args.fbp_correct_waves else FBP_WAVES_LEGACY
    log_msg(f"FBP band wavelengths passed to Clay: {waves}")
    encoder_model.eval()

    log_msg(f"Starting extraction for {len(all_tifs)} images into {output_base}")

    patches_per_image = None

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
                                 stride=args.stride,
                                 preload=False)

            if len(ds) == 0:
                continue

            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

            all_feats = []
            all_masks = []

            for batch_imgs, batch_msks in loader:
                feats = get_encoder_representation(batch_imgs.to(DEVICE), encoder_model, waves=waves)
                all_feats.append(feats.cpu())
                all_masks.append(batch_msks.cpu())

            features_tensor = torch.cat(all_feats, dim=0)
            masks_tensor = torch.cat(all_masks, dim=0)

            torch.save({
                'features': features_tensor,
                'masks': masks_tensor,
            }, target_file)

            # Record patches per image (same for all FBP images)
            patches_per_image = int(features_tensor.size(0))

            log_msg(f"Finished {img_p.stem}")

    # Only save metadata if we actually processed at least one image
    if patches_per_image is None:
        log_msg("All files already existed, skipping metadata update.")
    else:
        metadata = {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "encoder": "clay_v1_pruned",
            "timestamp": time.ctime(),
            "patches_per_image": patches_per_image
        }
        with open(output_base / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extraction Script")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--clay_native_forward", action="store_true",
                        help="Run Clay as its own Encoder.forward does: GSD-scaled position encoding, cls token, single final "
                             "norm (implies --clay_single_norm). Off by default = the original no-position-encoding path of "
                             "every existing embedding/checkpoint; a model trained with it on must be evaluated with it on.")
    parser.add_argument("--fbp_correct_waves", action="store_true",
                        help="Pass the correct GF-2 wavelengths for the tifs' blue/green/red bands ([0.485, 0.555, 0.660] um) "
                             "instead of the original [0.842, 0.665, 0.560] (NIR/red/green) every existing FBP embedding used.")
    parser.add_argument("--clay_single_norm", action="store_true",
                        help="Skip the second application of Clay's final LayerNorm (its transformer already applies it). "
                             "Off by default to match all existing embeddings/checkpoints; a model trained with it on "
                             "must also be evaluated with it on.")

    args = parser.parse_args()

    # We call the function here
    run_extraction(args)