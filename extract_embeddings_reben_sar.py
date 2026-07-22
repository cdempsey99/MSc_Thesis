import torch
import argparse
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader

from utils.dataset_e2e import ReBENSARRawDataset, load_reben_sar_splits
from models.encoder import initialize_clay_encoder, get_encoder_representation_partial
from utils.misc import log_msg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S1_WAVES = ReBENSARRawDataset.WAVELENGTHS


def save_shard(features, masks, out_dir, split_name, shard_idx):
    path = out_dir / f"{split_name}_shard_{shard_idx:04d}.pt"
    torch.save({'features': features, 'masks': masks}, path)
    log_msg(f"  Saved {path.name}: {features.shape[0]} patches")


def extract_split(ds, out_dir, split_name, encoder, shard_size, batch_size):
    if len(ds) == 0:
        return 0

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    feat_buf, mask_buf = [], []
    n_buf = 0
    shard_idx = 0

    log_msg(f"Extracting {split_name} ({len(ds)} patches)...")

    for batch_idx, (imgs, masks) in enumerate(loader):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                feats = get_encoder_representation_partial(imgs.to(DEVICE), encoder, waves=S1_WAVES)

        feat_buf.append(feats.cpu())
        mask_buf.append(masks.short())  # int16 sufficient for class indices 0-19
        n_buf += feats.shape[0]

        while n_buf >= shard_size:
            all_f = torch.cat(feat_buf, dim=0)
            all_m = torch.cat(mask_buf, dim=0)
            save_shard(all_f[:shard_size], all_m[:shard_size], out_dir, split_name, shard_idx)
            shard_idx += 1
            overflow_f = all_f[shard_size:]
            overflow_m = all_m[shard_size:]
            feat_buf = [overflow_f] if overflow_f.shape[0] > 0 else []
            mask_buf = [overflow_m] if overflow_m.shape[0] > 0 else []
            n_buf = overflow_f.shape[0]

        if (batch_idx + 1) % 100 == 0:
            log_msg(f"  {split_name}: {min((batch_idx + 1) * batch_size, len(ds))}/{len(ds)} patches")

    if n_buf > 0:
        save_shard(torch.cat(feat_buf, dim=0), torch.cat(mask_buf, dim=0),
                   out_dir, split_name, shard_idx)
        shard_idx += 1

    log_msg(f"{split_name} done: {len(ds)} patches → {shard_idx} shards")
    return shard_idx


def run_extraction(args):
    out_dir = Path(args.out_dir) / "embeddings" / "reben_sar"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_msg(f"Output dir: {out_dir}")

    train_ds, val_ds, test_ds = load_reben_sar_splits(
        metadata_path=args.metadata_path,
        s1_root=args.s1_root,
        ref_root=args.ref_root,
        exclude_snow=True,
        exclude_cloud=True,
        max_patches=args.max_patches,
        despeckle=args.despeckle,
    )

    encoder = initialize_clay_encoder()
    encoder.eval()

    split_info = {}
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        n_shards = extract_split(ds, out_dir, split_name, encoder,
                                 shard_size=args.shard_size, batch_size=args.batch_size)
        split_info[split_name] = {"n_shards": n_shards, "n_patches": len(ds)}

    metadata = {
        "shard_size": args.shard_size,
        "encoder": "clay_v1",
        "wavelengths": S1_WAVES.tolist(),
        "feature_shape": [1024, 28, 28],
        "mask_shape": [224, 224],
        "timestamp": time.ctime(),
        "splits": split_info,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log_msg("Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frozen Clay embeddings for reBEN SAR patches")
    parser.add_argument("--s1_root",       type=str, required=True)
    parser.add_argument("--ref_root",      type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--out_dir",       type=str, required=True)
    parser.add_argument("--shard_size",    type=int, default=1000)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--max_patches",   type=int, default=None,
                        help="Limit train patches for QA runs (val/test get max_patches//5)")
    parser.add_argument("--despeckle", action="store_true",
                        help="Apply a light 3x3 median filter to VV/VH bands (in dB, before z-scoring) for speckle reduction")
    args = parser.parse_args()
    run_extraction(args)
