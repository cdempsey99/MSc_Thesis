"""
calibrate_as1.py — post-hoc temperature scaling for the AS1 baseline (single encoder +
single decoder, no diversity, no student).

Fits one scalar T on the validation split (never on test) via fit_temperature, then
applies it to the test split to report calibration metrics (ECE, NLL, reliability
diagram) both with and without scaling. mIoU is identical in both cases by construction
- temperature scaling can't change argmax predictions, only confidence shape - so this
purely quantifies how much of AS1's calibration gap a single extra parameter can close,
as a baseline for the diversity-ensemble / distillation ECE claims elsewhere in the
thesis.

Usage:
    python calibrate_as1.py \
        --data_dir /beegfs/scratch/callumdempsey \
        --decoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/AS1_..._best_model.pth \
        --decoder_embed_dim 256 --num_classes 25 \
        --run_name AS1_calibration
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.dataset import BakedFeatureDataset
from utils.misc import log_msg, fit_temperature
from utils.visualisation import plot_reliability_diagram
from models.ensemble import DecoderEnsemble

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_val_test_split(data_dir, patch_size, stride, max_images=None):
    """Mirrors train_decoders.py's split exactly (same glob, same seed=42 shuffle, same
    70/15/15 cut) so val/test here are the same patches AS1 was actually validated/tested
    on — required for the fitted T and reported ECE to mean anything."""
    embedding_dir = Path(data_dir) / "embeddings" / "fbp" / "clay_v1" / f"patch{patch_size}_stride{stride}"
    all_files = sorted(list(embedding_dir.glob("*_embeddings.pt")))
    if max_images is not None:
        all_files = all_files[:max_images]
    random.seed(42)
    random.shuffle(all_files)

    n = len(all_files)
    if n < 5:
        val_files = all_files[1:2]
        test_files = all_files[2:]
    else:
        val_files = all_files[int(n * 0.7):int(n * 0.85)]
        test_files = all_files[int(n * 0.85):]
    return val_files, test_files


def collect_logits_and_targets(decoder, loader):
    """One pass over a loader; returns concatenated [N, K] logits and [N] targets for
    labelled pixels only (mask > 0). Fine for validation/test-set sizes that fit in
    memory once flattened to just the labelled pixels; not meant for huge streaming sets."""
    all_logits, all_targets = [], []
    with torch.no_grad():
        for features, masks in loader:
            features = features.to(DEVICE)
            logits = decoder(features)[0]  # M=1 for AS1 -> single head, [B, K, 224, 224]
            masks = masks.to(DEVICE).long()
            valid = masks > 0
            if not valid.any():
                continue
            logits_flat = logits.permute(0, 2, 3, 1)[valid]  # [N_valid, K]
            all_logits.append(logits_flat.cpu())
            all_targets.append(masks[valid].cpu())
    return torch.cat(all_logits), torch.cat(all_targets)


def compute_ece_and_nll(logits, targets, temperature=1.0, num_bins=10):
    probs = torch.softmax(logits / temperature, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == targets).float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_accs = torch.zeros(num_bins)
    bin_props = torch.zeros(num_bins)
    ece = 0.0
    for i in range(num_bins):
        in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
        prop = in_bin.float().mean().item()
        bin_props[i] = prop
        if in_bin.any():
            bin_accs[i] = correct[in_bin].mean().item()
            avg_conf = conf[in_bin].mean().item()
            ece += abs(bin_accs[i].item() - avg_conf) * prop

    nll = torch.nn.functional.cross_entropy(logits / temperature, targets).item()
    return ece, nll, bin_accs.numpy(), bin_props.numpy()


def main():
    parser = argparse.ArgumentParser(description="Post-hoc temperature scaling for the AS1 baseline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the embeddings folder root")
    parser.add_argument("--decoder_checkpoint", type=str, required=True, help="Path to AS1's saved decoder .pth")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--decoder_embed_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None, help="Must match the value AS1 was trained with, if it used one")
    parser.add_argument("--run_name", type=str, default="AS1_calibration")
    args = parser.parse_args()

    val_files, test_files = build_val_test_split(args.data_dir, args.patch_size, args.stride, args.max_images)
    log_msg(f"Split: {len(val_files)} Val | {len(test_files)} Test")

    val_loader = DataLoader(BakedFeatureDataset(val_files, augment=False),
                            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(BakedFeatureDataset(test_files, augment=False),
                             batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    decoder = DecoderEnsemble(M=1, in_channels=1024, embed_dim=args.decoder_embed_dim,
                              num_classes=args.num_classes)
    ckpt = torch.load(args.decoder_checkpoint, map_location=DEVICE)
    decoder.load_state_dict(ckpt['model_state_dict'])
    decoder.to(DEVICE)
    decoder.eval()
    log_msg(f"Loaded AS1 decoder from {args.decoder_checkpoint}")

    log_msg("Collecting validation logits...")
    val_logits, val_targets = collect_logits_and_targets(decoder, val_loader)

    log_msg(f"Fitting temperature on {val_targets.numel():,} validation pixels...")
    T = fit_temperature(val_logits.to(DEVICE), val_targets.to(DEVICE))

    log_msg("Collecting test logits...")
    test_logits, test_targets = collect_logits_and_targets(decoder, test_loader)

    raw_ece, raw_nll, raw_bin_accs, raw_bin_props = compute_ece_and_nll(test_logits, test_targets, temperature=1.0)
    cal_ece, cal_nll, cal_bin_accs, cal_bin_props = compute_ece_and_nll(test_logits, test_targets, temperature=T)

    pred_raw = test_logits.argmax(dim=1)
    pred_cal = (test_logits / T).argmax(dim=1)
    assert torch.equal(pred_raw, pred_cal), "temperature scaling changed predictions - should be impossible"

    log_msg(f"AS1 RAW:        ECE={raw_ece:.4f} | NLL={raw_nll:.4f}")
    log_msg(f"AS1 CALIBRATED: ECE={cal_ece:.4f} | NLL={cal_nll:.4f} | T={T:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": args.run_name,
        "temperature": T,
        "num_val_pixels": int(val_targets.numel()),
        "num_test_pixels": int(test_targets.numel()),
        "raw_ece": raw_ece, "raw_nll": raw_nll,
        "calibrated_ece": cal_ece, "calibrated_nll": cal_nll,
    }
    results_path = os.path.join(runs_dir, f"{args.run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")

    plot_reliability_diagram(raw_bin_accs, raw_bin_props, save_name=f"{args.run_name}_raw_reliability")
    plot_reliability_diagram(cal_bin_accs, cal_bin_props, save_name=f"{args.run_name}_calibrated_reliability")


if __name__ == "__main__":
    main()
