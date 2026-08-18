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

Both stages stream through their DataLoader one batch at a time and never hold more
than a capped number of pixels (validation) or a single batch (test) in memory - FBP's
~7300x7300 images mean "every labelled pixel in the split" is tens of GB, not something
to naively collect.

Usage:
    python calibrate_as1.py \
        --data_dir /beegfs/scratch/callumdempsey/results \
        --decoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/AS1_..._best_model.pth \
        --decoder_embed_dim 512 --num_classes 25 \
        --patch_size 224 --stride 224 --max_images 150 \
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
    on - required for the fitted T and reported ECE to mean anything."""
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


def collect_val_logits_for_temperature(decoder, loader, max_pixels=2_000_000):
    """
    Streams through the validation loader, collecting per-pixel logits/targets only
    until max_pixels is reached, then stops early. Temperature scaling fits a single
    scalar, so a capped subsample (patches arrive in the already-shuffled file order
    from build_val_test_split) is standard practice and statistically sufficient -
    unlike test-set ECE/NLL below, this does not need every pixel in the split.
    """
    all_logits, all_targets = [], []
    collected = 0
    with torch.no_grad():
        for features, masks in loader:
            features = features.to(DEVICE)
            logits = decoder(features)[0]  # M=1 for AS1 -> single head, [B, K, 224, 224]
            masks = masks.to(DEVICE).long()
            valid = masks > 0
            if not valid.any():
                continue
            logits_flat = logits.permute(0, 2, 3, 1)[valid]  # [N_valid, K]
            targets_flat = masks[valid]
            all_logits.append(logits_flat.cpu())
            all_targets.append(targets_flat.cpu())
            collected += targets_flat.numel()
            if collected >= max_pixels:
                break
    return torch.cat(all_logits), torch.cat(all_targets)


def stream_test_calibration(decoder, test_loader, temperature, num_bins=10):
    """
    One pass over the test set computing raw (T=1) and temperature-scaled calibration
    metrics simultaneously, so the (large) test set only needs to be read once. Never
    stores per-pixel data - accumulates running confidence/accuracy histograms and a
    running NLL sum per batch, discarding each batch's logits immediately, matching the
    streaming pattern already used by evaluate_test_set/evaluate_test_set_reben
    elsewhere in this codebase. Also counts prediction mismatches between raw and
    calibrated (should always be zero, by construction) as a correctness check.
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    stats = {
        tag: {
            "bin_conf_sums": torch.zeros(num_bins),
            "bin_acc_sums": torch.zeros(num_bins),
            "bin_counts": torch.zeros(num_bins),
            "nll_sum": 0.0,
            "n": 0,
        }
        for tag in ("raw", "calibrated")
    }
    mismatches = 0

    with torch.no_grad():
        for features, masks in test_loader:
            features = features.to(DEVICE)
            logits = decoder(features)[0]
            masks = masks.to(DEVICE).long()
            valid = masks > 0
            if not valid.any():
                continue
            logits_flat = logits.permute(0, 2, 3, 1)[valid]  # [N, K]
            targets_flat = masks[valid]                       # [N]

            pred_raw = None
            for tag, T in (("raw", 1.0), ("calibrated", temperature)):
                scaled = logits_flat / T
                probs = torch.softmax(scaled, dim=1)
                conf, pred = probs.max(dim=1)
                correct = (pred == targets_flat).float()

                s = stats[tag]
                s["nll_sum"] += torch.nn.functional.cross_entropy(scaled, targets_flat, reduction="sum").item()
                s["n"] += targets_flat.numel()

                conf_cpu, correct_cpu = conf.cpu(), correct.cpu()
                for i in range(num_bins):
                    in_bin = (conf_cpu > bin_boundaries[i]) & (conf_cpu <= bin_boundaries[i + 1])
                    if in_bin.any():
                        s["bin_conf_sums"][i] += conf_cpu[in_bin].sum()
                        s["bin_acc_sums"][i] += correct_cpu[in_bin].sum()
                        s["bin_counts"][i] += in_bin.sum()

                if tag == "raw":
                    pred_raw = pred
                else:
                    mismatches += (pred != pred_raw).sum().item()

    results = {"mismatches": mismatches}
    for tag in ("raw", "calibrated"):
        s = stats[tag]
        n = max(s["n"], 1)
        bin_props = s["bin_counts"] / n
        bin_accs = torch.where(s["bin_counts"] > 0, s["bin_acc_sums"] / s["bin_counts"].clamp(min=1), torch.zeros(num_bins))
        bin_avg_conf = torch.where(s["bin_counts"] > 0, s["bin_conf_sums"] / s["bin_counts"].clamp(min=1), torch.zeros(num_bins))
        ece = float((bin_props * (bin_accs - bin_avg_conf).abs()).sum())
        results[tag] = {
            "ece": ece,
            "nll": s["nll_sum"] / n,
            "bin_accs": bin_accs.numpy(),
            "bin_props": bin_props.numpy(),
            "n_pixels": s["n"],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Post-hoc temperature scaling for the AS1 baseline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root containing embeddings/ (i.e. train_decoders.py's --data_dir / --out_dir), NOT the raw data/fbp images+labels dir")
    parser.add_argument("--decoder_checkpoint", type=str, required=True, help="Path to AS1's saved decoder .pth")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--decoder_embed_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None, help="Must match the value AS1 was trained with, if it used one")
    parser.add_argument("--max_val_pixels", type=int, default=2_000_000, help="Cap on pixels collected for temperature fitting")
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

    log_msg(f"Collecting up to {args.max_val_pixels:,} validation pixels...")
    val_logits, val_targets = collect_val_logits_for_temperature(decoder, val_loader, args.max_val_pixels)

    log_msg(f"Fitting temperature on {val_targets.numel():,} validation pixels...")
    T = fit_temperature(val_logits.to(DEVICE), val_targets.to(DEVICE))

    log_msg("Streaming test set calibration metrics (raw + calibrated in one pass)...")
    results_stream = stream_test_calibration(decoder, test_loader, temperature=T)

    mismatches = results_stream["mismatches"]
    total_pixels = results_stream["raw"]["n_pixels"]
    mismatch_frac = mismatches / max(total_pixels, 1)
    # Dividing every logit by the same positive T can't change the argmax in exact
    # arithmetic, but float32 across tens of millions of pixels can flip an extremely
    # rare near-exact tie by a fraction of an ulp - tolerate a negligible fraction of
    # these rather than requiring bit-exact equality; anything above this points to an
    # actual logic error, not floating-point noise.
    max_allowed_frac = 1e-5
    log_msg(f"Prediction mismatches after scaling: {mismatches} / {total_pixels:,} ({mismatch_frac:.2e})")
    assert mismatch_frac <= max_allowed_frac, \
        f"temperature scaling changed {mismatches}/{total_pixels} ({mismatch_frac:.2e}) predictions - " \
        f"exceeds the floating-point noise tolerance ({max_allowed_frac:.0e}), likely a real bug"

    raw, cal = results_stream["raw"], results_stream["calibrated"]
    log_msg(f"AS1 RAW:        ECE={raw['ece']:.4f} | NLL={raw['nll']:.4f} | n={raw['n_pixels']:,}")
    log_msg(f"AS1 CALIBRATED: ECE={cal['ece']:.4f} | NLL={cal['nll']:.4f} | T={T:.4f} | n={cal['n_pixels']:,}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": args.run_name,
        "temperature": T,
        "num_val_pixels": int(val_targets.numel()),
        "num_test_pixels": raw["n_pixels"],
        "raw_ece": raw["ece"], "raw_nll": raw["nll"],
        "calibrated_ece": cal["ece"], "calibrated_nll": cal["nll"],
    }
    results_path = os.path.join(runs_dir, f"{args.run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")

    plot_reliability_diagram(raw["bin_accs"], raw["bin_props"], save_name=f"{args.run_name}_raw_reliability")
    plot_reliability_diagram(cal["bin_accs"], cal["bin_props"], save_name=f"{args.run_name}_calibrated_reliability")


if __name__ == "__main__":
    main()
