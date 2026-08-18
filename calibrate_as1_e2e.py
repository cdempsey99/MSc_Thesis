"""
calibrate_as1_e2e.py — post-hoc temperature scaling for the reBEN/SAR AS1 baseline
(single finetuned encoder + single decoder, no diversity, no student).

Dataset-agnostic between reBEN (optical, S2) and SAR via --dataset, since both use the
same e2e (finetuned encoder) pipeline, unlike FBP's frozen-embedding AS1 - see
calibrate_as1_frozen.py for that version, which does not apply here (no encoder in that
pipeline at all). Fits one scalar T on the validation split via fit_temperature, then
applies it to the test split to report calibration metrics (ECE, NLL, reliability
diagram) both with and without scaling. mIoU is identical in both cases by construction
- temperature scaling can't change argmax predictions, only confidence shape.

Both stages stream through their DataLoader one batch at a time (running histograms/NLL
sums, discarding each batch's logits immediately) rather than accumulating full-set
logits - reBEN/SAR test sets are ~120K patches, large enough that "collect everything"
would repeat the OOM mistake calibrate_as1_frozen.py originally made on FBP, just worse.

This is meaningfully more expensive than the FBP version: every patch needs a real
encoder forward pass (no pre-baked embeddings here), over test sets ~5x larger than
FBP's - expect this to run closer to the multi-hour eval passes discussed for reBEN
elsewhere, not the few-minutes FBP run.

Usage (reBEN example):
    python calibrate_as1_e2e.py \
        --dataset reben \
        --metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
        --img_root /beegfs/scratch/callumdempsey/data/reben/BigEarthNet-S2 \
        --ref_root /beegfs/scratch/callumdempsey/data/reben/Reference_Maps \
        --decoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/<AS1_run>_best_decoder.pth \
        --encoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/<AS1_run>_best_encoder.pth \
        --n_unfrozen_blocks <value from the AS1 job log> \
        --decoder_embed_dim <value from the AS1 job log> \
        --run_name AS1_reben_calibration

SAR example: same, but --dataset sar, --img_root points at the SAR root, and optionally
--despeckle / --lee_filter if the AS1 run used them (check the job log).
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from utils.dataset_e2e import load_reben_splits, load_reben_sar_splits, ReBENRawDataset, ReBENSARRawDataset
from utils.misc import log_msg, fit_temperature
from utils.visualisation import plot_reliability_diagram
from models.ensemble import DecoderEnsemble
from models.encoder import initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAVES = {
    "reben": ReBENRawDataset.WAVELENGTHS,
    "sar": ReBENSARRawDataset.WAVELENGTHS,
}


def load_splits(args):
    if args.dataset == "reben":
        return load_reben_splits(
            metadata_path=args.metadata_path, s2_root=args.img_root, ref_root=args.ref_root,
            exclude_snow=not args.include_snow, exclude_cloud=not args.include_cloud,
            max_patches=args.max_patches, max_val_patches=args.max_val_patches,
        )
    else:
        return load_reben_sar_splits(
            metadata_path=args.metadata_path, s1_root=args.img_root, ref_root=args.ref_root,
            exclude_snow=not args.include_snow, exclude_cloud=not args.include_cloud,
            max_patches=args.max_patches, max_val_patches=args.max_val_patches,
            despeckle=args.despeckle, lee_filter_despeckle=args.lee_filter,
        )


def collect_val_logits_for_temperature(encoder_model, decoder, loader, waves, max_pixels=2_000_000):
    """
    Streams the validation loader, running the real encoder forward pass each batch,
    stopping once max_pixels labelled pixels are collected. Temperature scaling fits a
    single scalar, so a capped subsample is standard practice and statistically
    sufficient - it does not need every pixel in the split.
    """
    all_logits, all_targets = [], []
    collected = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            with torch.cuda.amp.autocast():
                features = get_encoder_representation_partial(imgs, encoder_model, waves=waves)
                logits = decoder(features)[0].float()  # M=1 for AS1 -> single head
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


def stream_test_calibration(encoder_model, decoder, test_loader, waves, temperature, num_bins=10):
    """
    One pass over the test set computing raw (T=1) and temperature-scaled calibration
    metrics simultaneously (each patch's real encoder forward pass only run once), so
    the (large) test set only needs to be read once. Never stores per-pixel data -
    accumulates running confidence/accuracy histograms and a running NLL sum per batch,
    matching the streaming pattern already used by evaluate_test_set_reben elsewhere in
    this codebase. Also counts prediction mismatches between raw and calibrated (should
    always be ~0, by construction) as a correctness check.
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
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE)
            with torch.cuda.amp.autocast():
                features = get_encoder_representation_partial(imgs, encoder_model, waves=waves)
                logits = decoder(features)[0].float()
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
    parser = argparse.ArgumentParser(description="Post-hoc temperature scaling for the reBEN/SAR AS1 baseline")
    parser.add_argument("--dataset", choices=["reben", "sar"], required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True, help="s2_root for reben, s1_root for sar")
    parser.add_argument("--ref_root", type=str, required=True)
    parser.add_argument("--decoder_checkpoint", type=str, required=True)
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--n_unfrozen_blocks", type=int, required=True,
                        help="Must match the AS1 run's value - check its job log, do not assume")
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--decoder_embed_dim", type=int, required=True,
                        help="Must match the AS1 run's value - check its job log, do not assume")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--max_val_patches", type=int, default=None)
    parser.add_argument("--include_snow", action="store_true")
    parser.add_argument("--include_cloud", action="store_true")
    parser.add_argument("--despeckle", action="store_true", help="SAR only")
    parser.add_argument("--lee_filter", action="store_true", help="SAR only")
    parser.add_argument("--max_val_pixels", type=int, default=2_000_000)
    parser.add_argument("--run_name", type=str, default="AS1_calibration")
    args = parser.parse_args()

    waves = WAVES[args.dataset]

    log_msg(f"Loading {args.dataset} splits...")
    _, val_ds, test_ds = load_splits(args)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    log_msg(f"Val: {len(val_ds)} patches | Test: {len(test_ds)} patches")

    encoder_model = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=args.n_unfrozen_blocks)
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=DEVICE)
    encoder_model.load_state_dict(enc_ckpt['encoder_state_dict'], strict=False)
    encoder_model.to(DEVICE)
    encoder_model.eval()
    log_msg(f"Loaded encoder from {args.encoder_checkpoint}")

    decoder = DecoderEnsemble(M=args.ensemble_size, in_channels=1024, embed_dim=args.decoder_embed_dim,
                              num_classes=args.num_classes)
    dec_ckpt = torch.load(args.decoder_checkpoint, map_location=DEVICE)
    decoder.load_state_dict(dec_ckpt['model_state_dict'])
    decoder.to(DEVICE)
    decoder.eval()
    log_msg(f"Loaded decoder from {args.decoder_checkpoint}")

    log_msg(f"Collecting up to {args.max_val_pixels:,} validation pixels...")
    val_logits, val_targets = collect_val_logits_for_temperature(
        encoder_model, decoder, val_loader, waves, args.max_val_pixels
    )

    log_msg(f"Fitting temperature on {val_targets.numel():,} validation pixels...")
    T = fit_temperature(val_logits.to(DEVICE), val_targets.to(DEVICE))

    log_msg("Streaming test set calibration metrics (raw + calibrated in one pass)...")
    results_stream = stream_test_calibration(encoder_model, decoder, test_loader, waves, temperature=T)

    mismatches = results_stream["mismatches"]
    total_pixels = results_stream["raw"]["n_pixels"]
    mismatch_frac = mismatches / max(total_pixels, 1)
    max_allowed_frac = 1e-5
    log_msg(f"Prediction mismatches after scaling: {mismatches} / {total_pixels:,} ({mismatch_frac:.2e})")
    assert mismatch_frac <= max_allowed_frac, \
        f"temperature scaling changed {mismatches}/{total_pixels} ({mismatch_frac:.2e}) predictions - " \
        f"exceeds the floating-point noise tolerance ({max_allowed_frac:.0e}), likely a real bug"

    raw, cal = results_stream["raw"], results_stream["calibrated"]
    log_msg(f"{args.dataset.upper()} AS1 RAW:        ECE={raw['ece']:.4f} | NLL={raw['nll']:.4f} | n={raw['n_pixels']:,}")
    log_msg(f"{args.dataset.upper()} AS1 CALIBRATED: ECE={cal['ece']:.4f} | NLL={cal['nll']:.4f} | T={T:.4f} | n={cal['n_pixels']:,}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": args.run_name,
        "dataset": args.dataset,
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
