"""
calibrate_as1_frozen.py — post-hoc temperature scaling for the AS1 baseline (single encoder +
single decoder, no diversity, no student).

Fits one scalar T on the validation split (never on test) via fit_temperature, then
applies it to the test split to report calibration metrics (ECE, NLL, AUROC, reliability
diagram) both with and without scaling, plus mIoU/fw-IoU/Acc (identical raw vs calibrated
by construction - temperature scaling can't change argmax predictions, only confidence
shape). This is the baseline the diversity-ensemble / distillation UQ claims elsewhere in
the thesis need to beat, not just match AS1 itself.

Validation pixels are now collected PER IMAGE (capped independently per file) rather than
as one flat stream capped globally - BakedFeatureDataset lays every file's patches out
contiguously, so with shuffle=False the original global cap could be (and likely was)
exhausted within the very first val image alone, meaning T was effectively fit on one
image's pixels while being described as "22 validation images". Per-image collection also
enables an optional image-level bootstrap (--bootstrap_n) to check whether the fitted T is
actually stable across the val split or just noise from a small/correlated pixel pool -
image-level, not pixel-level, since pixels within one image are highly correlated and a
naive per-pixel bootstrap would understate the real small-sample risk.

Both stages stream through their DataLoader one batch at a time and never hold more
than a capped number of pixels (validation) or one batch (test) in memory - FBP's
~7300x7300 images mean "every labelled pixel in the split" is tens of GB, not something
to naively collect.

Usage:
    python calibrate_as1_frozen.py \
        --data_dir /beegfs/scratch/callumdempsey/results \
        --decoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/AS1_..._best_model.pth \
        --decoder_embed_dim 512 --num_classes 25 \
        --patch_size 224 --stride 224 --max_images 150 \
        --run_name AS1_calibration \
        --bootstrap_n 100
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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


def collect_val_logits_per_image(decoder, val_files, batch_size, max_pixels_per_image=None):
    """
    Collects validation logits/targets grouped by SOURCE IMAGE, each independently capped,
    instead of one flat pool capped globally. See module docstring for why: with
    shuffle=False and BakedFeatureDataset laying every file's patches out contiguously, a
    single global pixel cap is almost certainly exhausted within the first val image alone
    (FBP images are huge - roughly 1000+ 224x224 patches/image, tens of millions of
    labelled pixels/image at typical class density), silently fitting T on a single
    image's pixels while being described as "22 validation images".

    Returns a list of (logits [N_i, K], targets [N_i], image_name) tuples, one per val
    image that actually contributed pixels - kept as a list (not concatenated) so callers
    can do image-level bootstrap resampling.
    """
    per_image = []
    for f in val_files:
        loader = DataLoader(BakedFeatureDataset([f], augment=False), batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
        logits_list, targets_list = [], []
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
                logits_list.append(logits_flat.cpu())
                targets_list.append(targets_flat.cpu())
                collected += targets_flat.numel()
                if max_pixels_per_image is not None and collected >= max_pixels_per_image:
                    break
        if logits_list:
            per_image.append((torch.cat(logits_list), torch.cat(targets_list), f.name))
    return per_image


def bootstrap_temperature_stability(per_image, n_bootstrap):
    """
    Refits T on n_bootstrap resamples of the VALIDATION IMAGES (not pixels) drawn with
    replacement - a cluster/block bootstrap over images. Pixels within the same image are
    highly correlated (shared scene content, class mix, lighting), so a naive per-pixel
    bootstrap over millions of pixels drawn from a fixed small set of images would barely
    perturb the fit each resample (law of large numbers) and give a falsely narrow,
    stable-looking T distribution - it would not capture the real small-sample risk, which
    lives at the image level. Resampling whole images is the correct unit here.

    Purely diagnostic: quantifies sampling uncertainty in the existing val pool, doesn't
    introduce any new data. A wide spread means the original single-shot T shouldn't be
    trusted as a stable estimate; a narrow spread means it probably can be.
    """
    n_images = len(per_image)
    temperatures = []
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n_images, size=n_images)
        boot_logits = torch.cat([per_image[j][0] for j in idx]).to(DEVICE)
        boot_targets = torch.cat([per_image[j][1] for j in idx]).to(DEVICE)
        T = fit_temperature(boot_logits, boot_targets)
        temperatures.append(T)
        if (i + 1) % max(1, n_bootstrap // 10) == 0:
            log_msg(f"  Bootstrap fit {i + 1}/{n_bootstrap} done (T={T:.4f})")
    temperatures = np.array(temperatures)
    return {
        "n_bootstrap": n_bootstrap,
        "n_images": n_images,
        "mean": float(temperatures.mean()),
        "std": float(temperatures.std()),
        "min": float(temperatures.min()),
        "max": float(temperatures.max()),
        "p5": float(np.percentile(temperatures, 5)),
        "p95": float(np.percentile(temperatures, 95)),
        "all_temperatures": temperatures.tolist(),
    }


def stream_test_calibration(decoder, test_loader, temperature, num_classes, num_bins=10, auroc_max=2_000_000):
    """
    One pass over the test set computing raw (T=1) and temperature-scaled calibration
    metrics simultaneously, so the (large) test set only needs to be read once. Never
    stores per-pixel data beyond a capped AUROC sample and a fixed-size confusion matrix -
    accumulates running confidence/accuracy histograms and a running NLL sum per batch,
    discarding each batch's logits immediately, matching the streaming pattern already
    used by evaluate_test_set elsewhere in this codebase. Also counts prediction
    mismatches between raw and calibrated (should always be zero, by construction) as a
    correctness check.

    AUROC uses the same recipe as evaluate_test_set elsewhere (roc_auc_score(is_wrong,
    entropy), capped at auroc_max pixels) - entropy of softmax(logits/T) as the
    uncertainty score. Note this genuinely can differ between raw and calibrated (unlike
    mIoU): dividing every pixel's logits by the same T doesn't preserve entropy's RANK
    ORDER across different pixels with different original logit distributions, so it's
    not just a monotonic relabelling - worth computing separately, not assumed equal.

    mIoU/fw-IoU/Acc are computed once (from the confusion matrix built off "raw"
    predictions) since argmax - and therefore every prediction-based stat - is identical
    for raw and calibrated by construction.
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    stats = {
        tag: {
            "bin_conf_sums": torch.zeros(num_bins),
            "bin_acc_sums": torch.zeros(num_bins),
            "bin_counts": torch.zeros(num_bins),
            "nll_sum": 0.0,
            "n": 0,
            "auroc_entropy": [],
            "auroc_errors": [],
            "auroc_collected": 0,
        }
        for tag in ("raw", "calibrated")
    }
    mismatches = 0
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

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

                if s["auroc_collected"] < auroc_max:
                    entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=1)
                    err = 1.0 - correct
                    s["auroc_entropy"].append(entropy.cpu().numpy())
                    s["auroc_errors"].append(err.cpu().numpy())
                    s["auroc_collected"] += err.numel()

                if tag == "raw":
                    pred_raw = pred
                    np.add.at(conf_matrix, (targets_flat.cpu().numpy(), pred.cpu().numpy()), 1)
                else:
                    mismatches += (pred != pred_raw).sum().item()

    # mIoU / fw-IoU / Acc from confusion matrix - only average over classes present in
    # ground truth, identical for raw and calibrated by construction (see docstring).
    iou_per_class = np.zeros(num_classes - 1)
    present = []
    for c in range(1, num_classes):
        if conf_matrix[c, :].sum() > 0:
            tp = conf_matrix[c, c]
            fp = conf_matrix[:, c].sum() - tp
            fn = conf_matrix[c, :].sum() - tp
            denom = tp + fp + fn
            iou_per_class[c - 1] = tp / denom if denom > 0 else 0.0
            present.append(c - 1)
    miou = float(np.mean(iou_per_class[present])) if present else 0.0
    class_pixel_counts = conf_matrix[1:, :].sum(axis=1)
    total_labelled = class_pixel_counts.sum()
    fw_iou = float((class_pixel_counts / max(total_labelled, 1) * iou_per_class).sum())
    acc = float(np.diag(conf_matrix)[1:].sum() / max(conf_matrix[1:, :].sum(), 1))

    results = {"mismatches": mismatches, "miou": miou, "fw_iou": fw_iou, "acc": acc}
    for tag in ("raw", "calibrated"):
        s = stats[tag]
        n = max(s["n"], 1)
        bin_props = s["bin_counts"] / n
        bin_accs = torch.where(s["bin_counts"] > 0, s["bin_acc_sums"] / s["bin_counts"].clamp(min=1), torch.zeros(num_bins))
        bin_avg_conf = torch.where(s["bin_counts"] > 0, s["bin_conf_sums"] / s["bin_counts"].clamp(min=1), torch.zeros(num_bins))
        ece = float((bin_props * (bin_accs - bin_avg_conf).abs()).sum())

        all_ent = np.concatenate(s["auroc_entropy"]) if s["auroc_entropy"] else np.array([])
        all_err = np.concatenate(s["auroc_errors"]) if s["auroc_errors"] else np.array([])
        auroc = float(roc_auc_score(all_err, all_ent)) if len(np.unique(all_err)) > 1 else 0.0

        results[tag] = {
            "ece": ece,
            "nll": s["nll_sum"] / n,
            "auroc": auroc,
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
    parser.add_argument("--max_val_pixels", type=int, default=2_000_000,
                        help="Total cap on pixels collected for temperature fitting, split evenly across val images "
                             "(so every val image contributes, not just whichever loads first)")
    parser.add_argument("--bootstrap_n", type=int, default=0,
                        help="If >0, refit T on this many image-level bootstrap resamples of the val split and log "
                             "the spread, to check whether the single-shot T is a stable estimate or just noise "
                             "from a small/correlated pixel pool. Off (0) by default - adds bootstrap_n extra "
                             "LBFGS fits, each logged individually by fit_temperature.")
    parser.add_argument("--run_name", type=str, default="AS1_calibration")
    args = parser.parse_args()

    val_files, test_files = build_val_test_split(args.data_dir, args.patch_size, args.stride, args.max_images)
    log_msg(f"Split: {len(val_files)} Val | {len(test_files)} Test")

    test_loader = DataLoader(BakedFeatureDataset(test_files, augment=False),
                             batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    decoder = DecoderEnsemble(M=1, in_channels=1024, embed_dim=args.decoder_embed_dim,
                              num_classes=args.num_classes)
    ckpt = torch.load(args.decoder_checkpoint, map_location=DEVICE)
    decoder.load_state_dict(ckpt['model_state_dict'])
    decoder.to(DEVICE)
    decoder.eval()
    log_msg(f"Loaded AS1 decoder from {args.decoder_checkpoint}")

    max_pixels_per_image = max(args.max_val_pixels // max(len(val_files), 1), 1)
    log_msg(f"Collecting up to {max_pixels_per_image:,} validation pixels PER IMAGE "
            f"({args.max_val_pixels:,} total budget / {len(val_files)} val images)...")
    per_image = collect_val_logits_per_image(decoder, val_files, args.batch_size, max_pixels_per_image)
    total_val_pixels = sum(t.numel() for _, t, _ in per_image)
    log_msg(f"Collected {total_val_pixels:,} val pixels from {len(per_image)}/{len(val_files)} images "
            f"(images with zero labelled pixels in their capped sample are dropped)")

    val_logits = torch.cat([l for l, t, n in per_image])
    val_targets = torch.cat([t for l, t, n in per_image])

    log_msg(f"Fitting temperature on {val_targets.numel():,} validation pixels...")
    T = fit_temperature(val_logits.to(DEVICE), val_targets.to(DEVICE))

    bootstrap_results = None
    if args.bootstrap_n > 0:
        log_msg(f"Running image-level bootstrap ({args.bootstrap_n} resamples over {len(per_image)} images) "
                f"to test T-fit stability...")
        bootstrap_results = bootstrap_temperature_stability(per_image, args.bootstrap_n)
        log_msg(
            f"Bootstrap T: mean={bootstrap_results['mean']:.4f} std={bootstrap_results['std']:.4f} "
            f"min={bootstrap_results['min']:.4f} max={bootstrap_results['max']:.4f} "
            f"[5th-95th pct: {bootstrap_results['p5']:.4f}-{bootstrap_results['p95']:.4f}] "
            f"(single-shot fit was T={T:.4f})"
        )

    log_msg("Streaming test set calibration metrics (raw + calibrated in one pass)...")
    results_stream = stream_test_calibration(decoder, test_loader, temperature=T, num_classes=args.num_classes)

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
    log_msg(f"AS1 mIoU={results_stream['miou']:.4f} | fw-IoU={results_stream['fw_iou']:.4f} | "
            f"Acc={results_stream['acc']:.4f} (identical raw vs calibrated by construction)")
    log_msg(f"AS1 RAW:        ECE={raw['ece']:.4f} | NLL={raw['nll']:.4f} | AUROC={raw['auroc']:.4f} | n={raw['n_pixels']:,}")
    log_msg(f"AS1 CALIBRATED: ECE={cal['ece']:.4f} | NLL={cal['nll']:.4f} | AUROC={cal['auroc']:.4f} | T={T:.4f} | n={cal['n_pixels']:,}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": args.run_name,
        "temperature": T,
        "num_val_pixels": int(val_targets.numel()),
        "num_val_images_used": len(per_image),
        "num_val_images_available": len(val_files),
        "num_test_pixels": raw["n_pixels"],
        "miou": results_stream["miou"],
        "fw_iou": results_stream["fw_iou"],
        "acc": results_stream["acc"],
        "raw_ece": raw["ece"], "raw_nll": raw["nll"], "raw_auroc": raw["auroc"],
        "calibrated_ece": cal["ece"], "calibrated_nll": cal["nll"], "calibrated_auroc": cal["auroc"],
        "bootstrap": bootstrap_results,
    }
    results_path = os.path.join(runs_dir, f"{args.run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")

    plot_reliability_diagram(raw["bin_accs"], raw["bin_props"], save_name=f"{args.run_name}_raw_reliability")
    plot_reliability_diagram(cal["bin_accs"], cal["bin_props"], save_name=f"{args.run_name}_calibrated_reliability")


if __name__ == "__main__":
    main()
