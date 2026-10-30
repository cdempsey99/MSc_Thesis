"""
calibrate_as1_e2e.py — post-hoc temperature scaling for the reBEN/SAR AS1 baseline
(single finetuned encoder + single decoder, no diversity, no student).

Dataset-agnostic between reBEN (optical, S2) and SAR via --dataset, since both use the
same e2e (finetuned encoder) pipeline, unlike FBP's frozen-embedding AS1 - see
calibrate_as1_frozen.py for that version, which does not apply here (no encoder in that
pipeline at all). Fits one scalar T on the validation split via fit_temperature, then
applies it to the test split and reports the full reeval_auroc.py metric set (acc, mIoU,
ECE, NLL, total AUROC, ...) both with and without scaling. mIoU is identical in both cases
by construction - temperature scaling can't change argmax predictions, only confidence shape.

Test metrics come from reeval_auroc.py's _Tracker, so they follow the evaluators' conventions
exactly (ECE on patches with >= 100 labelled pixels, the seeded ~5M-pixel AUROC sample) and the
raw numbers match a fresh evaluation of the same checkpoint. Raw and calibrated are scored on the
same AUROC pixels. --calibration_dir also saves <run_name>__raw.npz / __calibrated.npz in
reeval_auroc.py's format, so plot_calibration.py can draw both.

T is fitted on, and applied to, log(mean head probability). For M=1 that is exactly standard
temperature scaling of the logits; for M>1 (--ensemble_size) it scales the ensemble's averaged
prediction. For M>1 the calibrated result reports no aleatoric/epistemic split (scaling the
averaged prediction does not define per-head distributions); the raw result keeps it.

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
        --decoder_checkpoint /home/users/c/callumdempsey/results/checkpoints/<AS1_run>_best_decoder.pth \
        --encoder_checkpoint /home/users/c/callumdempsey/results/checkpoints/<AS1_run>_best_encoder.pth \
        --n_unfrozen_blocks <value from the AS1 job log> \
        --decoder_embed_dim <value from the AS1 job log> \
        --clay_native_forward --clay_normalise_s2 \
        --calibration_dir /home/users/c/callumdempsey/results/calibration/reben_as1_ts \
        --run_name AS1_reben_calibration

(_best_ or _last_ / _final_ checkpoints, whichever the AS1 numbers being compared against used; the
encoder flags must match how the AS1 run was trained; --max_patches as in training for subset runs.)

SAR example: same, but --dataset sar, --img_root points at the SAR root, and optionally
--despeckle / --lee_filter if the AS1 run used them (check the job log).
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset_e2e import load_reben_splits, load_reben_sar_splits, ReBENRawDataset, ReBENSARRawDataset
from utils.misc import log_msg, fit_temperature, auroc_pixels_per_batch, auroc_pixel_sample
from utils.visualisation import plot_reliability_diagram
from models.ensemble import DecoderEnsemble
from models.encoder import initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial, REBEN_GSD
from reeval_auroc import _Tracker

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
            clay_normalise=args.clay_normalise_s2,
        )
    else:
        return load_reben_sar_splits(
            metadata_path=args.metadata_path, s1_root=args.img_root, ref_root=args.ref_root,
            exclude_snow=not args.include_snow, exclude_cloud=not args.include_cloud,
            max_patches=args.max_patches, max_val_patches=args.max_val_patches,
            despeckle=args.despeckle, lee_filter_despeckle=args.lee_filter,
        )


def ensemble_log_probs(preds):
    """[M, B, C, H, W] head logits -> [B, C, H, W] log of the mean head probability. For M=1 this
    is log_softmax(logits), and softmax(log_softmax(z) / T) == softmax(z / T), so fitting/applying T
    to it is exactly standard temperature scaling."""
    return torch.logsumexp(torch.log_softmax(preds.float(), dim=2), dim=0) - math.log(preds.shape[0])


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
                preds = decoder(features)
            logits = ensemble_log_probs(preds)  # [B, C, H, W]
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


def stream_test_calibration(encoder_model, decoder, test_loader, waves, temperature, num_classes):
    """
    One pass over the test set scoring raw (T=1) and temperature-scaled predictions
    simultaneously (each patch's real encoder forward pass only run once), so the (large)
    test set only needs to be read once. Metrics are accumulated by reeval_auroc.py's _Tracker
    (running sums, plus the seeded AUROC pixel sample), with the same sample used for raw and
    calibrated. Also counts prediction mismatches between raw and calibrated (should always be
    ~0, by construction) as a correctness check.
    """
    M = decoder.M
    per_batch = auroc_pixels_per_batch(test_loader)
    gen = torch.Generator().manual_seed(0)   # same seed as the fixed evaluators
    trackers = {tag: _Tracker(num_classes, M, fbp_style=False) for tag in ("raw", "calibrated")}
    mismatches, n_scored = 0, 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE)
            with torch.cuda.amp.autocast():
                features = get_encoder_representation_partial(imgs, encoder_model, waves=waves)
                preds = decoder(features)                 # [M, B, C, H, W]
            y = masks.to(DEVICE).long()
            if y.dim() == 4:
                y = y.squeeze(1)
            lab = y > 0
            if not lab.any():
                continue
            conf_lab = lab & (lab.flatten(1).sum(dim=1) >= 100).view(-1, 1, 1)
            smp = auroc_pixel_sample(lab, per_batch, gen)
            log_mean_p = ensemble_log_probs(preds)        # [B, C, H, W]

            pred_raw = None
            for tag, T in (("raw", 1.0), ("calibrated", temperature)):
                mean_p = torch.softmax(log_mean_p / T, dim=1)
                tot = -(mean_p * torch.log(mean_p.clamp(min=1e-10))).sum(dim=1)
                if tag == "raw" and M > 1:
                    probs = torch.softmax(preds.float(), dim=2)
                    ale = (-(probs * torch.log(probs.clamp(1e-10))).sum(dim=2)).mean(dim=0)
                    trackers[tag].add(mean_p, y, lab, conf_lab, smp, tot, ale, head_pred=probs.argmax(dim=2))
                    del probs
                else:
                    trackers[tag].add(mean_p, y, lab, conf_lab, smp, tot, tot)

                pred = mean_p.argmax(dim=1)[lab]
                if tag == "raw":
                    pred_raw = pred
                    n_scored += pred.numel()
                else:
                    mismatches += (pred != pred_raw).sum().item()

    results = {"mismatches": mismatches, "n_pixels": n_scored}
    for tag, tr in trackers.items():
        summ = tr.summary(ensemble=(M > 1 and tag == "raw"))
        if tag == "calibrated" and M > 1:   # no per-head distributions after scaling the mean
            for k in ("mean_aleatoric", "mean_epistemic", "epistemic_share"):
                summ.pop(k, None)
        n = max(tr.bin_n.sum().item(), 1)
        summ["bin_accs"] = (tr.bin_acc / tr.bin_n.clamp(min=1)).cpu().numpy()
        summ["bin_props"] = (tr.bin_n / n).cpu().numpy()
        results[tag] = summ
        results[f"{tag}_arrays"] = tr.calibration_arrays()
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
    parser.add_argument("--calibration_dir", type=str, default=None,
                        help="Also save <run_name>__raw.npz and <run_name>__calibrated.npz (reeval_auroc.py's "
                             "format) for plot_calibration.py")
    parser.add_argument("--clay_native_forward", action="store_true",
                        help="Clay position encoding, cls token, single final norm - must match how the checkpoint "
                             "was trained (off = original path)")
    parser.add_argument("--clay_normalise_s2", action="store_true",
                        help="reBEN optical only: Clay sentinel-2-l2a z-score inputs - must match training")
    parser.add_argument("--clay_single_norm", action="store_true",
                        help="Skip the second final LayerNorm - must match training")
    args = parser.parse_args()

    waves = WAVES[args.dataset]

    log_msg(f"Loading {args.dataset} splits...")
    _, val_ds, test_ds = load_splits(args)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    log_msg(f"Val: {len(val_ds)} patches | Test: {len(test_ds)} patches")

    encoder_model = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=args.n_unfrozen_blocks,
                                                             single_norm=args.clay_single_norm,
                                                             native_forward=args.clay_native_forward, gsd=REBEN_GSD)
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
    results_stream = stream_test_calibration(encoder_model, decoder, test_loader, waves, temperature=T,
                                             num_classes=args.num_classes)

    mismatches = results_stream["mismatches"]
    total_pixels = results_stream["n_pixels"]
    mismatch_frac = mismatches / max(total_pixels, 1)
    max_allowed_frac = 1e-5
    log_msg(f"Prediction mismatches after scaling: {mismatches} / {total_pixels:,} ({mismatch_frac:.2e})")
    assert mismatch_frac <= max_allowed_frac, \
        f"temperature scaling changed {mismatches}/{total_pixels} ({mismatch_frac:.2e}) predictions - " \
        f"exceeds the floating-point noise tolerance ({max_allowed_frac:.0e}), likely a real bug"

    raw, cal = results_stream["raw"], results_stream["calibrated"]
    tag = args.dataset.upper()
    for name, r, extra in (("RAW:       ", raw, ""), ("CALIBRATED:", cal, f" | T={T:.4f}")):
        log_msg(f"{tag} AS1 {name} mIoU={r['miou']:.4f} | Acc={r['acc']:.4f} | ECE={r['ece']:.4f} | NLL={r['nll']:.4f} | "
                f"AUROC total={r['auroc_total']:.4f}{extra}")
    if "auroc_epistemic" in raw:
        log_msg(f"{tag} RAW ensemble: AUROC epistemic={raw['auroc_epistemic']:.4f} aleatoric={raw['auroc_aleatoric']:.4f}"
                f" | epistemic share={raw['epistemic_share']:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    def _jsonable(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    results = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "ensemble_size": decoder.M,
        "temperature": T,
        "num_val_pixels": int(val_targets.numel()),
        "num_test_pixels": total_pixels,
        "raw_ece": raw["ece"], "raw_nll": raw["nll"],
        "calibrated_ece": cal["ece"], "calibrated_nll": cal["nll"],
        "raw": _jsonable(raw),
        "calibrated": _jsonable(cal),
    }
    results_path = os.path.join(runs_dir, f"{args.run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")

    if args.calibration_dir:
        os.makedirs(args.calibration_dir, exist_ok=True)
        for who in ("raw", "calibrated"):
            np.savez_compressed(os.path.join(args.calibration_dir, f"{args.run_name}__{who}.npz"),
                                run=args.run_name, who=who, tag="calibrate_as1_e2e",
                                **results_stream[f"{who}_arrays"])
        log_msg(f"Calibration arrays saved to {args.calibration_dir}")

    plot_reliability_diagram(raw["bin_accs"], raw["bin_props"], save_name=f"{args.run_name}_raw_reliability")
    plot_reliability_diagram(cal["bin_accs"], cal["bin_props"], save_name=f"{args.run_name}_calibrated_reliability")


if __name__ == "__main__":
    main()
