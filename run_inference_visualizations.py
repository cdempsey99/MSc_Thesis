"""
run_inference_visualizations.py

Standalone inference/visualisation pass over a trained checkpoint's TEST split - no training
loop, no full metric recomputation (mIoU/ECE/etc.). Built to pull many more qualitative example
patches (heatmap panels) than the 6 the training/eval scripts generate automatically, for
picking thesis figures without rerunning a full multi-hour eval job.

Dataset-agnostic via --dataset {fbp, reben, reben_sar}: each dataset needs a different loading
path (FBP is precomputed frozen embeddings; reBEN/reBEN SAR are raw images through a finetuned
encoder+partial-unfreeze), but renders through the same utils.visualisation panel functions
either way.

Student-only by default (shared encoder + StudentHead, matching the "run inference through
just the student, skip the M teacher decoder heads" goal) - pass --include_teacher to also
render teacher-ensemble panels (visualise_all_metrics) alongside the student ones
(visualise_student_uncertainty).

Restricted to the TEST split only (each dataset's own established split logic, same seed/
official split as training/eval) - no train/val option.

By default patches are picked randomly via the same rejection-sampling the training/eval
scripts use (--num_vis controls how many). Pass --patch_ids <idx> [<idx> ...] instead to
render specific, chosen global test-split indices - e.g. to run the exact same patch through
several different checkpoints for a like-for-like comparison figure.

Set OUT_DIR before running (not a CLI flag here) - configs.config.BASE_OUT reads it once at
import time, same as every other script in this project.

Usage (FBP, student-only):
    export OUT_DIR=/beegfs/scratch/callumdempsey/results
    export SCRATCH_DATA=/beegfs/scratch/callumdempsey/data/fbp
    python run_inference_visualizations.py --dataset fbp \
        --data_dir $OUT_DIR --checkpoint_path .../AS4_..._final_student_....pth \
        --decoder_embed_dim 512 --num_vis 40 --run_name AS4_fbp_corrected_inference

Usage (reBEN, student + teacher):
    python run_inference_visualizations.py --dataset reben \
        --s2_root ... --ref_root ... --metadata_path ... \
        --checkpoint_path .../..._last_student.pth \
        --encoder_checkpoint_path .../..._last_encoder.pth \
        --include_teacher --teacher_checkpoint_path .../..._last_decoder.pth --ensemble_size 10 \
        --num_vis 40 --run_name AS4_reben_inference
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.misc import log_msg, get_decoder_output_maps, _select_visualization_patches
from utils.visualisation import visualise_all_metrics, visualise_student_uncertainty
from models.ensemble import DecoderEnsemble, StudentHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _student_uncertainty_full(alphas):
    """Like utils.misc.dirichlet_uncertainty_and_pred, but also returns aleatoric and alpha0 -
    needed for visualise_student_uncertainty's extra panels, which the shared function (fixed
    3-tuple return, used by evaluate_error_localization/evaluate_uncertainty_correlation) doesn't
    expose. Same formula, just not throwing away two of the intermediate values."""
    alpha0 = alphas.sum(dim=1, keepdim=True)
    mean_probs = alphas / alpha0
    alpha0_sq = alpha0.squeeze(1)
    total_ent = -(mean_probs * torch.log(mean_probs.clamp(min=1e-10))).sum(dim=1)
    aleatoric = (torch.digamma(alpha0_sq + 1) - (mean_probs * torch.digamma(alphas + 1)).sum(dim=1))
    epistemic = (total_ent - aleatoric).clamp(min=0)
    pred_class = torch.argmax(mean_probs, dim=1).cpu().numpy()
    return total_ent, aleatoric, epistemic, alpha0_sq, pred_class


def _build_fbp_test_dataset(data_dir, patch_size, stride, max_images):
    """Mirrors train_decoders.py's / reeval_vb_checkpoint_frozen.py's split exactly (same glob,
    same seed=42 shuffle, same 70/15/15 cut) - only the test portion is needed here."""
    from utils.dataset import BakedFeatureDataset
    embedding_dir = Path(data_dir) / "embeddings" / "fbp" / "clay_v1" / f"patch{patch_size}_stride{stride}"
    all_files = sorted(embedding_dir.glob("*_embeddings.pt"))
    if max_images is not None:
        all_files = all_files[:max_images]
    random.seed(42)
    random.shuffle(all_files)
    n = len(all_files)
    test_files = all_files[2:] if n < 5 else all_files[int(n * 0.85):]
    log_msg(f"FBP test split: {len(test_files)} images")
    return BakedFeatureDataset(test_files, augment=False)


def _fbp_raw_patch(test_ds, g_idx, stride):
    """Mirrors utils.misc.evaluate_test_set's raw-image lookup exactly (same SCRATCH_DATA env,
    same patch-grid math) so panels here match what the training eval produces."""
    img_pt_path, local_idx = test_ds.get_patch_info(g_idx)
    img_stem = img_pt_path.stem.replace('_embeddings', '')
    scratch_data_env = os.getenv("SCRATCH_DATA")
    if scratch_data_env:
        data_dir = Path(scratch_data_env)
    else:
        data_dir = img_pt_path.parent.parent.parent.parent.parent / "data" / "fbp"
    raw_img_path = data_dir / f"{img_stem}.tif"
    patches_per_row = len(range(0, 7300 - 224, stride))
    x = (local_idx % patches_per_row) * stride
    y = (local_idx // patches_per_row) * stride
    if not raw_img_path.exists():
        return None, f"{img_stem} x={x} y={y}"
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(raw_img_path) as src:
        img = src.read([1, 2, 3], window=Window(x, y, 224, 224)).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return np.transpose(img, (1, 2, 0)), f"{img_stem} x={x} y={y}"


def _reben_raw_patch(raw_input, dataset_kind):
    """reBEN/SAR test datasets return the model's own input tensor directly (no separate raw-
    image file to look up) - denormalise it for display. SAR is 2-channel VV/VH with no natural
    RGB rendering, so it's skipped (returns None) rather than faking a false-colour composite."""
    if dataset_kind == "reben_sar":
        return None
    img = raw_input.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return np.transpose(img, (1, 2, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["fbp", "reben", "reben_sar"])
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--checkpoint_path", required=True, help="Student checkpoint")
    parser.add_argument("--encoder_checkpoint_path", default=None, help="Required for reben/reben_sar")
    parser.add_argument("--include_teacher", action="store_true")
    parser.add_argument("--teacher_checkpoint_path", default=None)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--architecture_variation", action="store_true")
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Defaults to 25 for fbp, 20 for reben/reben_sar")
    parser.add_argument("--num_vis", type=int, default=30,
                        help="Ignored if --patch_ids is given")
    parser.add_argument("--patch_ids", type=int, nargs="+", default=None,
                        help="Explicit global test-split indices to render, e.g. --patch_ids 4075 5060 28604 "
                             "- bypasses the random rejection-sampling entirely, so the same patch can be "
                             "run through different checkpoints for a like-for-like comparison. Indices are "
                             "global positions into the TEST split only (same numbering as the 'Vis patch'/"
                             "'patch_<id>' filenames logged by the training/eval scripts).")
    parser.add_argument("--n_unfrozen_blocks", type=int, default=4,
                        help="Only controls which encoder blocks would accumulate gradients - "
                             "irrelevant for this no-grad inference pass, kept only because the "
                             "encoder constructor requires it")
    # FBP
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=None)
    # reBEN / reBEN SAR
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--ref_root", default=None)
    parser.add_argument("--s2_root", default=None)
    parser.add_argument("--s1_root", default=None)
    parser.add_argument("--despeckle", action="store_true")
    parser.add_argument("--lee_filter", action="store_true")
    args = parser.parse_args()

    if args.num_classes is None:
        args.num_classes = 25 if args.dataset == "fbp" else 20
    if args.include_teacher and not args.teacher_checkpoint_path:
        raise ValueError("--include_teacher requires --teacher_checkpoint_path")
    if args.dataset != "fbp" and not args.encoder_checkpoint_path:
        raise ValueError(f"--dataset {args.dataset} requires --encoder_checkpoint_path")

    # 1. Test dataset (test split only, same seed/official split as training)
    waves = None
    if args.dataset == "fbp":
        test_ds = _build_fbp_test_dataset(args.data_dir, args.patch_size, args.stride, args.max_images)
    elif args.dataset == "reben":
        from utils.dataset_e2e import load_reben_splits, ReBENRawDataset
        _, _, test_ds = load_reben_splits(metadata_path=args.metadata_path, s2_root=args.s2_root,
                                          ref_root=args.ref_root)
        waves = ReBENRawDataset.WAVELENGTHS
    else:
        from utils.dataset_e2e import load_reben_sar_splits, ReBENSARRawDataset
        _, _, test_ds = load_reben_sar_splits(metadata_path=args.metadata_path, s1_root=args.s1_root,
                                              ref_root=args.ref_root, despeckle=args.despeckle,
                                              lee_filter_despeckle=args.lee_filter)
        waves = ReBENSARRawDataset.WAVELENGTHS

    # 2. Pick patches - explicit --patch_ids if given (lets the same patch be compared across
    # checkpoints), otherwise the same rejection-sampling the training/eval scripts use, just
    # asking for more of them.
    if args.patch_ids:
        vis_indices = args.patch_ids
        log_msg(f"Using {len(vis_indices)} explicitly given patch indices: {vis_indices}")
    else:
        vis_indices = _select_visualization_patches(test_ds, num_vis=args.num_vis, max_unlabelled_frac=0.10)
        log_msg(f"Selected {len(vis_indices)}/{args.num_vis} requested visualisation patches")
    vis_loader = DataLoader(Subset(test_ds, vis_indices), batch_size=1, shuffle=False)

    # 3. Models
    encoder_model = None
    if args.dataset != "fbp":
        from models.encoder import initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial
        encoder_model = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=args.n_unfrozen_blocks)
        enc_ckpt = torch.load(args.encoder_checkpoint_path, map_location=DEVICE)
        encoder_model.load_state_dict(enc_ckpt['encoder_state_dict'], strict=False)
        encoder_model.to(DEVICE).eval()
        log_msg(f"Loaded encoder from {args.encoder_checkpoint_path}")

    student = StudentHead(in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
    student_ckpt = torch.load(args.checkpoint_path, map_location=DEVICE)
    student.load_state_dict(student_ckpt['model_state_dict'])
    student.to(DEVICE).eval()
    log_msg(f"Loaded student from {args.checkpoint_path}")

    teacher = None
    if args.include_teacher:
        teacher = DecoderEnsemble(M=args.ensemble_size, in_channels=1024, embed_dim=args.decoder_embed_dim,
                                  num_classes=args.num_classes, architecture_variation=args.architecture_variation)
        teacher_ckpt = torch.load(args.teacher_checkpoint_path, map_location=DEVICE)
        teacher.load_state_dict(teacher_ckpt['model_state_dict'])
        teacher.to(DEVICE).eval()
        log_msg(f"Loaded teacher (M={args.ensemble_size}) from {args.teacher_checkpoint_path}")

    # 4. Run inference on just the selected patches and render panels
    with torch.no_grad():
        for i, (raw_input, mask) in enumerate(vis_loader):
            g_idx = vis_indices[i]
            raw_input = raw_input.to(DEVICE)
            gt = mask.squeeze().cpu().numpy()

            if args.dataset == "fbp":
                features = raw_input
                raw_patch, patch_info = _fbp_raw_patch(test_ds, g_idx, args.stride)
            else:
                features = get_encoder_representation_partial(raw_input, encoder_model, waves=waves)
                raw_patch = _reben_raw_patch(raw_input[0], args.dataset)
                patch_info = f"patch_{g_idx}"

            alphas = student(features)
            total_ent, aleatoric, epistemic, alpha0, pred_class = _student_uncertainty_full(alphas)

            visualise_student_uncertainty(
                class_map=pred_class[0], total_entropy=total_ent[0], aleatoric=aleatoric[0],
                epistemic=epistemic[0], alpha0_map=alpha0[0], ground_truth=gt, hide_unlabelled=True,
                save_name=f"{args.run_name}_student_patch_{g_idx}", raw_patch=raw_patch,
                patch_info=patch_info, num_classes=args.num_classes,
            )

            if teacher is not None:
                mean_probs, class_map, variance_map, total_entropy, mi_map = get_decoder_output_maps(
                    teacher, features, save_name=f"{args.run_name}_teacher_patch_{g_idx}"
                )
                visualise_all_metrics(
                    class_map=class_map, variance_map=variance_map, total_entropy=total_entropy,
                    mi_map=mi_map, ground_truth=gt, hide_unlabelled=True,
                    save_name=f"{args.run_name}_teacher_patch_{g_idx}", raw_patch=raw_patch,
                    patch_info=patch_info, num_classes=args.num_classes,
                )

            log_msg(f"[{i + 1}/{len(vis_indices)}] rendered patch {g_idx}")

    log_msg(f"Done - {len(vis_indices)} patches rendered under BASE_OUT/metrics"
            f"{' (and BASE_OUT/heads, from --include_teacher)' if teacher is not None else ''}")


if __name__ == "__main__":
    main()
