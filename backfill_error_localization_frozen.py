"""
backfill_error_localization_frozen.py — backfills the spatial uncertainty-error
localization metric (per-patch Spearman(uncertainty, correctness): full distribution +
fraction-positive + illustrative heatmap, see utils.misc.evaluate_error_localization)
onto an already-completed FBP frozen-decoder checkpoint (AS1-4) trained before this
metric existed in train_decoders.py / train_student.py's eval path.

Checkpoint-only, no retraining: loads a saved decoder (and optionally student)
checkpoint and evaluates it against the test split. Mirrors calibrate_as1_frozen.py's
checkpoint-loading conventions exactly, generalised to arbitrary ensemble_size (not just
AS1's M=1) and an optional student checkpoint.

Usage:
    python backfill_error_localization_frozen.py \
        --data_dir /beegfs/scratch/callumdempsey/results \
        --decoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/<run>_best_model.pth \
        --ensemble_size 10 --decoder_embed_dim 512 --num_classes 25 \
        --patch_size 224 --stride 224 --max_images 150 \
        --run_name AS3_frozen_encoder_m=10_ensemble_20260716_1500

    # add --student_checkpoint <path> to also backfill the student head
"""
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.dataset import BakedFeatureDataset
from utils.misc import (log_msg, evaluate_error_localization,
                        ensemble_uncertainty_and_pred, dirichlet_uncertainty_and_pred)
from models.ensemble import DecoderEnsemble, StudentHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_test_split(data_dir, patch_size, stride, max_images=None):
    """Mirrors train_decoders.py's split exactly (same glob, same seed=42 shuffle, same
    70/15/15 cut) - only the test portion is needed for this metric."""
    embedding_dir = Path(data_dir) / "embeddings" / "fbp" / "clay_v1" / f"patch{patch_size}_stride{stride}"
    all_files = sorted(list(embedding_dir.glob("*_embeddings.pt")))
    if max_images is not None:
        all_files = all_files[:max_images]
    random.seed(42)
    random.shuffle(all_files)

    n = len(all_files)
    if n < 5:
        test_files = all_files[2:]
    else:
        test_files = all_files[int(n * 0.85):]
    return test_files


def main():
    parser = argparse.ArgumentParser(description="Backfill error-localization metric onto an existing FBP frozen-decoder checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root containing embeddings/ (train_decoders.py's --data_dir/--out_dir), not the raw data/fbp images+labels dir")
    parser.add_argument("--decoder_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Optional - also backfills the student head if given")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--decoder_embed_dim", type=int, required=True,
                        help="Must match the checkpoint's actual training config - check its job log, do not assume")
    parser.add_argument("--ensemble_size", type=int, required=True,
                        help="M - must match the checkpoint's actual training config")
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--run_name", type=str, required=True,
                        help="Output filename prefix - use the checkpoint's own run_name for traceability")
    args = parser.parse_args()

    test_files = build_test_split(args.data_dir, args.patch_size, args.stride, args.max_images)
    log_msg(f"Test: {len(test_files)} images")
    test_loader = DataLoader(BakedFeatureDataset(test_files, augment=False),
                             batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    decoder = DecoderEnsemble(M=args.ensemble_size, in_channels=1024, embed_dim=args.decoder_embed_dim,
                              num_classes=args.num_classes)
    ckpt = torch.load(args.decoder_checkpoint, map_location=DEVICE)
    decoder.load_state_dict(ckpt['model_state_dict'])
    decoder.to(DEVICE)
    decoder.eval()
    log_msg(f"Loaded decoder (M={args.ensemble_size}) from {args.decoder_checkpoint}")

    evaluate_error_localization(
        lambda feats: ensemble_uncertainty_and_pred(decoder(feats), args.num_classes),
        test_loader, args, run_name=args.run_name, who="teacher"
    )

    if args.student_checkpoint is not None:
        student = StudentHead(in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
        student_ckpt = torch.load(args.student_checkpoint, map_location=DEVICE)
        student.load_state_dict(student_ckpt['model_state_dict'])
        student.to(DEVICE)
        student.eval()
        log_msg(f"Loaded student from {args.student_checkpoint}")

        evaluate_error_localization(
            lambda feats: dirichlet_uncertainty_and_pred(student(feats).float()),
            test_loader, args, run_name=args.run_name, who="student"
        )


if __name__ == "__main__":
    main()
