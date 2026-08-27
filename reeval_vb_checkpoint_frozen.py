"""
reeval_vb_checkpoint_frozen.py — re-evaluates a trained FBP frozen-decoder
VariationalBottleneck checkpoint with the corrected eval path (feeding each head its own
mu_m(features) - the deterministic posterior mean it was actually trained around - instead
of the raw encoder features or a single shared mu). No retraining: loads the saved decoder +
bottleneck (and optionally student) and re-runs evaluate_test_set + evaluate_error_localization
against the FBP test split.

NOTE: only works against checkpoints saved with the per-head VariationalBottleneck (one
mu_conv/logvar_conv pair per head). The earlier shared-bottleneck checkpoint
(AS3_fbp_variational_layer_40ep_qa_20260821_1708_best_model.pth) has a different
bottleneck_state_dict shape and will NOT load here - that run's results are already fully
evaluated (see its own _reeval_results.json), nothing is lost by this script no longer
supporting it.

See models.ensemble.VBEvalWrapper for the fix itself.

Usage:
    python reeval_vb_checkpoint_frozen.py \
        --data_dir /beegfs/scratch/callumdempsey/results \
        --decoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/<per_head_vb_run>_best_model.pth \
        --ensemble_size 5 --decoder_embed_dim 512 --num_classes 25 \
        --sigma_prior 0.5 --patch_size 224 --stride 224 --max_images 150 \
        --run_name <per_head_vb_run>_reeval

    # add --student_checkpoint <path> to also re-evaluate the student head
"""
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.dataset import BakedFeatureDataset
from utils.misc import (log_msg, evaluate_test_set, evaluate_error_localization,
                        ensemble_uncertainty_and_pred, dirichlet_uncertainty_and_pred)
from models.ensemble import DecoderEnsemble, StudentHead, VariationalBottleneck, VBEvalWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_test_split(data_dir, patch_size, stride, max_images=None):
    """Mirrors train_decoders.py's split exactly (same glob, same seed=42 shuffle, same
    70/15/15 cut) - only the test portion is needed for this re-eval."""
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
    parser = argparse.ArgumentParser(description="Re-evaluate an FBP frozen-decoder VariationalBottleneck checkpoint with the corrected (mu-fed) eval path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root containing embeddings/ (train_decoders.py's --data_dir/--out_dir)")
    parser.add_argument("--decoder_checkpoint", type=str, required=True,
                        help="Must contain both model_state_dict and bottleneck_state_dict (e.g. a _best_model.pth saved with --use_variational_bottleneck)")
    parser.add_argument("--student_checkpoint", type=str, default=None,
                        help="Optional - only pass this if the checkpoint's run actually trained a student")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--decoder_embed_dim", type=int, required=True,
                        help="Must match the checkpoint's actual training config")
    parser.add_argument("--ensemble_size", type=int, required=True,
                        help="M - must match the checkpoint's actual training config")
    parser.add_argument("--sigma_prior", type=float, required=True,
                        help="Must match the checkpoint's actual training config - doesn't affect eval output (mu-only), but is needed to construct the module")
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--diversity_methods", type=str, nargs="+", default=[],
                        help="Must match the checkpoint's actual training config - only used for evaluate_test_set's results.json metadata, doesn't affect eval computation")
    parser.add_argument("--lam_jsd", type=float, default=0.0)
    parser.add_argument("--lam_pearson", type=float, default=0.0)
    parser.add_argument("--lam_orth", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--hide_unlabelled_pixels", action="store_true", default=True)
    parser.add_argument("--run_name", type=str, required=True,
                        help="Output filename prefix - use a distinct name from the original run so you don't overwrite the buggy results, e.g. '<original_run_name>_reeval'")
    args = parser.parse_args()

    test_files = build_test_split(args.data_dir, args.patch_size, args.stride, args.max_images)
    log_msg(f"Test: {len(test_files)} images")
    test_loader = DataLoader(BakedFeatureDataset(test_files, augment=False),
                             batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    decoder = DecoderEnsemble(M=args.ensemble_size, in_channels=1024, embed_dim=args.decoder_embed_dim,
                              num_classes=args.num_classes)
    bottleneck = VariationalBottleneck(channels=1024, num_heads=args.ensemble_size, sigma_prior=args.sigma_prior)

    ckpt = torch.load(args.decoder_checkpoint, map_location=DEVICE)
    decoder.load_state_dict(ckpt['model_state_dict'])
    if ckpt.get('bottleneck_state_dict') is None:
        raise ValueError(f"{args.decoder_checkpoint} has no bottleneck_state_dict - this doesn't look like "
                         f"a --use_variational_bottleneck checkpoint, use the plain backfill script instead")
    bottleneck.load_state_dict(ckpt['bottleneck_state_dict'])
    decoder.to(DEVICE).eval()
    bottleneck.to(DEVICE).eval()
    log_msg(f"Loaded decoder (M={args.ensemble_size}) and bottleneck from {args.decoder_checkpoint}")

    eval_model = VBEvalWrapper(decoder, bottleneck)

    results = evaluate_test_set(eval_model, test_loader, None, args, run_name=args.run_name)
    evaluate_error_localization(
        lambda feats: ensemble_uncertainty_and_pred(eval_model(feats), args.num_classes),
        test_loader, args, run_name=args.run_name, who="teacher"
    )

    if args.student_checkpoint is not None:
        student = StudentHead(in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
        student_ckpt = torch.load(args.student_checkpoint, map_location=DEVICE)
        student.load_state_dict(student_ckpt['model_state_dict'])
        student.to(DEVICE).eval()
        log_msg(f"Loaded student from {args.student_checkpoint}")

        # Student always trains on raw encoder features regardless of the bottleneck, so no
        # wrapper is needed here (unlike the teacher, which needs VBEvalWrapper to route each
        # head to its own mu_m).
        evaluate_error_localization(
            lambda feats: dirichlet_uncertainty_and_pred(student(feats)),
            test_loader, args, run_name=args.run_name, who="student"
        )


if __name__ == "__main__":
    main()
