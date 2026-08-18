"""
backfill_error_localization_e2e.py — backfills the spatial uncertainty-error
localization metric (per-patch Spearman(uncertainty, correctness): full distribution +
fraction-positive + illustrative heatmap, see utils.misc.evaluate_error_localization)
onto an already-completed reBEN/SAR e2e (finetuned encoder) checkpoint, trained before
this metric existed in train_e2e_reben.py / train_e2e_reben_sar.py / train_decoders_reben.py
/ train_decoders_reben_sar.py's eval paths.

Dataset-agnostic between reBEN (optical, S2) and SAR via --dataset, same as
calibrate_as1_e2e.py. Checkpoint-only, no retraining: loads a saved encoder+decoder
(and optionally student, reBEN only - SAR has no student pipeline) and evaluates against
the test split.

Usage (reBEN example):
    python backfill_error_localization_e2e.py \
        --dataset reben \
        --metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
        --img_root /beegfs/scratch/callumdempsey/data/reben/BigEarthNet-S2 \
        --ref_root /beegfs/scratch/callumdempsey/data/reben/Reference_Maps \
        --decoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/<run>_best_decoder.pth \
        --encoder_checkpoint /beegfs/scratch/callumdempsey/results/checkpoints/<run>_best_encoder.pth \
        --n_unfrozen_blocks 4 --ensemble_size 10 --decoder_embed_dim 512 \
        --run_name AS2_finetuned_m=10_full_dataset_run_20260715_1801

    # add --student_checkpoint <path> for reBEN's student head (SAR has none)
    # SAR: --dataset sar, --img_root points at the SAR root, optionally --despeckle/--lee_filter
"""
import argparse

import torch
from torch.utils.data import DataLoader

from utils.dataset_e2e import load_reben_splits, load_reben_sar_splits, ReBENRawDataset, ReBENSARRawDataset
from utils.misc import (log_msg, evaluate_error_localization,
                        ensemble_uncertainty_and_pred, dirichlet_uncertainty_and_pred)
from models.ensemble import DecoderEnsemble, StudentHead
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


def main():
    parser = argparse.ArgumentParser(description="Backfill error-localization metric onto an existing reBEN/SAR e2e checkpoint")
    parser.add_argument("--dataset", choices=["reben", "sar"], required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True, help="s2_root for reben, s1_root for sar")
    parser.add_argument("--ref_root", type=str, required=True)
    parser.add_argument("--decoder_checkpoint", type=str, required=True)
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, default=None,
                        help="Optional, reBEN only - also backfills the student head if given")
    parser.add_argument("--n_unfrozen_blocks", type=int, required=True,
                        help="Must match the checkpoint's actual training config - check its job log, do not assume")
    parser.add_argument("--ensemble_size", type=int, required=True,
                        help="M - must match the checkpoint's actual training config")
    parser.add_argument("--decoder_embed_dim", type=int, required=True,
                        help="Must match the checkpoint's actual training config")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--max_val_patches", type=int, default=None)
    parser.add_argument("--include_snow", action="store_true")
    parser.add_argument("--include_cloud", action="store_true")
    parser.add_argument("--despeckle", action="store_true", help="SAR only")
    parser.add_argument("--lee_filter", action="store_true", help="SAR only")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Output filename prefix - use the checkpoint's own run_name for traceability")
    args = parser.parse_args()

    waves = WAVES[args.dataset]

    log_msg(f"Loading {args.dataset} splits...")
    _, _, test_ds = load_splits(args)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    log_msg(f"Test: {len(test_ds)} patches")

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
    log_msg(f"Loaded decoder (M={args.ensemble_size}) from {args.decoder_checkpoint}")

    def _teacher_forward(imgs):
        with torch.cuda.amp.autocast():
            features = get_encoder_representation_partial(imgs, encoder_model, waves=waves)
            return decoder(features)

    evaluate_error_localization(
        lambda imgs: ensemble_uncertainty_and_pred(_teacher_forward(imgs), args.num_classes),
        test_loader, args, run_name=args.run_name, who="teacher"
    )

    if args.student_checkpoint is not None:
        student = StudentHead(in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
        student_ckpt = torch.load(args.student_checkpoint, map_location=DEVICE)
        student.load_state_dict(student_ckpt['model_state_dict'])
        student.to(DEVICE)
        student.eval()
        log_msg(f"Loaded student from {args.student_checkpoint}")

        def _student_forward(imgs):
            with torch.cuda.amp.autocast():
                features = get_encoder_representation_partial(imgs, encoder_model, waves=waves)
                return student(features)

        evaluate_error_localization(
            lambda imgs: dirichlet_uncertainty_and_pred(_student_forward(imgs).float()),
            test_loader, args, run_name=args.run_name, who="student"
        )


if __name__ == "__main__":
    main()
