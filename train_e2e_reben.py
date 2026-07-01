import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from configs.config import *
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse
import random
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from utils.misc import log_msg, save_checkpoint, FocalLoss
from utils.dataset_e2e import load_reben_splits, ReBENRawDataset
S2_WAVES = ReBENRawDataset.WAVELENGTHS  # [0.6646, 0.5598, 0.4924] μm — S2 B04/B03/B02
from utils.visualisation import plot_loss_curves, visualise_all_metrics
from models.ensemble import DecoderEnsemble
from models.encoder import (
    initialize_clay_encoder_partial_unfreeze,
    get_encoder_representation_partial,
)
from utils.misc import get_decoder_output_maps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_e2e_reben(args):
    run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M')}"
    log_msg(f"Run name: {run_name}")
    log_msg(f"Running reBEN E2E training with inputs:\n {vars(args)}")

    # 1. Data — official splits from metadata.parquet
    train_ds, val_ds, test_ds = load_reben_splits(
        metadata_path=args.metadata_path,
        s2_root=args.s2_root,
        ref_root=args.ref_root,
        exclude_snow=not args.include_snow,
        exclude_cloud=not args.include_cloud,
        max_patches=args.max_patches,
        max_val_patches=args.max_val_patches,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # 2. Models
    log_msg(f"Initialising encoder with {args.n_unfrozen_blocks} unfrozen blocks...")
    encoder_model = initialize_clay_encoder_partial_unfreeze(
        n_unfrozen_blocks=args.n_unfrozen_blocks
    )

    log_msg("Initialising decoder ensemble...")
    decoder = DecoderEnsemble(
        M=args.ensemble_size,
        in_channels=1024,
        embed_dim=args.decoder_embed_dim,
        num_classes=args.num_classes,
    )
    decoder.to(DEVICE)

    # 3. Optimiser with differential LRs
    trainable_enc = [p for p in encoder_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': trainable_enc,         'lr': args.lr_encoder, 'weight_decay': 0.01},
        {'params': decoder.parameters(),  'lr': args.lr_decoder, 'weight_decay': 0.05},
    ])

    def warmup_lambda(epoch):
        return float(epoch + 1) / float(max(1, args.warmup_epochs))

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-8
    )
    scaler = torch.cuda.amp.GradScaler()

    if args.use_focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, ignore_index=0)
        log_msg(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        log_msg("Using CrossEntropyLoss")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # 4. Resume logic
    start_epoch = 0
    loss_history = {"train": [], "val": []}

    if args.resume:
        decoder_ckpt = Path(args.resume_decoder_path) if args.resume_decoder_path else None
        encoder_ckpt = Path(args.resume_encoder_path) if args.resume_encoder_path else None
        if decoder_ckpt and decoder_ckpt.exists():
            log_msg(f"Resuming decoder from {decoder_ckpt}...")
            ckpt = torch.load(decoder_ckpt, map_location=DEVICE)
            decoder.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if encoder_ckpt and encoder_ckpt.exists():
            log_msg(f"Resuming encoder from {encoder_ckpt}...")
            enc_ckpt = torch.load(encoder_ckpt, map_location=DEVICE)
            encoder_model.load_state_dict(enc_ckpt['encoder_state_dict'], strict=False)
        start_epoch = args.resume_epoch
        loss_path = os.path.join(runs_dir, f"{args.run_name}_loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_history = json.load(f)
        log_msg(f"Resuming from epoch {start_epoch + 1}")
        for _ in range(min(start_epoch, args.warmup_epochs)):
            warmup_scheduler.step()

    # 5. Training loop
    best_val_loss = float('inf')
    log_msg("Starting training...")

    for epoch in range(start_epoch, args.num_epochs):
        log_msg(f"Starting epoch {epoch + 1}...")
        epoch_task_loss = 0

        encoder_model.eval()
        transformer = encoder_model.model.encoder.transformer
        for block in transformer.layers[-args.n_unfrozen_blocks:]:
            block.train()
        transformer.norm.train()
        decoder.train()

        for batch_imgs, batch_masks in train_loader:
            optimizer.zero_grad()
            batch_imgs  = batch_imgs.to(DEVICE)
            batch_masks = batch_masks.to(DEVICE).long()

            with torch.cuda.amp.autocast():
                features = get_encoder_representation_partial(batch_imgs, encoder_model, waves=S2_WAVES)
                all_preds = decoder(features)

                total_task_loss = 0
                for head_idx in range(decoder.M):
                    total_task_loss += criterion(all_preds[head_idx], batch_masks)
                mean_logits = all_preds.mean(dim=0)
                total_task_loss += criterion(mean_logits, batch_masks)
                task_loss = total_task_loss / (decoder.M + 1)

            scaler.scale(task_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in encoder_model.parameters() if p.requires_grad] +
                list(decoder.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_task_loss += task_loss.item()

        avg_task = epoch_task_loss / len(train_loader)
        loss_history["train"].append(avg_task)
        enc_lr = optimizer.param_groups[0]['lr']
        dec_lr = optimizer.param_groups[1]['lr']
        log_msg(f"Epoch [{epoch + 1}/{args.num_epochs}] - Task Loss: {avg_task:.4f} | "
                f"LR encoder: {enc_lr:.2e} decoder: {dec_lr:.2e}")

        # Validation
        encoder_model.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs  = v_imgs.to(DEVICE)
                v_masks = v_masks.to(DEVICE).long()
                with torch.cuda.amp.autocast():
                    features = get_encoder_representation_partial(v_imgs, encoder_model, waves=S2_WAVES)
                    all_preds = decoder(features)
                    total_val_loss = 0
                    for head_idx in range(decoder.M):
                        total_val_loss += criterion(all_preds[head_idx], v_masks)
                    mean_logits = all_preds.mean(dim=0)
                    total_val_loss += criterion(mean_logits, v_masks)
                    v_loss = total_val_loss / (decoder.M + 1)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        loss_history["val"].append(avg_val_loss)
        log_msg(f"Validation Loss: {avg_val_loss:.8f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint({'epoch': epoch + 1, 'model_state_dict': decoder.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()},
                            str(CHECKPOINT_DIR), filename=f"{run_name}_best_decoder.pth")
            save_checkpoint({'epoch': epoch + 1, 'encoder_state_dict': encoder_model.state_dict()},
                            str(CHECKPOINT_DIR), filename=f"{run_name}_best_encoder.pth")
            log_msg(f"New best val loss {best_val_loss:.6f} — saved")

        if (epoch + 1) % 5 == 0:
            save_checkpoint({'epoch': epoch + 1, 'model_state_dict': decoder.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()},
                            str(CHECKPOINT_DIR), filename=f"{run_name}_last_decoder.pth")
            save_checkpoint({'epoch': epoch + 1, 'encoder_state_dict': encoder_model.state_dict()},
                            str(CHECKPOINT_DIR), filename=f"{run_name}_last_encoder.pth")
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(avg_val_loss)

    # Final save
    timestamp = time.strftime("%Y%m%d_%H%M")
    save_checkpoint({'epoch': args.num_epochs, 'model_state_dict': decoder.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()},
                    str(CHECKPOINT_DIR), filename=f"{run_name}_final_decoder_{timestamp}.pth")
    save_checkpoint({'epoch': args.num_epochs, 'encoder_state_dict': encoder_model.state_dict()},
                    str(CHECKPOINT_DIR), filename=f"{run_name}_final_encoder_{timestamp}.pth")
    loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    log_msg("Training completed.")
    plot_loss_curves(loss_path, save_name=f"{run_name}_loss_curves")

    # Load best for evaluation
    best_decoder_path = CHECKPOINT_DIR / f"{run_name}_best_decoder.pth"
    if best_decoder_path.exists():
        ckpt = torch.load(best_decoder_path, map_location=DEVICE)
        decoder.load_state_dict(ckpt['model_state_dict'])
    best_encoder_path = CHECKPOINT_DIR / f"{run_name}_best_encoder.pth"
    if best_encoder_path.exists():
        enc_ckpt = torch.load(best_encoder_path, map_location=DEVICE)
        encoder_model.load_state_dict(enc_ckpt['encoder_state_dict'], strict=False)

    log_msg("Running evaluation...")
    evaluate_test_set_reben(encoder_model, decoder, test_loader, args, run_name)

    return decoder, encoder_model


def evaluate_test_set_reben(encoder_model, decoder, test_loader, args, run_name):
    encoder_model.eval()
    decoder.eval()

    num_classes = args.num_classes
    class_names = REBEN_CLASSES
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_conf_sums  = np.zeros(num_bins)
    bin_acc_sums   = np.zeros(num_bins, dtype=np.int64)
    bin_counts     = np.zeros(num_bins, dtype=np.int64)
    patch_count = 0
    nll_sum = 0.0
    nll_count = 0
    auroc_entropy = []
    auroc_errors  = []
    AUROC_MAX = 2_000_000

    with torch.no_grad():
        for batch_idx, (test_imgs, test_masks) in enumerate(test_loader):
            test_imgs = test_imgs.to(DEVICE)

            with torch.cuda.amp.autocast():
                features  = get_encoder_representation_partial(test_imgs, encoder_model, waves=S2_WAVES)
                all_preds = decoder(features)
                mean_logits = all_preds.mean(dim=0)
                mean_probs  = torch.softmax(mean_logits, dim=1)
                class_maps  = torch.argmax(mean_probs, dim=1).cpu().numpy()
                conf_maps   = torch.max(mean_probs, dim=1)[0].cpu().numpy()

            mean_probs_f32 = mean_probs.float()
            test_masks_t   = test_masks.to(DEVICE).long()
            labelled_t     = test_masks_t > 0
            if labelled_t.any():
                log_probs = torch.log(mean_probs_f32.clamp(min=1e-10))
                gt_idx    = test_masks_t.unsqueeze(1).clamp(0, num_classes - 1)
                nll_sum  += (-log_probs.gather(1, gt_idx).squeeze(1)[labelled_t]).sum().item()
                nll_count += labelled_t.sum().item()
                auroc_collected = sum(len(x) for x in auroc_entropy) if auroc_entropy else 0
                if auroc_collected < AUROC_MAX:
                    ent_t = -(mean_probs_f32 * torch.log(mean_probs_f32.clamp(min=1e-10))).sum(dim=1)
                    err_t = (torch.argmax(mean_probs_f32, dim=1) != test_masks_t).float()
                    auroc_entropy.append(ent_t[labelled_t].cpu().numpy())
                    auroc_errors.append(err_t[labelled_t].cpu().numpy())

            for b in range(test_imgs.shape[0]):
                gt = test_masks[b].squeeze().cpu().numpy()
                if (gt > 0).sum() < 100:
                    continue
                mask      = (gt > 0) & (gt < num_classes)
                pred_flat = class_maps[b][mask]
                true_flat = gt[mask]
                conf_flat = conf_maps[b][mask]

                np.add.at(conf_matrix, (true_flat, pred_flat), 1)

                bin_indices = np.digitize(conf_flat, bin_boundaries[1:-1])
                for bin_idx in range(num_bins):
                    in_bin = bin_indices == bin_idx
                    if in_bin.sum() > 0:
                        bin_conf_sums[bin_idx] += conf_flat[in_bin].sum()
                        bin_acc_sums[bin_idx]  += (pred_flat[in_bin] == true_flat[in_bin]).sum()
                        bin_counts[bin_idx]    += in_bin.sum()
                patch_count += 1

    iou_per_class = np.zeros(num_classes - 1)
    present = []
    for c in range(1, num_classes):
        tp    = conf_matrix[c, c]
        fp    = conf_matrix[:, c].sum() - tp
        fn    = conf_matrix[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou_per_class[c - 1] = tp / denom
            present.append(c - 1)
    global_miou = float(np.mean(iou_per_class[present])) if present else 0.0

    class_pixel_counts = conf_matrix[1:, :].sum(axis=1)
    total_labelled = class_pixel_counts.sum()
    fw_iou = float((class_pixel_counts / max(total_labelled, 1) * iou_per_class).sum())

    global_nll = nll_sum / max(nll_count, 1)
    if auroc_entropy:
        all_ent = np.concatenate(auroc_entropy)
        all_err = np.concatenate(auroc_errors)
        global_auroc = float(roc_auc_score(all_err, all_ent)) if len(np.unique(all_err)) > 1 else 0.0
    else:
        global_auroc = 0.0

    global_acc = np.diag(conf_matrix).sum() / max(conf_matrix.sum(), 1)

    bin_accs  = np.where(bin_counts > 0, bin_acc_sums / bin_counts, 0.0)
    bin_confs = np.where(bin_counts > 0, bin_conf_sums / bin_counts, 0.0)
    total_samples = bin_counts.sum()
    global_ece = (np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / total_samples
                  if total_samples > 0 else 0.0)

    log_msg(f"REBEN TEST RESULTS ({patch_count} patches):")
    log_msg(f"Global mIoU: {global_miou:.4f} | fw-IoU: {fw_iou:.4f} | "
            f"Acc: {global_acc:.4f} | ECE: {global_ece:.4f} | "
            f"NLL: {global_nll:.4f} | AUROC: {global_auroc:.4f}")
    log_msg("Per-class IoU (descending frequency):")
    freq_order = np.argsort(class_pixel_counts)[::-1]
    for class_idx in freq_order:
        log_msg(f"  {class_names[class_idx + 1]}: {iou_per_class[class_idx]:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": run_name,
        "global_miou": float(global_miou),
        "global_fwiou": fw_iou,
        "global_accuracy": float(global_acc),
        "global_ece": float(global_ece),
        "global_nll": global_nll,
        "global_auroc": global_auroc,
        "num_patches": patch_count,
        "per_class_iou": {class_names[i + 1]: float(iou) for i, iou in enumerate(iou_per_class)},
    }
    results_path = os.path.join(runs_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reBEN end-to-end partial-unfreeze training")

    # Paths
    parser.add_argument("--s2_root",       type=str, required=True,
                        help="Path to BigEarthNet-S2/ directory")
    parser.add_argument("--ref_root",      type=str, required=True,
                        help="Path to Reference_Maps/ directory")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to metadata.parquet")

    # Data filters
    parser.add_argument("--include_snow",  action="store_true",
                        help="Include patches flagged as containing seasonal snow")
    parser.add_argument("--include_cloud", action="store_true",
                        help="Include patches flagged as containing cloud or shadow")

    # Model
    parser.add_argument("--ensemble_size",     type=int,   default=5)
    parser.add_argument("--decoder_embed_dim", type=int,   default=512)
    parser.add_argument("--num_classes",       type=int,   default=20)
    parser.add_argument("--n_unfrozen_blocks", type=int,   default=4)

    # Training
    parser.add_argument("--num_epochs",    type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr_encoder",   type=float, default=1e-6)
    parser.add_argument("--lr_decoder",   type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--hide_unlabelled_pixels", action="store_true")

    # Loss
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma",    type=float, default=2.0)

    # Resume
    parser.add_argument("--resume",              action="store_true")
    parser.add_argument("--resume_decoder_path", type=str, default=None)
    parser.add_argument("--resume_encoder_path", type=str, default=None)
    parser.add_argument("--resume_epoch",        type=int, default=0)

    # Misc
    parser.add_argument("--run_name",    type=str, default="reben_e2e")
    parser.add_argument("--max_patches", type=int, default=None,
                        help="Limit train patches for QA runs (val/test get max_patches//5)")
    parser.add_argument("--max_val_patches", type=int, default=None,
                        help="Cap val patches independently of --max_patches")

    args = parser.parse_args()
    train_e2e_reben(args)
