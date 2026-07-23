import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from configs.config import *
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse
import numpy as np
import os

from utils.misc import log_msg, save_checkpoint, FocalLoss
from utils.dataset_e2e import BakedReBENDataset
from utils.visualisation import plot_loss_curves, plot_confusion_matrix
from models.ensemble import DecoderEnsemble
from profile_flops import ensure_flops_profile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_baked_reben(decoder, test_loader, args, run_name):
    decoder.eval()

    num_classes = args.num_classes
    class_names = REBEN_CLASSES
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_conf_sums = np.zeros(num_bins)
    bin_acc_sums  = np.zeros(num_bins, dtype=np.int64)
    bin_counts    = np.zeros(num_bins, dtype=np.int64)
    patch_count = 0
    nll_sum = 0.0
    nll_count = 0
    auroc_entropy = []
    auroc_errors  = []
    AUROC_MAX = 2_000_000
    n_pairs = decoder.M * (decoder.M - 1) // 2
    head_conf_matrices = [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in range(decoder.M)]
    total_ent_sum = 0.0
    aleatoric_sum = 0.0
    jsd_sum = 0.0
    uq_count = 0

    with torch.no_grad():
        for features, masks in test_loader:
            features = features.to(DEVICE)

            with torch.cuda.amp.autocast():
                all_preds   = decoder(features)

            all_head_probs_f32 = torch.stack([torch.softmax(all_preds[m].float(), dim=1)
                                              for m in range(decoder.M)])  # [M, B, C, H, W]
            mean_probs_f32 = all_head_probs_f32.mean(dim=0)  # true probability mixture: mean(softmax), not softmax(mean)
            class_maps  = torch.argmax(mean_probs_f32, dim=1).cpu().numpy()
            conf_maps   = torch.max(mean_probs_f32, dim=1)[0].cpu().numpy()
            masks_t   = masks.to(DEVICE).long()
            labelled_t = masks_t > 0
            if labelled_t.any():
                log_probs = torch.log(mean_probs_f32.clamp(min=1e-10))
                gt_idx    = masks_t.unsqueeze(1).clamp(0, num_classes - 1)
                nll_sum  += (-log_probs.gather(1, gt_idx).squeeze(1)[labelled_t]).sum().item()
                nll_count += labelled_t.sum().item()
                ent_t = -(mean_probs_f32 * torch.log(mean_probs_f32.clamp(min=1e-10))).sum(dim=1)
                collected = sum(len(x) for x in auroc_entropy) if auroc_entropy else 0
                if collected < AUROC_MAX:
                    err_t = (torch.argmax(mean_probs_f32, dim=1) != masks_t).float()
                    auroc_entropy.append(ent_t[labelled_t].cpu().numpy())
                    auroc_errors.append(err_t[labelled_t].cpu().numpy())
                # UQ decomposition
                total_ent_sum += ent_t[labelled_t].sum().item()
                aleat = torch.zeros_like(ent_t)
                for m in range(decoder.M):
                    hp = all_head_probs_f32[m]
                    aleat += -(hp * torch.log(hp.clamp(1e-10))).sum(dim=1)
                aleat /= decoder.M
                aleatoric_sum += aleat[labelled_t].sum().item()
                uq_count += labelled_t.sum().item()
                if n_pairs > 0:
                    for i in range(decoder.M):
                        for j in range(i + 1, decoder.M):
                            p, q = all_head_probs_f32[i], all_head_probs_f32[j]
                            m_pq = 0.5 * (p + q)
                            jsd = (-(m_pq * torch.log(m_pq.clamp(1e-10))).sum(dim=1)
                                   + 0.5 * (p * torch.log(p.clamp(1e-10))).sum(dim=1)
                                   + 0.5 * (q * torch.log(q.clamp(1e-10))).sum(dim=1))
                            jsd_sum += jsd[labelled_t].sum().item()

            masks_np = masks.numpy()
            head_preds_np = torch.argmax(all_head_probs_f32, dim=2).cpu().numpy()  # [M, B, H, W]
            for b in range(features.shape[0]):
                gt = masks_np[b]
                if (gt > 0).sum() < 100:
                    continue
                valid = (gt > 0) & (gt < num_classes)
                pred_flat = class_maps[b][valid]
                true_flat = gt[valid]
                conf_flat = conf_maps[b][valid]

                np.add.at(conf_matrix, (true_flat, pred_flat), 1)
                for m in range(decoder.M):
                    np.add.at(head_conf_matrices[m], (true_flat, head_preds_np[m, b][valid]), 1)

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

    # Per-head mIoU
    head_mious = []
    for m in range(decoder.M):
        h_iou = np.zeros(num_classes - 1)
        h_present = []
        for c in range(1, num_classes):
            tp = head_conf_matrices[m][c, c]
            fp = head_conf_matrices[m][:, c].sum() - tp
            fn = head_conf_matrices[m][c, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                h_iou[c - 1] = tp / denom
                h_present.append(c - 1)
        head_mious.append(float(np.mean(h_iou[h_present])) if h_present else 0.0)

    # UQ decomposition
    mean_total_ent    = total_ent_sum / max(uq_count, 1)
    mean_aleatoric    = aleatoric_sum / max(uq_count, 1)
    mean_epistemic    = max(mean_total_ent - mean_aleatoric, 0.0)
    mean_pairwise_jsd = jsd_sum / max(uq_count * n_pairs, 1) if n_pairs > 0 else 0.0

    global_acc = np.diag(conf_matrix).sum() / max(conf_matrix.sum(), 1)

    bin_accs  = np.where(bin_counts > 0, bin_acc_sums / bin_counts, 0.0)
    bin_confs = np.where(bin_counts > 0, bin_conf_sums / bin_counts, 0.0)
    total_samples = bin_counts.sum()
    global_ece = (np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / total_samples
                  if total_samples > 0 else 0.0)

    log_msg(f"REBEN BAKED TEST RESULTS ({patch_count} patches):")
    log_msg(f"Global mIoU: {global_miou:.4f} | fw-IoU: {fw_iou:.4f} | "
            f"Acc: {global_acc:.4f} | ECE: {global_ece:.4f} | "
            f"NLL: {global_nll:.4f} | AUROC: {global_auroc:.4f}")
    log_msg(f"Uncertainty: total={mean_total_ent:.4f} | aleatoric={mean_aleatoric:.4f} | "
            f"epistemic={mean_epistemic:.4f} | pairwise_JSD={mean_pairwise_jsd:.4f}")
    log_msg(f"Per-head mIoU: {' | '.join(f'head{m}={v:.4f}' for m, v in enumerate(head_mious))}")
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
        "mean_total_entropy": mean_total_ent,
        "mean_aleatoric": mean_aleatoric,
        "mean_epistemic": mean_epistemic,
        "mean_pairwise_jsd": mean_pairwise_jsd,
        "per_head_miou": {f"head_{m}": v for m, v in enumerate(head_mious)},
        "num_patches": patch_count,
        "per_class_iou": {class_names[i + 1]: float(iou) for i, iou in enumerate(iou_per_class)},
        "confusion_matrix": conf_matrix.tolist(),
    }
    results_path = os.path.join(runs_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")
    plot_confusion_matrix(results_path, save_name=f"{run_name}_confusion")
    return results


def train_decoders_reben(args):
    run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M')}"
    log_msg(f"Run name: {run_name}")
    log_msg(f"reBEN frozen decoder training: {vars(args)}")

    # 1. Datasets
    embedding_dir = Path(args.embedding_dir)
    log_msg("Loading train dataset...")
    train_ds = BakedReBENDataset(embedding_dir, split="train", augment=True)
    log_msg("Loading val dataset...")
    val_ds   = BakedReBENDataset(embedding_dir, split="val",   augment=False)
    log_msg("Loading test dataset...")
    test_ds  = BakedReBENDataset(embedding_dir, split="test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # 2. Model
    decoder = DecoderEnsemble(
        M=args.ensemble_size,
        in_channels=1024,
        embed_dim=args.decoder_embed_dim,
        num_classes=args.num_classes,
    )
    decoder.to(DEVICE)

    ensure_flops_profile(
        config={
            "dataset": "reben",
            "encoder_type": "frozen",
            "n_unfrozen_blocks": None,
            "ensemble_size": args.ensemble_size,
            "decoder_embed_dim": args.decoder_embed_dim,
            "num_classes": args.num_classes,
            "in_channels": None,
            "batch_size": args.batch_size,
            "include_student": False,
        },
        decoder=decoder,
    )

    # 3. Optimiser
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.05)

    def warmup_lambda(epoch):
        return min(1.0, float(epoch + 1) / float(max(1, args.warmup_epochs)))

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

    # 4. Resume
    start_epoch = 0
    loss_history = {"train": [], "val": []}

    if args.resume and args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if ckpt_path.exists():
            log_msg(f"Resuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            decoder.load_state_dict(ckpt['model_state_dict'])
            if not args.no_load_optimizer:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = args.resume_epoch
            loss_path = os.path.join(runs_dir, f"{args.run_name}_loss_history.json")
            if os.path.exists(loss_path):
                with open(loss_path) as f:
                    loss_history = json.load(f)
            log_msg(f"Resuming from epoch {start_epoch + 1}")
        else:
            log_msg(f"WARNING: checkpoint not found at {ckpt_path}, starting fresh")

    if start_epoch < args.warmup_epochs:
        for _ in range(start_epoch):
            warmup_scheduler.step()

    # 5. Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_msg("Starting training...")

    for epoch in range(start_epoch, args.num_epochs):
        log_msg(f"Starting epoch {epoch + 1}...")
        decoder.train()
        epoch_loss = 0.0

        for batch_features, batch_masks in train_loader:
            optimizer.zero_grad()
            batch_features = batch_features.to(DEVICE)
            batch_masks    = batch_masks.to(DEVICE)

            with torch.cuda.amp.autocast():
                all_preds = decoder(batch_features)
                total_loss = 0
                for head_idx in range(decoder.M):
                    total_loss += criterion(all_preds[head_idx], batch_masks)
                mean_logits = all_preds.mean(dim=0)
                total_loss += criterion(mean_logits, batch_masks)
                loss = total_loss / (decoder.M + 1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        loss_history["train"].append(avg_train_loss)
        log_msg(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {avg_train_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Validation
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_features, v_masks in val_loader:
                v_features = v_features.to(DEVICE)
                v_masks    = v_masks.to(DEVICE)
                with torch.cuda.amp.autocast():
                    all_preds = decoder(v_features)
                    total_val = 0
                    for head_idx in range(decoder.M):
                        total_val += criterion(all_preds[head_idx], v_masks)
                    mean_logits = all_preds.mean(dim=0)
                    total_val += criterion(mean_logits, v_masks)
                    val_loss += (total_val / (decoder.M + 1)).item()

        avg_val_loss = val_loss / len(val_loader)
        loss_history["val"].append(avg_val_loss)
        log_msg(f"Validation Loss: {avg_val_loss:.8f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(
                {'epoch': epoch + 1, 'model_state_dict': decoder.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                str(CHECKPOINT_DIR), filename=f"{run_name}_best_decoder.pth"
            )
            log_msg(f"New best val loss {best_val_loss:.6f} — saved")
        else:
            epochs_no_improve += 1
            log_msg(f"No improvement for {epochs_no_improve}/{args.early_stopping_patience} epochs")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                {'epoch': epoch + 1, 'model_state_dict': decoder.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                str(CHECKPOINT_DIR), filename=f"{run_name}_last_decoder.pth"
            )
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(avg_val_loss)

        if epochs_no_improve >= args.early_stopping_patience:
            log_msg(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Final save
    timestamp = time.strftime("%Y%m%d_%H%M")
    save_checkpoint(
        {'epoch': epoch + 1, 'model_state_dict': decoder.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()},
        str(CHECKPOINT_DIR), filename=f"{run_name}_final_decoder_{timestamp}.pth"
    )
    loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    log_msg("Training complete.")
    plot_loss_curves(loss_path, save_name=f"{run_name}_loss_curves")

    # Load best for evaluation
    best_path = CHECKPOINT_DIR / f"{run_name}_best_decoder.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE)
        decoder.load_state_dict(ckpt['model_state_dict'])
        log_msg("Loaded best checkpoint for evaluation")

    log_msg("Running evaluation...")
    evaluate_baked_reben(decoder, test_loader, args, run_name)

    return decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reBEN frozen decoder training on pre-baked embeddings")

    # Paths
    parser.add_argument("--embedding_dir", type=str, required=True,
                        help="Path to embeddings/reben/ directory containing shard .pt files")
    parser.add_argument("--out_dir", type=str, default="./results")

    # Model
    parser.add_argument("--ensemble_size",     type=int,   default=5)
    parser.add_argument("--decoder_embed_dim", type=int,   default=512)
    parser.add_argument("--num_classes",       type=int,   default=20)

    # Training
    parser.add_argument("--num_epochs",              type=int,   default=200)
    parser.add_argument("--batch_size",              type=int,   default=32)
    parser.add_argument("--lr",                      type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",           type=int,   default=5)
    parser.add_argument("--early_stopping_patience", type=int,   default=20)

    # Loss
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma",    type=float, default=2.0)

    # Resume
    parser.add_argument("--resume",             action="store_true")
    parser.add_argument("--resume_checkpoint",  type=str, default=None,
                        help="Path to decoder checkpoint to resume from")
    parser.add_argument("--resume_epoch",       type=int, default=0)
    parser.add_argument("--no_load_optimizer",  action="store_true",
                        help="Skip loading optimizer state on resume — resets LR")

    # Misc
    parser.add_argument("--run_name", type=str, default="reben_frozen")

    args = parser.parse_args()
    train_decoders_reben(args)
