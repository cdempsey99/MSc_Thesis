import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
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
import math
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from utils.misc import get_ece
from utils.dataset_e2e import FBPRawDataset
from utils.misc import log_msg, save_checkpoint
from utils.visualisation import plot_loss_curves
from models.ensemble import DecoderEnsemble
from models.encoder import (
    initialize_clay_encoder_partial_unfreeze,
    get_encoder_representation_partial
)
from utils.visualisation import *
from utils.misc import get_decoder_output_maps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_e2e(args):
    run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M')}"
    log_msg(f"Run name: {run_name}")
    log_msg(f"Running E2E training with inputs:\n {vars(args)}")

    # 1. Data discovery
    data_dir = Path(args.data_dir)
    all_tifs = sorted(list(data_dir.glob("*.tif")))
    mask_paths = [data_dir / (p.stem + "_24label.png") for p in all_tifs]

    if args.max_images:
        all_tifs = all_tifs[:args.max_images]
        mask_paths = mask_paths[:args.max_images]

    # 2. Train/val/test split by image
    random.seed(42)
    indices = list(range(len(all_tifs)))
    random.shuffle(indices)

    n = len(indices)
    if n < 5:
        log_msg(f"Small dataset ({n} images). Using manual QA splits.")
        train_idx = indices[0:1]
        val_idx = indices[1:2]
        test_idx = indices[2:]
    else:
        train_idx = indices[:int(n * 0.7)]
        val_idx = indices[int(n * 0.7):int(n * 0.85)]
        test_idx = indices[int(n * 0.85):]

    train_imgs = [all_tifs[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_imgs = [all_tifs[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]
    test_imgs = [all_tifs[i] for i in test_idx]
    test_masks = [mask_paths[i] for i in test_idx]

    log_msg(f"Split: {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")

    # 3. Datasets and loaders
    log_msg("Initialising datasets...")
    train_ds = FBPRawDataset(train_imgs, train_masks,
                             patch_size=args.patch_size,
                             stride=args.stride,
                             augment=True)
    val_ds = FBPRawDataset(val_imgs, val_masks,
                           patch_size=args.patch_size,
                           stride=args.stride,
                           augment=False)
    test_ds = FBPRawDataset(test_imgs, test_masks,
                            patch_size=args.patch_size,
                            stride=args.stride,
                            augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialise partially unfrozen encoder and decoder ensemble
    log_msg(f"Initialising encoder with {args.n_unfrozen_blocks} unfrozen blocks...")
    encoder_model = initialize_clay_encoder_partial_unfreeze(
        n_unfrozen_blocks=args.n_unfrozen_blocks
    )

    log_msg("Initialising decoder ensemble...")
    decoder = DecoderEnsemble(
        M=args.ensemble_size,
        in_channels=1024,
        embed_dim=args.decoder_embed_dim,
        num_classes=args.num_classes
    )
    decoder.to(DEVICE)

    # 5. Optimiser — separate LR for unfrozen encoder params and decoder
    trainable_encoder_params = [p for p in encoder_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': trainable_encoder_params, 'lr': args.lr_encoder, 'weight_decay': 0.01},
        {'params': decoder.parameters(), 'lr': args.lr_decoder, 'weight_decay': 0.05}
    ])

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        progress = float(epoch - args.warmup_epochs) / float(max(1, args.num_epochs - args.warmup_epochs))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # 6. Resume logic
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
            log_msg("Decoder weights loaded.")

        if encoder_ckpt and encoder_ckpt.exists():
            log_msg(f"Resuming encoder from {encoder_ckpt}...")
            enc_ckpt = torch.load(encoder_ckpt, map_location=DEVICE)
            encoder_model.load_state_dict(enc_ckpt['encoder_state_dict'], strict=False)
            log_msg("Encoder weights loaded.")

        start_epoch = args.resume_epoch
        loss_path = os.path.join(runs_dir, f"{args.run_name}_loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_history = json.load(f)
            log_msg(f"Loss history loaded — {len(loss_history['train'])} epochs so far")

        log_msg(f"Resuming from epoch {start_epoch + 1}")
        for _ in range(start_epoch):
            scheduler.step()

    # 7. Training loop
    best_val_loss = float('inf')
    log_msg("Starting training...")

    for epoch in range(start_epoch, args.num_epochs):
        log_msg(f"Starting epoch {epoch + 1}...")
        epoch_task_loss = 0

        # Only unfrozen layers in train mode — frozen blocks stay in eval
        # so their dropout (if any) is not activated during training
        encoder_model.eval()
        transformer = encoder_model.model.encoder.transformer
        for block in transformer.layers[-args.n_unfrozen_blocks:]:
            block.train()
        transformer.norm.train()
        decoder.train()

        for batch_imgs, batch_masks in train_loader:
            optimizer.zero_grad()
            batch_imgs = batch_imgs.to(DEVICE)
            batch_masks = batch_masks.to(DEVICE).long()

            with torch.cuda.amp.autocast():
                features = get_encoder_representation_partial(batch_imgs, encoder_model)
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
        log_msg(f"Epoch [{epoch + 1}/{args.num_epochs}] - Task Loss: {avg_task:.4f} | LR encoder: {enc_lr:.2e} decoder: {dec_lr:.2e}")

        # Validation
        encoder_model.eval()
        decoder.eval()
        val_loss = 0

        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs = v_imgs.to(DEVICE)
                v_masks = v_masks.to(DEVICE).long()

                with torch.cuda.amp.autocast():
                    features = get_encoder_representation_partial(v_imgs, encoder_model)
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
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(CHECKPOINT_DIR), filename=f"{run_name}_best_decoder.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder_model.state_dict(),
            }, str(CHECKPOINT_DIR), filename=f"{run_name}_best_encoder.pth")
            log_msg(f"New best val loss {best_val_loss:.6f} — saved best model")

        if (epoch + 1) % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(CHECKPOINT_DIR), filename=f"{run_name}_last_decoder.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder_model.state_dict(),
            }, str(CHECKPOINT_DIR), filename=f"{run_name}_last_encoder.pth")
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

        scheduler.step()

    # Final save
    timestamp = time.strftime("%Y%m%d_%H%M")
    save_checkpoint({
        'epoch': args.num_epochs,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(CHECKPOINT_DIR), filename=f"{run_name}_final_decoder_{timestamp}.pth")
    save_checkpoint({
        'epoch': args.num_epochs,
        'encoder_state_dict': encoder_model.state_dict(),
    }, str(CHECKPOINT_DIR), filename=f"{run_name}_final_encoder_{timestamp}.pth")

    loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    log_msg("Training completed.")
    plot_loss_curves(loss_path, save_name=f"{run_name}_loss_curves")

    # Load best model for evaluation
    best_decoder_path = CHECKPOINT_DIR / f"{run_name}_best_decoder.pth"
    if best_decoder_path.exists():
        log_msg("Loading best decoder for evaluation...")
        ckpt = torch.load(best_decoder_path, map_location=DEVICE)
        decoder.load_state_dict(ckpt['model_state_dict'])

    best_encoder_path = CHECKPOINT_DIR / f"{run_name}_best_encoder.pth"
    if best_encoder_path.exists():
        log_msg("Loading best encoder for evaluation...")
        enc_ckpt = torch.load(best_encoder_path, map_location=DEVICE)
        encoder_model.load_state_dict(enc_ckpt['encoder_state_dict'], strict=False)

    log_msg("Running evaluation...")
    evaluate_test_set_e2e(encoder_model, decoder, test_loader, criterion, args, run_name)

    return decoder, encoder_model


def evaluate_test_set_e2e(encoder_model, decoder, test_loader, criterion, args, run_name):
    encoder_model.eval()
    decoder.eval()

    num_classes = args.num_classes
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_conf_sums = np.zeros(num_bins)
    bin_acc_sums = np.zeros(num_bins, dtype=np.int64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)
    patch_count = 0

    vis_global_indices = set(random.sample(
        range(len(test_loader.dataset)),
        min(3, len(test_loader.dataset))
    ))
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (test_imgs, test_masks) in enumerate(test_loader):
            test_imgs = test_imgs.to(DEVICE)

            with torch.cuda.amp.autocast():
                features = get_encoder_representation_partial(test_imgs, encoder_model)
                all_preds = decoder(features)
                mean_logits = all_preds.mean(dim=0)
                mean_probs = torch.softmax(mean_logits, dim=1)
                class_maps = torch.argmax(mean_probs, dim=1).cpu().numpy()
                conf_maps = torch.max(mean_probs, dim=1)[0].cpu().numpy()

            for b in range(test_imgs.shape[0]):
                g_idx = batch_idx * test_loader.batch_size + b

                if g_idx in vis_global_indices and vis_count < 3:
                    img_p, mask_p, x, y = test_loader.dataset.get_patch_info(g_idx)
                    img_stem = Path(img_p).stem

                    log_msg(f"Vis patch: {img_stem} x={x} y={y}")

                    raw_patch = None
                    if Path(img_p).exists():
                        with rasterio.open(img_p) as src:
                            win = Window(x, y, 224, 224)
                            img = src.read([1, 2, 3], window=win).astype(np.float32)
                            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                            raw_patch = np.transpose(img, (1, 2, 0))

                    single_feat = features[b].unsqueeze(0)
                    mean_probs_vis, class_map_vis, var_map, ent_map, mi_map = get_decoder_output_maps(
                        decoder, single_feat, save_name=f"{run_name}_patch_{g_idx}"
                    )
                    gt_vis = test_masks[b].squeeze().cpu().numpy()

                    visualise_all_metrics(
                        class_map=class_map_vis,
                        variance_map=var_map,
                        total_entropy=ent_map,
                        mi_map=mi_map,
                        ground_truth=gt_vis,
                        hide_unlabelled=args.hide_unlabelled_pixels,
                        save_name=f"{run_name}_test_patch_{g_idx}",
                        raw_patch=raw_patch,
                        patch_info=f"{img_stem} x={x} y={y}"
                    )
                    del mean_probs_vis, class_map_vis, var_map, ent_map, mi_map, raw_patch
                    plt.close('all')
                    torch.cuda.empty_cache()
                    gc.collect()
                    vis_count += 1

                gt = test_masks[b].squeeze().cpu().numpy()
                if (gt > 0).sum() < 100:
                    continue
                mask = gt > 0
                pred_flat = class_maps[b][mask]
                true_flat = gt[mask]
                conf_flat = conf_maps[b][mask]

                np.add.at(conf_matrix, (true_flat, pred_flat), 1)

                bin_indices = np.digitize(conf_flat, bin_boundaries[1:-1])
                for bin_idx in range(num_bins):
                    in_bin = bin_indices == bin_idx
                    if in_bin.sum() > 0:
                        bin_conf_sums[bin_idx] += conf_flat[in_bin].sum()
                        bin_acc_sums[bin_idx] += (pred_flat[in_bin] == true_flat[in_bin]).sum()
                        bin_counts[bin_idx] += in_bin.sum()

                patch_count += 1

    iou_per_class = np.zeros(num_classes - 1)
    for c in range(1, num_classes):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou_per_class[c - 1] = tp / denom

    present = conf_matrix[1:, :].sum(axis=1) > 0
    global_miou = iou_per_class[present].mean() if present.any() else 0.0
    global_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

    bin_accs = np.where(bin_counts > 0, bin_acc_sums / bin_counts, 0.0)
    bin_confs = np.where(bin_counts > 0, bin_conf_sums / bin_counts, 0.0)
    total_samples = bin_counts.sum()
    global_ece = (np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / total_samples
                  if total_samples > 0 else 0.0)
    plot_reliability_diagram(bin_accs, bin_counts, save_name=f"{run_name}_reliability")

    log_msg(f"FINAL TEST RESULTS ({patch_count} patches):")
    log_msg(f"Global mIoU: {global_miou:.4f} | Acc: {global_acc:.4f} | ECE: {global_ece:.4f}")

    log_msg("Per-class IoU:")
    for class_idx, iou in enumerate(iou_per_class):
        log_msg(f"  {FBP_CLASSES[class_idx + 1]}: {iou:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    results = {
        "run_name": run_name,
        "global_miou": float(global_miou),
        "global_accuracy": float(global_acc),
        "global_ece": float(global_ece),
        "num_patches": patch_count,
        "per_class_iou": {FBP_CLASSES[i + 1]: float(iou) for i, iou in enumerate(iou_per_class)}
    }
    results_path = os.path.join(runs_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end partial-unfreeze encoder + decoder training")

    # Paths
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./results")

    # Data
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=None)

    # Model
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=25)

    # Encoder fine-tuning
    parser.add_argument("--n_unfrozen_blocks", type=int, default=4,
                        help="Number of final transformer blocks to unfreeze (plus norm layer)")

    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-6,
                        help="LR for unfrozen encoder blocks — keep low to avoid disrupting Clay features")
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Epochs to linearly ramp LR from 0 to target before cosine decay")
    parser.add_argument("--hide_unlabelled_pixels", action="store_true")

    # Diversity
    parser.add_argument("--diversity_methods", type=str, nargs="+", default=[],
                        choices=["jsd", "pearson", "orthogonality"])
    parser.add_argument("--lam_jsd", type=float, default=0.0)
    parser.add_argument("--lam_pearson", type=float, default=0.0)
    parser.add_argument("--lam_orth", type=float, default=0.0)

    # Misc
    parser.add_argument("--run_name", type=str, default="e2e_partial_unfreeze")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_decoder_path", type=str, default=None)
    parser.add_argument("--resume_encoder_path", type=str, default=None)
    parser.add_argument("--resume_epoch", type=int, default=0)

    args = parser.parse_args()
    train_e2e(args)
