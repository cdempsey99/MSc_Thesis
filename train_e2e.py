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

from utils.dataset_e2e import FBPRawDataset
from utils.misc import log_msg, save_checkpoint, evaluate_test_set
from utils.visualisation import plot_loss_curves
from models.ensemble import DecoderEnsemble
from models.lora_encoder import (
    initialize_clay_encoder_with_lora,
    get_encoder_representation_grad,
    save_lora_weights
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_e2e(args):
    run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M')}"
    log_msg(f"Run name: {run_name}")

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

    # 4. Initialise LoRA encoder and decoder ensemble
    log_msg("Initialising LoRA encoder...")
    base_encoder, lora_encoder = initialize_clay_encoder_with_lora(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    base_encoder.to(DEVICE)
    lora_encoder.to(DEVICE)

    log_msg("Initialising decoder ensemble...")
    decoder = DecoderEnsemble(
        M=args.ensemble_size,
        in_channels=1024,
        embed_dim=args.decoder_embed_dim,
        num_classes=args.num_classes
    )
    decoder.to(DEVICE)

    # 5. Optimiser — separate LR for encoder and decoder
    optimizer = torch.optim.AdamW([
        {'params': lora_encoder.parameters(), 'lr': args.lr_encoder, 'weight_decay': 0.01},
        {'params': decoder.parameters(), 'lr': args.lr_decoder, 'weight_decay': 0.05}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 6. Training loop
    best_val_loss = float('inf')
    loss_history = {"train": [], "val": []}
    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    log_msg("Starting training...")

    for epoch in range(args.num_epochs):
        log_msg(f"Starting epoch {epoch + 1}...")
        epoch_task_loss = 0

        lora_encoder.train()
        decoder.train()

        for batch_imgs, batch_masks in train_loader:
            optimizer.zero_grad()

            batch_imgs = batch_imgs.to(DEVICE)
            batch_masks = batch_masks.to(DEVICE).long()

            # Forward pass through LoRA encoder
            features = get_encoder_representation_grad(
                batch_imgs, base_encoder, lora_encoder
            )  # [B, 1024, 28, 28]

            # Forward pass through decoder ensemble
            all_preds = decoder(features)  # [M, B, 25, 224, 224]

            # Loss — per head + mean
            total_task_loss = 0
            for head_idx in range(decoder.M):
                total_task_loss += criterion(all_preds[head_idx], batch_masks)

            mean_logits = all_preds.mean(dim=0)
            total_task_loss += criterion(mean_logits, batch_masks)
            task_loss = total_task_loss / (decoder.M + 1)

            task_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()

        avg_task = epoch_task_loss / len(train_loader)
        loss_history["train"].append(avg_task)
        log_msg(f"Epoch [{epoch + 1}/{args.num_epochs}] - Task Loss: {avg_task:.4f}")

        # Validation
        lora_encoder.eval()
        decoder.eval()
        val_loss = 0

        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs = v_imgs.to(DEVICE)
                v_masks = v_masks.to(DEVICE).long()

                features = get_encoder_representation_grad(
                    v_imgs, base_encoder, lora_encoder
                )
                all_preds = decoder(features)

                total_val_loss = 0
                for head_idx in range(decoder.M):
                    total_val_loss += criterion(all_preds[head_idx], v_masks)
                mean_logits = all_preds.mean(dim=0)
                total_val_loss += criterion(mean_logits, v_masks)
                val_loss += (total_val_loss / (decoder.M + 1)).item()

        avg_val_loss = val_loss / len(val_loader)
        loss_history["val"].append(avg_val_loss)
        log_msg(f"Validation Loss: {avg_val_loss:.8f}")

        # Best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save decoder weights
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(CHECKPOINT_DIR), filename=f"{run_name}_best_decoder.pth")
            # Save LoRA weights
            save_lora_weights(lora_encoder,
                              str(CHECKPOINT_DIR / f"{run_name}_best_lora"))
            log_msg(f"New best val loss {best_val_loss:.6f} — saved best model")

        # Periodic save every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(CHECKPOINT_DIR), filename=f"{run_name}_last_decoder.pth")
            save_lora_weights(lora_encoder,
                              str(CHECKPOINT_DIR / f"{run_name}_last_lora"))
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

        scheduler.step()

    # Final save
    save_checkpoint({
        'epoch': args.num_epochs,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(CHECKPOINT_DIR), filename=f"{run_name}_final_decoder.pth")
    save_lora_weights(lora_encoder, str(CHECKPOINT_DIR / f"{run_name}_final_lora"))

    loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    log_msg("Training completed.")

    # Plot loss curves
    plot_loss_curves(loss_path, save_name=f"{run_name}_loss_curves")

    # Load best model for evaluation
    best_decoder_path = CHECKPOINT_DIR / f"{run_name}_best_decoder.pth"
    if best_decoder_path.exists():
        log_msg("Loading best decoder for evaluation...")
        checkpoint = torch.load(best_decoder_path, map_location=DEVICE)
        decoder.load_state_dict(checkpoint['model_state_dict'])

    # Evaluation — need a wrapper that runs encoder first
    log_msg("Running evaluation...")
    results = evaluate_test_set_e2e(
        base_encoder, lora_encoder, decoder,
        test_loader, criterion, args, run_name
    )

    return decoder, lora_encoder


def evaluate_test_set_e2e(base_encoder, lora_encoder, decoder, test_loader, criterion, args, run_name):
    """
    Evaluation for end-to-end model — runs raw images through
    LoRA encoder before decoder evaluation.
    """
    base_encoder.eval()
    lora_encoder.eval()
    decoder.eval()

    all_preds_global = []
    all_gts_global = []
    all_conf_global = []
    patch_count = 0

    with torch.no_grad():
        for batch_idx, (test_imgs, test_masks) in enumerate(test_loader):
            test_imgs = test_imgs.to(DEVICE)

            # Run through encoder
            features = get_encoder_representation_grad(
                test_imgs, base_encoder, lora_encoder
            )

            # Run through decoder
            all_preds = decoder(features)
            mean_logits = all_preds.mean(dim=0)
            mean_probs = torch.softmax(mean_logits, dim=1)
            class_maps = torch.argmax(mean_probs, dim=1).cpu().numpy()
            conf_maps = torch.max(mean_probs, dim=1)[0].cpu().numpy()

            for b in range(test_imgs.shape[0]):
                gt = test_masks[b].squeeze().cpu().numpy()
                if (gt > 0).sum() < 100:
                    continue
                mask = gt > 0
                all_preds_global.append(class_maps[b][mask])
                all_gts_global.append(gt[mask])
                all_conf_global.append(conf_maps[b][mask])
                patch_count += 1

    all_preds_arr = np.concatenate(all_preds_global)
    all_gts_arr = np.concatenate(all_gts_global)
    all_conf_arr = np.concatenate(all_conf_global)

    global_miou = jaccard_score(all_gts_arr, all_preds_arr,
                                average="macro", labels=np.unique(all_gts_arr),
                                zero_division=0)
    global_acc = np.mean(all_preds_arr == all_gts_arr)

    log_msg(f"FINAL TEST RESULTS ({patch_count} patches):")
    log_msg(f"Global mIoU: {global_miou:.4f} | Acc: {global_acc:.4f}")

    for class_idx in range(24):
        per_class = jaccard_score(all_gts_arr, all_preds_arr,
                                  average=None, labels=list(range(1, 25)),
                                  zero_division=0)
    log_msg("Per-class IoU:")
    for class_idx, iou in enumerate(per_class):
        log_msg(f"  {FBP_CLASSES[class_idx + 1]}: {iou:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    results = {
        "run_name": run_name,
        "global_miou": float(global_miou),
        "global_accuracy": float(global_acc),
        "num_patches": patch_count,
    }
    results_path = os.path.join(runs_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end LoRA encoder + decoder training")

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

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--hide_unlabelled_pixels", action="store_true")

    # Diversity
    parser.add_argument("--diversity_methods", type=str, nargs="+", default=[],
                        choices=["jsd", "pearson", "orthogonality"])
    parser.add_argument("--lam_jsd", type=float, default=0.0)
    parser.add_argument("--lam_pearson", type=float, default=0.0)
    parser.add_argument("--lam_orth", type=float, default=0.0)

    # Misc
    parser.add_argument("--run_name", type=str, default="e2e_run")

    args = parser.parse_args()
    train_e2e(args)


