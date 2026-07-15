"""
train_student.py — EnDD student head training.

Loads a pre-trained, frozen DecoderEnsemble (teacher) and trains a single StudentHead
to match the teacher's predictive distribution via the Dirichlet NLL loss (EnDD).

Loss = lam_distill * endd_loss(alphas, teacher_logits, T)
     + lam_task   * CE(log(alpha/alpha0), targets, ignore_index=0)

Temperature T anneals linearly from T_start → 1.0 over training, softening teacher
distributions early on for stability (recommended in the EnDD paper).

Usage (HPC example):
    python train_student.py \
        --data_dir /beegfs/scratch/callumdempsey \
        --teacher_checkpoint /beegfs/scratch/callumdempsey/checkpoints/my_run_best_model.pth \
        --teacher_ensemble_size 3 \
        --num_epochs 30 \
        --lam_distill 1.0 \
        --lam_task 0.5 \
        --T_start 4.0 \
        --run_name student_endd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import random
import time
from pathlib import Path

from configs.config import *
from models.ensemble import DecoderEnsemble, StudentHead
from utils.dataset import BakedFeatureDataset
from utils.misc import endd_loss, evaluate_student_test_set, evaluate_uncertainty_correlation, get_temperature, save_checkpoint, log_msg
from utils.visualisation import plot_loss_curves
from torch.utils.data import DataLoader


def train_student(teacher_model, student_model, train_loader, val_loader, args, run_name):
    student_model.to(DEVICE)
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_last_checkpoint.pth"
    best_model_path = CHECKPOINT_DIR / f"{run_name}_best_model.pth"

    # "train"/"val" are kept so plot_loss_curves works; detailed keys track both losses
    loss_history = {"train": [], "val": [], "train_task": []}
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and checkpoint_path.exists():
        log_msg(f"--> Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        student_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_history = json.load(f)
        log_msg(f"--> Resumed from epoch {start_epoch}")
    elif not args.resume and checkpoint_path.exists():
        log_msg("!! WARNING: Checkpoint exists but --resume not set. Will overwrite. !!")

    log_msg(f"Starting student training | lam_distill={args.lam_distill} lam_task={args.lam_task} "
            f"T_start={args.T_start} epochs={args.num_epochs}")

    save_interval = 5

    for epoch in range(start_epoch, args.num_epochs):
        T = get_temperature(epoch, args.num_epochs, args.T_start)
        student_model.train()

        epoch_distill = 0.0
        epoch_task = 0.0

        for features, targets in train_loader:
            optimizer.zero_grad()
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()

            # Teacher forward — frozen, no gradients
            with torch.no_grad():
                teacher_logits = teacher_model(features)  # [M, B, K, 224, 224]

            # Student forward — Dirichlet alphas
            student_alphas = student_model(features)      # [B, K, 224, 224]

            d_loss = endd_loss(student_alphas, teacher_logits, temperature=T)

            t_loss = torch.tensor(0.0, device=DEVICE)
            if args.lam_task > 0:
                alpha0 = student_alphas.sum(dim=1, keepdim=True)
                log_probs = torch.log(student_alphas / alpha0 + 1e-10)
                t_loss = F.nll_loss(log_probs, targets, ignore_index=0)

            total_loss = args.lam_distill * d_loss + args.lam_task * t_loss
            total_loss.backward()
            optimizer.step()

            epoch_distill += d_loss.item()
            epoch_task += t_loss.item()

        avg_distill = epoch_distill / len(train_loader)
        avg_task = epoch_task / len(train_loader)
        loss_history["train"].append(avg_distill)
        loss_history["train_task"].append(avg_task)

        log_msg(f"Epoch [{epoch+1}/{args.num_epochs}] T={T:.2f} | "
                f"Distill: {avg_distill:.4f} | Task: {avg_task:.4f}")

        # Validation on distillation loss
        if val_loader is not None:
            student_model.eval()
            val_distill = 0.0
            with torch.no_grad():
                for v_features, _ in val_loader:
                    v_features = v_features.to(DEVICE)
                    teacher_logits_v = teacher_model(v_features)
                    student_alphas_v = student_model(v_features)
                    val_distill += endd_loss(student_alphas_v, teacher_logits_v, temperature=T).item()

            avg_val = val_distill / len(val_loader)
            loss_history["val"].append(avg_val)
            log_msg(f"Validation Distill Loss: {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_checkpoint(
                    {"epoch": epoch + 1, "model_state_dict": student_model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict()},
                    str(CHECKPOINT_DIR), filename=f"{run_name}_best_model.pth"
                )
                log_msg(f"New best val distill loss {best_val_loss:.6f} — best model saved")

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                {"epoch": epoch + 1, "model_state_dict": student_model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()},
                str(CHECKPOINT_DIR), filename=f"{run_name}_last_checkpoint.pth"
            )
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

        scheduler.step()

    # Final save
    timestamp = time.strftime("%Y%m%d_%H%M")
    save_checkpoint(
        {"epoch": args.num_epochs, "model_state_dict": student_model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        str(CHECKPOINT_DIR), filename=f"{run_name}_final_model_{timestamp}.pth"
    )
    loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    log_msg("Training complete.")

    return student_model


def run_student_training(args):
    version_str = f"patch{args.patch_size}_stride{args.stride}"
    embedding_dir = BASE_OUT / "embeddings" / "clay_v1" / version_str

    meta_path = embedding_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata at {meta_path}. Did you run extract_embeddings.py?")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta["patch_size"] != args.patch_size:
        raise ValueError(f"Metadata mismatch: data extracted with patch {meta['patch_size']}")

    if args.resume:
        run_name = args.run_name  # use exact name to locate existing checkpoint
    else:
        run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M')}"
    log_msg(f"Run name: {run_name}")

    # Identical split to train_decoders.py — same seed and ordering
    all_files = sorted(list(embedding_dir.glob("*_embeddings.pt")))
    if args.max_images is not None:
        all_files = all_files[:args.max_images]
        log_msg(f"Limited to {args.max_images} images")
    random.seed(42)
    random.shuffle(all_files)

    n = len(all_files)
    if n < 5:
        train_files = all_files[0:1]
        val_files = all_files[1:2]
        test_files = all_files[2:]
    else:
        train_files = all_files[:int(n * 0.7)]
        val_files = all_files[int(n * 0.7):int(n * 0.85)]
        test_files = all_files[int(n * 0.85):]

    log_msg(f"Split: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

    train_ds = BakedFeatureDataset(train_files, augment=True)
    val_ds = BakedFeatureDataset(val_files, augment=False)
    test_ds = BakedFeatureDataset(test_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Load teacher ensemble (frozen)
    log_msg(f"Loading teacher ensemble (M={args.teacher_ensemble_size}) from {args.teacher_checkpoint}")
    teacher_model = DecoderEnsemble(
        M=args.teacher_ensemble_size,
        in_channels=args.decoder_in_channels,
        embed_dim=args.decoder_embed_dim,
        num_classes=args.num_classes,
    )
    ckpt = torch.load(args.teacher_checkpoint, map_location=DEVICE)
    teacher_model.load_state_dict(ckpt["model_state_dict"])
    teacher_model.to(DEVICE)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    log_msg("Teacher loaded and frozen.")

    # Instantiate student
    student_model = StudentHead(
        in_channels=args.decoder_in_channels,
        embed_dim=args.decoder_embed_dim,
        num_classes=args.num_classes,
    )

    log_msg(f"Student params: {sum(p.numel() for p in student_model.parameters()):,}")

    # Train
    start_time = time.time()
    trained_student = train_student(teacher_model, student_model, train_loader, val_loader, args, run_name)
    log_msg(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    plot_loss_curves(
        os.path.join(os.getenv("OUT_DIR", "results"), "runs", f"{run_name}_loss_history.json"),
        save_name=f"{run_name}_loss_curves"
    )

    # Free teacher from GPU before evaluation — not needed from here on
    teacher_model.to("cpu")
    torch.cuda.empty_cache()

    # Load best model for evaluation
    best_model_path = CHECKPOINT_DIR / f"{run_name}_best_model.pth"
    if best_model_path.exists():
        log_msg(f"Loading best model from {best_model_path} for evaluation")
        ckpt = torch.load(best_model_path, map_location=DEVICE)
        trained_student.load_state_dict(ckpt["model_state_dict"])
    else:
        log_msg("No best model found, evaluating final model")

    evaluate_student_test_set(trained_student, test_loader, args, run_name=run_name)

    # Teacher needed again to compare teacher/student uncertainty map spatial correlation
    teacher_model.to(DEVICE)
    teacher_model.eval()
    evaluate_uncertainty_correlation(teacher_model, trained_student, test_loader, args, run_name=run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnDD Student Head Training")
    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory (containing embeddings/)")
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to trained teacher .pth checkpoint")
    parser.add_argument("--out_dir", type=str, default="./results")
    # Data params — must match the embedding extraction run
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    # Teacher params — must match the teacher training run
    parser.add_argument("--teacher_ensemble_size", type=int, default=3)
    parser.add_argument("--decoder_in_channels", type=int, default=1024)
    parser.add_argument("--decoder_embed_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=25)
    # Training params
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    # EnDD-specific params
    parser.add_argument("--lam_distill", type=float, default=1.0, help="Weight on Dirichlet NLL distillation loss")
    parser.add_argument("--lam_task", type=float, default=0.0, help="Weight on CE task loss from ground truth")
    parser.add_argument("--T_start", type=float, default=4.0, help="Initial temperature for teacher softmax annealing")
    # Misc
    parser.add_argument("--hide_unlabelled_pixels", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default="student")
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()
    run_student_training(args)
