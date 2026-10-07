import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from configs.config import *
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse
import math
import numpy as np
import os

from utils.misc import log_msg, save_checkpoint, FocalLoss, evaluate_error_localization, ensemble_uncertainty_and_pred
from utils.dataset_e2e import BakedReBENDataset
from utils.visualisation import plot_loss_curves, plot_confusion_matrix
from models.ensemble import DecoderEnsemble
from profile_flops import ensure_flops_profile, start_compute_tracking, record_compute_cost

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _WithIndex(torch.utils.data.Dataset):
    """Wraps a dataset so each item also returns its index - needed by --head_mask_mode fixed to
    look up a patch's permanent head assignment. Module-level so DataLoader workers can pickle it.
    (Same as train_e2e_reben._WithIndex; kept separate so the two pipelines stay independent.)"""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        feats, mask = self.ds[i]
        return feats, mask, i


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
    auroc_epistemic = []  # same pixels as auroc_entropy: does head disagreement (MI) rank the errors?
    auroc_aleatoric = []  # ... and the mean per-head entropy, for comparison
    # Decision-level diversity counts over all labelled pixels, kept on GPU until the end
    # (identical definitions to train_e2e_reben.evaluate_test_set_reben)
    div_pixels = 0
    pair_disagree = torch.zeros(decoder.M, decoder.M, device=DEVICE)
    pair_double_fault = torch.zeros(decoder.M, decoder.M, device=DEVICE)
    head_errors = torch.zeros(decoder.M, device=DEVICE)
    ens_errors = torch.zeros((), device=DEVICE)
    oracle_correct = torch.zeros((), device=DEVICE)

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
                collect_auroc = collected < AUROC_MAX
                ens_wrong_t = torch.argmax(mean_probs_f32, dim=1) != masks_t
                if collect_auroc:
                    auroc_entropy.append(ent_t[labelled_t].cpu().numpy())
                    auroc_errors.append(ens_wrong_t.float()[labelled_t].cpu().numpy())
                # UQ decomposition
                total_ent_sum += ent_t[labelled_t].sum().item()
                aleat = torch.zeros_like(ent_t)
                for m in range(decoder.M):
                    hp = all_head_probs_f32[m]
                    aleat += -(hp * torch.log(hp.clamp(1e-10))).sum(dim=1)
                aleat /= decoder.M
                aleatoric_sum += aleat[labelled_t].sum().item()
                uq_count += labelled_t.sum().item()
                if collect_auroc and decoder.M > 1:
                    auroc_aleatoric.append(aleat[labelled_t].cpu().numpy())
                    auroc_epistemic.append((ent_t - aleat).clamp(min=0)[labelled_t].cpu().numpy())
                if decoder.M > 1:
                    gt_l = masks_t[labelled_t]                                           # [P]
                    head_pred_l = torch.argmax(all_head_probs_f32, dim=2)[:, labelled_t]  # [M, P]
                    head_wrong_l = head_pred_l != gt_l                                   # [M, P]
                    div_pixels += gt_l.numel()
                    head_errors += head_wrong_l.sum(dim=1)
                    ens_errors += ens_wrong_t[labelled_t].sum()
                    oracle_correct += (~head_wrong_l).any(dim=0).sum()
                    for i in range(decoder.M):
                        for j in range(i + 1, decoder.M):
                            pair_disagree[i, j] += (head_pred_l[i] != head_pred_l[j]).sum()
                            pair_double_fault[i, j] += (head_wrong_l[i] & head_wrong_l[j]).sum()
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

    # Decision-level diversity (M > 1 only; None otherwise) - same definitions as the fine-tuned
    # pipeline: normalised disagreement (Fort et al. 2019) = mean pairwise disagreement / ensemble
    # error rate; error correlation ratio = per-pair double-fault / (e_i * e_j), averaged over pairs
    # (1 = independent errors); oracle accuracy = at least one head right.
    norm_disagreement = error_corr_ratio = oracle_acc = ens_acc_labelled = None
    auroc_epistemic_score = auroc_aleatoric_score = None
    if decoder.M > 1 and div_pixels > 0:
        n = float(div_pixels)
        head_err_rate = (head_errors / n).cpu().numpy()
        ens_err_rate = ens_errors.item() / n
        pair_dis = (pair_disagree / n).cpu().numpy()
        pair_df = (pair_double_fault / n).cpu().numpy()
        pairs = [(i, j) for i in range(decoder.M) for j in range(i + 1, decoder.M)]
        mean_disagreement = float(np.mean([pair_dis[i, j] for i, j in pairs]))
        norm_disagreement = mean_disagreement / ens_err_rate if ens_err_rate > 0 else None
        ratios = [pair_df[i, j] / (head_err_rate[i] * head_err_rate[j]) for i, j in pairs
                  if head_err_rate[i] > 0 and head_err_rate[j] > 0]
        error_corr_ratio = float(np.mean(ratios)) if ratios else None
        oracle_acc = oracle_correct.item() / n
        ens_acc_labelled = 1.0 - ens_err_rate
        if auroc_epistemic and len(np.unique(all_err)) > 1:
            auroc_epistemic_score = float(roc_auc_score(all_err, np.concatenate(auroc_epistemic)))
            auroc_aleatoric_score = float(roc_auc_score(all_err, np.concatenate(auroc_aleatoric)))

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
    if norm_disagreement is not None or error_corr_ratio is not None:
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        log_msg(f"Diversity: norm_disagreement={fmt(norm_disagreement)} | error_corr_ratio={fmt(error_corr_ratio)} | "
                f"oracle_acc={fmt(oracle_acc)} (ensemble acc on same pixels={fmt(ens_acc_labelled)}) | "
                f"AUROC epistemic={fmt(auroc_epistemic_score)} aleatoric={fmt(auroc_aleatoric_score)} "
                f"total={global_auroc:.4f}")
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
        "normalised_disagreement": norm_disagreement,
        "error_correlation_ratio": error_corr_ratio,
        "oracle_accuracy": oracle_acc,
        "ensemble_accuracy_labelled": ens_acc_labelled,
        "auroc_epistemic": auroc_epistemic_score,
        "auroc_aleatoric": auroc_aleatoric_score,
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

    # Same subset semantics as load_reben_splits(max_patches, max_val_patches) in the fine-tuned
    # pipeline: first N train, first N//5 val and test, max_val_patches overriding val. The
    # extractor wrote each split in metadata order (shuffle=False), so the first N baked patches
    # are exactly the patches the fine-tuned --max_patches N runs use.
    if args.max_patches:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.max_patches, len(train_ds))))
        val_ds   = torch.utils.data.Subset(val_ds,   range(min(args.max_patches // 5, len(val_ds))))
        test_ds  = torch.utils.data.Subset(test_ds,  range(min(args.max_patches // 5, len(test_ds))))
    if args.max_val_patches:
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.max_val_patches, len(val_ds))))
    if args.max_patches or args.max_val_patches:
        log_msg(f"Using subset: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    # persistent_workers avoids respawning workers every epoch (only valid with num_workers > 0).
    # Training loader also yields each patch's index, used by --head_mask_mode fixed.
    persistent = args.num_workers > 0
    train_loader = DataLoader(_WithIndex(train_ds), batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Per-head sample masks (Bootstrapped-DQN-style, Osband et al. 2016) - same mechanism as the
    # fine-tuned pipeline: each head's task loss only counts the patches its mask selects.
    use_head_masks = args.head_mask_prob < 1.0
    fixed_head_masks = None
    if use_head_masks:
        if args.head_mask_mode == "fixed":
            g = torch.Generator().manual_seed(args.head_mask_seed)
            fixed_head_masks = (torch.rand(len(train_ds), args.ensemble_size, generator=g)
                                < args.head_mask_prob).to(DEVICE)  # [N, M], drawn once for the whole run
        log_msg(f"Per-head sample masking: p={args.head_mask_prob}, mode={args.head_mask_mode} "
                f"(each head trains on ~{args.head_mask_prob * args.batch_size:.0f} of {args.batch_size} patches per batch)")
    if args.no_mean_logit_loss:
        log_msg("Ensemble-mean (averaged-logit) task loss term disabled: per-head losses only")

    # 2. Model
    decoder = DecoderEnsemble(
        M=args.ensemble_size,
        in_channels=1024,
        embed_dim=args.decoder_embed_dim,
        num_classes=args.num_classes,
        architecture_variation=args.architecture_variation,
    )
    decoder.to(DEVICE)
    if args.architecture_variation:
        log_msg(f"Per-head architecture variation enabled - refine_blocks: {decoder.refine_block_counts}, "
                f"embed_dim: {decoder.head_embed_dims}")

    flops_profile = ensure_flops_profile(
        config={
            "dataset": "reben",
            "encoder_type": "frozen",
            "n_unfrozen_blocks": None,
            "ensemble_size": args.ensemble_size,
            "decoder_embed_dim": args.decoder_embed_dim,
            "num_classes": args.num_classes,
            "architecture_variation": args.architecture_variation,
            "in_channels": None,
            "batch_size": args.batch_size,
            "include_student": False,
        },
        decoder=decoder,
    )

    # 3. Optimiser. fused=True runs the AdamW update as a few multi-tensor kernels (as in the
    # fine-tuned pipeline).
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.05, fused=True)

    def warmup_lambda(epoch):
        return min(1.0, float(epoch + 1) / float(max(1, args.warmup_epochs)))

    # --lr_schedule cosine: identical to train_e2e_reben - linear warmup, then cosine decay from the
    # base lr (reached on the last warmup epoch) to lr_min_ratio * base on the final epoch.
    # Extensions (resume at/after the saved planned end with a larger --num_epochs) get an SGDR warm
    # restart over the added epochs; resuming mid-plan with a different --num_epochs raises.
    cosine_state = {'end': args.num_epochs, 'restart': None}

    def _cosine(t):
        return args.lr_min_ratio + (1.0 - args.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    def cosine_lambda(epoch):
        if cosine_state['restart'] is not None and epoch >= cosine_state['restart']:
            restart = cosine_state['restart']
            return _cosine((epoch - restart) / max(1, args.num_epochs - 1 - restart))
        if epoch < args.warmup_epochs - 1:
            return warmup_lambda(epoch)
        return _cosine((epoch - (args.warmup_epochs - 1)) / max(1, cosine_state['end'] - args.warmup_epochs))

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lambda if args.lr_schedule == "cosine" else warmup_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-8
    )
    log_msg(f"LR schedule: {args.lr_schedule} ({args.warmup_epochs} warmup epochs"
            + (f", cosine to {args.lr_min_ratio}x base by epoch {args.num_epochs})" if args.lr_schedule == "cosine"
               else ", then halve on 5-epoch val plateau)"))
    # Early stopping would cut the cosine curve short and make runs incomparable - off for cosine.
    use_early_stopping = args.lr_schedule != "cosine"
    if not use_early_stopping:
        log_msg("Early stopping disabled (cosine schedule runs the full num_epochs)")
    scaler = torch.cuda.amp.GradScaler()

    if args.compile:
        # In-place compile keeps state_dict keys unchanged (no '_orig_mod.' prefix)
        decoder.compile()
        log_msg("Decoder compiled (torch.compile, in place)")

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

    saved_lr_plan = None
    if args.resume and args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if ckpt_path.exists():
            log_msg(f"Resuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            decoder.load_state_dict(ckpt['model_state_dict'])
            if not args.no_load_optimizer:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                for g in optimizer.param_groups:  # load_state_dict overwrites fused - re-enable it
                    g['fused'] = True
            if ckpt.get('lr_schedule') == "cosine":
                saved_lr_plan = (ckpt.get('lr_schedule_num_epochs'), ckpt.get('lr_schedule_restart_epoch'))
            start_epoch = args.resume_epoch
            loss_path = os.path.join(runs_dir, f"{args.run_name}_loss_history.json")
            if os.path.exists(loss_path):
                with open(loss_path) as f:
                    loss_history = json.load(f)
            log_msg(f"Resuming from epoch {start_epoch + 1}")
        else:
            log_msg(f"WARNING: checkpoint not found at {ckpt_path}, starting fresh")

    if args.lr_schedule == "cosine" and start_epoch > 0 and start_epoch < args.num_epochs:
        if saved_lr_plan is None:
            raise ValueError("--lr_schedule cosine resume needs a last_decoder checkpoint saved by a cosine "
                             "run; this one has no cosine schedule info (plateau run, or a 'best' checkpoint?)")
        saved_num_epochs, saved_restart = saved_lr_plan
        if saved_num_epochs == args.num_epochs:
            cosine_state.update(end=saved_num_epochs, restart=saved_restart)
        elif args.num_epochs > saved_num_epochs and start_epoch >= saved_num_epochs:
            cosine_state.update(end=saved_num_epochs, restart=start_epoch)
            log_msg(f"Extension {saved_num_epochs} -> {args.num_epochs} epochs: cosine warm restart "
                    f"from base lr over epochs {start_epoch + 1}-{args.num_epochs}")
        else:
            raise ValueError(f"Cosine schedule was planned for {saved_num_epochs} epochs; resuming at epoch "
                             f"{start_epoch} with --num_epochs {args.num_epochs} is ambiguous. Finish the "
                             f"original {saved_num_epochs} epochs first, then extend from there.")

    # LambdaLR sets lr = base * lambda(step count) absolutely, so stepping start_epoch times
    # repositions it exactly. Cosine needs this for every epoch, plateau only inside warmup.
    if args.lr_schedule == "cosine" or start_epoch < args.warmup_epochs:
        for _ in range(start_epoch):
            warmup_scheduler.step()

    # 5. Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_msg("Starting training...")

    compute_start_epoch = start_epoch
    training_start_time = start_compute_tracking()

    clip_params = list(decoder.parameters())  # built once rather than every batch

    for epoch in range(start_epoch, args.num_epochs):
        log_msg(f"Starting epoch {epoch + 1}...")
        decoder.train()
        # Loss accumulated on the GPU and read back once per epoch (a per-batch .item() forces a sync)
        epoch_loss = torch.zeros((), device=DEVICE)

        train_phase_start = time.perf_counter()
        for batch_features, batch_masks, batch_idx in train_loader:
            optimizer.zero_grad()
            # non_blocking lets the host->GPU copy overlap with compute (loaders use pin_memory=True)
            batch_features = batch_features.to(DEVICE, non_blocking=True)
            batch_masks    = batch_masks.to(DEVICE, non_blocking=True).long()

            with torch.cuda.amp.autocast():
                all_preds = decoder(batch_features)

                if use_head_masks:
                    if fixed_head_masks is not None:
                        head_masks = fixed_head_masks[batch_idx.to(DEVICE)].T  # [M, B]
                    else:
                        head_masks = torch.rand(decoder.M, batch_features.shape[0], device=DEVICE) < args.head_mask_prob
                total_loss = 0
                n_loss_terms = 0
                for head_idx in range(decoder.M):
                    if use_head_masks:
                        sel = head_masks[head_idx]
                        if not sel.any():  # vanishingly rare at p=0.5, batch 32: head simply skips this step
                            continue
                        total_loss += criterion(all_preds[head_idx][sel], batch_masks[sel])
                    else:
                        total_loss += criterion(all_preds[head_idx], batch_masks)
                    n_loss_terms += 1
                if not args.no_mean_logit_loss:
                    mean_logits = all_preds.mean(dim=0)
                    total_loss += criterion(mean_logits, batch_masks)
                    n_loss_terms += 1
                loss = total_loss / max(n_loss_terms, 1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.detach()

        train_phase_time = time.perf_counter() - train_phase_start
        n_batches = len(train_loader)
        log_msg(f"Train phase: {train_phase_time:.1f}s | {train_phase_time / n_batches:.4f} s/batch | "
                f"{train_phase_time / (n_batches * args.batch_size):.4f} s/patch")

        avg_train_loss = epoch_loss.item() / n_batches
        loss_history["train"].append(avg_train_loss)
        log_msg(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {avg_train_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Validation (always includes the averaged-logit term, as in the fine-tuned pipeline, so
        # val losses stay comparable across settings)
        decoder.eval()
        val_loss = torch.zeros((), device=DEVICE)
        with torch.no_grad():
            for v_features, v_masks in val_loader:
                v_features = v_features.to(DEVICE, non_blocking=True)
                v_masks    = v_masks.to(DEVICE, non_blocking=True).long()
                with torch.cuda.amp.autocast():
                    all_preds = decoder(v_features)
                    total_val = 0
                    for head_idx in range(decoder.M):
                        total_val += criterion(all_preds[head_idx], v_masks)
                    mean_logits = all_preds.mean(dim=0)
                    total_val += criterion(mean_logits, v_masks)
                    val_loss += (total_val / (decoder.M + 1)).detach()

        avg_val_loss = val_loss.item() / len(val_loader)
        loss_history["val"].append(avg_val_loss)
        log_msg(f"Validation Loss: {avg_val_loss:.8f}")

        # Stepped before any checkpoint is written, so a saved checkpoint carries the lr for the
        # NEXT epoch (same ordering fix as the fine-tuned pipeline)
        if args.lr_schedule == "cosine" or epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(avg_val_loss)

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
            if use_early_stopping:
                log_msg(f"No improvement for {epochs_no_improve}/{args.early_stopping_patience} epochs")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                {'epoch': epoch + 1, 'model_state_dict': decoder.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'lr_schedule': args.lr_schedule,
                 'lr_schedule_num_epochs': args.num_epochs,
                 'lr_schedule_restart_epoch': cosine_state['restart']},
                str(CHECKPOINT_DIR), filename=f"{run_name}_last_decoder.pth"
            )
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

        if use_early_stopping and epochs_no_improve >= args.early_stopping_patience:
            log_msg(f"Early stopping triggered at epoch {epoch + 1}")
            break

    record_compute_cost(flops_profile, epochs_completed=epoch + 1 - compute_start_epoch,
                        batches_per_epoch=len(train_loader), start_time=training_start_time,
                        run_name=run_name)

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

    def _teacher_forward(feats):
        with torch.cuda.amp.autocast():
            return decoder(feats)

    evaluate_error_localization(
        lambda feats: ensemble_uncertainty_and_pred(_teacher_forward(feats), args.num_classes),
        test_loader, args, run_name=run_name, who="teacher"
    )

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
    parser.add_argument("--early_stopping_patience", type=int,   default=20,
                        help="Plateau schedule only; ignored with --lr_schedule cosine")
    parser.add_argument("--lr_schedule", type=str, default="plateau", choices=["plateau", "cosine"],
                        help="plateau: warmup then halve lr after 5 epochs without a new best val loss "
                             "(original behaviour). cosine: warmup then cosine decay to "
                             "lr_min_ratio * base lr at the final epoch (as train_e2e_reben).")
    parser.add_argument("--lr_min_ratio", type=float, default=0.01,
                        help="Cosine schedule only: final lr as a fraction of the base lr.")
    parser.add_argument("--max_patches", type=int, default=None,
                        help="Use only the first N train patches (and first N//5 val/test), matching "
                             "train_e2e_reben --max_patches N. Default: all extracted patches.")
    parser.add_argument("--max_val_patches", type=int, default=None,
                        help="Cap val patches independently of --max_patches")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes for train/val/test loaders")
    parser.add_argument("--compile", action="store_true",
                        help="In-place torch.compile of the decoder ensemble")

    # Diversity mechanisms (same flags and behaviour as train_e2e_reben)
    parser.add_argument("--architecture_variation", action="store_true",
                        help="Per-head decoder architecture variation (depth/width)")
    parser.add_argument("--head_mask_prob", type=float, default=1.0,
                        help="Per-head sample masking (Bootstrapped DQN): each head's task loss only uses a "
                             "fraction p of each batch's patches. 1.0 = off (original behaviour).")
    parser.add_argument("--head_mask_mode", type=str, default="batch", choices=["batch", "fixed"],
                        help="batch: new random masks every batch. fixed: each patch permanently assigned "
                             "to a random subset of heads for the whole run.")
    parser.add_argument("--head_mask_seed", type=int, default=0,
                        help="Seed for the fixed per-patch masks")
    parser.add_argument("--no_mean_logit_loss", action="store_true",
                        help="Drop the averaged-logit term from the training task loss (per-head losses only)")

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
