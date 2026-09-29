"""
profile_reben_training_step.py — torch.profiler breakdown of a single reBEN e2e training step
(encoder forward, decoder forward, loss, backward, optimizer step), to see which ops actually
dominate GPU time.

Why: record_compute_cost's FLOPs/wall-clock arithmetic gives an aggregate MFU number (~32% at
the corrected V100S FP16 Tensor Core peak of ~130 TFLOPS) but can't say *where* the missing ~68%
goes - whether it's the expected dominant cost (encoder transformer matmuls) or something
unexpected eating more time than it should. This profiles the real training step directly instead
of inferring from FLOPs counts.

Mirrors train_e2e_reben.py's exact training-step logic (M=1, no diversity, no student, matching
the config whose MFU we're trying to explain) - same encoder/decoder construction, same
autocast/GradScaler/optimizer sequence - but as a standalone, read-only investigation that never
touches the actual training script.

A handful of real batches are pre-fetched once, then reused for every profiled step, so the
profiler is measuring compute time specifically, not DataLoader variability (already established
as a solved problem this session).

Usage:
    python profile_reben_training_step.py \
        --s2_root /beegfs/scratch/callumdempsey/data/reben/BigEarthNet-S2 \
        --ref_root /beegfs/scratch/callumdempsey/data/reben/Reference_Maps \
        --metadata_path /beegfs/scratch/callumdempsey/data/reben/metadata.parquet \
        --batch_size 16
"""
import argparse

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule
from torch.utils.data import DataLoader

from utils.misc import log_msg, FocalLoss
from utils.dataset_e2e import load_reben_splits, ReBENRawDataset
from utils.training import _ensemble_forward
from models.ensemble import DecoderEnsemble
from models.encoder import initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial

S2_WAVES = ReBENRawDataset.WAVELENGTHS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s2_root", type=str, required=True)
    parser.add_argument("--ref_root", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_unfrozen_blocks", type=int, default=4)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--max_patches", type=int, default=500,
                        help="Just needs to be enough for a handful of batches - doesn't affect "
                             "profiling time since only the batches actually fetched get read.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--trace_out", type=str, default="profiler_trace.json",
                        help="Chrome trace file for visual inspection (chrome://tracing or "
                             "https://ui.perfetto.dev), in addition to the printed table.")
    args = parser.parse_args()

    log_msg(f"Profiling reBEN training step: batch_size={args.batch_size}, "
            f"n_unfrozen_blocks={args.n_unfrozen_blocks}")

    # 1. Data — just enough real batches to profile, pre-fetched once so DataLoader timing
    # doesn't contaminate the compute-side profile.
    train_ds, _, _ = load_reben_splits(
        metadata_path=args.metadata_path,
        s2_root=args.s2_root,
        ref_root=args.ref_root,
        exclude_snow=True,
        exclude_cloud=True,
        max_patches=args.max_patches,
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    n_steps_needed = 6  # 1 wait + 1 warmup + 3 active + 1 spare
    cached_batches = []
    for imgs, masks in loader:
        cached_batches.append((imgs.to(DEVICE), masks.to(DEVICE).long()))
        if len(cached_batches) >= n_steps_needed:
            break
    log_msg(f"Pre-fetched {len(cached_batches)} real batches for profiling.")

    # 2. Models — mirrors train_e2e_reben.py's M=1, no-diversity, no-student setup exactly.
    encoder_model = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=args.n_unfrozen_blocks)
    decoder = DecoderEnsemble(M=1, in_channels=1024, embed_dim=args.decoder_embed_dim,
                              num_classes=args.num_classes)
    decoder.to(DEVICE)

    trainable_enc = [p for p in encoder_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': trainable_enc, 'lr': 1e-6, 'weight_decay': 0.01},
        {'params': decoder.parameters(), 'lr': 1e-4, 'weight_decay': 0.05},
    ])
    scaler = torch.cuda.amp.GradScaler()
    criterion = FocalLoss(gamma=2.0, ignore_index=0)

    encoder_model.eval()
    transformer = encoder_model.model.encoder.transformer
    for block in transformer.layers[-args.n_unfrozen_blocks:]:
        block.train()
    transformer.norm.train()
    decoder.train()

    def training_step(batch_imgs, batch_masks):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            features = get_encoder_representation_partial(batch_imgs, encoder_model, waves=S2_WAVES)
            all_preds, _, _ = _ensemble_forward(decoder, features, None)
            total_task_loss = 0
            for head_idx in range(decoder.M):
                total_task_loss += criterion(all_preds[head_idx], batch_masks)
            mean_logits = all_preds.mean(dim=0)
            total_task_loss += criterion(mean_logits, batch_masks)
            task_loss = total_task_loss / (decoder.M + 1)

        scaler.scale(task_loss).backward()
        scaler.unscale_(optimizer)
        clip_params = [p for p in encoder_model.parameters() if p.requires_grad] + list(decoder.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    # 3. Profile — wait=1 (skip entirely), warmup=1 (measured but discarded), active=3 (recorded).
    prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for i in range(5):
            imgs, masks = cached_batches[i % len(cached_batches)]
            training_step(imgs, masks)
            prof.step()

    log_msg("Profiling complete. Top 25 ops by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    prof.export_chrome_trace(args.trace_out)
    log_msg(f"Chrome trace saved to {args.trace_out} (view at chrome://tracing or https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
