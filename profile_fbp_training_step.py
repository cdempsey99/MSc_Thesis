"""
profile_fbp_training_step.py - where does an FBP frozen-encoder training step spend its time?

Why: the FBP pipeline runs at ~2.2 s/batch (V100S, M=10, JSD, batch 32) despite doing about a
third of the compute per patch of the reBEN pipeline. It isn't data-bound (4 vs 8 workers gave
identical speed, 30/09) and it isn't conv-bound either (--amp made it ~10% SLOWER, 30/09). The
working hypothesis is memory-bandwidth-bound elementwise work on the full-resolution 224x224
logits (10 heads x 32 x 25 classes x 224^2 ~ 1.6 GB fp32): the heads' progressive upsampling,
11 focal losses and the JSD, forward and backward. This measures it instead of guessing again.

Mirrors utils/training.py train_model's non-bagged training step exactly (same _ensemble_forward,
_to_224, per-head + mean focal losses, JSD, BatchNorm snapshot, clip, AdamW) on a few real
pre-fetched batches, so data loading is excluded. Two measurements:

  1. Phase timing: each phase (forward / task loss / JSD / backward / optimizer) is bracketed by
     torch.cuda.synchronize(), averaged over several steps. Coarse but unambiguous.
  2. torch.profiler: top ops by CUDA time, plus a Chrome trace for detail.

--amp and --compile reproduce those variants, so their effect can be seen phase by phase.
Read-only: never touches the training scripts or writes checkpoints.

Usage:
    python profile_fbp_training_step.py --data_dir /beegfs/scratch/callumdempsey/results \\
        --trace_out /home/users/c/callumdempsey/results/profiler_trace_fbp.json
"""
import argparse
import random
import time
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from torch.utils.data import DataLoader

from utils.misc import log_msg, FocalLoss, js_divergence_loss
from utils.dataset import BakedFeatureDataset
from utils.training import _ensemble_forward, _to_224, _bn_snapshot
from models.ensemble import DecoderEnsemble

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHASES = ["forward", "task_loss", "jsd", "backward", "optimizer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Same as train_decoders.py --data_dir")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--n_files", type=int, default=3,
                        help="Embedding files (images) to draw batches from - only a few batches are used.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--decoder_in_channels", type=int, default=1024)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--lam_jsd", type=float, default=0.1, help="0 skips the JSD phase entirely")
    parser.add_argument("--n_timed_steps", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="fp16 autocast + GradScaler, as train_decoders.py --amp")
    parser.add_argument("--compile", action="store_true",
                        help="In-place nn.Module.compile() of the decoder ensemble (default mode)")
    parser.add_argument("--trace_out", type=str, default="profiler_trace_fbp.json")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True  # as train_decoders.py
    log_msg(f"Profiling FBP training step: M={args.ensemble_size}, batch={args.batch_size}, "
            f"lam_jsd={args.lam_jsd}, amp={args.amp}, compile={args.compile}")

    # 1. A few real batches, pre-fetched to the GPU once (data loading excluded)
    embedding_dir = Path(args.data_dir) / "embeddings" / "fbp" / "clay_v1" / f"patch{args.patch_size}_stride{args.stride}"
    files = sorted(embedding_dir.glob("*_embeddings.pt"))
    random.seed(42)
    random.shuffle(files)
    ds = BakedFeatureDataset(files[:args.n_files], augment=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    n_needed = max(args.n_timed_steps, 5) + 3
    batches = []
    for feats, targets in loader:
        batches.append((feats.to(DEVICE), targets.to(DEVICE).long()))
        if len(batches) >= n_needed:
            break
    log_msg(f"Pre-fetched {len(batches)} batches: features {tuple(batches[0][0].shape)}, "
            f"targets {tuple(batches[0][1].shape)}")

    # 2. Same model / loss / optimizer as full_decoder_training_run (no hyperparameter variation)
    decoder = DecoderEnsemble(args.ensemble_size, args.decoder_in_channels, args.decoder_embed_dim,
                              args.num_classes).to(DEVICE)
    decoder.train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = FocalLoss(gamma=2.0, ignore_index=0)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    if args.compile:
        decoder.compile()

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def training_step(features, targets, timings=None):
        """One step of train_model's non-bagged path. With timings, each phase is synchronised
        and timed; without, it runs unsynchronised exactly like real training (for the profiler)."""
        def mark(name, t0):
            if timings is not None:
                sync()
                timings[name] += time.perf_counter() - t0
            return time.perf_counter()

        optimizer.zero_grad()
        _bn_snapshot(decoder)  # per-batch cost in the real loop too
        sync() if timings is not None else None
        t = time.perf_counter()

        with record_function("forward"), torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
            all_preds, kl_loss, _ = _ensemble_forward(decoder, features, None)
        t = mark("forward", t)

        with record_function("task_loss"), torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
            total_task_loss = 0
            for head_idx in range(decoder.M):
                total_task_loss += criterion(_to_224(all_preds[head_idx]), targets)
            total_task_loss += criterion(_to_224(all_preds.mean(dim=0)), targets)
            task_loss = total_task_loss / (decoder.M + 1)
        t = mark("task_loss", t)

        with record_function("jsd"):
            div = js_divergence_loss(all_preds.float()) if args.lam_jsd > 0 else 0.0
            total_loss = task_loss + args.lam_jsd * div + 0.0 * kl_loss
        t = mark("jsd", t)

        with record_function("backward"):
            scaler.scale(total_loss).backward()
        t = mark("backward", t)

        with record_function("optimizer"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        mark("optimizer", t)

    # 3. Warmup (cuDNN benchmark search, compile tracing) - excluded from all numbers
    warm_start = time.perf_counter()
    for i in range(3):
        training_step(*batches[i % len(batches)])
    sync()
    log_msg(f"Warmup done in {time.perf_counter() - warm_start:.1f}s (excluded)")
    torch.cuda.reset_peak_memory_stats()

    # 4. Phase timing
    timings = {p: 0.0 for p in PHASES}
    for i in range(args.n_timed_steps):
        training_step(*batches[i % len(batches)], timings=timings)
    total = sum(timings.values())
    log_msg(f"Phase timing (synchronised, mean of {args.n_timed_steps} steps): "
            f"{total / args.n_timed_steps:.3f} s/step total")
    for p in PHASES:
        per_step = timings[p] / args.n_timed_steps
        print(f"    {p:<10} {per_step:8.3f} s   {100 * timings[p] / total:5.1f}%")
    log_msg(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # 5. torch.profiler (unsynchronised, like real training)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
                 record_shapes=True, profile_memory=True) as prof:
        for i in range(5):
            training_step(*batches[i % len(batches)])
            prof.step()
    log_msg("Top 30 ops by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_chrome_trace(args.trace_out)
    log_msg(f"Chrome trace saved to {args.trace_out} (open at https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
