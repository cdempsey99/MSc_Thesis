"""
profile_flops.py — one-time FLOPs profiling for a given model configuration.

FLOPs are a static property of the architecture/config, not something that varies
run to run like wall-clock time. So this only needs running once per distinct
config (M, decoder_embed_dim, n_unfrozen_blocks, with/without student), not
instrumented into the real training jobs. Multiply the reported per-batch
forward+backward FLOPs by known epoch/batch counts for a total training-cost
estimate.

Usage:
    # Frozen-embeddings decoder only (FBP AS1-AS4, reBEN frozen) — encoder isn't
    # part of the training loop here, embeddings are pre-baked.
    python profile_flops.py --mode decoder_only --ensemble_size 10 --decoder_embed_dim 512 \
        --num_classes 25 --batch_size 16 --include_student

    # Full e2e (reBEN/SAR finetuned) — encoder is trained jointly.
    python profile_flops.py --mode e2e --n_unfrozen_blocks 4 --ensemble_size 1 \
        --decoder_embed_dim 512 --num_classes 20 --in_channels 2 --batch_size 16
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from models.ensemble import DecoderEnsemble, StudentHead
from utils.misc import endd_loss, log_msg, js_divergence_loss, pearson_diversity_loss, orthogonality_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _config_key(config):
    """Deterministic, human-readable filename for a given FLOPs-relevant config."""
    diversity = config.get("diversity_methods") or []
    parts = [
        config.get("dataset", "unknown"),
        config.get("encoder_type", "unknown"),
        f"nblocks{config['n_unfrozen_blocks']}" if config.get("n_unfrozen_blocks") is not None else "nblocksNA",
        f"M{config['ensemble_size']}",
        f"embed{config['decoder_embed_dim']}",
        f"classes{config['num_classes']}",
        f"inch{config['in_channels']}" if config.get("in_channels") is not None else "inchNA",
        f"batch{config['batch_size']}",
        "student" if config.get("include_student") else "nostudent",
        ("div_" + "-".join(sorted(diversity))) if diversity else "nodiv",
    ]
    return "_".join(str(p) for p in parts)


def ensure_flops_profile(config, decoder, student=None, encoder=None, waves=None):
    """
    Checks results/flops_costs/{key}.json for a saved FLOPs profile matching `config`.
    If found, logs and returns it without recomputing anything. If not, profiles one
    forward+backward pass using the actual (already-instantiated, already-on-device)
    decoder/student/encoder objects from the real training run — no separate dummy
    models, no redundant encoder weight reload — then saves the result and returns it.

    Safe to call from inside a real training script: uses dummy random inputs/targets
    and zeroes gradients before and after, so it doesn't contaminate the real
    optimizer state or accumulated gradients.

    config keys: dataset, encoder_type ("frozen"/"finetuned"), n_unfrozen_blocks
    (None for frozen), ensemble_size, decoder_embed_dim, num_classes, in_channels
    (None for decoder_only), batch_size, include_student, diversity_methods
    (list of "jsd"/"pearson"/"orthogonality", or None/[] — these have real FLOPs
    cost at M>1 and are included in the profiled forward pass if present).
    """
    out_dir = Path(os.getenv("OUT_DIR", "results")) / "flops_costs"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _config_key(config)
    path = out_dir / f"{key}.json"

    if path.exists():
        with open(path) as f:
            profile = json.load(f)
        log_msg(f"FLOPs profile already exists for config '{key}', skipping "
                f"({profile['forward_backward_flops']:,} FLOPs/batch)")
        return profile

    log_msg(f"No FLOPs profile found for config '{key}', profiling now...")

    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    targets = torch.randint(0, num_classes, (batch_size, 224, 224), device=DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    decoder.zero_grad(set_to_none=True)
    if student is not None:
        student.zero_grad(set_to_none=True)
    if encoder is not None:
        encoder.zero_grad(set_to_none=True)

    with FlopCounterMode(display=False) as fc:
        if encoder is not None:
            from models.encoder import get_encoder_representation_partial
            in_channels = config["in_channels"]
            images = torch.randn(batch_size, in_channels, 224, 224, device=DEVICE)
            enc_waves = waves if waves is not None else torch.ones(in_channels, dtype=torch.float32)
            features = get_encoder_representation_partial(images, encoder, waves=enc_waves)
        else:
            features = torch.randn(batch_size, 1024, 28, 28, device=DEVICE)

        all_preds = decoder(features)
        loss = sum(criterion(all_preds[m], targets) for m in range(decoder.M)) / decoder.M

        diversity_methods = config.get("diversity_methods") or []
        if "jsd" in diversity_methods:
            loss = loss + js_divergence_loss(all_preds.float())
        if "pearson" in diversity_methods:
            loss = loss + pearson_diversity_loss(all_preds.float())
        if "orthogonality" in diversity_methods:
            loss = loss + orthogonality_loss(decoder)

        if student is not None:
            alphas = student(features)
            loss = loss + endd_loss(alphas, all_preds, temperature=1.0)
        loss.backward()

    total_flops = fc.get_total_flops()

    decoder.zero_grad(set_to_none=True)
    if student is not None:
        student.zero_grad(set_to_none=True)
    if encoder is not None:
        encoder.zero_grad(set_to_none=True)

    profile = {**config, "forward_backward_flops": total_flops}
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    log_msg(f"FLOPs profile saved to {path} ({total_flops:,} FLOPs/batch)")
    return profile


def start_compute_tracking():
    """
    Call once right before the training loop begins. Resets CUDA's peak-memory
    high-water-mark so the value read at the end of training reflects only this
    run, not whatever happened earlier in the process (e.g. the FLOPs profiling
    forward/backward pass itself). Returns a start timestamp for wall-clock timing.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return time.time()


def record_compute_cost(flops_profile, epochs_completed, batches_per_epoch, start_time, run_name):
    """
    Combines the ahead-of-time FLOPs profile (static, from ensure_flops_profile,
    doesn't vary run to run) with per-run wall-clock time and peak GPU memory
    (both only knowable after the fact, and genuinely vary run to run due to
    cluster contention) into one combined summary. Saved to
    results/runs/{run_name}_compute_cost.json.
    """
    wall_clock_seconds = time.time() - start_time
    per_batch_flops = flops_profile["forward_backward_flops"]
    total_training_flops = per_batch_flops * batches_per_epoch * epochs_completed

    peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None

    summary = {
        "run_name": run_name,
        "flops_per_batch": per_batch_flops,
        "batches_per_epoch": batches_per_epoch,
        "epochs_completed": epochs_completed,
        "total_training_flops": total_training_flops,
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_minutes": wall_clock_seconds / 60.0,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{run_name}_compute_cost.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    mem_str = f"{peak_gpu_memory_gb:.2f} GB peak GPU" if peak_gpu_memory_gb is not None else "GPU memory unavailable"
    log_msg(f"Compute cost: {total_training_flops:,.0f} total FLOPs ({epochs_completed} epochs x "
            f"{batches_per_epoch} batches x {per_batch_flops:,} FLOPs/batch) | "
            f"{wall_clock_seconds / 60:.1f} min wall-clock | {mem_str}")
    log_msg(f"Compute cost summary saved to {path}")
    return summary


def profile_decoder_only(args):
    decoder = DecoderEnsemble(M=args.ensemble_size, in_channels=1024,
                               embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
    decoder.to(DEVICE)

    student = None
    if args.include_student:
        student = StudentHead(in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
        student.to(DEVICE)

    ensure_flops_profile(
        config={
            "dataset": args.dataset,
            "encoder_type": "frozen",
            "n_unfrozen_blocks": None,
            "ensemble_size": args.ensemble_size,
            "decoder_embed_dim": args.decoder_embed_dim,
            "num_classes": args.num_classes,
            "in_channels": None,
            "batch_size": args.batch_size,
            "include_student": args.include_student,
        },
        decoder=decoder,
        student=student,
    )


def profile_e2e(args):
    from models.encoder import initialize_clay_encoder_partial_unfreeze

    encoder = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=args.n_unfrozen_blocks)
    decoder = DecoderEnsemble(M=args.ensemble_size, in_channels=1024,
                               embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
    decoder.to(DEVICE)

    ensure_flops_profile(
        config={
            "dataset": args.dataset,
            "encoder_type": "finetuned",
            "n_unfrozen_blocks": args.n_unfrozen_blocks,
            "ensemble_size": args.ensemble_size,
            "decoder_embed_dim": args.decoder_embed_dim,
            "num_classes": args.num_classes,
            "in_channels": args.in_channels,
            "batch_size": args.batch_size,
            "include_student": False,
        },
        decoder=decoder,
        encoder=encoder,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually trigger a cached FLOPs profile for a given model configuration")
    parser.add_argument("--mode", choices=["decoder_only", "e2e"], required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="e.g. fbp, reben, reben_sar — must match what the real training script used")
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--include_student", action="store_true",
                        help="decoder_only mode: also profile a StudentHead trained in parallel")
    parser.add_argument("--n_unfrozen_blocks", type=int, default=4, help="e2e mode only")
    parser.add_argument("--in_channels", type=int, default=3,
                        help="e2e mode only: 3 for optical (FBP/reBEN), 2 for SAR")
    args = parser.parse_args()

    if args.mode == "decoder_only":
        profile_decoder_only(args)
    else:
        profile_e2e(args)
