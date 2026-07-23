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
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from models.ensemble import DecoderEnsemble, StudentHead
from utils.misc import endd_loss, log_msg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _config_key(config):
    """Deterministic, human-readable filename for a given FLOPs-relevant config."""
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
    (None for decoder_only), batch_size, include_student.
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


def profile_decoder_only(args):
    decoder = DecoderEnsemble(M=args.ensemble_size, in_channels=1024,
                               embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
    decoder.to(DEVICE)

    student = None
    if args.include_student:
        student = StudentHead(in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
        student.to(DEVICE)

    features = torch.randn(args.batch_size, 1024, 28, 28, device=DEVICE)
    targets = torch.randint(0, args.num_classes, (args.batch_size, 224, 224), device=DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with FlopCounterMode(display=True) as fc:
        all_preds = decoder(features)
        loss = sum(criterion(all_preds[m], targets) for m in range(decoder.M)) / decoder.M
        if student is not None:
            alphas = student(features)
            loss = loss + endd_loss(alphas, all_preds, temperature=1.0)
        loss.backward()

    log_msg(f"Decoder-only config: M={args.ensemble_size}, embed_dim={args.decoder_embed_dim}, "
            f"student={args.include_student}, batch_size={args.batch_size}")
    log_msg(f"Forward+backward FLOPs per batch: {fc.get_total_flops():,}")


def profile_e2e(args):
    from models.encoder import initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial

    encoder = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=args.n_unfrozen_blocks)
    decoder = DecoderEnsemble(M=args.ensemble_size, in_channels=1024,
                               embed_dim=args.decoder_embed_dim, num_classes=args.num_classes)
    decoder.to(DEVICE)

    waves = torch.ones(args.in_channels, dtype=torch.float32)  # values don't affect FLOPs, only shape does
    images = torch.randn(args.batch_size, args.in_channels, 224, 224, device=DEVICE)
    targets = torch.randint(0, args.num_classes, (args.batch_size, 224, 224), device=DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with FlopCounterMode(display=True) as fc:
        features = get_encoder_representation_partial(images, encoder, waves=waves)
        all_preds = decoder(features)
        loss = sum(criterion(all_preds[m], targets) for m in range(decoder.M)) / decoder.M
        loss.backward()

    log_msg(f"E2E config: n_unfrozen_blocks={args.n_unfrozen_blocks}, M={args.ensemble_size}, "
            f"embed_dim={args.decoder_embed_dim}, in_channels={args.in_channels}, batch_size={args.batch_size}")
    log_msg(f"Forward+backward FLOPs per batch: {fc.get_total_flops():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time FLOPs profiling for a given model configuration")
    parser.add_argument("--mode", choices=["decoder_only", "e2e"], required=True)
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
