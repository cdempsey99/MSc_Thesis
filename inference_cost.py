"""
inference_cost.py - inference cost per 224x224 patch of every model configuration in the thesis.

Measures each component separately (Clay encoder, one default decoder head, the ten-head AS2 teacher, the
ten-head architecture-variation AS4 teacher, the EnDD student) and composes the configurations from them:

    AS1 (single model)          encoder + 1 head
    AS2 teacher (M=10)          encoder + 10 heads
    AS4 AV teacher (M=10)       encoder + 10 AV heads
    student                     encoder + student
    deep ensemble (M=10)        10 x (encoder + 1 head)   - computed, not trained

Per component and configuration: forward FLOPs (torch.utils.flop_counter.FlopCounterMode, the same counter as
profile_flops.py, so inference and training costs are comparable), parameter count, latency (CUDA events after
warm-up, mean over repeats), throughput (patches/s) and peak GPU memory. Heads include the bilinear upsampling to
224x224; the teacher rows also include the uncertainty computation (softmax mean, entropy, mutual information),
the student row its Dirichlet decomposition. FlopCounterMode counts matmul/conv FLOPs only, so element-wise work
(softmax, entropies, interpolation) adds latency but no FLOPs.

The encoder runs at inference on every dataset: on FBP its features were pre-computed for training, but a new image
still has to pass through it. Weights are random (cost does not depend on weight values); architectures are built
exactly as in training. Encoder per dataset: FBP 3 bands (GF-2 B/G/R, GSD 4.0 m), reBEN 3 bands (S2 R/G/B,
GSD 10*120/224 m), SAR 2 bands (VV/VH); all with --clay_native_forward as in the final runs.

Usage (one short GPU job; batch 32 = throughput, batch 1 = per-patch latency; fp32 and fp16 autocast):
    export OUT_DIR=/home/users/c/callumdempsey/results
    python inference_cost.py --batch_sizes 32 1 --tag v100s
Output: <OUT_DIR>/runs/inference_cost_<tag>.json and a printed table.
"""
import argparse
import json
import os
import time

import torch

from models.encoder import (initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial,
                            FBP_GSD, REBEN_GSD, FBP_WAVES_GF2_BGR)
from models.ensemble import DecoderEnsemble, StudentHead
from utils.dataset_e2e import ReBENRawDataset, ReBENSARRawDataset
from utils.misc import log_msg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FBP_PATCHES_PER_SCENE = 960   # 30 x 32 non-overlapping 224 px patches of a 6908 x 7300 scene

DATASETS = {
    "FBP": {"waves": torch.tensor(FBP_WAVES_GF2_BGR, dtype=torch.float32), "gsd": FBP_GSD, "classes": 25},
    "reBEN": {"waves": ReBENRawDataset.WAVELENGTHS, "gsd": REBEN_GSD, "classes": 20},
    "SAR": {"waves": ReBENSARRawDataset.WAVELENGTHS, "gsd": REBEN_GSD, "classes": 20},
}


def teacher_uncertainty(logits):
    """logits [M, B, C, H, W] -> mean probs, total entropy, mutual information (as reeval_auroc.py)."""
    probs = torch.softmax(logits.float(), dim=2)
    mean_p = probs.mean(dim=0)
    total = -(mean_p * torch.log(mean_p.clamp(min=1e-10))).sum(dim=1)
    ale = (-(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=2)).mean(dim=0)
    return mean_p.argmax(dim=1), total, (total - ale).clamp(min=0)


def student_uncertainty(alphas):
    """Dirichlet alphas [B, C, H, W] -> prediction, total, epistemic (as reeval_auroc.py)."""
    alphas = alphas.float()
    a0 = alphas.sum(dim=1, keepdim=True)
    mp = alphas / a0
    total = -(mp * torch.log(mp.clamp(min=1e-10))).sum(dim=1)
    ale = torch.digamma(a0.squeeze(1) + 1) - (mp * torch.digamma(alphas + 1)).sum(dim=1)
    return mp.argmax(dim=1), total, (total - ale).clamp(min=0)


def measure(fn, n_warmup, n_iters):
    """Mean latency (ms) of fn() and peak memory (GB) over n_iters timed runs after n_warmup untimed ones."""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters, torch.cuda.max_memory_allocated() / 1024 ** 3


def count_flops(fn):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def main():
    p = argparse.ArgumentParser(description="Inference cost per patch of the thesis configurations")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 1])
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--ensemble_size", type=int, default=10)
    p.add_argument("--decoder_embed_dim", type=int, default=512)
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_iters", type=int, default=20)
    p.add_argument("--tag", default="gpu")
    args = p.parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("needs a GPU (latency and memory are measured with CUDA events)")
    gpu = torch.cuda.get_device_name(0)
    log_msg(f"Inference cost on {gpu}: {vars(args)}")
    torch.backends.cudnn.benchmark = True

    encoder = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=4, native_forward=True, gsd=REBEN_GSD)
    encoder = encoder.to(DEVICE).eval()
    n_params_encoder = sum(t.numel() for t in encoder.parameters())

    results = {"gpu": gpu, "args": vars(args), "datasets": {}}
    for ds in args.datasets:
        cfg = DATASETS[ds]
        encoder.clay_gsd = cfg["gsd"]   # read by the native-forward position encoding at every call
        waves = cfg["waves"]
        C = cfg["classes"]
        heads = {
            "head": DecoderEnsemble(M=1, in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=C),
            "as2_heads": DecoderEnsemble(M=args.ensemble_size, in_channels=1024, embed_dim=args.decoder_embed_dim,
                                         num_classes=C),
            "av_heads": DecoderEnsemble(M=args.ensemble_size, in_channels=1024, embed_dim=args.decoder_embed_dim,
                                        num_classes=C, architecture_variation=True),
            "student": StudentHead(in_channels=1024, embed_dim=args.decoder_embed_dim, num_classes=C),
        }
        for m in heads.values():
            m.to(DEVICE).eval()
        n_params = {"encoder": n_params_encoder, **{k: sum(t.numel() for t in m.parameters()) for k, m in heads.items()}}
        out = {"n_params": n_params, "runs": []}

        for prec in args.precisions:
            amp = prec == "fp16"
            for bs in args.batch_sizes:
                x = torch.randn(bs, len(waves), 224, 224, device=DEVICE)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    feats = get_encoder_representation_partial(x, encoder, waves=waves)
                feats = feats.detach()

                def run(name):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        if name == "encoder":
                            return get_encoder_representation_partial(x, encoder, waves=waves)
                        if name == "head":
                            return heads["head"](feats)
                        if name in ("as2_heads", "av_heads"):
                            return teacher_uncertainty(heads[name](feats))
                        return student_uncertainty(heads["student"](feats))

                comp = {}
                for name in ("encoder", "head", "as2_heads", "av_heads", "student"):
                    flops = count_flops(lambda: run(name)) / bs
                    ms, mem = measure(lambda: run(name), args.n_warmup, args.n_iters)
                    comp[name] = {"gflops_per_patch": flops / 1e9, "ms_per_batch": ms, "ms_per_patch": ms / bs,
                                  "peak_mem_gb": mem}
                    log_msg(f"[{ds} {prec} bs={bs}] {name}: {flops / 1e9:.2f} GFLOPs/patch | {ms:.2f} ms/batch | "
                            f"{ms / bs:.3f} ms/patch | peak {mem:.2f} GB")

                M = args.ensemble_size
                def combo(parts):
                    g = sum(f * comp[c]["gflops_per_patch"] for c, f in parts)
                    t = sum(f * comp[c]["ms_per_patch"] for c, f in parts)
                    return {"gflops_per_patch": g, "ms_per_patch": t, "patches_per_s": 1000.0 / t,
                            "params": sum(f * n_params[c] for c, f in parts)}
                configs = {
                    "AS1": combo([("encoder", 1), ("head", 1)]),
                    "AS2 teacher": combo([("encoder", 1), ("as2_heads", 1)]),
                    "AS4 AV teacher": combo([("encoder", 1), ("av_heads", 1)]),
                    "student": combo([("encoder", 1), ("student", 1)]),
                    f"deep ensemble (M={M}, computed)": combo([("encoder", M), ("head", M)]),
                }
                base = configs["AS1"]
                for c in configs.values():
                    c["rel_flops_vs_AS1"] = c["gflops_per_patch"] / base["gflops_per_patch"]
                    c["rel_time_vs_AS1"] = c["ms_per_patch"] / base["ms_per_patch"]
                    if ds == "FBP":
                        c["s_per_fbp_scene"] = c["ms_per_patch"] * FBP_PATCHES_PER_SCENE / 1000.0
                out["runs"].append({"precision": prec, "batch_size": bs, "components": comp, "configs": configs})
                del x, feats
                torch.cuda.empty_cache()
        results["datasets"][ds] = out

    out_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"inference_cost_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    log_msg(f"SUMMARY ({gpu}) - GFLOPs/patch | ms/patch | patches/s | x AS1 (FLOPs) | x AS1 (time)")
    for ds, out in results["datasets"].items():
        for run in out["runs"]:
            log_msg(f"  {ds} {run['precision']} bs={run['batch_size']}:")
            for name, c in run["configs"].items():
                scene = f" | {c['s_per_fbp_scene']:.1f} s/scene" if "s_per_fbp_scene" in c else ""
                log_msg(f"    {name:28s} {c['gflops_per_patch']:8.1f} | {c['ms_per_patch']:7.3f} | "
                        f"{c['patches_per_s']:8.1f} | {c['rel_flops_vs_AS1']:5.2f} | {c['rel_time_vs_AS1']:5.2f}{scene}")
    log_msg(f"Saved to {path}")


if __name__ == "__main__":
    main()
