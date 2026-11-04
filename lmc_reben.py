"""
lmc_reben.py - linear mode connectivity between two fine-tuned reBEN models.

Question: do two independently fine-tuned models (each with its own fine-tuned Clay encoder and its own
decoder, both started from the same pretrained Clay weights) sit in the same loss basin? Walk the straight
line between them in weight space, theta(a) = (1-a) * theta_A + a * theta_B, and evaluate every blended
model. A flat path (no barrier) = same basin (Neyshabur et al. 2020; Frankle et al. 2020); a rise in the
middle = separate basins (Fort et al. 2019).

Two corrections make the test meaningful:
  1. Decoder alignment (activation matching, cf. Ainsworth et al. 2022 "Git Re-Basin"). The decoders start
     from different random inits, so channel k of A need not do the same job as channel k of B; blending
     unmatched channels creates an artificial barrier. Each hidden decoder layer's channels of B are
     matched one-to-one to A's by maximising the activation correlation (Hungarian algorithm) and B is
     permuted accordingly - B computes exactly the same function afterwards (checked). The encoders need
     no alignment: both start from the same pretrained weights, so their units correspond already.
  2. BatchNorm recalibration (Jordan et al. 2022 "REPAIR"). At every a the decoder's BatchNorm running
     statistics are reset and re-estimated on training batches, otherwise the stale statistics of a
     blended net give a spurious barrier. (The encoder uses LayerNorm - no running statistics.)

The unaligned path is evaluated too, for comparison. Output: <OUT_DIR>/runs/lmc_<tag>.json (all metrics
per a, barrier heights) and <OUT_DIR>/losses/lmc_<tag>.png.

Checkpoints are given as stems, as in reeval_auroc.py (<stem>_decoder.pth / <stem>_encoder.pth, e.g.
<run_name>_last or <run_name>_ep20). For ensemble checkpoints pick one head of each with --head_a/--head_b;
the two heads must have the same architecture (arch-var heads of different sizes cannot be interpolated).
Sanity check: --run_b equal to --run_a (and the same head) must give a flat path.

Forward passes only, so it runs fine on the older tesla cards, away from the main training jobs:
    sbatch -J lmc_reben -p TestAndBuild,gpu_short --gres=gpu:tesla:1 --mem=32G --cpus-per-task=8 --time=03:00:00 \
        -o ~/logs/lmc_reben_%j.out --wrap "cd ~/MSc_Thesis && export OUT_DIR=/home/users/c/callumdempsey/results && \
        /beegfs/scratch/callumdempsey/miniconda/envs/thesis_env/bin/python3 -u lmc_reben.py \
        --run_a <stem_A> --run_b <stem_B> --tag <name> --clay_native_forward --clay_normalise_s2 \
        --metadata_path .../metadata.parquet --s2_root .../BigEarthNet-S2 --ref_root .../Reference_Maps"
"""
import argparse
import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset

from configs.config import CHECKPOINT_DIR
from models.encoder import get_encoder_representation_partial
from reeval_auroc import load_models, _miou
from utils.dataset_e2e import load_reben_splits, ReBENRawDataset
from utils.misc import log_msg, FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------
# Decoder head as a chain of (conv, bn, act) stages + final classifier
# ---------------------------------------------------------------------------------------------
def head_stages(head):
    """The SegFormerDecoderHead in forward order: list of (conv, bn, act) hidden stages and the
    classifier. Every hidden stage's output channels can be permuted freely (the next conv absorbs it)."""
    stages = [(head.linear_fusion, head.bn1, head.activation1),
              (head.spatial_refine1, head.bn2, head.activation2)]
    if head.num_refine_blocks >= 2:
        stages.append((head.spatial_refine2, head.bn3, head.activation3))
    for block in head.extra_refine_blocks:
        stages.append((block[0], block[1], block[2]))
    stages.append((head.bottleneck, head.bn4, head.activation4))
    return stages, head.classifier


class SingleModel(torch.nn.Module):
    """One encoder + one decoder head, returning [B, C, 224, 224] logits."""

    def __init__(self, encoder, head, waves):
        super().__init__()
        self.encoder, self.head, self.waves = encoder, head, waves

    def features(self, x):
        with torch.cuda.amp.autocast():
            return get_encoder_representation_partial(x, self.encoder, waves=self.waves)

    def forward(self, x):
        feats = self.features(x)
        with torch.cuda.amp.autocast():
            return self.head(feats)


# ---------------------------------------------------------------------------------------------
# Step 1: activation matching
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def match_heads(model_a, model_b, loader):
    """Per hidden stage, the permutation of B's channels that maximises their correlation with A's
    channels (post-activation, over all pixels of the alignment patches). Returns a list of index
    tensors perm, meaning B's channel perm[i] plays the role of A's channel i."""
    stages_a, _ = head_stages(model_a.head)
    stages_b, _ = head_stages(model_b.head)
    acts = {"a": {}, "b": {}}
    hooks = []
    for name, stages in (("a", stages_a), ("b", stages_b)):
        for k, (_, _, act) in enumerate(stages):
            hooks.append(act.register_forward_hook(
                lambda m, i, o, name=name, k=k: acts[name].__setitem__(k, o.detach())))
    sums = None
    n = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        model_a(x)
        model_b(x)
        if sums is None:
            sums = [{"a": 0., "b": 0., "aa": 0., "bb": 0., "ab": 0.} for _ in stages_a]
        for k in range(len(stages_a)):
            a = acts["a"][k].float().permute(0, 2, 3, 1).reshape(-1, acts["a"][k].shape[1])  # [P, C]
            b = acts["b"][k].float().permute(0, 2, 3, 1).reshape(-1, acts["b"][k].shape[1])
            s = sums[k]
            s["a"] = s["a"] + a.sum(0); s["b"] = s["b"] + b.sum(0)
            s["aa"] = s["aa"] + (a * a).sum(0); s["bb"] = s["bb"] + (b * b).sum(0)
            s["ab"] = s["ab"] + a.T @ b
        n += a.shape[0]
    for h in hooks:
        h.remove()
    perms = []
    for k, s in enumerate(sums):
        mean_a, mean_b = s["a"] / n, s["b"] / n
        std_a = (s["aa"] / n - mean_a ** 2).clamp(min=1e-8).sqrt()
        std_b = (s["bb"] / n - mean_b ** 2).clamp(min=1e-8).sqrt()
        corr = (s["ab"] / n - mean_a[:, None] * mean_b[None, :]) / (std_a[:, None] * std_b[None, :])
        rows, cols = linear_sum_assignment(corr.cpu().numpy(), maximize=True)
        perm = torch.as_tensor(cols[np.argsort(rows)], device=DEVICE)
        matched = corr[torch.arange(len(perm), device=DEVICE), perm]
        log_msg(f"  stage {k}: {len(perm)} channels | mean matched corr {matched.mean():.3f} "
                f"(identity pairing: {corr.diagonal().mean():.3f})")
        perms.append(perm)
    return perms


@torch.no_grad()
def permute_head(head, perms):
    """Reorder each hidden stage's output channels by perms[k] (conv out, BN params/stats) and the
    following layer's input channels. The head computes exactly the same function afterwards."""
    stages, classifier = head_stages(head)
    for k, (conv, bn, _) in enumerate(stages):
        p = perms[k]
        conv.weight.copy_(conv.weight[p])
        if conv.bias is not None:
            conv.bias.copy_(conv.bias[p])
        for t in (bn.weight, bn.bias, bn.running_mean, bn.running_var):
            t.copy_(t[p])
        nxt = stages[k + 1][0] if k + 1 < len(stages) else classifier
        nxt.weight.copy_(nxt.weight[:, p])


# ---------------------------------------------------------------------------------------------
# Steps 2-4: interpolate, recalibrate BatchNorm, evaluate
# ---------------------------------------------------------------------------------------------
def blend_state(sd_a, sd_b, alpha, keys):
    """(1 - alpha) * A + alpha * B on the given float keys; everything else from A."""
    out = dict(sd_a)
    for k in keys:
        out[k] = (1.0 - alpha) * sd_a[k] + alpha * sd_b[k]
    return out


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches):
    """Reset the decoder's BatchNorm statistics and re-estimate them (cumulative average) on training
    batches; everything else stays in eval mode (no dropout, encoder untouched)."""
    model.eval()
    bns = [m for m in model.head.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(DEVICE))
    model.eval()


@torch.no_grad()
def evaluate(model, loader, num_classes, focal):
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=DEVICE)
    nll_sum, focal_sum, n_lab = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE).long()
        if y.dim() == 4:
            y = y.squeeze(1)
        logits = model(x).float()
        lab = y > 0
        if not lab.any():
            continue
        n = int(lab.sum())
        nll_sum += F.cross_entropy(logits, y, ignore_index=0, reduction="sum").item()
        focal_sum += focal(logits, y).item() * n
        n_lab += n
        pred = logits.argmax(dim=1)
        conf += torch.bincount(y[lab] * num_classes + pred[lab], minlength=num_classes ** 2).view(num_classes, num_classes)
    conf = conf.cpu().numpy()
    return {"nll": nll_sum / max(n_lab, 1), "focal": focal_sum / max(n_lab, 1),
            "acc": float(np.trace(conf) / max(conf.sum(), 1)), "miou": _miou(conf, False)}


def barrier(values, alphas, higher_is_worse=True):
    """Largest excess of the path over the straight line between its endpoints."""
    v = np.asarray(values, dtype=float)
    a = np.asarray(alphas, dtype=float)
    line = (1 - a) * v[0] + a * v[-1]
    excess = (v - line) if higher_is_worse else (line - v)
    i = int(np.argmax(excess))
    return {"height": float(excess[i]), "at_alpha": float(a[i])}


def run_path(name, model, sd_a_enc, sd_b_enc, enc_keys, sd_a_dec, sd_b_dec, dec_keys, alphas,
             bn_loader, train_loader, val_loader, args, focal):
    rows = []
    for alpha in alphas:
        model.encoder.load_state_dict(blend_state(sd_a_enc, sd_b_enc, alpha, enc_keys), strict=False)
        model.head.load_state_dict(blend_state(sd_a_dec, sd_b_dec, alpha, dec_keys))
        recalibrate_bn(model, bn_loader, args.n_bn_batches)
        tr = evaluate(model, train_loader, args.num_classes, focal)
        va = evaluate(model, val_loader, args.num_classes, focal)
        rows.append({"alpha": float(alpha), "train": tr, "val": va})
        log_msg(f"[{name}] a={alpha:.2f} | train nll {tr['nll']:.4f} acc {tr['acc']:.4f} | "
                f"val nll {va['nll']:.4f} acc {va['acc']:.4f} miou {va['miou']:.4f}")
    barriers = {f"{split}_{m}": barrier([r[split][m] for r in rows], alphas, higher_is_worse=(m in ("nll", "focal")))
                for split in ("train", "val") for m in ("nll", "focal", "acc", "miou")}
    for k in ("train_nll", "val_nll", "val_acc"):
        log_msg(f"[{name}] barrier {k}: {barriers[k]['height']:.4f} at a={barriers[k]['at_alpha']:.2f}")
    return {"path": rows, "barriers": barriers}


def main():
    p = argparse.ArgumentParser(description="Linear mode connectivity between two fine-tuned reBEN models")
    p.add_argument("--run_a", required=True, help="checkpoint stem of model A (without _decoder.pth)")
    p.add_argument("--run_b", required=True, help="checkpoint stem of model B")
    p.add_argument("--head_a", type=int, default=0, help="head index of A (ensemble checkpoints)")
    p.add_argument("--head_b", type=int, default=0, help="head index of B")
    p.add_argument("--tag", required=True, help="name for the output files")
    p.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    p.add_argument("--metadata_path", required=True)
    p.add_argument("--s2_root", required=True)
    p.add_argument("--ref_root", required=True)
    p.add_argument("--n_unfrozen_blocks", type=int, default=4)
    p.add_argument("--clay_native_forward", action="store_true")
    p.add_argument("--clay_normalise_s2", action="store_true")
    p.add_argument("--clay_single_norm", action="store_true")
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--n_alphas", type=int, default=11, help="points on the path, endpoints included")
    p.add_argument("--n_align", type=int, default=2000, help="training patches for activation matching")
    p.add_argument("--n_train_eval", type=int, default=2000, help="training patches evaluated along the path")
    p.add_argument("--n_val", type=int, default=3000, help="validation patches evaluated along the path")
    p.add_argument("--n_bn_batches", type=int, default=50, help="training batches for BatchNorm recalibration")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_unaligned", action="store_true", help="only evaluate the aligned path")
    args = p.parse_args()
    log_msg(f"LMC: {vars(args)}")

    # Data: fixed, seeded patch subsets, no augmentation anywhere
    train_ds, val_ds, _ = load_reben_splits(metadata_path=args.metadata_path, s2_root=args.s2_root,
                                            ref_root=args.ref_root, clay_normalise=args.clay_normalise_s2)
    train_ds.augment = False
    g = np.random.default_rng(args.seed)
    tr_idx = g.permutation(len(train_ds))
    n_bn = args.n_bn_batches * args.batch_size
    align_idx = tr_idx[:args.n_align]
    bn_idx = tr_idx[args.n_align:args.n_align + n_bn]
    tr_eval_idx = tr_idx[args.n_align + n_bn:args.n_align + n_bn + args.n_train_eval]
    va_idx = g.permutation(len(val_ds))[:args.n_val]
    mk = lambda ds, idx: DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    align_loader, bn_loader = mk(train_ds, align_idx), mk(train_ds, bn_idx)
    train_loader, val_loader = mk(train_ds, tr_eval_idx), mk(val_ds, va_idx)
    log_msg(f"Patches: align {len(align_idx)} | BN {len(bn_idx)} | train eval {len(tr_eval_idx)} | val {len(va_idx)}")

    # Models
    waves = ReBENRawDataset.WAVELENGTHS
    stems = [r if os.path.isabs(r) else os.path.join(args.checkpoint_dir, r) for r in (args.run_a, args.run_b)]
    enc_a, dec_a, _ = load_models(stems[0], "finetuned", args.n_unfrozen_blocks, args.clay_single_norm, args.clay_native_forward)
    enc_b, dec_b, _ = load_models(stems[1], "finetuned", args.n_unfrozen_blocks, args.clay_single_norm, args.clay_native_forward)
    head_a = copy.deepcopy(dec_a.heads[args.head_a]).to(DEVICE).eval()
    head_b = copy.deepcopy(dec_b.heads[args.head_b]).to(DEVICE).eval()
    shapes_a = {k: v.shape for k, v in head_a.state_dict().items()}
    shapes_b = {k: v.shape for k, v in head_b.state_dict().items()}
    if shapes_a != shapes_b:
        raise ValueError("Heads A and B have different architectures - cannot interpolate (arch-var heads?)")
    model_a, model_b = SingleModel(enc_a, head_a, waves), SingleModel(enc_b, head_b, waves)
    focal = FocalLoss(gamma=2.0, ignore_index=0)

    sd_a_enc = {k: v.detach().clone() for k, v in enc_a.state_dict().items()}
    sd_b_enc = {k: v.detach().clone() for k, v in enc_b.state_dict().items()}
    enc_keys = [k for k, v in sd_a_enc.items() if v.is_floating_point() and not torch.equal(v, sd_b_enc[k])]
    log_msg(f"Encoder: {len(enc_keys)} of {len(sd_a_enc)} tensors differ between A and B (interpolated); "
            f"the rest (frozen blocks) are identical")

    # Endpoint check before anything is changed
    ref_a = evaluate(model_a, val_loader, args.num_classes, focal)
    ref_b = evaluate(model_b, val_loader, args.num_classes, focal)
    log_msg(f"Endpoints as loaded: A val nll {ref_a['nll']:.4f} acc {ref_a['acc']:.4f} | "
            f"B val nll {ref_b['nll']:.4f} acc {ref_b['acc']:.4f}")

    # Step 1: align B's head to A's, and check B's function is unchanged
    log_msg("Activation matching (B's decoder channels -> A's)...")
    perms = match_heads(model_a, model_b, align_loader)
    head_b_aligned = copy.deepcopy(head_b)
    permute_head(head_b_aligned, perms)
    with torch.no_grad():
        x, _ = next(iter(val_loader))
        feats = model_b.features(x.to(DEVICE)).float()
        out_orig, out_perm = head_b(feats), head_b_aligned(feats)
        max_diff = (out_orig - out_perm).abs().max().item()
    log_msg(f"Permutation check: max |B - B_aligned| on one batch = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise RuntimeError(f"Permuted B differs from B (max diff {max_diff:.2e}) - alignment bug")

    sd_a_dec = {k: v.detach().clone() for k, v in head_a.state_dict().items()}
    sd_b_dec = {k: v.detach().clone() for k, v in head_b.state_dict().items()}
    sd_b_dec_al = {k: v.detach().clone() for k, v in head_b_aligned.state_dict().items()}
    dec_keys = [k for k, v in sd_a_dec.items() if v.is_floating_point()]

    # Steps 2-4 on a working copy of A (its encoder and head get overwritten at every alpha)
    alphas = np.linspace(0.0, 1.0, args.n_alphas)
    work = SingleModel(enc_a, head_a, waves)
    results = {"args": vars(args), "endpoints_as_loaded": {"A": ref_a, "B": ref_b},
               "permutation_check_max_diff": max_diff, "n_encoder_tensors_interpolated": len(enc_keys)}
    results["aligned"] = run_path("aligned", work, sd_a_enc, sd_b_enc, enc_keys, sd_a_dec, sd_b_dec_al, dec_keys,
                                  alphas, bn_loader, train_loader, val_loader, args, focal)
    if not args.skip_unaligned:
        results["unaligned"] = run_path("unaligned", work, sd_a_enc, sd_b_enc, enc_keys, sd_a_dec, sd_b_dec, dec_keys,
                                        alphas, bn_loader, train_loader, val_loader, args, focal)

    out_dir = os.getenv("OUT_DIR", "results")
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "losses"), exist_ok=True)
    json_path = os.path.join(out_dir, "runs", f"lmc_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {json_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for path in ("aligned", "unaligned"):
        if path not in results:
            continue
        rows = results[path]["path"]
        a = [r["alpha"] for r in rows]
        ls = "-" if path == "aligned" else "--"
        axes[0].plot(a, [r["val"]["nll"] for r in rows], ls, marker="o", label=f"val ({path})")
        axes[0].plot(a, [r["train"]["nll"] for r in rows], ls, marker=".", alpha=0.6, label=f"train ({path})")
        axes[1].plot(a, [r["val"]["acc"] for r in rows], ls, marker="o", label=f"val ({path})")
    axes[0].set(xlabel="interpolation coefficient a (0 = A, 1 = B)", ylabel="NLL", title="Loss along the linear path")
    axes[1].set(xlabel="interpolation coefficient a (0 = A, 1 = B)", ylabel="pixel accuracy", title="Accuracy along the linear path")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Linear mode connectivity: {args.tag}")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "losses", f"lmc_{args.tag}.png")
    fig.savefig(png_path, dpi=150)
    log_msg(f"Plot saved to {png_path}")


if __name__ == "__main__":
    main()
