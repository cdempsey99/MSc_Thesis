"""
posthoc_ensemble_reben.py - evaluate a POST-HOC ensemble built from already-trained reBEN
checkpoints, using the exact same evaluation code as the training scripts (so every number is
directly comparable with the per-run tables). Checkpoint-only, no retraining, no changes to any
existing pipeline code.

Each --member is one trained run, optionally restricted to some of its heads:
    --member AS1_reben_LR_cosine_run1_20261007_0610          (all heads of that run)
    --member AS2_reben_baseline_LR_cosine_run1_20261005_1541:0,2,4   (heads 0, 2 and 4 only)
The ensemble's "heads" are all selected heads of all members, averaged as probabilities exactly
like a normal M-head ensemble. Use cases:
  - separately trained single models (e.g. 3 fine-tuned AS1 runs -> a small deep ensemble whose
    members each fine-tuned their own last encoder blocks)
  - a head subset of one jointly trained run (e.g. 3 of the 5 AS2 heads, for an M-matched control)
  - a single run on its own (sanity check: must reproduce that run's logged test numbers)

--pipeline finetuned: members are train_e2e_reben.py runs (<run>_best_encoder.pth + _best_decoder.pth);
    each member runs its OWN encoder. Evaluated with train_e2e_reben.evaluate_test_set_reben.
--pipeline frozen: members are train_decoders_reben.py runs (<run>_best_decoder.pth only), all on the
    same baked embeddings. Evaluated with train_decoders_reben.evaluate_baked_reben.

Decoder architecture (M, per-head width/depth, arch var or not) is inferred from each checkpoint's
state_dict, so members with different configurations can be mixed.

After the standard evaluation, an extra pass reports the "unanimous error" breakdown: the share of
ensemble errors where EVERY selected head is wrong (errors head disagreement cannot flag), how often
they all agree on the same wrong class, and the most common true->predicted class pairs among them.

Usage (fine-tuned, 3 separately trained AS1 runs):
    python posthoc_ensemble_reben.py --pipeline finetuned \
        --s2_root $REBEN/BigEarthNet-S2 --ref_root $REBEN/Reference_Maps \
        --metadata_path $REBEN/metadata.parquet --max_patches 25000 --n_unfrozen_blocks 4 \
        --member AS1_reben_LR_cosine_run1_20261007_0610 \
        --member AS1_reben_LR_cosine_run2_20261007_0715 \
        --member AS1_reben_LR_cosine_run3_20261007_0716 \
        --run_name POSTHOC_ft_AS1x3

Usage (frozen, 3 separately trained frozen AS1 runs):
    python posthoc_ensemble_reben.py --pipeline frozen --embedding_dir $EMBED_DIR --max_patches 25000 \
        --member AS1_frozen_reben_run1_20261007_1220 --member AS1_frozen_reben_run2_20261007_1220 \
        --member AS1_frozen_reben_run3_20261007_1220 --run_name POSTHOC_frozen_AS1x3
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import CHECKPOINT_DIR, REBEN_CLASSES
from models.ensemble import SegFormerDecoderHead
from models.encoder import initialize_clay_encoder_partial_unfreeze
from utils.misc import log_msg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------
# Decoder reconstruction from a checkpoint (no config needed)
# ---------------------------------------------------------------------------------------------
class InferredDecoder(nn.Module):
    """Rebuilds a DecoderEnsemble's heads from its state_dict alone: number of heads, each head's
    embed_dim and refine-block count are read off the weight shapes/keys, so arch-var and uniform
    checkpoints both load. State-dict keys are identical to DecoderEnsemble's ('heads.i.*').
    Dropout rates are irrelevant at eval (eval mode), so they are not reconstructed."""

    def __init__(self, state_dict):
        super().__init__()
        head_ids = sorted({int(m.group(1)) for k in state_dict
                           for m in [re.match(r"heads\.(\d+)\.", k)] if m})
        heads = []
        self.head_configs = []
        for i in head_ids:
            w = state_dict[f"heads.{i}.linear_fusion.weight"]
            embed_dim, in_channels = w.shape[0], w.shape[1]
            num_classes = state_dict[f"heads.{i}.classifier.weight"].shape[0]
            n_extra = len({int(m.group(1)) for k in state_dict
                           for m in [re.match(rf"heads\.{i}\.extra_refine_blocks\.(\d+)\.", k)] if m})
            n_blocks = 1 + int(f"heads.{i}.spatial_refine2.weight" in state_dict) + n_extra
            heads.append(SegFormerDecoderHead(in_channels, embed_dim, num_classes, num_refine_blocks=n_blocks))
            self.head_configs.append((embed_dim, n_blocks))
        self.heads = nn.ModuleList(heads)
        self.M = len(heads)
        self.load_state_dict(state_dict)

    def forward(self, x):
        return torch.stack([head(x) for head in self.heads])


# ---------------------------------------------------------------------------------------------
# The post-hoc ensemble: looks like a DecoderEnsemble (has .M, returns [M, B, C, H, W]) to the
# existing evaluators
# ---------------------------------------------------------------------------------------------
class PostHocEnsemble(nn.Module):
    """For --pipeline finetuned, forward() takes raw images and runs every member's own encoder
    (through the real get_encoder_representation_partial); for --pipeline frozen it takes the baked
    features directly. Either way it returns the selected heads of all members stacked along dim 0."""

    def __init__(self, members, encoder_fn=None, waves=None):
        super().__init__()
        # members: list of (encoder or None, InferredDecoder, list of head indices)
        self.encoders = nn.ModuleList([m[0] for m in members if m[0] is not None])
        self.decoders = nn.ModuleList([m[1] for m in members])
        self._enc_idx = []
        e = 0
        for enc, _, _ in members:
            self._enc_idx.append(e if enc is not None else None)
            e += enc is not None
        self.head_sel = [m[2] for m in members]
        self.M = sum(len(h) for h in self.head_sel)
        self.encoder_fn = encoder_fn
        self.waves = waves

    def forward(self, x):
        outs = []
        for k, dec in enumerate(self.decoders):
            if self._enc_idx[k] is not None:
                feats = self.encoder_fn(x, self.encoders[self._enc_idx[k]], waves=self.waves)
            else:
                feats = x
            outs.append(dec(feats)[self.head_sel[k]])
        return torch.cat(outs, dim=0)


def parse_member(spec):
    run, _, heads = spec.partition(":")
    heads = [int(h) for h in heads.split(",")] if heads else None
    return run, heads


def build_members(args, encoder_fn):
    ckpt_dir = Path(args.checkpoint_dir)
    members = []
    for spec in args.member:
        run, heads = parse_member(spec)
        dec_path = ckpt_dir / f"{run}_best_decoder.pth"
        dec_sd = torch.load(dec_path, map_location=DEVICE)["model_state_dict"]
        decoder = InferredDecoder(dec_sd).to(DEVICE).eval()
        heads = heads if heads is not None else list(range(decoder.M))
        if max(heads) >= decoder.M:
            raise ValueError(f"{run}: asked for heads {heads} but checkpoint has M={decoder.M}")
        log_msg(f"Member {run}: decoder M={decoder.M}, head configs (embed_dim, refine_blocks)="
                f"{decoder.head_configs}, using heads {heads}")

        encoder = None
        if args.pipeline == "finetuned":
            enc_path = ckpt_dir / f"{run}_best_encoder.pth"
            encoder = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=args.n_unfrozen_blocks)
            enc_sd = torch.load(enc_path, map_location=DEVICE)["encoder_state_dict"]
            # strict=False exactly as train_e2e_reben's own best-checkpoint reload; key counts logged
            # so a mismatch (e.g. wrong --n_unfrozen_blocks) is visible rather than silent
            res = encoder.load_state_dict(enc_sd, strict=False)
            log_msg(f"  encoder {enc_path.name}: {len(enc_sd)} keys loaded, "
                    f"{len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected")
            encoder = encoder.to(DEVICE).eval()
        members.append((encoder, decoder, heads))
    return PostHocEnsemble(members, encoder_fn=encoder_fn, waves=args._waves)


# ---------------------------------------------------------------------------------------------
# Unanimous-error breakdown (extra pass)
# ---------------------------------------------------------------------------------------------
def unanimous_error_breakdown(model, loader, num_classes, run_name, top_k=10):
    """Of the pixels the ensemble gets wrong, how many are wrong for EVERY selected head (no head
    disagreement to flag them), and which true->predicted class pairs dominate those."""
    n_lab = n_ens_wrong = n_unan = n_unan_same = 0
    unan_conf = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, ensemble pred]
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long()
            if y.dim() == 4:
                y = y.squeeze(1)
            with torch.cuda.amp.autocast():
                preds = model(x)
            probs = torch.softmax(preds.float(), dim=2)       # [M, B, C, H, W]
            ens_pred = probs.mean(dim=0).argmax(dim=1)         # [B, H, W]
            head_pred = probs.argmax(dim=2)                    # [M, B, H, W]
            lab = (y > 0) & (y < num_classes)
            if not lab.any():
                continue
            gt = y[lab]
            ens_wrong = ens_pred[lab] != gt
            hp = head_pred[:, lab]                             # [M, P]
            all_wrong = (hp != gt).all(dim=0)
            same_wrong = all_wrong & (hp == hp[0:1]).all(dim=0)
            unan = ens_wrong & all_wrong
            n_lab += gt.numel()
            n_ens_wrong += ens_wrong.sum().item()
            n_unan += unan.sum().item()
            n_unan_same += (ens_wrong & same_wrong).sum().item()
            t = gt[unan].cpu().numpy()
            p = ens_pred[lab][unan].cpu().numpy()
            np.add.at(unan_conf, (t, p), 1)

    share = n_unan / max(n_ens_wrong, 1)
    share_same = n_unan_same / max(n_ens_wrong, 1)
    log_msg(f"UNANIMOUS ERRORS ({run_name}): ensemble error rate={n_ens_wrong / max(n_lab, 1):.4f} | "
            f"errors where every head is wrong={share:.4f} of ensemble errors "
            f"(all heads on the SAME wrong class={share_same:.4f})")
    flat = [(unan_conf[i, j], i, j) for i in range(1, num_classes) for j in range(num_classes) if i != j]
    flat.sort(reverse=True)
    top = []
    log_msg(f"Top {top_k} true -> predicted pairs among unanimous errors (share of unanimous errors):")
    for cnt, i, j in flat[:top_k]:
        frac = cnt / max(n_unan, 1)
        log_msg(f"  {REBEN_CLASSES[i]} -> {REBEN_CLASSES[j]}: {frac:.4f}")
        top.append({"true": REBEN_CLASSES[i], "pred": REBEN_CLASSES[j], "share_of_unanimous": frac})

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out = {"run_name": run_name, "labelled_pixels": n_lab, "ensemble_error_rate": n_ens_wrong / max(n_lab, 1),
           "unanimous_share_of_errors": share, "unanimous_same_class_share_of_errors": share_same,
           "top_unanimous_pairs": top, "unanimous_confusion": unan_conf.tolist()}
    path = os.path.join(runs_dir, f"{run_name}_unanimous_errors.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log_msg(f"Unanimous-error breakdown saved to {path}")


# ---------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Post-hoc ensemble evaluation of trained reBEN checkpoints")
    p.add_argument("--pipeline", choices=["finetuned", "frozen"], required=True)
    p.add_argument("--member", action="append", required=True,
                   help="RUN_NAME (with timestamp, as in the checkpoint filename) optionally followed by "
                        ":h1,h2,... to use only those heads. Repeat for each member.")
    p.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    p.add_argument("--run_name", type=str, required=True, help="Output filename prefix")
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_patches", type=int, default=None,
                   help="Same subset semantics as training: test = first max_patches//5 test patches")
    # finetuned pipeline
    p.add_argument("--s2_root", type=str)
    p.add_argument("--ref_root", type=str)
    p.add_argument("--metadata_path", type=str)
    p.add_argument("--n_unfrozen_blocks", type=int, default=4,
                   help="Must match the members' training config (all fine-tuned runs so far: 4)")
    p.add_argument("--include_snow", action="store_true")
    p.add_argument("--include_cloud", action="store_true")
    p.add_argument("--test_countries", nargs="+", default=None,
                   help="Finetuned only: evaluate on test patches from ONLY these countries (geographic-shift "
                        "test, e.g. Portugal). Same max_patches semantics: first max_patches//5 of them.")
    p.add_argument("--exclude_test_countries", nargs="+", default=None,
                   help="Finetuned only: drop these countries from the test patches (in-distribution reference "
                        "for a model trained with train_e2e_reben_country.py --exclude_countries)")
    # frozen pipeline
    p.add_argument("--embedding_dir", type=str)
    p.add_argument("--skip_unanimous", action="store_true", help="Skip the extra unanimous-error pass")
    args = p.parse_args()

    # Fields the existing evaluators read from args (not otherwise meaningful for a post-hoc ensemble)
    args.hide_unlabelled_pixels = True
    args.diversity_methods = []
    args.lam_jsd = args.lam_pearson = args.lam_orth = 0.0

    log_msg(f"Post-hoc ensemble: {vars(args)}")

    if args.pipeline == "finetuned":
        if not (args.s2_root and args.ref_root and args.metadata_path):
            p.error("--pipeline finetuned needs --s2_root, --ref_root and --metadata_path")
        import train_e2e_reben as fte
        from utils.dataset_e2e import load_reben_splits
        args._waves = fte.S2_WAVES
        _, _, test_ds = load_reben_splits(
            metadata_path=args.metadata_path, s2_root=args.s2_root, ref_root=args.ref_root,
            exclude_snow=not args.include_snow, exclude_cloud=not args.include_cloud,
            max_patches=args.max_patches, only_countries=args.test_countries,
            exclude_countries=args.exclude_test_countries)
        real_encoder_fn = fte.get_encoder_representation_partial
        model = build_members(args, encoder_fn=real_encoder_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        log_msg(f"Post-hoc ensemble: {len(args.member)} member(s), M={model.M} heads in total, "
                f"{len(test_ds)} test patches")
        # The evaluator calls get_encoder_representation_partial(imgs, encoder) then decoder(features).
        # Within the evaluator's module only, make the first call a pass-through, so the images reach
        # PostHocEnsemble, which runs each member's own encoder via the real function above.
        fte.get_encoder_representation_partial = lambda imgs, encoder_model, waves=None: imgs
        try:
            fte.evaluate_test_set_reben(nn.Identity(), model, test_loader, args, run_name=args.run_name)
        finally:
            fte.get_encoder_representation_partial = real_encoder_fn
    else:
        if not args.embedding_dir:
            p.error("--pipeline frozen needs --embedding_dir")
        if args.test_countries or args.exclude_test_countries:
            p.error("--test_countries / --exclude_test_countries are only supported for --pipeline finetuned")
        import train_decoders_reben as frz
        from utils.dataset_e2e import BakedReBENDataset
        args._waves = None
        test_ds = BakedReBENDataset(Path(args.embedding_dir), split="test", augment=False)
        if args.max_patches:
            test_ds = torch.utils.data.Subset(test_ds, range(min(args.max_patches // 5, len(test_ds))))
        model = build_members(args, encoder_fn=None)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        log_msg(f"Post-hoc ensemble: {len(args.member)} member(s), M={model.M} heads in total, "
                f"{len(test_ds)} test patches")
        frz.evaluate_baked_reben(model, test_loader, args, run_name=args.run_name)

    if not args.skip_unanimous and model.M > 1:
        unanimous_error_breakdown(model, test_loader, args.num_classes, args.run_name)


if __name__ == "__main__":
    main()
