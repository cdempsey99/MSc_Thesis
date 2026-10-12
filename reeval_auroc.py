"""
reeval_auroc.py - recompute ONLY the AUROC metrics (plus Acc/mIoU as a sanity check) for already-trained
reBEN / reBEN-SAR checkpoints, after the AUROC sampling fix (08/10/2026: the evaluators used to score
AUROC on the first ~2M labelled test pixels only, i.e. the first ~50 test patches).

One pass over the test set per checkpoint - no error localisation, no plots, no correlation pass - so a
full-scale run takes roughly as long as one inference sweep instead of the 7-20 h of the full eval.

AUROC is computed exactly as in the fixed evaluators: the same seeded per-batch pixel sample
(utils.misc.auroc_pixel_sample, generator seed 0), the same uncertainty definitions
(total = entropy of the mean of the heads' softmaxes, aleatoric = mean per-head entropy,
epistemic = total - aleatoric clamped at 0) and the same error definition (argmax of the mean != label),
scored with sklearn's roc_auc_score. With the same test set and batch size the numbers match a fresh run
of the training-time evaluator.

Each --run is a checkpoint STEM: the path prefix shared by its files, without "_decoder.pth", e.g.
    AS1_reben_qa_full_resume_20260706_1108_best          -> ..._best_decoder.pth (+ _best_encoder.pth)
    AS4_finetuned_m=10_..._resumed4_20260817_1123_last   -> ..._last_decoder/_encoder/_student.pth
If <stem>_student.pth exists, the AS4 student (Dirichlet head) is scored too, from the same forward pass.

--pipeline finetuned : reBEN optical, train_e2e_reben.py checkpoints (decoder + encoder)
--pipeline sar       : reBEN SAR, train_e2e_reben_sar.py checkpoints (decoder + encoder)
--pipeline frozen    : reBEN optical frozen encoder, train_decoders_reben.py checkpoints (decoder only,
                       on the baked embeddings)

All runs in one call share one test set, so group them by pipeline and --max_patches (25k subset runs vs
full-scale runs) and call the script once per group. A summary table is printed at the end and every result
is appended to <OUT_DIR>/runs/<--tag>_auroc_reeval.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from configs.config import CHECKPOINT_DIR
from models.ensemble import StudentHead
from models.encoder import initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial
from posthoc_ensemble_reben import InferredDecoder
from utils.misc import log_msg, auroc_pixels_per_batch, auroc_pixel_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(stem, pipeline, n_unfrozen_blocks):
    dec_sd = torch.load(f"{stem}_decoder.pth", map_location=DEVICE)["model_state_dict"]
    decoder = InferredDecoder(dec_sd).to(DEVICE).eval()   # strict load: a wrong/odd checkpoint fails loudly
    log_msg(f"  decoder: M={decoder.M}, head configs (embed_dim, refine_blocks)={decoder.head_configs}")

    encoder = None
    if pipeline != "frozen":
        encoder = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=n_unfrozen_blocks)
        enc_sd = torch.load(f"{stem}_encoder.pth", map_location=DEVICE)["encoder_state_dict"]
        res = encoder.load_state_dict(enc_sd, strict=False)   # as the training scripts' own reload
        log_msg(f"  encoder: {len(enc_sd)} keys loaded, {len(res.missing_keys)} missing, "
                f"{len(res.unexpected_keys)} unexpected")
        encoder = encoder.to(DEVICE).eval()

    student = None
    if Path(f"{stem}_student.pth").exists():
        st_sd = torch.load(f"{stem}_student.pth", map_location=DEVICE)["model_state_dict"]
        w = st_sd["linear_fusion.weight"]
        student = StudentHead(in_channels=w.shape[1], embed_dim=w.shape[0],
                              num_classes=st_sd["classifier.weight"].shape[0])
        student.load_state_dict(st_sd)
        student = student.to(DEVICE).eval()
        log_msg(f"  student: found {Path(stem).name}_student.pth, scoring it too")
    return encoder, decoder, student


def _auroc(err, score):
    return float(roc_auc_score(err, score)) if len(np.unique(err)) > 1 else None


def _miou(conf):
    ious = []
    for c in range(1, conf.shape[0]):   # class 0 = unlabelled, as in the evaluators
        tp = conf[c, c]
        denom = conf[:, c].sum() + conf[c, :].sum() - tp
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


def evaluate(encoder, decoder, student, loader, waves, num_classes):
    per_batch = auroc_pixels_per_batch(loader)
    gen_t = torch.Generator().manual_seed(0)   # same seed as the fixed evaluators
    gen_s = torch.Generator().manual_seed(0)   # student evaluator draws its own sample with the same seed
    t = {"err": [], "tot": [], "ale": [], "epi": []}
    s = {"err": [], "tot": [], "ale": [], "epi": []}
    conf_t = torch.zeros(num_classes, num_classes, dtype=torch.long, device=DEVICE)
    conf_s = torch.zeros_like(conf_t)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long()
            if y.dim() == 4:
                y = y.squeeze(1)
            with torch.cuda.amp.autocast():
                feats = x if encoder is None else get_encoder_representation_partial(x, encoder, waves=waves)
                preds = decoder(feats)
                alphas = student(feats) if student is not None else None
            lab = y > 0
            if not lab.any():
                continue

            probs = torch.softmax(preds.float(), dim=2)             # [M, B, C, H, W]
            mean_p = probs.mean(dim=0)
            pred = mean_p.argmax(dim=1)
            tot = -(mean_p * torch.log(mean_p.clamp(min=1e-10))).sum(dim=1)
            ale = (-(probs * torch.log(probs.clamp(1e-10))).sum(dim=2)).mean(dim=0)
            smp = auroc_pixel_sample(lab, per_batch, gen_t)
            t["err"].append((pred != y)[smp].float().cpu().numpy())
            t["tot"].append(tot[smp].cpu().numpy())
            t["ale"].append(ale[smp].cpu().numpy())
            t["epi"].append((tot - ale).clamp(min=0)[smp].cpu().numpy())
            idx = y[lab] * num_classes + pred[lab]
            conf_t += torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)

            if alphas is not None:
                alphas = alphas.float()
                a0 = alphas.sum(dim=1, keepdim=True)
                mp = alphas / a0
                spred = mp.argmax(dim=1)
                stot = -(mp * torch.log(mp.clamp(min=1e-10))).sum(dim=1)
                sale = torch.digamma(a0.squeeze(1) + 1) - (mp * torch.digamma(alphas + 1)).sum(dim=1)
                ssmp = auroc_pixel_sample(lab, per_batch, gen_s)
                s["err"].append((spred != y)[ssmp].float().cpu().numpy())
                s["tot"].append(stot[ssmp].cpu().numpy())
                s["ale"].append(sale[ssmp].cpu().numpy())
                s["epi"].append((stot - sale).clamp(min=0)[ssmp].cpu().numpy())
                idx = y[lab] * num_classes + spred[lab]
                conf_s += torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)

    def summarise(d, conf, ensemble):
        err = np.concatenate(d["err"])
        conf = conf.cpu().numpy()
        out = {"acc": float(np.trace(conf) / max(conf.sum(), 1)), "miou": _miou(conf),
               "auroc_total": _auroc(err, np.concatenate(d["tot"])),
               "auroc_pixels": int(err.size), "error_rate_sampled": float(err.mean())}
        if ensemble:
            out["auroc_aleatoric"] = _auroc(err, np.concatenate(d["ale"]))
            out["auroc_epistemic"] = _auroc(err, np.concatenate(d["epi"]))
        return out

    res = {"teacher": summarise(t, conf_t, decoder.M > 1)}
    if student is not None:
        res["student"] = summarise(s, conf_s, True)
    return res


def main():
    p = argparse.ArgumentParser(description="AUROC-only re-evaluation of trained reBEN / SAR checkpoints")
    p.add_argument("--pipeline", choices=["finetuned", "sar", "frozen"], required=True)
    p.add_argument("--run", action="append", required=True,
                   help="Checkpoint stem without '_decoder.pth' (e.g. <run_name>_best). Repeat per checkpoint. "
                        "Relative stems are looked up in --checkpoint_dir.")
    p.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    p.add_argument("--tag", type=str, required=True, help="Name for the summary JSON")
    p.add_argument("--max_patches", type=int, default=None,
                   help="Same subset semantics as training (test = first max_patches//5). Omit for full scale.")
    p.add_argument("--metadata_path", type=str)
    p.add_argument("--s2_root", type=str)
    p.add_argument("--s1_root", type=str)
    p.add_argument("--ref_root", type=str)
    p.add_argument("--embedding_dir", type=str)
    p.add_argument("--n_unfrozen_blocks", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()
    log_msg(f"AUROC re-eval: {vars(args)}")

    if args.pipeline == "finetuned":
        from utils.dataset_e2e import load_reben_splits, ReBENRawDataset
        _, _, test_ds = load_reben_splits(metadata_path=args.metadata_path, s2_root=args.s2_root,
                                          ref_root=args.ref_root, max_patches=args.max_patches)
        waves = ReBENRawDataset.WAVELENGTHS
    elif args.pipeline == "sar":
        from utils.dataset_e2e import load_reben_sar_splits, ReBENSARRawDataset
        _, _, test_ds = load_reben_sar_splits(metadata_path=args.metadata_path, s1_root=args.s1_root,
                                              ref_root=args.ref_root, max_patches=args.max_patches)
        waves = ReBENSARRawDataset.WAVELENGTHS
    else:
        from utils.dataset_e2e import BakedReBENDataset
        test_ds = BakedReBENDataset(Path(args.embedding_dir), split="test", augment=False)
        if args.max_patches:
            test_ds = Subset(test_ds, range(min(args.max_patches // 5, len(test_ds))))
        waves = None
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    log_msg(f"Test set: {len(test_ds)} patches, {len(loader)} batches")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"{args.tag}_auroc_reeval.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for stem in args.run:
        full = stem if os.path.isabs(stem) else os.path.join(args.checkpoint_dir, stem)
        log_msg(f"=== {stem}")
        try:
            encoder, decoder, student = load_models(full, args.pipeline, args.n_unfrozen_blocks)
            res = evaluate(encoder, decoder, student, loader, waves, args.num_classes)
        except Exception as e:   # one bad checkpoint must not cost the rest of the batch
            log_msg(f"  FAILED: {type(e).__name__}: {e}")
            continue
        res.update({"pipeline": args.pipeline, "max_patches": args.max_patches, "test_patches": len(test_ds),
                    "M": decoder.M})
        results[stem] = res
        for who, r in res.items():
            if isinstance(r, dict):
                log_msg(f"  {who}: Acc {r['acc']:.4f} | mIoU {r['miou']:.4f} | AUROC total {r['auroc_total']:.4f}"
                        + (f" | epistemic {r['auroc_epistemic']:.4f} | aleatoric {r['auroc_aleatoric']:.4f}"
                           if r.get("auroc_epistemic") is not None else ""))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        del encoder, decoder, student
        torch.cuda.empty_cache()

    log_msg(f"SUMMARY ({args.tag}, {args.pipeline}, {len(test_ds)} test patches) - Acc | mIoU | "
            f"AUROC total | epistemic | aleatoric")
    for stem in args.run:
        if stem not in results:
            log_msg(f"  {stem}: FAILED")
            continue
        for who in ("teacher", "student"):
            r = results[stem].get(who)
            if r is None:
                continue
            fmt = lambda v: f"{v:.4f}" if v is not None else "  -   "
            log_msg(f"  {stem} [{who}]: {r['acc']:.4f} | {r['miou']:.4f} | {fmt(r['auroc_total'])} | "
                    f"{fmt(r.get('auroc_epistemic'))} | {fmt(r.get('auroc_aleatoric'))}")
    log_msg(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
