"""
reeval_auroc.py - re-evaluate already-trained checkpoints after the AUROC sampling fix (08/10/2026 for
reBEN / SAR, 20/10/2026 for FBP: the evaluators used to score AUROC on the first ~2M labelled test pixels
only, i.e. the first ~50 reBEN test patches / a fraction of one FBP test image). Alongside the AUROCs it
recomputes Acc / mIoU (sanity check against the original logs), ECE, NLL, the uncertainty decomposition
and the decision-level diversity metrics (normalised disagreement, error correlation ratio, oracle
accuracy, share of ensemble errors made by every head), so FBP gets the same diversity table as reBEN.

One pass over the test set per checkpoint - no error localisation, no plots, no correlation pass - so a
full-scale run takes roughly as long as one inference sweep instead of the 7-20 h of the full eval.

AUROC is computed exactly as in the fixed evaluators: the same seeded per-batch pixel sample
(utils.misc.auroc_pixel_sample, generator seed 0), the same uncertainty definitions
(total = entropy of the mean of the heads' softmaxes, aleatoric = mean per-head entropy,
epistemic = total - aleatoric clamped at 0) and the same error definition (argmax of the mean != label),
scored with sklearn's roc_auc_score. With the same test set and batch size the numbers match a fresh run
of the training-time evaluator. Diversity metrics use the definitions of evaluate_test_set_reben /
utils.misc.evaluate_test_set, over all labelled pixels.

--pipeline finetuned : reBEN optical, train_e2e_reben.py checkpoints (decoder + encoder)
--pipeline sar       : reBEN SAR, train_e2e_reben_sar.py checkpoints (decoder + encoder)
--pipeline frozen    : reBEN optical frozen encoder, train_decoders_reben.py checkpoints (decoder only,
                       on the baked embeddings)
--pipeline fbp       : FBP frozen encoder, train_decoders.py checkpoints (decoder only, on the baked
                       embeddings; per-head variational-bottleneck checkpoints are routed through
                       VBEvalWrapper exactly as at training time). Evaluated in fp32 with batch size 64,
                       as train_decoders.py's own evaluation, and with FBP's mIoU / ECE conventions.

reBEN / SAR: each --run is a checkpoint STEM: the path prefix shared by its files, without "_decoder.pth", e.g.
    AS1_reben_qa_full_resume_20260706_1108_best          -> ..._best_decoder.pth (+ _best_encoder.pth)
    AS4_finetuned_m=10_..._resumed4_20260817_1123_last   -> ..._last_decoder/_encoder/_student.pth
If <stem>_student.pth exists, the AS4 student (Dirichlet head) is scored too, from the same forward pass.

FBP: each --run is a checkpoint FILE name (e.g. <run_name>_best_model.pth, _final_model_<ts>.pth or
_last_checkpoint.pth). The AS4 student is picked up automatically from the same file (a _last_checkpoint.pth
holds student_model_state_dict) or the matching sibling (_best_model -> _best_student,
_final_model_<ts> -> _final_student_<ts>); give it explicitly as "<teacher file>::<student file>" otherwise.

--calibration_dir additionally saves, per checkpoint and model, the arrays plot_calibration.py turns into
reliability / error-confidence curves (pixel and error mass per confidence bin, equal-mass error curves,
correct-vs-wrong confidence histograms, risk-coverage, class-wise ECE). Pairwise error diversity is also
reported as the standard correlation coefficient rho and Yule's Q (Kuncheva & Whitaker 2003).

All runs in one call share one test set, so group them by pipeline and --max_patches (25k subset runs vs
full-scale runs) and call the script once per group. A summary table is printed at the end and every result
is appended to <OUT_DIR>/runs/<--tag>_auroc_reeval.json.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from configs.config import CHECKPOINT_DIR
from models.ensemble import StudentHead, VariationalBottleneck, VBEvalWrapper
from models.encoder import initialize_clay_encoder_partial_unfreeze, get_encoder_representation_partial, REBEN_GSD
from posthoc_ensemble_reben import InferredDecoder
from utils.misc import log_msg, auroc_pixels_per_batch, auroc_pixel_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 10
FINE_BINS = 100   # --calibration_dir: exact per-bin pixel / correct / confidence sums over all scored pixels
CLASS_BINS = 15   # --calibration_dir: per predicted-class reliability bins (class-wise ECE)


def _student_from_state_dict(st_sd):
    w = st_sd["linear_fusion.weight"]
    student = StudentHead(in_channels=w.shape[1], embed_dim=w.shape[0],
                          num_classes=st_sd["classifier.weight"].shape[0])
    student.load_state_dict(st_sd)
    return student.to(DEVICE).eval()


def load_models(stem, pipeline, n_unfrozen_blocks, single_norm=False, native_forward=False):
    dec_sd = torch.load(f"{stem}_decoder.pth", map_location=DEVICE)["model_state_dict"]
    decoder = InferredDecoder(dec_sd).to(DEVICE).eval()   # strict load: a wrong/odd checkpoint fails loudly
    log_msg(f"  decoder: M={decoder.M}, head configs (embed_dim, refine_blocks)={decoder.head_configs}")

    encoder = None
    if pipeline != "frozen":
        encoder = initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=n_unfrozen_blocks, single_norm=single_norm,
                                                           native_forward=native_forward, gsd=REBEN_GSD)
        enc_sd = torch.load(f"{stem}_encoder.pth", map_location=DEVICE)["encoder_state_dict"]
        res = encoder.load_state_dict(enc_sd, strict=False)   # as the training scripts' own reload
        log_msg(f"  encoder: {len(enc_sd)} keys loaded, {len(res.missing_keys)} missing, "
                f"{len(res.unexpected_keys)} unexpected")
        encoder = encoder.to(DEVICE).eval()

    student = None
    if Path(f"{stem}_student.pth").exists():
        student = _student_from_state_dict(torch.load(f"{stem}_student.pth", map_location=DEVICE)["model_state_dict"])
        log_msg(f"  student: found {Path(stem).name}_student.pth, scoring it too")
    return encoder, decoder, student


def load_models_fbp(run, checkpoint_dir):
    """FBP train_decoders.py checkpoint -> (None, teacher, student or None). The teacher is an
    InferredDecoder, wrapped in VBEvalWrapper if the checkpoint carries a per-head bottleneck."""
    teacher_file, _, student_file = run.partition("::")
    path = teacher_file if os.path.isabs(teacher_file) else os.path.join(checkpoint_dir, teacher_file)
    ckpt = torch.load(path, map_location=DEVICE)
    decoder = InferredDecoder(ckpt["model_state_dict"]).to(DEVICE).eval()
    log_msg(f"  decoder: M={decoder.M}, head configs (embed_dim, refine_blocks)={decoder.head_configs}")

    teacher = decoder
    bn_sd = ckpt.get("bottleneck_state_dict")
    if bn_sd is not None:
        # sigma_prior only enters the KL term, not the eval forward (each head gets its mu_m), so any value works
        n_heads = len({k.split(".")[1] for k in bn_sd if k.startswith("mu_convs.")})
        channels = bn_sd["mu_convs.0.weight"].shape[0]
        bottleneck = VariationalBottleneck(channels=channels, num_heads=n_heads)
        bottleneck.load_state_dict(bn_sd)
        teacher = VBEvalWrapper(decoder, bottleneck.to(DEVICE).eval()).eval()
        log_msg(f"  per-head variational bottleneck found ({n_heads} heads): evaluating each head on its mu_m")

    student = None
    if student_file:
        st_path = student_file if os.path.isabs(student_file) else os.path.join(checkpoint_dir, student_file)
        student = _student_from_state_dict(torch.load(st_path, map_location=DEVICE)["model_state_dict"])
        log_msg(f"  student: {Path(st_path).name} (given explicitly)")
    elif ckpt.get("student_model_state_dict") is not None:
        student = _student_from_state_dict(ckpt["student_model_state_dict"])
        log_msg("  student: student_model_state_dict found in the same checkpoint, scoring it too")
    else:
        name = Path(path).name
        sibling = None
        if name.endswith("_best_model.pth"):
            sibling = name[:-len("_best_model.pth")] + "_best_student.pth"
        elif "_final_model_" in name:
            sibling = name.replace("_final_model_", "_final_student_")
        if sibling and os.path.exists(os.path.join(os.path.dirname(path), sibling)):
            student = _student_from_state_dict(
                torch.load(os.path.join(os.path.dirname(path), sibling), map_location=DEVICE)["model_state_dict"])
            log_msg(f"  student: found {sibling}, scoring it too")
    return None, teacher, student


def build_fbp_test_files(data_dir, patch_size, stride, max_images=None):
    """Mirrors train_decoders.py's split exactly (same glob, same seed=42 shuffle, same 70/15/15 cut)."""
    embedding_dir = Path(data_dir) / "embeddings" / "fbp" / "clay_v1" / f"patch{patch_size}_stride{stride}"
    all_files = sorted(list(embedding_dir.glob("*_embeddings.pt")))
    if not all_files:
        raise FileNotFoundError(f"No *_embeddings.pt in {embedding_dir} - check --data_dir / --stride")
    if max_images is not None:
        all_files = all_files[:max_images]
    random.seed(42)
    random.shuffle(all_files)
    n = len(all_files)
    return all_files[2:] if n < 5 else all_files[int(n * 0.85):]


def _auroc(err, score):
    return float(roc_auc_score(err, score)) if len(np.unique(err)) > 1 else None


def _miou(conf, gt_present_only):
    """class 0 = unlabelled, as in the evaluators. reBEN averages over classes with a non-zero IoU
    denominator; FBP's evaluate_test_set only over classes present in the ground truth."""
    ious = []
    for c in range(1, conf.shape[0]):
        if gt_present_only and conf[c, :].sum() == 0:
            continue
        tp = conf[c, c]
        denom = conf[:, c].sum() + conf[c, :].sum() - tp
        if denom > 0:
            ious.append(tp / denom)
        elif gt_present_only:
            ious.append(0.0)
    return float(np.mean(ious)) if ious else 0.0


class _Tracker:
    """Running sums for one model (teacher ensemble or student) over the test set."""

    def __init__(self, num_classes, M, fbp_style):
        self.C, self.M, self.fbp_style = num_classes, M, fbp_style
        self.auroc = {"err": [], "tot": [], "ale": [], "epi": [], "conf": []}
        # calibration detail (saved with --calibration_dir): same pixels as ECE (patches with >= 100 labelled)
        self.cal_n = torch.zeros(FINE_BINS, dtype=torch.float64, device=DEVICE)
        self.cal_correct = torch.zeros(FINE_BINS, dtype=torch.float64, device=DEVICE)
        self.cal_conf = torch.zeros(FINE_BINS, dtype=torch.float64, device=DEVICE)
        self.cls_n = torch.zeros(num_classes * CLASS_BINS, dtype=torch.float64, device=DEVICE)
        self.cls_correct = torch.zeros_like(self.cls_n)
        self.cls_conf = torch.zeros_like(self.cls_n)
        self.conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=DEVICE)
        self.head_conf = torch.zeros(M, num_classes, num_classes, dtype=torch.long, device=DEVICE)
        self.bin_conf = torch.zeros(NUM_BINS, dtype=torch.float64, device=DEVICE)
        self.bin_acc = torch.zeros(NUM_BINS, dtype=torch.float64, device=DEVICE)
        self.bin_n = torch.zeros(NUM_BINS, dtype=torch.float64, device=DEVICE)
        self.nll = self.tot_sum = self.ale_sum = 0.0
        self.n_lab = 0
        self.div_n = 0
        self.pair_dis = torch.zeros(M, M, device=DEVICE)
        self.pair_df = torch.zeros(M, M, device=DEVICE)
        self.head_err = torch.zeros(M, device=DEVICE)
        self.ens_err = torch.zeros((), device=DEVICE)
        self.oracle = torch.zeros((), device=DEVICE)
        self.unanimous = torch.zeros((), device=DEVICE)

    def add(self, mean_p, y, lab, conf_lab, smp, tot, ale, head_pred=None):
        pred = mean_p.argmax(dim=1)
        wrong = pred != y
        epi = (tot - ale).clamp(min=0)
        self.auroc["err"].append(wrong[smp].float().cpu().numpy())
        self.auroc["tot"].append(tot[smp].cpu().numpy())
        self.auroc["ale"].append(ale[smp].cpu().numpy())
        self.auroc["epi"].append(epi[smp].cpu().numpy())
        conf_all = mean_p.max(dim=1)[0]
        self.auroc["conf"].append(conf_all[smp].float().cpu().numpy())

        gt_idx = y.unsqueeze(1).clamp(0, self.C - 1)
        self.nll += (-torch.log(mean_p.clamp(min=1e-10)).gather(1, gt_idx).squeeze(1)[lab]).sum().item()
        self.tot_sum += tot[lab].sum().item()
        self.ale_sum += ale[lab].sum().item()
        self.n_lab += int(lab.sum().item())

        # confusion matrix + ECE only on patches with >= 100 labelled pixels, as both evaluators do
        idx = y[conf_lab] * self.C + pred[conf_lab]
        self.conf += torch.bincount(idx, minlength=self.C ** 2).view(self.C, self.C)
        conf_v = mean_p.max(dim=1)[0][conf_lab].double()
        corr_v = (~wrong)[conf_lab].double()
        bounds = torch.linspace(0, 1, NUM_BINS + 1, device=DEVICE, dtype=torch.float64)
        if self.fbp_style:   # utils.misc.evaluate_test_set: lo <= conf < hi
            for i in range(NUM_BINS):
                inb = (conf_v >= bounds[i]) & (conf_v < bounds[i + 1])
                self.bin_conf[i] += conf_v[inb].sum()
                self.bin_acc[i] += corr_v[inb].sum()
                self.bin_n[i] += inb.sum()
        else:                # evaluate_test_set_reben: np.digitize on the inner boundaries
            b = torch.bucketize(conf_v, bounds[1:-1], right=True)
            self.bin_conf += torch.bincount(b, weights=conf_v, minlength=NUM_BINS)
            self.bin_acc += torch.bincount(b, weights=corr_v, minlength=NUM_BINS)
            self.bin_n += torch.bincount(b, minlength=NUM_BINS).double()
        fb = (conf_v * FINE_BINS).long().clamp(max=FINE_BINS - 1)
        self.cal_n += torch.bincount(fb, minlength=FINE_BINS).double()
        self.cal_correct += torch.bincount(fb, weights=corr_v, minlength=FINE_BINS)
        self.cal_conf += torch.bincount(fb, weights=conf_v, minlength=FINE_BINS)
        cb = pred[conf_lab] * CLASS_BINS + (conf_v * CLASS_BINS).long().clamp(max=CLASS_BINS - 1)
        n_cb = self.C * CLASS_BINS
        self.cls_n += torch.bincount(cb, minlength=n_cb).double()
        self.cls_correct += torch.bincount(cb, weights=corr_v, minlength=n_cb)
        self.cls_conf += torch.bincount(cb, weights=conf_v, minlength=n_cb)

        if head_pred is not None:   # [M, B, H, W]
            for m in range(self.M):
                hidx = y[conf_lab] * self.C + head_pred[m][conf_lab]
                self.head_conf[m] += torch.bincount(hidx, minlength=self.C ** 2).view(self.C, self.C)
            if self.M > 1:
                gt_l = y[lab]
                hp_l = head_pred[:, lab]
                hw_l = hp_l != gt_l
                ew_l = wrong[lab]
                self.div_n += gt_l.numel()
                self.head_err += hw_l.sum(dim=1)
                self.ens_err += ew_l.sum()
                self.oracle += (~hw_l).any(dim=0).sum()
                self.unanimous += (ew_l & hw_l.all(dim=0)).sum()
                for i in range(self.M):
                    for j in range(i + 1, self.M):
                        self.pair_dis[i, j] += (hp_l[i] != hp_l[j]).sum()
                        self.pair_df[i, j] += (hw_l[i] & hw_l[j]).sum()

    def summary(self, ensemble):
        err = np.concatenate(self.auroc["err"])
        conf = self.conf.cpu().numpy()
        n = max(self.bin_n.sum().item(), 1)
        ece = float(sum(abs(self.bin_acc[i].item() / self.bin_n[i].item() - self.bin_conf[i].item() / self.bin_n[i].item())
                        * self.bin_n[i].item() / n for i in range(NUM_BINS) if self.bin_n[i].item() > 0))
        tot = self.tot_sum / max(self.n_lab, 1)
        ale = self.ale_sum / max(self.n_lab, 1)
        epi = max(tot - ale, 0.0)
        out = {"acc": float(np.trace(conf) / max(conf.sum(), 1)), "miou": _miou(conf, self.fbp_style),
               "ece": ece, "nll": self.nll / max(self.n_lab, 1),
               "mean_total_entropy": tot, "mean_aleatoric": ale, "mean_epistemic": epi,
               "epistemic_share": epi / tot if tot > 0 else None,
               "auroc_total": _auroc(err, np.concatenate(self.auroc["tot"])),
               "auroc_pixels": int(err.size), "error_rate_sampled": float(err.mean())}
        if ensemble:
            out["auroc_aleatoric"] = _auroc(err, np.concatenate(self.auroc["ale"]))
            out["auroc_epistemic"] = _auroc(err, np.concatenate(self.auroc["epi"]))
        if self.M > 1 and self.div_n > 0:
            dn = float(self.div_n)
            her = (self.head_err / dn).cpu().numpy()
            ens_rate = self.ens_err.item() / dn
            pdis = (self.pair_dis / dn).cpu().numpy()
            pdf = (self.pair_df / dn).cpu().numpy()
            pairs = [(i, j) for i in range(self.M) for j in range(i + 1, self.M)]
            ratios = [pdf[i, j] / (her[i] * her[j]) for i, j in pairs if her[i] > 0 and her[j] > 0]
            # Standard pairwise error-diversity measures (Kuncheva & Whitaker 2003) on the heads' binary
            # error indicators: correlation coefficient rho and Yule's Q, from the 2x2 table of fractions
            # n11 = both wrong, n10 / n01 = only one wrong, n00 = both right. Unlike the ratio above they
            # are bounded and not capped at 1/e, so they compare across datasets with different error rates.
            rhos, qs = [], []
            for i, j in pairs:
                n11 = pdf[i, j]
                n10, n01 = her[i] - n11, her[j] - n11
                n00 = 1.0 - her[i] - her[j] + n11
                den_rho = np.sqrt(her[i] * (1 - her[i]) * her[j] * (1 - her[j]))
                if den_rho > 0:
                    rhos.append((n11 * n00 - n10 * n01) / den_rho)
                if n11 * n00 + n10 * n01 > 0:
                    qs.append((n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01))
            out.update({
                "error_correlation_rho": float(np.mean(rhos)) if rhos else None,
                "error_yule_q": float(np.mean(qs)) if qs else None,
                "normalised_disagreement": float(np.mean([pdis[i, j] for i, j in pairs])) / ens_rate
                if ens_rate > 0 else None,
                "error_correlation_ratio": float(np.mean(ratios)) if ratios else None,
                "oracle_accuracy": self.oracle.item() / dn,
                "unanimous_share_of_errors": self.unanimous.item() / self.ens_err.item()
                if self.ens_err.item() > 0 else None,
                "per_head_miou": [_miou(self.head_conf[m].cpu().numpy(), self.fbp_style) for m in range(self.M)],
            })
        return out

    def calibration_arrays(self):
        """Raw material for the error-confidence / reliability plots (plot_calibration.py): exact
        per-bin sums over every scored pixel (FINE_BINS equal-width confidence bins, and CLASS_BINS bins
        per predicted class) plus the AUROC pixel sample (confidence, error, uncertainties) for
        equal-mass curves and correct/wrong confidence histograms."""
        return {
            "fine_n": self.cal_n.cpu().numpy(), "fine_correct": self.cal_correct.cpu().numpy(),
            "fine_conf": self.cal_conf.cpu().numpy(),
            "class_n": self.cls_n.view(self.C, CLASS_BINS).cpu().numpy(),
            "class_correct": self.cls_correct.view(self.C, CLASS_BINS).cpu().numpy(),
            "class_conf": self.cls_conf.view(self.C, CLASS_BINS).cpu().numpy(),
            "sample_conf": np.concatenate(self.auroc["conf"]).astype(np.float32),
            "sample_err": np.concatenate(self.auroc["err"]).astype(np.uint8),
            "sample_total_unc": np.concatenate(self.auroc["tot"]).astype(np.float32),
            "sample_epistemic": np.concatenate(self.auroc["epi"]).astype(np.float32),
        }


def evaluate(encoder, decoder, student, loader, waves, num_classes, fbp=False):
    per_batch = auroc_pixels_per_batch(loader)
    gen_t = torch.Generator().manual_seed(0)   # same seed as the fixed evaluators
    gen_s = torch.Generator().manual_seed(0)   # student evaluator draws its own sample with the same seed
    t = _Tracker(num_classes, decoder.M, fbp)
    s = _Tracker(num_classes, 1, fbp) if student is not None else None

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long()
            if y.dim() == 4:
                y = y.squeeze(1)
            # FBP's train_decoders.py evaluates in fp32; the reBEN / SAR evaluators under autocast
            with torch.cuda.amp.autocast(enabled=not fbp):
                if fbp:
                    x = x.float()
                feats = x if encoder is None else get_encoder_representation_partial(x, encoder, waves=waves)
                preds = decoder(feats)
                alphas = student(feats) if student is not None else None
            lab = y > 0
            if not lab.any():
                continue
            conf_lab = lab & (lab.flatten(1).sum(dim=1) >= 100).view(-1, 1, 1)

            probs = torch.softmax(preds.float(), dim=2)             # [M, B, C, H, W]
            mean_p = probs.mean(dim=0)
            tot = -(mean_p * torch.log(mean_p.clamp(min=1e-10))).sum(dim=1)
            ale = (-(probs * torch.log(probs.clamp(1e-10))).sum(dim=2)).mean(dim=0)
            smp = auroc_pixel_sample(lab, per_batch, gen_t)
            t.add(mean_p, y, lab, conf_lab, smp, tot, ale, head_pred=probs.argmax(dim=2))
            del probs

            if alphas is not None:
                alphas = alphas.float()
                a0 = alphas.sum(dim=1, keepdim=True)
                mp = alphas / a0
                stot = -(mp * torch.log(mp.clamp(min=1e-10))).sum(dim=1)
                sale = torch.digamma(a0.squeeze(1) + 1) - (mp * torch.digamma(alphas + 1)).sum(dim=1)
                ssmp = auroc_pixel_sample(lab, per_batch, gen_s)
                s.add(mp, y, lab, conf_lab, ssmp, stot, sale)

    res = {"teacher": t.summary(decoder.M > 1)}
    cal = {"teacher": t.calibration_arrays()}
    if s is not None:
        res["student"] = s.summary(True)
        cal["student"] = s.calibration_arrays()
    return res, cal


def _fmt(v, pct=False):
    if v is None:
        return "  -   "
    return f"{100 * v:.2f}%" if pct else f"{v:.4f}"


def main():
    p = argparse.ArgumentParser(description="Re-evaluation (AUROC + diversity) of trained reBEN / SAR / FBP checkpoints")
    p.add_argument("--pipeline", choices=["finetuned", "sar", "frozen", "fbp"], required=True)
    p.add_argument("--run", action="append", required=True,
                   help="reBEN/SAR: checkpoint stem without '_decoder.pth' (e.g. <run_name>_best). FBP: checkpoint "
                        "file name, optionally '<teacher file>::<student file>'. Repeat per checkpoint. "
                        "Relative names are looked up in --checkpoint_dir.")
    p.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    p.add_argument("--tag", type=str, required=True, help="Name for the summary JSON")
    p.add_argument("--max_patches", type=int, default=None,
                   help="Same subset semantics as training (test = first max_patches//5). Omit for full scale.")
    p.add_argument("--metadata_path", type=str)
    p.add_argument("--s2_root", type=str)
    p.add_argument("--s1_root", type=str)
    p.add_argument("--ref_root", type=str)
    p.add_argument("--embedding_dir", type=str)
    p.add_argument("--data_dir", type=str, help="FBP: root containing embeddings/ (train_decoders.py's --data_dir)")
    p.add_argument("--patch_size", type=int, default=224, help="FBP: must match the embeddings used in training")
    p.add_argument("--stride", type=int, default=None, help="FBP: must match the embeddings used in training")
    p.add_argument("--max_images", type=int, default=None, help="FBP: as train_decoders.py's --max_images")
    p.add_argument("--n_unfrozen_blocks", type=int, default=4)
    p.add_argument("--clay_native_forward", action="store_true",
                   help="Run Clay as its own Encoder.forward does: GSD-scaled position encoding, cls token, single final "
                        "norm (implies --clay_single_norm). Off by default = the original no-position-encoding path of "
                        "every existing embedding/checkpoint; a model trained with it on must be evaluated with it on.")
    p.add_argument("--clay_normalise_s2", action="store_true",
                   help="z-score the S2 DN with Clay's sentinel-2-l2a band stats instead of DN/10000. Off by default = every "
                        "existing model; a model trained with it on must be evaluated with it on.")
    p.add_argument("--clay_single_norm", action="store_true",
                   help="Skip the second application of Clay's final LayerNorm (its transformer already applies it). "
                        "Off by default to match all existing embeddings/checkpoints; a model trained with it on "
                        "must also be evaluated with it on.")
    p.add_argument("--num_classes", type=int, default=None, help="Default 20 (reBEN/SAR) or 25 (FBP)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Default 16 (reBEN/SAR) or 64 (FBP, = train_decoders.py's test loader)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--skip_done", action="store_true",
                   help="Skip runs already in <tag>_auroc_reeval.json, so a resubmission after a timeout only "
                        "evaluates the missing ones")
    p.add_argument("--calibration_dir", type=str, default=None,
                   help="Also save per-checkpoint calibration arrays (<dir>/<tag>__<run>__<teacher|student>.npz) "
                        "for plot_calibration.py: reliability / error-confidence curves, pixel and error mass per "
                        "confidence bin, correct-vs-wrong confidence histograms, class-wise ECE")
    args = p.parse_args()
    fbp = args.pipeline == "fbp"
    if args.num_classes is None:
        args.num_classes = 25 if fbp else 20
    if args.batch_size is None:
        args.batch_size = 64 if fbp else 16
    log_msg(f"Re-eval: {vars(args)}")

    if args.pipeline == "finetuned":
        from utils.dataset_e2e import load_reben_splits, ReBENRawDataset
        _, _, test_ds = load_reben_splits(metadata_path=args.metadata_path, s2_root=args.s2_root,
                                          ref_root=args.ref_root, max_patches=args.max_patches,
                                          clay_normalise=args.clay_normalise_s2)
        waves = ReBENRawDataset.WAVELENGTHS
    elif args.pipeline == "sar":
        from utils.dataset_e2e import load_reben_sar_splits, ReBENSARRawDataset
        _, _, test_ds = load_reben_sar_splits(metadata_path=args.metadata_path, s1_root=args.s1_root,
                                              ref_root=args.ref_root, max_patches=args.max_patches)
        waves = ReBENSARRawDataset.WAVELENGTHS
    elif args.pipeline == "frozen":
        from utils.dataset_e2e import BakedReBENDataset
        test_ds = BakedReBENDataset(Path(args.embedding_dir), split="test", augment=False)
        if args.max_patches:
            test_ds = Subset(test_ds, range(min(args.max_patches // 5, len(test_ds))))
        waves = None
    else:
        from utils.dataset import BakedFeatureDataset
        if args.data_dir is None or args.stride is None:
            p.error("--pipeline fbp needs --data_dir and --stride")
        test_files = build_fbp_test_files(args.data_dir, args.patch_size, args.stride, args.max_images)
        log_msg(f"FBP test split: {len(test_files)} images: {[f.stem.replace('_embeddings', '') for f in test_files]}")
        test_ds = BakedFeatureDataset(test_files, augment=False)
        waves = None
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    log_msg(f"Test set: {len(test_ds)} patches, {len(loader)} batches")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"{args.tag}_auroc_reeval.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for run in args.run:
        log_msg(f"=== {run}")
        if args.skip_done and run in results:
            log_msg("  already in the results JSON, skipping (--skip_done)")
            continue
        try:
            if fbp:
                encoder, decoder, student = load_models_fbp(run, args.checkpoint_dir)
            else:
                full = run if os.path.isabs(run) else os.path.join(args.checkpoint_dir, run)
                encoder, decoder, student = load_models(full, args.pipeline, args.n_unfrozen_blocks, args.clay_single_norm, args.clay_native_forward)
            res, cal = evaluate(encoder, decoder, student, loader, waves, args.num_classes, fbp=fbp)
        except Exception as e:   # one bad checkpoint must not cost the rest of the batch
            log_msg(f"  FAILED: {type(e).__name__}: {e}")
            continue
        res.update({"pipeline": args.pipeline, "max_patches": args.max_patches, "test_patches": len(test_ds),
                    "M": decoder.M})
        results[run] = res
        if args.calibration_dir:
            os.makedirs(args.calibration_dir, exist_ok=True)
            safe_run = Path(run.split("::")[0]).name.replace(".pth", "")
            for who, arrays in cal.items():
                np.savez_compressed(os.path.join(args.calibration_dir, f"{args.tag}__{safe_run}__{who}.npz"),
                                    run=run, who=who, tag=args.tag, **arrays)
            log_msg(f"  calibration arrays saved to {args.calibration_dir}")
        for who, r in res.items():
            if isinstance(r, dict):
                log_msg(f"  {who}: Acc {r['acc']:.4f} | mIoU {r['miou']:.4f} | ECE {r['ece']:.4f} | NLL {r['nll']:.4f} | "
                        f"AUROC total {_fmt(r['auroc_total'])} epistemic {_fmt(r.get('auroc_epistemic'))} "
                        f"aleatoric {_fmt(r.get('auroc_aleatoric'))} | epistemic share {_fmt(r['epistemic_share'], True)}")
                if r.get("normalised_disagreement") is not None or r.get("oracle_accuracy") is not None:
                    log_msg(f"  {who} diversity: norm_disagreement {_fmt(r.get('normalised_disagreement'))} | "
                            f"error_corr_ratio {_fmt(r.get('error_correlation_ratio'))} "
                            f"(rho {_fmt(r.get('error_correlation_rho'))}, Q {_fmt(r.get('error_yule_q'))}) | "
                            f"oracle_acc {_fmt(r.get('oracle_accuracy'))} | "
                            f"unanimous errors {_fmt(r.get('unanimous_share_of_errors'), True)} | "
                            f"per-head mIoU {[round(v, 4) for v in r.get('per_head_miou', [])]}")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        del encoder, decoder, student
        torch.cuda.empty_cache()

    log_msg(f"SUMMARY ({args.tag}, {args.pipeline}, {len(test_ds)} test patches) - Acc | mIoU | ECE | NLL | "
            f"AUROC total | epistemic | aleatoric | epi share | norm_dis | err_corr | oracle | unanimous")
    for run in args.run:
        if run not in results:
            log_msg(f"  {run}: FAILED")
            continue
        for who in ("teacher", "student"):
            r = results[run].get(who)
            if r is None:
                continue
            log_msg(f"  {run} [{who}]: {r['acc']:.4f} | {r['miou']:.4f} | {_fmt(r.get('ece'))} | {_fmt(r.get('nll'))} | "
                    f"{_fmt(r['auroc_total'])} | {_fmt(r.get('auroc_epistemic'))} | {_fmt(r.get('auroc_aleatoric'))} | "
                    f"{_fmt(r.get('epistemic_share'), True)} | {_fmt(r.get('normalised_disagreement'))} | "
                    f"{_fmt(r.get('error_correlation_ratio'))} | {_fmt(r.get('oracle_accuracy'))} | "
                    f"{_fmt(r.get('unanimous_share_of_errors'), True)}")
    log_msg(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
