"""
plot_calibration.py - error-confidence / reliability figures from the calibration arrays that
reeval_auroc.py --calibration_dir saves (one .npz per checkpoint and model: <tag>__<run>__<teacher|student>.npz).

Per .npz, one figure with four panels:
  (a) reliability curve: one point per confidence bin at the bin's ACTUAL mean confidence (not the bin
      centre) against its accuracy, vs the diagonal; below it, the share of all pixels and the share of all
      ERRORS falling in each bin - i.e. where the mass and the errors sit on the confidence scale
  (b) error-confidence curve with equal-mass bins (every point holds the same number of pixels), vs the
      calibrated line error = 1 - confidence; above the line = overconfident, below = underconfident
  (c) confidence histograms of correct vs wrong pixels
  (d) risk-coverage: error rate of the most-confident x% of pixels, ranked by confidence and by total
      uncertainty; the flat line is a model whose ranking carries no information
plus a CSV row of summary numbers (ECE, equal-mass ECE, signed gap, share of pixels / errors below 0.5
confidence, class-wise ECE) and, with several inputs, an overlay figure comparing them.

Panels (a) and the ECE / mass numbers use the exact per-bin sums over every scored pixel; (b)-(d) use the
~5M-pixel AUROC sample. Usage:
    python plot_calibration.py results/calibration/*.npz --labels AS1 AS2 ... --classes reben --out_dir results/calibration_plots
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import FBP_CLASSES, REBEN_CLASSES

N_REL_BINS = 10
N_EQUAL_MASS = 50


def _coarse(d, n_out=N_REL_BINS):
    """Sum the fine (100) equal-width bins into n_out bins."""
    k = len(d["fine_n"]) // n_out
    n = d["fine_n"].reshape(n_out, k).sum(1)
    c = d["fine_correct"].reshape(n_out, k).sum(1)
    s = d["fine_conf"].reshape(n_out, k).sum(1)
    return n, c, s


def summarise(d, class_names):
    n, c, s = _coarse(d)
    tot = n.sum()
    ok = n > 0
    acc, conf = np.zeros_like(n), np.zeros_like(n)
    acc[ok], conf[ok] = c[ok] / n[ok], s[ok] / n[ok]
    ece = float(np.sum(n[ok] / tot * np.abs(acc[ok] - conf[ok])))
    signed = float(np.sum(n[ok] / tot * (conf[ok] - acc[ok])))          # > 0 overconfident, < 0 underconfident

    sc, se = d["sample_conf"].astype(np.float64), d["sample_err"].astype(np.float64)
    order = np.argsort(sc)
    groups = np.array_split(order, 15)
    ece_em = float(sum(len(g) / len(sc) * abs((1 - se[g].mean()) - sc[g].mean()) for g in groups))

    fine_err = d["fine_n"] - d["fine_correct"]
    half = len(d["fine_n"]) // 2
    pix_below = float(d["fine_n"][:half].sum() / tot)
    err_below = float(fine_err[:half].sum() / max(fine_err.sum(), 1))

    cn, cc, cs = d["class_n"], d["class_correct"], d["class_conf"]
    class_ece = {}
    for k in range(cn.shape[0]):
        nk = cn[k].sum()
        if nk < 1000 or k == 0:   # skip near-empty predicted classes and the 'unlabelled' index
            continue
        m = cn[k] > 0
        class_ece[class_names[k] if class_names else str(k)] = float(
            np.sum(cn[k][m] / nk * np.abs(cc[k][m] / cn[k][m] - cs[k][m] / cn[k][m])))
    worst = sorted(class_ece.items(), key=lambda kv: -kv[1])[:5]
    return {
        "ece_10bin": ece, "ece_equal_mass_15": ece_em, "signed_gap": signed,
        "accuracy": float(c.sum() / tot), "mean_confidence": float(s.sum() / tot),
        "share_pixels_conf_below_0.5": pix_below, "share_errors_conf_below_0.5": err_below,
        "classwise_ece_mean": float(np.mean(list(class_ece.values()))) if class_ece else None,
        "worst_classes": "; ".join(f"{k} {v:.3f}" for k, v in worst),
    }, (n, acc, conf, ok)


def equal_mass_curve(d, n_groups=N_EQUAL_MASS):
    sc, se = d["sample_conf"], d["sample_err"].astype(np.float64)
    order = np.argsort(sc)
    groups = np.array_split(order, n_groups)
    return np.array([sc[g].mean() for g in groups]), np.array([se[g].mean() for g in groups])


def risk_coverage(score_low_is_certain, err, n_points=100):
    order = np.argsort(score_low_is_certain)
    cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    idx = np.linspace(len(err) // n_points, len(err), n_points).astype(int) - 1
    return (idx + 1) / len(err), cum[idx]


def plot_one(d, label, summ, bins, out_path):
    n, acc, conf, ok = bins
    err_per_bin = n * (1 - acc)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 2])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(conf[ok], acc[ok], "o-", label="model (bin mean confidence)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="accuracy",
           title=f"(a) reliability  ECE={summ['ece_10bin']:.3f}  signed gap={summ['signed_gap']:+.3f}")
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    centres = (np.arange(N_REL_BINS) + 0.5) / N_REL_BINS
    w = 0.4 / N_REL_BINS
    ax2.bar(centres - w / 2, n / n.sum(), width=w, label="share of pixels")
    ax2.bar(centres + w / 2, err_per_bin / max(err_per_bin.sum(), 1), width=w, label="share of errors")
    ax2.set(xlabel="confidence bin", ylabel="share")
    ax2.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    mc, er = equal_mass_curve(d)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, 1 - xs, "--", color="grey", label="calibrated: error = 1 - conf")
    ax.plot(mc, er, ".-", label=f"model ({N_EQUAL_MASS} equal-mass bins)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="error rate",
           title=f"(b) error-confidence  equal-mass ECE={summ['ece_equal_mass_15']:.3f}")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    sc, se = d["sample_conf"], d["sample_err"].astype(bool)
    hb = np.linspace(0, 1, 51)
    ax.hist(sc[~se], bins=hb, density=True, alpha=0.6, label="correct pixels")
    ax.hist(sc[se], bins=hb, density=True, alpha=0.6, label="wrong pixels")
    ax.set(xlabel="confidence", ylabel="density",
           title=f"(c) confidence: {100 * summ['share_errors_conf_below_0.5']:.0f}% of errors and "
                 f"{100 * summ['share_pixels_conf_below_0.5']:.0f}% of pixels below 0.5")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, :2])
    err = se.astype(np.float64)
    cov, risk = risk_coverage(-sc, err)
    ax.plot(cov, risk, label="ranked by confidence")
    cov, risk = risk_coverage(d["sample_total_unc"], err)
    ax.plot(cov, risk, label="ranked by total uncertainty")
    if np.any(d["sample_epistemic"] > 0):
        cov, risk = risk_coverage(d["sample_epistemic"], err)
        ax.plot(cov, risk, label="ranked by epistemic uncertainty")
    ax.axhline(err.mean(), ls="--", color="grey", label="uninformative ranking")
    ax.set(xlabel="coverage (fraction of most-certain pixels kept)", ylabel="error rate of kept pixels",
           title="(d) risk-coverage")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ax.text(0, 1, "\n".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in summ.items()), va="top", fontsize=8, family="monospace", wrap=True)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Reliability / error-confidence plots from reeval_auroc.py calibration arrays")
    p.add_argument("npz", nargs="+")
    p.add_argument("--labels", nargs="+", default=None, help="One label per npz (default: file name)")
    p.add_argument("--classes", choices=["reben", "fbp", "none"], default="none")
    p.add_argument("--out_dir", default="results/calibration_plots")
    args = p.parse_args()
    class_names = {"reben": REBEN_CLASSES, "fbp": FBP_CLASSES, "none": None}[args.classes]
    labels = args.labels or [Path(f).stem for f in args.npz]
    if len(labels) != len(args.npz):
        p.error("--labels needs one label per npz")
    os.makedirs(args.out_dir, exist_ok=True)

    rows, curves = [], []
    for f, label in zip(args.npz, labels):
        d = dict(np.load(f, allow_pickle=True))
        summ, bins = summarise(d, class_names)
        plot_one(d, label, summ, bins, os.path.join(args.out_dir, f"{Path(f).stem}.png"))
        rows.append({"label": label, "file": Path(f).name, **summ})
        curves.append((label, bins, equal_mass_curve(d)))
        print(f"{label}: ECE {summ['ece_10bin']:.4f} | equal-mass ECE {summ['ece_equal_mass_15']:.4f} | "
              f"signed gap {summ['signed_gap']:+.4f} | pixels <0.5 conf {100 * summ['share_pixels_conf_below_0.5']:.1f}% | "
              f"errors <0.5 conf {100 * summ['share_errors_conf_below_0.5']:.1f}%")

    with open(os.path.join(args.out_dir, "calibration_summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if len(curves) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].plot([0, 1], [0, 1], "--", color="grey")
        xs = np.linspace(0, 1, 50)
        axes[1].plot(xs, 1 - xs, "--", color="grey")
        for label, (n, acc, conf, ok), (mc, er) in curves:
            axes[0].plot(conf[ok], acc[ok], "o-", label=label)
            axes[1].plot(mc, er, ".-", label=label)
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="accuracy", title="reliability (bin mean confidence)")
        axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="error rate", title="error-confidence (equal-mass bins)")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "calibration_overlay.png"), dpi=120)
        plt.close(fig)
    print(f"Figures and calibration_summary.csv written to {args.out_dir}")


if __name__ == "__main__":
    main()
