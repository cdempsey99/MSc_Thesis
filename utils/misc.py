import torch.nn as nn
import random
import time
import inspect
import psutil
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, roc_auc_score
from scipy.stats import spearmanr
from utils.visualisation import *
import json
import os
import gc
import matplotlib.pyplot as plt
import math

import rasterio
from rasterio.windows import Window

def log_msg(message):
    t_stamp = time.strftime("%H:%M:%S")
    caller = inspect.stack()[1].function

    # CPU RAM (Process only)
    process = psutil.Process(os.getpid())
    cpu_ram = process.memory_info().rss / (1024 ** 3)

    # GPU VRAM (Allocated on current device)
    gpu_vram = 0
    if torch.cuda.is_available():
        gpu_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    print(f"[{t_stamp}] [{caller:^15}] [CPU: {cpu_ram:5.2f}GB | GPU: {gpu_vram:5.2f}GB] | {message}")


# Aux fn to calculate ECE
def get_ece(y_pred, y_true, unc_map_flat, save_name="reliability_diagram"):

    # Bin the unc values
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)

    ece = 0
    # Keep track of avg acc and unc for each bin for plotting later
    bin_accs = np.zeros(num_bins)
    bin_props = np.zeros(num_bins)

    # Inside each bin
    for i in range(num_bins):
        bin_start = bin_boundaries[i]
        bin_end = bin_boundaries[i+1]

        # Create mask to find which pixels we want to take in this bin
        in_bin = (unc_map_flat > bin_start) & (unc_map_flat < bin_end)

        prop_in_bin = np.mean(in_bin)
        bin_props[i] = prop_in_bin

        if prop_in_bin > 0:

            bin_confidence = np.mean(unc_map_flat[in_bin])
            bin_accuracy = np.mean(y_true[in_bin] == y_pred[in_bin])

            bin_accs[i] = bin_accuracy

            ece += np.abs(bin_accuracy - bin_confidence) * prop_in_bin

    plot_reliability_diagram(bin_accs, bin_props, save_name=save_name)

    return ece

# Fn to calculate the JS divergence between
# Maybe TODO : this is calculating the mean and comparing all heads to that, is that the best?
# Ok I think it is for the moment anyway
# Fn to compute pairwise Jenson Shannon Divergence between M decoder heads
# all_preds : [M, B, C, H, W] tensor
def js_divergence_loss_old(all_preds):
    # all_preds shape: [M, B, C, H, W]
    M = all_preds.shape[0]

    # Softmax to get probabilities
    all_probs = torch.softmax(all_preds, dim=2)

    # Get mean distribution
    # Take average across M heads
    mean_probs = torch.mean(all_probs, dim=0)
    log_mean_probs = torch.log(mean_probs + 1e-10)

    # Calculate KL div of each head wrt to the mean
    # using F.kl_div
    total_kl = 0
    for i in range(M):
        # F.kl_div(input, target) where input is log-space
        #p_i = all_probs[i]
        #total_kl += F.kl_div(log_mean_probs, p_i, reduction="batchmean", log_target=False)
        kl = F.kl_div(log_mean_probs, all_probs[i], reduction="sum")
        # Divide by (Batch * #pixels) to get the avg per pixel
        total_kl += kl / (all_probs.size(1) * all_probs.size(3) * all_probs.size(4))

    return total_kl / M


def js_divergence_loss(all_preds):

    M = all_preds.shape[0]

    # Case of single decoder, skip this function
    if M == 1:
        return torch.tensor(0.0, device=all_preds.device)

    # Softmax predictions to get probabilities
    # Move these to CPU to avoid GPU OOM
    all_probs = torch.softmax(all_preds, dim=2) # [M, B, C, H, W]

    total_jsd = 0
    count = 0
    for i in range(M):
        for j in range(i + 1, M):
            # midpoint of the two distributions
            p_i = all_probs[i].detach()
            p_j = all_probs[j].detach()
            avg_ij = 0.5 * (p_i + p_j)

            kl_i = F.kl_div(torch.log(avg_ij + 1e-10), all_probs[i], reduction="sum")
            kl_j = F.kl_div(torch.log(avg_ij + 1e-10), all_probs[j], reduction="sum")

            del avg_ij, p_i, p_j
            torch.cuda.empty_cache()

            n_pixels = all_probs.size(1) * all_probs.size(3) * all_probs.size(4)
            # JSD is a symmetrised version of KLD
            jsd = 0.5 * (kl_i + kl_j) / n_pixels

            total_jsd += jsd
            count += 1

    # return JSD averaged over all the pairs
    # we want higher loss to be bad, mean less diversity between heads
    # now 0 = max diversity, ln(2) = no diversity
    return math.log(2) - total_jsd / count

# Fn to compute diversity loss based on Pearson correlation between decoder heads
# For each class C, computes correlation between head spatial activation maps
# all_preds: [M, B, C, H, W] raw logits
def pearson_diversity_loss(all_preds):

    M, B, C, H, W = all_preds.shape

    if M == 1:
        return torch.tensor(0.0, device=all_preds.device)

    all_probs = torch.softmax(all_preds, dim=2)  # [M, B, C, H, W]

    # Reshape to [M, C, B*H*W]
    all_probs = all_probs.permute(0, 2, 1, 3, 4).reshape(M, C, -1)

    # Centre each vector
    mean = all_probs.mean(dim=2, keepdim=True)  # [M, C, 1]
    centred = all_probs - mean  # [M, C, B*H*W]

    # Normalise by std
    std = torch.sqrt((centred ** 2).sum(dim=2, keepdim=True) + 1e-10)
    normed = centred / std

    # Compute full [M, M] correlation matrix for each class C
    # normed [M, C, B*H*W] --> corr [C, M, M]
    corr = torch.einsum('mcd,ncd->cmn', normed, normed)  # [C, M, M], values in [-1, 1]

    # Average over classes
    corr_mean = corr.mean(dim=0)  # [M, M]

    # Extract upper triangle
    mask = torch.triu(torch.ones(M, M, device=all_preds.device), diagonal=1).bool()
    pairwise_corr = corr_mean[mask]  # [M*(M-1)/2]

    return (pairwise_corr ** 2).mean()


# Computes pairwise orthogonality loss between weights of the different decoder models
# Uses squared cosine similarity between flattened weight vectors
# 0 = perfectly orthogonal, 1 = identical
def orthogonality_loss(decoder_ensemble):

    M = decoder_ensemble.M

    if M == 1 :
        return torch.tensor(0.0, device=next(decoder_ensemble.parameters()).device)

    layer_names = ['linear_fusion', 'spatial_refine1', 'spatial_refine2', 'bottleneck', 'classifier']

    total_orth = 0.0
    count = 0

    for name in layer_names:
        # [out_channels, -1] so each row is one filter (one output channel flattened)
        weights = [getattr(head, name).weight.view(getattr(head, name).weight.shape[0], -1)
                   for head in decoder_ensemble.heads]
        for i in range(M):
            for j in range(i + 1, M):
                # cos similarity per corresponding filter pair, averaged over out_channels
                cos_sims = F.cosine_similarity(weights[i], weights[j], dim=1)  # [out_channels]
                total_orth += (cos_sims ** 2).mean()
                count += 1

    return total_orth / count


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=0, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        num_classes = input.shape[1]
        probs = torch.softmax(input, dim=1)  # [B, C, H, W]

        valid = (target != self.ignore_index)  # [B, H, W]
        target_safe = target.clone()
        target_safe[~valid] = 0  # avoid out-of-range before one_hot

        target_one_hot = F.one_hot(target_safe, num_classes).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        target_one_hot = target_one_hot * mask

        dice_scores = []
        for c in range(1, num_classes):  # skip ignore_index class (0)
            p = probs[:, c].reshape(-1)
            t = target_one_hot[:, c].reshape(-1)
            if t.sum() == 0:
                continue
            intersection = (p * t).sum()
            dice_scores.append(
                (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            )

        if not dice_scores:
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        return 1.0 - torch.stack(dice_scores).mean()


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class segmentation.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    gamma=0 reduces to standard cross-entropy.
    gamma=2 is the standard value from Lin et al. (RetinaNet).
    Automatically handles class imbalance by downweighting confident/easy pixels,
    forcing the model to focus on hard/rare ones.
    """
    def __init__(self, gamma=2.0, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # Per-pixel CE loss — 0 for ignored pixels
        ce_loss = F.cross_entropy(input, target, ignore_index=self.ignore_index, reduction='none')
        # p_t = probability assigned to the correct class
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        # Average only over labelled pixels
        mask = target != self.ignore_index
        return focal_loss[mask].mean() if mask.sum() > 0 else focal_loss.mean()


def compute_class_weights(train_loader, num_classes, ignore_index=0):
    """
    Log-damped inverse frequency weighting (Tong et al.): W_k = 1 / log(1 + mu_k)
    where mu_k = pixels_of_class_k / total_labelled_pixels.
    Classes absent from training or equal to ignore_index get weight 0.
    """
    counts = torch.zeros(num_classes)
    for _, targets in train_loader:
        for c in range(num_classes):
            counts[c] += (targets == c).sum().item()

    counts[ignore_index] = 0
    total = counts.sum().item()
    if total == 0:
        return torch.ones(num_classes)

    mu = counts / total
    weights = torch.zeros(num_classes)
    present = counts > 0
    weights[present] = 1.0 / torch.log(1.0 + mu[present])
    weights.clamp_(max=10.0)
    return weights


# Fn to calculate some metrics for accuracy and uncertainty
# TODO : These could probably be upgraded in future
def evaluate_metrics(class_map, ground_truth, unc_map):

    # 1. Ensure ground_truth is a 2D numpy array [224, 224]
    # and not [1, 224, 224]
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.squeeze().cpu().numpy()
    else:
        ground_truth = ground_truth.squeeze()

    # Flatten arrays and filter out unlabelled pixels
    y_true = ground_truth[ground_truth != 0]
    y_pred = class_map[ground_truth != 0]
    unc_map_flat = unc_map[ground_truth != 0]

    # Calculate mIoU
    miou = jaccard_score(y_true, y_pred, average="macro", labels=np.unique(y_true))

    # Also overall accuracy
    overall_accuracy = (y_true == y_pred).mean()

    # Average Uncertainty
    # what are the units of uncertainty here?
    avg_unc = np.mean(unc_map[ground_truth != 0])

    # Expected Calibration Error
    ece = get_ece(y_pred, y_true, unc_map_flat)

    log_msg(f"mIoU : {miou} | Overall Accuracy : {overall_accuracy} | Average Uncertainty : {avg_unc} | ECE : {ece} ")

    return miou, overall_accuracy, avg_unc, ece

# Fn to evaluate trained model
def get_decoder_output_maps(trained_decoder_model, grid_features, save_name="heads_predictions"):
    # Evaluation
    trained_decoder_model.eval()
    with torch.no_grad():
        # eval_preds shape: [5, 1, 24, 224, 224]
        eval_preds = trained_decoder_model(grid_features)

        # Set up the plotting grid
        fig, axes = plt.subplots(1, trained_decoder_model.M, figsize=(20, 5), squeeze=False)
        axes = axes[0]
        fig.suptitle(f"Individual Ensemble Head Predictions", fontsize=16)

        for i in range(trained_decoder_model.M):
            # Extract the [24, 224, 224] tensor for this head and get the winning class per pixel
            head_logits = eval_preds[i].squeeze(0)
            head_map = torch.argmax(head_logits, dim=0).cpu().numpy()

            # Print stats to console
            log_msg(f"Head {i} unique predictions: {np.unique(head_map)}")

            # Plotting
            im = axes[i].imshow(head_map, cmap='tab20', vmin=0, vmax=NUM_CLASSES)
            axes[i].set_title(f"Head {i + 1}")
            axes[i].axis('off')

        # Save figure
        save_path = os.path.join(BASE_OUT, "heads", save_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        current_time = datetime.now().strftime("%Y%m%d%H%M")
        full_file_path = os.path.join(save_path, f"ensemble_heads_{save_name}_{current_time}.png")
        plt.savefig(full_file_path, bbox_inches='tight', dpi=150)
        # plt.show()
        plt.close()

        # Always calculate these regardless of M
        # Take softmax first, then calculate class predictions for every pixel
        all_probs = torch.softmax(eval_preds, dim=2) # softmaxxing over the class dimension: [5, 1, 25, 224, 224]
        mean_probs = all_probs.mean(dim=0)  # [1, 25, 224, 224]
        class_map = torch.argmax(mean_probs, dim=1).squeeze().cpu().numpy() # [224, 224]

        total_entropy = -1 * torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)

        # Ensemble-only uncertainty measures
        if trained_decoder_model.M > 1:
            variance_map = all_probs.var(dim=0).mean(dim=1).squeeze().cpu().numpy()
            individual_entropies = -1 * torch.sum(all_probs * torch.log(all_probs + 1e-10), dim=2)
            avg_entropy = torch.mean(individual_entropies, dim=0)
            mutual_info = total_entropy - avg_entropy
        else:
            variance_map = np.zeros((224, 224))
            mutual_info = torch.zeros_like(total_entropy)

    return mean_probs, class_map, variance_map, total_entropy, mutual_info

def get_temperature(epoch, num_epochs, T_start, T_end=1.0):
    """Linear annealing from T_start down to T_end over training."""
    if num_epochs <= 1:
        return T_end
    return T_start - (T_start - T_end) * epoch / (num_epochs - 1)


def fit_temperature(logits, targets, max_iter=50, lr=0.01):
    """
    Post-hoc calibration (Guo et al., 2017), unrelated to get_temperature above (that's
    the EnDD distillation-softmax annealing schedule; this is calibration temperature
    scaling). Fits a single scalar T minimizing NLL of softmax(logits / T) against
    targets. Dividing every class's logit by the same T never changes the argmax, so
    predictions — and therefore mIoU — are unaffected by construction; only confidence
    shape (ECE, NLL, reliability diagrams) moves.

    Fit T on VALIDATION data only, then apply it to TEST logits before computing
    calibration metrics — fitting and evaluating on the same split invalidates the
    result. logits: [N, K] raw pre-softmax, targets: [N] int class labels — both
    already restricted to labelled pixels by the caller (e.g. via a mask > 0).

    T is parameterised as exp(log_T) so it's guaranteed positive without clamping.
    """
    logits = logits.detach()
    targets = targets.detach()

    log_temperature = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(log_temperature.exp().item())
    log_msg(f"Fitted temperature T={temperature:.4f} on {targets.numel():,} labelled pixels")
    return temperature


def endd_loss(student_alphas, teacher_logits, temperature=1.0):
    """
    Ensemble Distribution Distillation loss: Dirichlet NLL.

    Trains the student to output Dirichlet parameters α that assign high probability
    to the teacher ensemble's predictive distributions, capturing not just the mean
    prediction but the full spread of teacher opinions.

    student_alphas:  [B, K, H, W]  concentration parameters, all > 0
    teacher_logits:  [M, B, K, H, W]  raw logits from the frozen teacher ensemble
    temperature:     T applied to teacher logits before softmax; anneal from T_start→1
                     to soften targets early in training for stability

    Returns: scalar mean NLL
    """
    M = teacher_logits.shape[0]

    with torch.no_grad():
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=2)  # [M, B, K, H, W]

    alpha = student_alphas.clamp(min=1e-6)   # [B, K, H, W]
    alpha0 = alpha.sum(dim=1)                # [B, H, W]

    # Log-normaliser: lgamma(Σα_k) - Σ lgamma(α_k)  →  [B, H, W]
    log_norm = torch.lgamma(alpha0) - torch.lgamma(alpha).sum(dim=1)

    # Expected log-likelihood averaged over M teachers
    # Σ_k (α_k - 1) * log(π_m,k), summed over K, averaged over M  →  [B, H, W]
    log_teacher = torch.log(teacher_probs + 1e-10)              # [M, B, K, H, W]
    ll_terms = ((alpha.unsqueeze(0) - 1) * log_teacher).sum(dim=2).mean(dim=0)  # [B, H, W]

    return (-(log_norm + ll_terms)).mean()


def evaluate_student_test_set(student_model, test_loader, args, run_name="student_test", class_names=FBP_CLASSES):
    """
    Evaluate a trained StudentHead on the test split.
    The student outputs Dirichlet alphas [B, K, H, W]; class predictions come from
    the Dirichlet mean α / Σα_k, and confidence from its max value.
    """
    student_model.eval()

    num_classes = args.num_classes
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Running ECE bins — avoids storing all confidence values in memory
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_conf_sums = np.zeros(num_bins)
    bin_acc_sums  = np.zeros(num_bins, dtype=np.int64)
    bin_counts    = np.zeros(num_bins, dtype=np.int64)

    patch_count = 0
    nll_sum = 0.0
    nll_count = 0
    auroc_entropy = []
    auroc_errors = []
    AUROC_MAX = 2_000_000

    vis_global_indices = set(random.sample(
        range(len(test_loader.dataset)),
        min(3, len(test_loader.dataset))
    ))
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (test_features, test_masks) in enumerate(test_loader):
            test_features = test_features.to(DEVICE)

            alphas = student_model(test_features)                          # [B, K, H, W]
            alpha0 = alphas.sum(dim=1, keepdim=True)                      # [B, 1, H, W]
            mean_probs = alphas / alpha0                                   # [B, K, H, W]

            test_masks_t = test_masks.to(DEVICE).long()
            labelled_t = test_masks_t > 0
            if labelled_t.any():
                log_probs = torch.log(mean_probs.clamp(min=1e-10))
                gt_idx = test_masks_t.unsqueeze(1).clamp(0, num_classes - 1)
                nll_sum += (-log_probs.gather(1, gt_idx).squeeze(1)[labelled_t]).sum().item()
                nll_count += labelled_t.sum().item()
                auroc_collected = sum(len(x) for x in auroc_entropy) if auroc_entropy else 0
                if auroc_collected < AUROC_MAX:
                    ent_t = -(mean_probs * torch.log(mean_probs.clamp(min=1e-10))).sum(dim=1)
                    err_t = (torch.argmax(mean_probs, dim=1) != test_masks_t).float()
                    auroc_entropy.append(ent_t[labelled_t].cpu().numpy())
                    auroc_errors.append(err_t[labelled_t].cpu().numpy())
            alpha0_sq = alpha0.squeeze(1)                                  # [B, H, W]

            class_maps = torch.argmax(mean_probs, dim=1).cpu().numpy()    # [B, H, W]
            conf_maps = torch.max(mean_probs, dim=1)[0].cpu().numpy()     # [B, H, W]

            # Dirichlet uncertainty decomposition
            total_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)       # [B, H, W]
            aleatoric = (torch.digamma(alpha0_sq + 1)
                         - (mean_probs * torch.digamma(alphas + 1)).sum(dim=1))            # [B, H, W]
            epistemic = (total_entropy - aleatoric).clamp(min=0)                           # [B, H, W]

            for b in range(test_features.shape[0]):
                g_idx = batch_idx * test_loader.batch_size + b
                gt = test_masks[b].squeeze().cpu().numpy()

                if g_idx in vis_global_indices and vis_count < 3:
                    img_pt_path, local_idx = test_loader.dataset.get_patch_info(g_idx)
                    img_stem = Path(img_pt_path).stem.replace('_embeddings', '')
                    data_dir = Path(img_pt_path).parent.parent.parent.parent.parent / "data" / "fbp"
                    raw_img_path = data_dir / f"{img_stem}.tif"

                    patches_per_row = len(range(0, 7300 - 224, args.stride))
                    x = (local_idx % patches_per_row) * args.stride
                    y = (local_idx // patches_per_row) * args.stride

                    raw_patch = None
                    if raw_img_path.exists():
                        with rasterio.open(raw_img_path) as src:
                            img = src.read([1, 2, 3], window=Window(x, y, 224, 224)).astype(np.float32)
                            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                            raw_patch = np.transpose(img, (1, 2, 0))

                    visualise_student_uncertainty(
                        class_map=class_maps[b],
                        total_entropy=total_entropy[b],
                        aleatoric=aleatoric[b],
                        epistemic=epistemic[b],
                        alpha0_map=alpha0_sq[b],
                        ground_truth=gt,
                        hide_unlabelled=args.hide_unlabelled_pixels,
                        save_name=f"{run_name}_patch_{g_idx}",
                        raw_patch=raw_patch,
                        patch_info=f"{img_stem} x={x} y={y}",
                    )
                    plt.close('all')
                    vis_count += 1

                if (gt > 0).sum() < 100:
                    continue

                mask      = gt > 0
                pred_flat = class_maps[b][mask]
                true_flat = gt[mask].astype(np.int64)
                conf_flat = conf_maps[b][mask]

                # Update confusion matrix — O(num_classes²) total memory
                np.add.at(conf_matrix, (true_flat, pred_flat), 1)

                # Update running ECE bins
                for bin_idx in range(num_bins):
                    in_bin = (conf_flat >= bin_boundaries[bin_idx]) & (conf_flat < bin_boundaries[bin_idx + 1])
                    n = in_bin.sum()
                    if n > 0:
                        bin_conf_sums[bin_idx] += conf_flat[in_bin].sum()
                        bin_acc_sums[bin_idx]  += (true_flat[in_bin] == pred_flat[in_bin]).sum()
                        bin_counts[bin_idx]    += n

                patch_count += 1

    # mIoU from confusion matrix — only average over classes present in ground truth
    iou_per_class = np.zeros(num_classes - 1)
    present = []
    for c in range(1, num_classes):
        if conf_matrix[c, :].sum() > 0:
            tp    = conf_matrix[c, c]
            fp    = conf_matrix[:, c].sum() - tp
            fn    = conf_matrix[c, :].sum() - tp
            denom = tp + fp + fn
            iou_per_class[c - 1] = tp / denom if denom > 0 else 0.0
            present.append(c - 1)
    global_miou = float(np.mean(iou_per_class[present])) if present else 0.0

    class_pixel_counts = conf_matrix[1:, :].sum(axis=1)
    total_labelled = class_pixel_counts.sum()
    fw_iou = float((class_pixel_counts / max(total_labelled, 1) * iou_per_class).sum())

    global_nll = nll_sum / max(nll_count, 1)
    if auroc_entropy:
        all_ent = np.concatenate(auroc_entropy)
        all_err = np.concatenate(auroc_errors)
        global_auroc = float(roc_auc_score(all_err, all_ent)) if len(np.unique(all_err)) > 1 else 0.0
    else:
        global_auroc = 0.0

    # Overall accuracy (labelled pixels only)
    global_acc = float(np.diag(conf_matrix)[1:].sum() / max(conf_matrix[1:, :].sum(), 1))

    # ECE and reliability diagram from running bins
    bin_accs  = np.zeros(num_bins)
    bin_props = np.zeros(num_bins)
    ece = 0.0
    total_px = bin_counts.sum()
    if total_px > 0:
        for i in range(num_bins):
            if bin_counts[i] > 0:
                avg_conf = bin_conf_sums[i] / bin_counts[i]
                avg_acc  = bin_acc_sums[i]  / bin_counts[i]
                prop     = bin_counts[i] / total_px
                bin_accs[i]  = avg_acc
                bin_props[i] = prop
                ece += abs(avg_acc - avg_conf) * prop
    plot_reliability_diagram(bin_accs, bin_props, save_name=f"{run_name}_reliability")
    global_ece = float(ece)

    log_msg(f"STUDENT TEST RESULTS ({patch_count} patches):")
    log_msg(f"Global mIoU: {global_miou:.4f} | fw-IoU: {fw_iou:.4f} | Acc: {global_acc:.4f} | ECE: {global_ece:.4f} | NLL: {global_nll:.4f} | AUROC: {global_auroc:.4f}")
    log_msg("Per-class IoU (descending frequency):")
    freq_order = np.argsort(class_pixel_counts)[::-1]
    for class_idx in freq_order:
        log_msg(f"  {class_names[class_idx + 1]}: {iou_per_class[class_idx]:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": run_name,
        "global_miou": global_miou,
        "global_fwiou": fw_iou,
        "global_accuracy": global_acc,
        "global_ece": global_ece,
        "global_nll": global_nll,
        "global_auroc": global_auroc,
        "num_patches": patch_count,
        "per_class_iou": {class_names[i + 1]: float(iou) for i, iou in enumerate(iou_per_class)},
        "confusion_matrix": conf_matrix.tolist(),
    }
    results_path = os.path.join(runs_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Student results saved to {results_path}")
    plot_confusion_matrix(results_path, save_name=f"{run_name}_confusion")

    return results


def evaluate_uncertainty_correlation(teacher_model, student_model, test_loader, args, run_name="uncertainty_correlation"):
    """
    Compares the spatial layout of teacher (ensemble) vs student (Dirichlet) uncertainty maps.
    For each test patch, flattens the labelled pixels of the epistemic, aleatoric, and total
    entropy maps from both models and computes the Spearman rank correlation between them,
    independently for each of the three. Maps are computed on the fly per batch and never
    written to disk — only the per-patch correlations are accumulated into a running mean.
    """
    teacher_model.eval()
    student_model.eval()

    num_classes = args.num_classes

    epi_corr_sum = 0.0
    epi_corr_count = 0
    alea_corr_sum = 0.0
    alea_corr_count = 0
    total_corr_sum = 0.0
    total_corr_count = 0
    skipped_degenerate = 0
    patch_count = 0

    with torch.no_grad():
        for test_features, test_masks in test_loader:
            test_features = test_features.to(DEVICE)

            # Teacher: ensemble mixture, epistemic/aleatoric decomposition
            all_preds = teacher_model(test_features)                          # [M, B, K, h, w]
            M_sz, B_sz = teacher_model.M, test_features.shape[0]
            all_preds_up = F.interpolate(
                all_preds.view(M_sz * B_sz, num_classes, *all_preds.shape[-2:]),
                size=(224, 224), mode='bilinear', align_corners=False
            ).view(M_sz, B_sz, num_classes, 224, 224)
            teacher_head_probs = torch.softmax(all_preds_up.float(), dim=2)
            teacher_mean_probs = teacher_head_probs.mean(dim=0)               # [B, K, H, W]

            teacher_total_ent = -(teacher_mean_probs * torch.log(teacher_mean_probs.clamp(min=1e-10))).sum(dim=1)
            teacher_aleatoric = torch.zeros_like(teacher_total_ent)
            for m in range(teacher_model.M):
                hp = teacher_head_probs[m]
                teacher_aleatoric += -(hp * torch.log(hp.clamp(1e-10))).sum(dim=1)
            teacher_aleatoric /= teacher_model.M
            teacher_epistemic = (teacher_total_ent - teacher_aleatoric).clamp(min=0)

            # Student: Dirichlet decomposition
            alphas = student_model(test_features)                             # [B, K, 224, 224]
            alpha0 = alphas.sum(dim=1, keepdim=True)
            student_mean_probs = alphas / alpha0
            alpha0_sq = alpha0.squeeze(1)
            student_total_ent = -(student_mean_probs * torch.log(student_mean_probs + 1e-10)).sum(dim=1)
            student_aleatoric = (torch.digamma(alpha0_sq + 1)
                                  - (student_mean_probs * torch.digamma(alphas + 1)).sum(dim=1))
            student_epistemic = (student_total_ent - student_aleatoric).clamp(min=0)

            teacher_epi_np = teacher_epistemic.cpu().numpy()
            teacher_alea_np = teacher_aleatoric.cpu().numpy()
            teacher_total_np = teacher_total_ent.cpu().numpy()
            student_epi_np = student_epistemic.cpu().numpy()
            student_alea_np = student_aleatoric.cpu().numpy()
            student_total_np = student_total_ent.cpu().numpy()

            for b in range(test_features.shape[0]):
                gt = test_masks[b].squeeze().cpu().numpy()
                mask = gt > 0
                if mask.sum() < 100:
                    continue
                patch_count += 1

                epi_rho, _ = spearmanr(teacher_epi_np[b][mask], student_epi_np[b][mask])
                if not math.isnan(epi_rho):
                    epi_corr_sum += epi_rho
                    epi_corr_count += 1
                else:
                    skipped_degenerate += 1

                alea_rho, _ = spearmanr(teacher_alea_np[b][mask], student_alea_np[b][mask])
                if not math.isnan(alea_rho):
                    alea_corr_sum += alea_rho
                    alea_corr_count += 1
                else:
                    skipped_degenerate += 1

                total_rho, _ = spearmanr(teacher_total_np[b][mask], student_total_np[b][mask])
                if not math.isnan(total_rho):
                    total_corr_sum += total_rho
                    total_corr_count += 1
                else:
                    skipped_degenerate += 1

    mean_epi_corr = epi_corr_sum / max(epi_corr_count, 1)
    mean_alea_corr = alea_corr_sum / max(alea_corr_count, 1)
    mean_total_corr = total_corr_sum / max(total_corr_count, 1)

    log_msg(f"TEACHER/STUDENT UNCERTAINTY CORRELATION ({patch_count} patches, {skipped_degenerate} degenerate skipped):")
    log_msg(f"Mean Spearman epistemic corr: {mean_epi_corr:.4f} | Mean Spearman aleatoric corr: {mean_alea_corr:.4f} | Mean Spearman total corr: {mean_total_corr:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": run_name,
        "mean_epistemic_spearman": mean_epi_corr,
        "mean_aleatoric_spearman": mean_alea_corr,
        "mean_total_spearman": mean_total_corr,
        "num_patches": patch_count,
        "num_epistemic_correlations": epi_corr_count,
        "num_aleatoric_correlations": alea_corr_count,
        "num_total_correlations": total_corr_count,
    }
    results_path = os.path.join(runs_dir, f"{run_name}_uncertainty_correlation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Uncertainty correlation results saved to {results_path}")

    return results


def _spearman_error_localization_batch(uncertainty_t, pred_class_np, test_masks, stats):
    """
    Accumulates per-patch Spearman(uncertainty, error) into stats dict in place.
    uncertainty_t: [B, H, W] torch tensor. pred_class_np: [B, H, W] numpy array.
    test_masks: raw batch of ground-truth masks as loaded from the DataLoader.
    stats keys: corr_values (list of per-patch rho), skipped, patch_count. Keeping
    every patch's rho (not just a running sum) is what lets _log_and_save_error_localization
    report the full distribution, not just its mean.
    """
    uncertainty_np = uncertainty_t.cpu().numpy()
    for b in range(uncertainty_np.shape[0]):
        gt = test_masks[b].squeeze().cpu().numpy()
        mask = gt > 0
        if mask.sum() < 100:
            continue
        stats['patch_count'] += 1
        error_flat = (pred_class_np[b][mask] != gt[mask]).astype(np.float64)
        rho, _ = spearmanr(uncertainty_np[b][mask], error_flat)
        if not math.isnan(rho):
            stats['corr_values'].append(rho)
        else:
            stats['skipped'] += 1


def _log_and_save_error_localization(who, total_stats, epi_stats, run_name):
    total_vals = np.array(total_stats['corr_values'], dtype=np.float64)
    epi_vals = np.array(epi_stats['corr_values'], dtype=np.float64)
    mean_total = float(total_vals.mean()) if total_vals.size else 0.0
    mean_epi = float(epi_vals.mean()) if epi_vals.size else 0.0
    frac_total_pos = float((total_vals > 0).mean()) if total_vals.size else 0.0
    frac_epi_pos = float((epi_vals > 0).mean()) if epi_vals.size else 0.0

    log_msg(f"{who.upper()} ERROR LOCALIZATION ({total_stats['patch_count']} patches, "
             f"{total_stats['skipped'] + epi_stats['skipped']} degenerate skipped): "
             f"total mean={mean_total:.4f} (>{0:.0f}: {frac_total_pos:.1%}) | "
             f"epistemic mean={mean_epi:.4f} (>{0:.0f}: {frac_epi_pos:.1%})")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": run_name,
        "who": who,
        "mean_total_localization": mean_total,
        "mean_epistemic_localization": mean_epi,
        "frac_total_localization_positive": frac_total_pos,
        "frac_epistemic_localization_positive": frac_epi_pos,
        "per_patch_total_localization": total_vals.tolist(),
        "per_patch_epistemic_localization": epi_vals.tolist(),
        "num_patches": total_stats['patch_count'],
        "num_total_correlations": int(total_vals.size),
        "num_epistemic_correlations": int(epi_vals.size),
        "num_skipped_degenerate": total_stats['skipped'] + epi_stats['skipped'],
    }
    results_path = os.path.join(runs_dir, f"{run_name}_{who}_error_localization.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"{who.capitalize()} error localization results saved to {results_path}")

    plot_error_localization_histogram(
        total_vals, epi_vals, save_name=f"{run_name}_{who}_error_localization_histogram"
    )
    return results


def ensemble_uncertainty_and_pred(all_preds, num_classes):
    """
    Entropy decomposition for any M-head ensemble (teacher-style): total = H(mean probs),
    aleatoric = mean(H(head probs)), epistemic = total - aleatoric.
    all_preds: [M, B, K, h, w] raw logits (pre-softmax, any spatial resolution).
    Returns (total_entropy, epistemic_entropy, predicted_class_np), all [B, 224, 224].
    """
    M_sz, B_sz = all_preds.shape[0], all_preds.shape[1]
    all_preds_up = F.interpolate(
        all_preds.reshape(M_sz * B_sz, num_classes, *all_preds.shape[-2:]),
        size=(224, 224), mode='bilinear', align_corners=False
    ).view(M_sz, B_sz, num_classes, 224, 224)
    head_probs = torch.softmax(all_preds_up.float(), dim=2)
    mean_probs = head_probs.mean(dim=0)

    total_ent = -(mean_probs * torch.log(mean_probs.clamp(min=1e-10))).sum(dim=1)
    aleatoric = torch.zeros_like(total_ent)
    for m in range(M_sz):
        hp = head_probs[m]
        aleatoric += -(hp * torch.log(hp.clamp(min=1e-10))).sum(dim=1)
    aleatoric /= M_sz
    epistemic = (total_ent - aleatoric).clamp(min=0)
    pred_class = torch.argmax(mean_probs, dim=1).cpu().numpy()
    return total_ent, epistemic, pred_class


def dirichlet_uncertainty_and_pred(alphas):
    """
    Digamma decomposition for any Dirichlet-output student (EnDD-style).
    alphas: [B, K, 224, 224] concentration parameters.
    Returns (total_entropy, epistemic_entropy, predicted_class_np), all [B, 224, 224].
    """
    alpha0 = alphas.sum(dim=1, keepdim=True)
    mean_probs = alphas / alpha0
    alpha0_sq = alpha0.squeeze(1)
    total_ent = -(mean_probs * torch.log(mean_probs.clamp(min=1e-10))).sum(dim=1)
    aleatoric = (torch.digamma(alpha0_sq + 1) - (mean_probs * torch.digamma(alphas + 1)).sum(dim=1))
    epistemic = (total_ent - aleatoric).clamp(min=0)
    pred_class = torch.argmax(mean_probs, dim=1).cpu().numpy()
    return total_ent, epistemic, pred_class


def evaluate_error_localization(compute_fn, test_loader, args, run_name="error_localization", who="model"):
    """
    Does this model's own uncertainty spatially locate its own errors? Agnostic to what
    produced the uncertainty — compute_fn(batch_input) -> (total_entropy, epistemic_entropy,
    predicted_class_np) for one batch. Correlates each against that batch's own correctness
    map, per patch, then averages — the spatial analogue of a model's own global AUROC.
    (Different question from evaluate_uncertainty_correlation, which compares a student's
    uncertainty against the teacher's rather than against ground-truth correctness.)

    Two ready-made compute_fn ingredients exist — wrap whichever applies in a small closure
    at the call site: ensemble_uncertainty_and_pred(all_preds, num_classes) for any M-head
    ensemble (teacher-style), or dirichlet_uncertainty_and_pred(alphas) for any Dirichlet
    student. who is just a label ("teacher"/"student"/etc.) used in logging and the output
    filename.
    """
    total_stats = {'corr_values': [], 'skipped': 0, 'patch_count': 0}
    epi_stats = {'corr_values': [], 'skipped': 0, 'patch_count': 0}

    # A handful of patches for an illustrative uncertainty-vs-error figure — separate
    # from the quantitative distribution above, which is what actually backs the claim.
    num_vis = min(3, len(test_loader.dataset))
    vis_global_indices = set(random.sample(range(len(test_loader.dataset)), num_vis)) if num_vis > 0 else set()
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (batch_input, test_masks) in enumerate(test_loader):
            batch_input = batch_input.to(DEVICE)
            total_ent, epistemic, pred_class = compute_fn(batch_input)
            _spearman_error_localization_batch(total_ent, pred_class, test_masks, total_stats)
            _spearman_error_localization_batch(epistemic, pred_class, test_masks, epi_stats)

            if vis_count < num_vis:
                total_ent_np = total_ent.cpu().numpy()
                for b_offset in range(batch_input.shape[0]):
                    g_idx = batch_idx * test_loader.batch_size + b_offset
                    if g_idx not in vis_global_indices:
                        continue
                    gt = test_masks[b_offset].squeeze().cpu().numpy()
                    valid_mask = gt > 0
                    if valid_mask.sum() < 100:
                        continue
                    correct_mask = (pred_class[b_offset] == gt)
                    plot_error_localization_heatmap(
                        total_ent_np[b_offset], correct_mask, valid_mask,
                        save_name=f"{run_name}_{who}_patch_{g_idx}_error_localization"
                    )
                    vis_count += 1
                    if vis_count >= num_vis:
                        break

    return _log_and_save_error_localization(who, total_stats, epi_stats, run_name)


# Unsure if this is a good idea or not
# Fn to make sure we initialise weights with very small values to ensure some class doesn't accidentally start off as
# being picked and then the model becomes confident on this class only because of a random fluctuation
def init_weights(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialise with mean 0 and std very low
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_random_batch(loader, device):
    """
        Grabs a random batch from the provided DataLoader.
        """
    # 1. Pick a random index based on the number of batches
    random_idx = random.randint(0, len(loader) - 1)

    # 2. Iterate through until we hit that index
    for i, (images, targets) in enumerate(loader):
        if i == random_idx:
            # Send to device (HPC/GPU ready)
            test_images = images.to(device)
            test_targets = targets.to(device)
            return test_images, test_targets

    # Fallback just in case (shouldn't happen)
    return next(iter(loader))

# fn to save checkpoints of the model so as not to lose too much progress
def save_checkpoint(state, out_dir, filename="last_checkpoint.pth"):

    # Save latest version
    last_path = os.path.join(out_dir, filename)
    torch.save(state, last_path)
    log_msg(f"=> Saving checkpoint to {last_path}")

def evaluate_test_set(trained_model, test_loader, criterion, args, run_name="test", class_names=FBP_CLASSES):
    trained_model.eval()

    num_classes = args.num_classes
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Running ECE bins — avoids storing all confidence values in memory
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_conf_sums = np.zeros(num_bins)
    bin_acc_sums  = np.zeros(num_bins, dtype=np.int64)
    bin_counts    = np.zeros(num_bins, dtype=np.int64)

    patch_count = 0
    nll_sum = 0.0
    nll_count = 0
    auroc_entropy = []
    auroc_errors = []
    AUROC_MAX = 2_000_000

    n_pairs = trained_model.M * (trained_model.M - 1) // 2
    head_conf_matrices = [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in range(trained_model.M)]
    total_ent_sum = 0.0
    aleatoric_sum = 0.0
    jsd_sum = 0.0
    uq_count = 0

    # Pick 3 random GLOBAL indices from entire test set for visualisation
    vis_global_indices = set(random.sample(
        range(len(test_loader.dataset)),
        min(3, len(test_loader.dataset))
    ))
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (test_features, test_masks) in enumerate(test_loader):
            test_features = test_features.to(DEVICE)

            # Batched forward pass
            all_preds = trained_model(test_features)

            M_sz, B_sz = trained_model.M, test_features.shape[0]
            all_preds_up = F.interpolate(
                all_preds.view(M_sz * B_sz, num_classes, *all_preds.shape[-2:]),
                size=(224, 224), mode='bilinear', align_corners=False
            ).view(M_sz, B_sz, num_classes, 224, 224)
            all_head_probs_f32 = torch.softmax(all_preds_up.float(), dim=2)
            mean_probs = all_head_probs_f32.mean(dim=0)  # true probability mixture: mean(softmax), not softmax(mean)
            head_preds_np = torch.argmax(all_head_probs_f32, dim=2).cpu().numpy()

            class_maps = torch.argmax(mean_probs, dim=1).cpu().numpy()
            conf_maps = torch.max(mean_probs, dim=1)[0].cpu().numpy()

            test_masks_t = test_masks.to(DEVICE).long()
            labelled_t = test_masks_t > 0
            if labelled_t.any():
                log_probs = torch.log(mean_probs.clamp(min=1e-10))
                gt_idx = test_masks_t.unsqueeze(1).clamp(0, num_classes - 1)
                nll_sum += (-log_probs.gather(1, gt_idx).squeeze(1)[labelled_t]).sum().item()
                nll_count += labelled_t.sum().item()
                ent_t = -(mean_probs * torch.log(mean_probs.clamp(min=1e-10))).sum(dim=1)
                auroc_collected = sum(len(x) for x in auroc_entropy) if auroc_entropy else 0
                if auroc_collected < AUROC_MAX:
                    err_t = (torch.argmax(mean_probs, dim=1) != test_masks_t).float()
                    auroc_entropy.append(ent_t[labelled_t].cpu().numpy())
                    auroc_errors.append(err_t[labelled_t].cpu().numpy())
                total_ent_sum += ent_t[labelled_t].sum().item()
                aleat = torch.zeros_like(ent_t)
                for m in range(trained_model.M):
                    hp = all_head_probs_f32[m]
                    aleat += -(hp * torch.log(hp.clamp(1e-10))).sum(dim=1)
                aleat /= trained_model.M
                aleatoric_sum += aleat[labelled_t].sum().item()
                uq_count += labelled_t.sum().item()
                if n_pairs > 0:
                    for i in range(trained_model.M):
                        for j in range(i + 1, trained_model.M):
                            p, q = all_head_probs_f32[i], all_head_probs_f32[j]
                            m_pq = 0.5 * (p + q)
                            jsd = (-(m_pq * torch.log(m_pq.clamp(1e-10))).sum(dim=1)
                                   + 0.5 * (p * torch.log(p.clamp(1e-10))).sum(dim=1)
                                   + 0.5 * (q * torch.log(q.clamp(1e-10))).sum(dim=1))
                            jsd_sum += jsd[labelled_t].sum().item()

            # Check each item in batch for visualisation
            for b_offset in range(test_features.shape[0]):
                g_idx = batch_idx * test_loader.batch_size + b_offset

                if g_idx in vis_global_indices and vis_count < 3:
                    img_pt_path, local_idx = test_loader.dataset.get_patch_info(g_idx)

                    img_stem = img_pt_path.stem.replace('_embeddings', '')
                    data_dir = Path(img_pt_path).parent.parent.parent.parent.parent / "data" / "fbp"
                    raw_img_path = data_dir / f"{img_stem}.tif"

                    # Reconstruct x, y coordinates
                    patches_per_row = len(range(0, 7300 - 224, args.stride))
                    x = (local_idx % patches_per_row) * args.stride
                    y = (local_idx // patches_per_row) * args.stride

                    log_msg(f"Vis patch: {img_stem} local_idx={local_idx} x={x} y={y}")

                    mask_path = data_dir / f"{img_stem}_24label.png"
                    if mask_path.exists():
                        with rasterio.open(mask_path) as src:
                            mask_check = src.read(1, window=Window(x, y, 224, 224))
                            log_msg(f"Mask from disk unique values: {np.unique(mask_check)}")
                            log_msg(f"Mask from dataset unique values: {np.unique(test_masks[b_offset].squeeze().cpu().numpy())}")

                    # Load raw image patch
                    raw_patch = None
                    if raw_img_path.exists():
                        with rasterio.open(raw_img_path) as src:
                            win = Window(x, y, 224, 224)
                            img = src.read([1, 2, 3], window=win).astype(np.float32)
                            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                            raw_patch = np.transpose(img, (1, 2, 0))

                    # Get single feature for visualisation
                    single_feat = test_features[b_offset].unsqueeze(0)
                    mean_probs_vis, class_map_vis, var_map, ent_map, mi_map = get_decoder_output_maps(
                        trained_model, single_feat, save_name=f"{run_name}_patch_{g_idx}"
                    )
                    gt_vis = test_masks[b_offset].squeeze().cpu().numpy()

                    visualise_all_metrics(
                        class_map=class_map_vis,
                        variance_map=var_map,
                        total_entropy=ent_map,
                        mi_map=mi_map,
                        ground_truth=gt_vis,
                        hide_unlabelled=args.hide_unlabelled_pixels,
                        save_name=f"{run_name}_test_patch_{g_idx}",
                        raw_patch=raw_patch,
                        patch_info=f"{img_stem} x={x} y={y}"
                    )
                    del mean_probs_vis, class_map_vis, var_map, ent_map, mi_map, raw_patch
                    plt.close('all')
                    torch.cuda.empty_cache()
                    gc.collect()
                    vis_count += 1

            # Accumulate metrics via confusion matrix — O(num_classes²) total memory
            for b in range(test_features.shape[0]):
                gt = test_masks[b].squeeze().cpu().numpy()
                if (gt > 0).sum() < 100:
                    continue
                mask = gt > 0
                pred_flat = class_maps[b][mask]
                true_flat = gt[mask].astype(np.int64)
                conf_flat = conf_maps[b][mask]

                np.add.at(conf_matrix, (true_flat, pred_flat), 1)
                for m in range(trained_model.M):
                    np.add.at(head_conf_matrices[m], (true_flat, head_preds_np[m, b][mask]), 1)

                for bin_idx in range(num_bins):
                    in_bin = (conf_flat >= bin_boundaries[bin_idx]) & (conf_flat < bin_boundaries[bin_idx + 1])
                    n = in_bin.sum()
                    if n > 0:
                        bin_conf_sums[bin_idx] += conf_flat[in_bin].sum()
                        bin_acc_sums[bin_idx]  += (true_flat[in_bin] == pred_flat[in_bin]).sum()
                        bin_counts[bin_idx]    += n

                patch_count += 1

    # mIoU from confusion matrix — only average over classes present in ground truth
    iou_per_class = np.zeros(num_classes - 1)
    present = []
    for c in range(1, num_classes):
        if conf_matrix[c, :].sum() > 0:
            tp    = conf_matrix[c, c]
            fp    = conf_matrix[:, c].sum() - tp
            fn    = conf_matrix[c, :].sum() - tp
            denom = tp + fp + fn
            iou_per_class[c - 1] = tp / denom if denom > 0 else 0.0
            present.append(c - 1)
    global_miou = float(np.mean(iou_per_class[present])) if present else 0.0

    class_pixel_counts = conf_matrix[1:, :].sum(axis=1)
    total_labelled = class_pixel_counts.sum()
    fw_iou = float((class_pixel_counts / max(total_labelled, 1) * iou_per_class).sum())

    global_nll = nll_sum / max(nll_count, 1)
    if auroc_entropy:
        all_ent = np.concatenate(auroc_entropy)
        all_err = np.concatenate(auroc_errors)
        global_auroc = float(roc_auc_score(all_err, all_ent)) if len(np.unique(all_err)) > 1 else 0.0
    else:
        global_auroc = 0.0

    # Overall accuracy (labelled pixels only)
    global_acc = float(np.diag(conf_matrix)[1:].sum() / max(conf_matrix[1:, :].sum(), 1))

    mean_total_ent = total_ent_sum / max(uq_count, 1)
    mean_aleatoric = aleatoric_sum / max(uq_count, 1)
    mean_epistemic = max(mean_total_ent - mean_aleatoric, 0.0)
    mean_pairwise_jsd = jsd_sum / max(n_pairs * uq_count, 1)

    per_head_miou = []
    for m in range(trained_model.M):
        hcm = head_conf_matrices[m]
        iou_h = []
        for c in range(1, num_classes):
            if hcm[c, :].sum() > 0:
                tp = hcm[c, c]
                fp = hcm[:, c].sum() - tp
                fn = hcm[c, :].sum() - tp
                denom = tp + fp + fn
                iou_h.append(tp / denom if denom > 0 else 0.0)
        per_head_miou.append(float(np.mean(iou_h)) if iou_h else 0.0)

    # ECE and reliability diagram from running bins
    bin_accs  = np.zeros(num_bins)
    bin_props = np.zeros(num_bins)
    ece = 0.0
    total_px = bin_counts.sum()
    if total_px > 0:
        for i in range(num_bins):
            if bin_counts[i] > 0:
                avg_conf = bin_conf_sums[i] / bin_counts[i]
                avg_acc  = bin_acc_sums[i]  / bin_counts[i]
                prop     = bin_counts[i] / total_px
                bin_accs[i]  = avg_acc
                bin_props[i] = prop
                ece += abs(avg_acc - avg_conf) * prop
    plot_reliability_diagram(bin_accs, bin_props, save_name=f"{run_name}_reliability")
    global_ece = float(ece)

    log_msg(f"FINAL TEST RESULTS ({patch_count} patches):")
    log_msg(f"Global mIoU: {global_miou:.4f} | fw-IoU: {fw_iou:.4f} | Acc: {global_acc:.4f} | ECE: {global_ece:.4f} | NLL: {global_nll:.4f} | AUROC: {global_auroc:.4f}")
    log_msg(f"UQ: Total H={mean_total_ent:.4f} | Aleatoric={mean_aleatoric:.4f} | Epistemic={mean_epistemic:.4f} | Pairwise JSD={mean_pairwise_jsd:.4f}")
    log_msg(f"Per-head mIoU: {[f'{x:.4f}' for x in per_head_miou]}")
    log_msg("Per-class IoU (descending frequency):")
    freq_order = np.argsort(class_pixel_counts)[::-1]
    for class_idx in freq_order:
        log_msg(f"  {class_names[class_idx + 1]}: {iou_per_class[class_idx]:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": run_name,
        "diversity_methods": args.diversity_methods,
        "lam_jsd": args.lam_jsd,
        "lam_pearson": args.lam_pearson,
        "lam_orth": args.lam_orth,
        "global_miou": global_miou,
        "global_fwiou": fw_iou,
        "global_accuracy": global_acc,
        "global_ece": global_ece,
        "global_nll": global_nll,
        "global_auroc": global_auroc,
        "mean_total_entropy": mean_total_ent,
        "mean_aleatoric": mean_aleatoric,
        "mean_epistemic": mean_epistemic,
        "mean_pairwise_jsd": mean_pairwise_jsd,
        "per_head_miou": per_head_miou,
        "num_patches": patch_count,
        "per_class_iou": {class_names[i + 1]: float(iou) for i, iou in enumerate(iou_per_class)},
        "confusion_matrix": conf_matrix.tolist(),
    }
    results_path = os.path.join(runs_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Results saved to {results_path}")
    plot_confusion_matrix(results_path, save_name=f"{run_name}_confusion")

    return results