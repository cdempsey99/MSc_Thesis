import torch.nn as nn
import random
import time
import inspect
import psutil
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
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
def get_ece(y_pred, y_true, unc_map_flat):

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

    plot_reliability_diagram(bin_accs, bin_props)

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
    corr = torch.einsum('mcd,ncd->cmn', normed, normed) / (B * H * W)  # [C, M, M]

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

    # Get classifier weights for each head
    weights = [head.classifier.weight for head in decoder_ensemble.heads]

    total_orth = 0.0
    count = 0

    for i in range(M):
        for j in range(i + 1, M):

            # Flatten weight matrices to vectors
            w_i = weights[i].flatten()  # [num_classes * embed_dim]
            w_j = weights[j].flatten()  # [num_classes * embed_dim]

            # Squared cosine similarity
            cos_sim = F.cosine_similarity(w_i.unsqueeze(0), w_j.unsqueeze(0))
            log_msg(f"pair ({i}, {j}) cos_sim={cos_sim.item():.6f} cos_sim^2={cos_sim.item()**2:.8f}")
            total_orth += cos_sim ** 2
            count += 1

    result = total_orth/count
    log_msg(f"orthogonality_loss result={result.item():.8f}")
    return result


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


def evaluate_student_test_set(student_model, test_loader, args, run_name="student_test"):
    """
    Evaluate a trained StudentHead on the test split.
    The student outputs Dirichlet alphas [B, K, H, W]; class predictions come from
    the Dirichlet mean α / Σα_k, and confidence from its max value.
    """
    student_model.eval()

    all_preds_global = []
    all_gts_global = []
    all_conf_global = []
    patch_count = 0

    vis_global_indices = set(random.sample(
        range(len(test_loader.dataset)),
        min(3, len(test_loader.dataset))
    ))
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (test_features, test_masks) in enumerate(test_loader):
            test_features = test_features.to(DEVICE)

            alphas = student_model(test_features)                        # [B, K, H, W]
            mean_probs = alphas / alphas.sum(dim=1, keepdim=True)       # Dirichlet mean
            class_maps = torch.argmax(mean_probs, dim=1).cpu().numpy()  # [B, H, W]
            conf_maps = torch.max(mean_probs, dim=1)[0].cpu().numpy()   # [B, H, W]

            if vis_count < 3:
                entropy_maps = (-mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)  # [B, H, W]

            for b in range(test_features.shape[0]):
                g_idx = batch_idx * test_loader.batch_size + b
                gt = test_masks[b].squeeze().cpu().numpy()

                if g_idx in vis_global_indices and vis_count < 3:
                    ent_map = entropy_maps[b].cpu().numpy()
                    zero_map = np.zeros_like(ent_map)
                    visualise_all_metrics(
                        class_map=class_maps[b],
                        variance_map=zero_map,
                        total_entropy=torch.tensor(ent_map),
                        mi_map=torch.zeros_like(torch.tensor(ent_map)),
                        ground_truth=gt,
                        hide_unlabelled=args.hide_unlabelled_pixels,
                        save_name=f"{run_name}_patch_{g_idx}",
                    )
                    plt.close('all')
                    vis_count += 1

                if (gt > 0).sum() < 100:
                    continue
                mask = gt > 0
                all_preds_global.append(class_maps[b][mask])
                all_gts_global.append(gt[mask])
                all_conf_global.append(conf_maps[b][mask])
                patch_count += 1

    all_preds_arr = np.concatenate(all_preds_global)
    all_gts_arr = np.concatenate(all_gts_global)
    all_conf_arr = np.concatenate(all_conf_global)

    global_miou = jaccard_score(all_gts_arr, all_preds_arr, average="macro", labels=np.unique(all_gts_arr))
    global_acc = np.mean(all_preds_arr == all_gts_arr)
    global_ece = get_ece(all_preds_arr, all_gts_arr, all_conf_arr)
    per_class_iou = jaccard_score(all_gts_arr, all_preds_arr, average=None, labels=list(range(1, 25)))

    log_msg(f"STUDENT TEST RESULTS ({patch_count} patches):")
    log_msg(f"Global mIoU: {global_miou:.4f} | Acc: {global_acc:.4f} | ECE: {global_ece:.4f}")
    log_msg("Per-class IoU:")
    for class_idx, iou in enumerate(per_class_iou):
        log_msg(f"  {FBP_CLASSES[class_idx + 1]}: {iou:.4f}")

    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    results = {
        "run_name": run_name,
        "global_miou": float(global_miou),
        "global_accuracy": float(global_acc),
        "global_ece": float(global_ece),
        "num_patches": patch_count,
        "per_class_iou": {FBP_CLASSES[i + 1]: float(iou) for i, iou in enumerate(per_class_iou)}
    }
    results_path = os.path.join(runs_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Student results saved to {results_path}")

    return results


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

def evaluate_test_set(trained_model, test_loader, criterion, args, run_name="test"):
    trained_model.eval()

    all_preds_global = []
    all_gts_global = []
    all_conf_global = []
    patch_count = 0

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
            mean_logits = all_preds.mean(dim=0)
            mean_logits_high = F.interpolate(mean_logits, size=(224, 224), mode='bilinear', align_corners=False)
            mean_probs = torch.softmax(mean_logits_high, dim=1)
            class_maps = torch.argmax(mean_probs, dim=1).cpu().numpy()
            conf_maps = torch.max(mean_probs, dim=1)[0].cpu().numpy()

            # Check each item in batch for visualisation
            for b_offset in range(test_features.shape[0]):
                g_idx = batch_idx * test_loader.batch_size + b_offset

                if g_idx in vis_global_indices and vis_count < 3:
                    img_pt_path, local_idx = test_loader.dataset.get_patch_info(g_idx)

                    img_stem = img_pt_path.stem.replace('_embeddings', '')
                    data_dir = Path(img_pt_path).parent.parent.parent.parent.parent / "data"
                    raw_img_path = data_dir / f"{img_stem}.tif"

                    # Reconstruct x, y coordinates
                    patches_per_row = len(range(0, 7300 - 224, args.stride))
                    x = (local_idx % patches_per_row) * args.stride
                    y = (local_idx // patches_per_row) * args.stride

                    log_msg(f"Vis patch: {img_stem} local_idx={local_idx} x={x} y={y}")

                    # ADD DEBUG CHECK

                    mask_path = data_dir / f"{img_stem}_24label.png"
                    if mask_path.exists():
                        with rasterio.open(mask_path) as src:
                            mask_check = src.read(1, window=Window(x, y, 224, 224))
                            log_msg(f"Mask from disk unique values: {np.unique(mask_check)}")
                            log_msg(
                                f"Mask from dataset unique values: {np.unique(test_masks[b_offset].squeeze().cpu().numpy())}")

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

            # Accumulate metrics for each item in batch
            for b in range(test_features.shape[0]):
                gt = test_masks[b].squeeze().cpu().numpy()
                if (gt > 0).sum() < 100:
                    continue
                mask = gt > 0
                all_preds_global.append(class_maps[b][mask])
                all_gts_global.append(gt[mask])
                all_conf_global.append(conf_maps[b][mask])
                patch_count += 1

        # Convert to arrays
        all_preds_arr = np.concatenate(all_preds_global)
        all_gts_arr = np.concatenate(all_gts_global)
        all_conf_arr = np.concatenate(all_conf_global)

        # Global metrics
        global_miou = jaccard_score(all_gts_arr, all_preds_arr,
                                    average="macro", labels=np.unique(all_gts_arr))
        global_acc = np.mean(all_preds_arr == all_gts_arr)
        global_ece = get_ece(all_preds_arr, all_gts_arr, all_conf_arr)

        # Per class IoU
        per_class_iou = jaccard_score(all_gts_arr, all_preds_arr,
                                      average=None, labels=list(range(1, 25)))

        avg_test_loss = 0

        # Log results
        log_msg(f"FINAL TEST RESULTS ({patch_count} patches):")
        log_msg(
            f"Global mIoU: {global_miou:.4f} | Acc: {global_acc:.4f} | ECE: {global_ece:.4f} | Loss: {avg_test_loss:.4f}")
        log_msg("Per-class IoU:")
        for class_idx, iou in enumerate(per_class_iou):
            log_msg(f"  {FBP_CLASSES[class_idx + 1]}: {iou:.4f}")

        # Save to JSON
        runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
        os.makedirs(runs_dir, exist_ok=True)
        results = {
            "run_name": run_name,
            "diversity_methods": args.diversity_methods,
            "lam_jsd": args.lam_jsd,
            "lam_pearson": args.lam_pearson,
            "lam_orth": args.lam_orth,
            "global_miou": float(global_miou),
            "global_accuracy": float(global_acc),
            "global_ece": float(global_ece),
            "test_loss": float(avg_test_loss),
            "num_patches": patch_count,
            "per_class_iou": {FBP_CLASSES[i + 1]: float(iou)
                              for i, iou in enumerate(per_class_iou)}
        }
        results_path = os.path.join(runs_dir, f"{run_name}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log_msg(f"Results saved to {results_path}")

        return results