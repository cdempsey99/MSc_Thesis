import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from sklearn.metrics import jaccard_score
from utils.visualisation import *

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
def js_divergence_loss(all_preds):
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

    print(f"mIoU : {miou} | Overall Accuracy : {overall_accuracy} | Average Uncertainty : {avg_unc} | ECE : {ece} ")

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
            print(f"Head {i} unique predictions: {np.unique(head_map)}")

            # Plotting
            im = axes[i].imshow(head_map, cmap='tab20', vmin=0, vmax=NUM_CLASSES)
            axes[i].set_title(f"Head {i + 1}")
            axes[i].axis('off')

        # Save figure
        save_path = os.path.join(BASE_OUT, "heads", save_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        current_time = datetime.now().strftime("%Y%m%d%H%M")
        full_file_path = os.path.join(save_path, f"ensemble_heads_{current_time}.png")
        plt.savefig(full_file_path, bbox_inches='tight', dpi=150)
        # plt.show()

        # Take softmax first, then calculate class predictions for every pixel
        all_probs = torch.softmax(eval_preds, dim=2) # softmaxxing over the class dimension: [5, 1, 25, 224, 224]
        mean_probs = all_probs.mean(dim=0)  # [1, 25, 224, 224]
        class_map = torch.argmax(mean_probs, dim=1).squeeze().cpu().numpy() # [224, 224]

        # Always calculate these regardless of M
        all_probs = torch.softmax(eval_preds, dim=2)
        mean_probs = all_probs.mean(dim=0)
        class_map = torch.argmax(mean_probs, dim=1).squeeze().cpu().numpy()
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
    print(f"=> Saving checkpoint to {last_path}", flush=True)
