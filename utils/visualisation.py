import os
import numpy as np
import matplotlib.pyplot as plt
from configs.config import *

# Helper to ensure directory exists
def ensure_dir(path="results"):
    if not os.path.exists(path):
        os.makedirs(path)

# Fn to take the [1, 1024, 28, 28] tensor and produce a heatmap of 'attention'
def visualise_encoder_output(grid_features, save_name="encoder_heatmap.png"):
    ensure_dir("results/heatmaps")

    heatmap = grid_features.detach().cpu().mean(dim=1).squeeze()

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='viridis')  # 'viridis' is great for heatmaps
    plt.colorbar(label='Mean Activation Intensity')
    plt.title(f"Clay Encoder Feature Map (28x28)\nModel Size: Large | Channels: 1024")
    plt.xlabel("Spatial Width (Patches)")
    plt.ylabel("Spatial Height (Patches)")

    # Save logic
    plt.savefig(os.path.join("results/heatmaps", save_name), bbox_inches='tight')
    #plt.show()

# The 5 pane figure showing the ground truth, class predictions, variance, entropy and mutual information
def visualise_all_metrics(class_map, variance_map, total_entropy, mi_map, ground_truth, hide_unlabelled, save_name="all_metrics.png"):
    """
    Displays a comprehensive 5-pane figure showing
    ground truth, class prediction, average variance, entropy and mutual information
    """
    ensure_dir("results/metrics")

    # Ensure everything is 2D by removing any leading '1' dimensions
    class_map = np.squeeze(class_map)
    variance_map = np.squeeze(variance_map)
    total_entropy = np.squeeze(total_entropy)
    mi_map = np.squeeze(mi_map)
    ground_truth = np.squeeze(ground_truth)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 2. Toggle: Grey out unlabelled pixels (where ground_truth == 0)
    hide_unlabelled = True
    if hide_unlabelled:
        # Create a mask of pixels to ignore
        ignore_mask = (ground_truth == 0)

        # We can use a masked array for the prediction map
        class_map_masked = np.ma.masked_where(ignore_mask, class_map)
    else:
        class_map_masked = class_map

    # --- 1. Ground Truth ---
    axes[0].imshow(ground_truth, cmap='tab20', vmin=0, vmax=24)
    axes[0].set_title("Ground Truth")

    # --- 2. Predicted Classes ---
    axes[1].imshow(class_map_masked, cmap='tab20', vmin=0, vmax=24)
    axes[1].set_title("Predicted Classes")

    # --- 3. Variance Map (Raw Disagreement) ---
    im2 = axes[2].imshow(variance_map, cmap='magma')
    axes[2].set_title("Ensemble Variance")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # --- 4. Total Entropy (Total Uncertainty) ---
    # Max value is ln(25) ~ 3.2
    im3 = axes[3].imshow(total_entropy, cmap='magma', vmin=0, vmax=3.22)
    axes[3].set_title("Total Entropy (H)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # --- 5. Mutual Information (Epistemic Disagreement) ---
    im4 = axes[4].imshow(mi_map, cmap='magma')
    axes[4].set_title("Mutual Information (MI)")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join("results/metrics", save_name), bbox_inches='tight', dpi=300)
    plt.show()


# Fn to print dictionary of the proportions of each smenatic category in a given ground_truth map
def display_truth_proportions(ground_truth):

    unique, counts = np.unique(ground_truth.cpu().numpy(), return_counts=True)

    total_pixels = sum(counts)
    proportions = {}
    for u, c in zip(unique, counts):
        proportions[FBP_CLASSES_DICT[int(u)]] = f"{100 * c / total_pixels:.1f}%"

    print("Ground Truth Pixel Fractions for this tile:")
    print(proportions)


#Aux fn for the next aux fn
def plot_reliability_diagram(bin_accs_all, bin_counts, save_name="reliability_diagram"):

    ensure_dir("results/reliability")

    # bin_accs_all should be an array of 10 values
    bin_centers = np.linspace(0.05, 0.95, 10)

    plt.figure(figsize=(6, 6))

    # Plot the "Ideal" line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    # Plot the actual bars
    plt.bar(bin_centers, bin_accs_all, width=0.1, alpha=0.8,
            edgecolor='black', color='blue', label='Model Accuracy')

    # Optional: Add a gap visualization
    # The gap is |Identity - Accuracy|

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Confidence (Mean Max Probability)')
    plt.ylabel('Observed Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results/reliability", save_name), bbox_inches='tight')
    plt.show()


# Fn used to plot various accuracy metrics for different values of lambda (diversity hyperparameter)
def plot_lambda_results(lams, mious, overall_accs, avg_uncs, save_name="lambda_curves.png"):

    ensure_dir("results/lambda_curves")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy Metrics
    ax1.plot(lams, mious, marker='o', label='mIoU', color='blue')
    ax1.plot(lams, overall_accs, marker='s', label='Overall Acc', color='green')
    ax1.set_xlabel('Lambda (Diversity Penalty)')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Accuracy vs. Lambda')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Uncertainty Metric
    ax2.plot(lams, avg_uncs, marker='^', label='Avg Uncertainty', color='red')
    ax2.set_xlabel('Lambda (Diversity Penalty)')
    ax2.set_ylabel('Mean Variance')
    ax2.set_title('Uncertainty vs. Lambda')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join("results/lambda_curves", save_name), bbox_inches='tight')
    plt.show()

