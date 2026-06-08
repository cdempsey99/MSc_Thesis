from utils.misc import *
from datetime import datetime#
import numpy as np
import matplotlib.pyplot as plt
from configs.config import *
#from utils.misc import *
import json

# Auxiliary lambda
to_np = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x

# Helper to ensure directory exists
def ensure_dir(path="results"):
    if not os.path.exists(path):
        os.makedirs(path)

# Fn to take the [1, 1024, 28, 28] tensor and produce a heatmap of 'attention'
def visualise_encoder_output(grid_features, save_name="encoder_heatmap.png"):
    
    #base_results_dir = os.getenv("OUT_DIR", "results")

    save_path = os.path.join(BASE_OUT, "heatmaps")
    ensure_dir(save_path)

    heatmap = grid_features.detach().cpu().mean(dim=1).squeeze()

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='viridis')  # 'viridis' is great for heatmaps
    plt.colorbar(label='Mean Activation Intensity')
    plt.title(f"Clay Encoder Feature Map (28x28)\nModel Size: Large | Channels: 1024")
    plt.xlabel("Spatial Width (Patches)")
    plt.ylabel("Spatial Height (Patches)")

    # Save logic
    full_file_path = os.path.join(save_path, save_name)
    plt.savefig(full_file_path, bbox_inches='tight')
    # plt.show()

# The 5 pane figure showing the ground truth, class predictions, variance, entropy and mutual information
def visualise_all_metrics_old(class_map, variance_map, total_entropy, mi_map, ground_truth, hide_unlabelled, save_name="all_metrics"):
    """
    Displays a comprehensive 5-pane figure showing
    ground truth, class prediction, average variance, entropy and mutual information
    """
    # Use the env var for the res dir, default to results for Windows setup
    #base_results_dir = os.getenv("OUT_DIR", "results")

    save_path = os.path.join(BASE_OUT, "metrics")
    ensure_dir(save_path)

    to_np = lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x

    # Ensure everything is 2D by removing any leading '1' dimensions
    # and ensure everything is on the cpu
    class_map = np.squeeze(to_np(class_map))
    variance_map = np.squeeze(to_np(variance_map))
    total_entropy = np.squeeze(to_np(total_entropy))
    mi_map = np.squeeze(to_np(mi_map))
    ground_truth = np.squeeze(to_np(ground_truth))

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 2. Toggle: Grey out unlabelled pixels (where ground_truth == 0)
    hide_unlabelled = True
    if hide_unlabelled:
        # Create a mask of pixels to ignore
        ignore_mask = (ground_truth == 0)

        # We can use a masked array for the prediction map
        class_map_np = to_np(class_map)
        ignore_mask_np = to_np(ignore_mask)
        class_map_masked = np.ma.masked_where(ignore_mask_np, class_map_np)
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
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    save_name += f"_{current_time}.png"
    full_file_path = os.path.join(save_path, save_name)
    plt.savefig(full_file_path, bbox_inches='tight', dpi=300)
    # plt.show()

def visualise_all_metrics(class_map, variance_map, total_entropy, mi_map,
                          ground_truth, hide_unlabelled, save_name="all_metrics",
                          raw_patch=None, patch_info=""):

    save_path = os.path.join(BASE_OUT, "metrics")
    ensure_dir(save_path)

    to_np = lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x

    class_map = np.squeeze(to_np(class_map))
    variance_map = np.squeeze(to_np(variance_map))
    total_entropy = np.squeeze(to_np(total_entropy))
    mi_map = np.squeeze(to_np(mi_map))
    ground_truth = np.squeeze(to_np(ground_truth))

    # Determine number of panels
    n_panels = 6 if raw_patch is not None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    hide_unlabelled = True
    if hide_unlabelled:
        ignore_mask = (ground_truth == 0)
        class_map_masked = np.ma.masked_where(to_np(ignore_mask), to_np(class_map))
    else:
        class_map_masked = class_map

    panel = 0

    # --- 0. Raw Image with mask overlay ---
    if raw_patch is not None:
        axes[panel].imshow(raw_patch)
        axes[panel].imshow(ground_truth, cmap='tab20', vmin=0, vmax=24, alpha=0.3)
        axes[panel].set_title(f"Raw Image\n{patch_info}", fontsize=8)
        axes[panel].axis('off')
        panel += 1

    # --- Ground Truth ---
    axes[panel].imshow(ground_truth, cmap='tab20', vmin=0, vmax=24)
    axes[panel].set_title("Ground Truth")
    axes[panel].axis('off')
    panel += 1

    # --- Predicted Classes ---
    axes[panel].imshow(class_map_masked, cmap='tab20', vmin=0, vmax=24)
    axes[panel].set_title("Predicted Classes")
    axes[panel].axis('off')
    panel += 1

    # --- Variance ---
    im2 = axes[panel].imshow(variance_map, cmap='magma')
    axes[panel].set_title("Ensemble Variance")
    fig.colorbar(im2, ax=axes[panel], fraction=0.046, pad=0.04)
    axes[panel].axis('off')
    panel += 1

    # --- Total Entropy ---
    im3 = axes[panel].imshow(total_entropy, cmap='magma', vmin=0, vmax=3.22)
    axes[panel].set_title("Total Entropy (H)")
    fig.colorbar(im3, ax=axes[panel], fraction=0.046, pad=0.04)
    axes[panel].axis('off')
    panel += 1

    # --- Mutual Information ---
    im4 = axes[panel].imshow(mi_map, cmap='magma')
    axes[panel].set_title("Mutual Information (MI)")
    fig.colorbar(im4, ax=axes[panel], fraction=0.046, pad=0.04)
    axes[panel].axis('off')

    plt.tight_layout()
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    save_name += f"_{current_time}.png"
    full_file_path = os.path.join(save_path, save_name)
    plt.savefig(full_file_path, bbox_inches='tight', dpi=300)
    plt.close('all')


# Fn to print dictionary of the proportions of each smenatic category in a given ground_truth map
def display_truth_proportions(ground_truth):

    unique, counts = np.unique(ground_truth.cpu().numpy(), return_counts=True)

    total_pixels = sum(counts)
    proportions = {}
    for u, c in zip(unique, counts):
        proportions[FBP_CLASSES_DICT[int(u)]] = f"{100 * c / total_pixels:.1f}%"

    print("Ground Truth Pixel Fractions for this tile:")
    print(proportions)


# Plot the accuracy versus confidence of the model 
def plot_reliability_diagram(bin_accs_all, bin_counts, save_name="reliability_diagram"):

    # Use env var for results dir, with default results for Windows setup
    #base_results_dir = os.getenv("OUT_DIR", "results")

    save_path = os.path.join(BASE_OUT, "reliability")
    ensure_dir(save_path)

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
    current_time = (datetime.now().strftime("%Y%m%d%H%M"))
    save_name += f"_{current_time}.png"
    full_file_path = os.path.join(save_path, save_name)
    plt.savefig(full_file_path, bbox_inches='tight')
    # plt.show()


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
    # plt.show()

def visualise_student_uncertainty(class_map, total_entropy, aleatoric, epistemic, alpha0_map,
                                   ground_truth, hide_unlabelled, save_name="student_unc"):
    """
    6-panel Dirichlet uncertainty figure for the EnDD student:
    GT | Prediction | Total H(p̄) | Aleatoric E[H[p]] | Epistemic MI | Concentration α₀
    """
    save_path = os.path.join(BASE_OUT, "metrics")
    ensure_dir(save_path)

    to_np = lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x

    class_map    = np.squeeze(to_np(class_map))
    total_entropy = np.squeeze(to_np(total_entropy))
    aleatoric    = np.squeeze(to_np(aleatoric))
    epistemic    = np.squeeze(to_np(epistemic))
    alpha0_map   = np.squeeze(to_np(alpha0_map))
    ground_truth = np.squeeze(to_np(ground_truth))

    if hide_unlabelled:
        class_map = np.ma.masked_where(ground_truth == 0, class_map)

    fig, axes = plt.subplots(1, 6, figsize=(36, 5))

    axes[0].imshow(ground_truth, cmap='tab20', vmin=0, vmax=24)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    axes[1].imshow(class_map, cmap='tab20', vmin=0, vmax=24)
    axes[1].set_title("Predicted Classes")
    axes[1].axis('off')

    im2 = axes[2].imshow(total_entropy, cmap='magma', vmin=0, vmax=3.22)
    axes[2].set_title("Total Entropy H(p̄)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    im3 = axes[3].imshow(aleatoric, cmap='magma')
    axes[3].set_title("Aleatoric E[H[p]]")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis('off')

    im4 = axes[4].imshow(epistemic, cmap='magma')
    axes[4].set_title("Epistemic MI")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    axes[4].axis('off')

    # alpha0 uses viridis (high = certain) to visually distinguish from the uncertainty maps
    im5 = axes[5].imshow(alpha0_map, cmap='viridis')
    axes[5].set_title("Concentration α₀")
    fig.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
    axes[5].axis('off')

    plt.tight_layout()
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    full_file_path = os.path.join(save_path, f"{save_name}_{current_time}.png")
    plt.savefig(full_file_path, bbox_inches='tight', dpi=300)
    plt.close('all')


def plot_loss_curves(loss_history_path, save_name="loss_curves"):
    save_path = os.path.join(BASE_OUT, "losses")
    ensure_dir(save_path)

    with open(loss_history_path) as f:
        history = json.load(f)

    train_losses = history["train"]
    val_losses = history["val"]
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    current_time = datetime.now().strftime("%Y%m%d%H%M")
    full_path = os.path.join(save_path, f"{save_name}_{current_time}.png")
    plt.savefig(full_path, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {full_path}")

