import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from configs.config import *
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse
import random
import numpy as np

from utils.dataset import *
from utils.training import full_decoder_training_run, log_msg
from utils.misc import *
from utils.visualisation import *

# Set Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(args):
    # 1. Locate Embeddings & Validate Metadata
    version_str = f"patch{args.patch_size}_stride{args.stride}"
    embedding_dir = Path(args.data_dir) / "embeddings" / "clay_v1" / version_str

    meta_path = embedding_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata at {meta_path}. Did you run extract_embeddings.py?")

    with open(meta_path, "r") as f:
        meta = json.load(f)
        if meta["patch_size"] != args.patch_size:
            raise ValueError(f"Metadata mismatch! Data was extracted with patch {meta['patch_size']}")

    run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M')}"
    log_msg(f"Run name: {run_name}")

    # 2. Split Discovery
    # We find all .pt files and split them into Train/Val/Test
    all_files = sorted(list(embedding_dir.glob("*_embeddings.pt")))
    random.seed(42)  # Keep splits consistent across different decoder experiments
    random.shuffle(all_files)

    n = len(all_files)
    if n < 5:
        # QA Mode: Manually ensure no set is empty
        log_msg(f"Small dataset detected ({n} files). Using manual QA splits.")
        train_files = all_files[0:1]
        val_files = all_files[1:2]
        test_files = all_files[2:]
    else:
        # Production Mode: Use your standard percentages
        train_files = all_files[:int(n * 0.7)]
        val_files = all_files[int(n * 0.7):int(n * 0.85)]
        test_files = all_files[int(n * 0.85):]

    log_msg(f"Split: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

    # 3. Data Loaders (The "Safe" Config)
    # We pass the list of files to the Dataset
    log_msg("Initialising train dataset...")
    train_ds = BakedFeatureDataset(train_files)
    log_msg("Initialising valid dataset...")
    val_ds = BakedFeatureDataset(val_files)
    log_msg("Initialising test dataset...")
    test_ds = BakedFeatureDataset(test_files)
    log_msg("All datasets initialised")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Training
    # Pack parameters into your input_dict for the training function
    input_dict = vars(args)
    input_dict["lambda_div"] = args.lam  # Mapping arg name to dict key
    input_dict["in_channels"] = args.decoder_in_channels
    input_dict["embed_dim"] = args.decoder_embed_dim
    input_dict["device"] = DEVICE
    input_dict["learning_rate"] = args.lr
    input_dict["run_name"] = run_name

    start_time = time.time()
    trained_model = full_decoder_training_run(input_dict, train_loader, val_loader)

    log_msg(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    """
    # 5. Final Evaluation
    log_msg("\n" + "=" * 30 + "\nSTARTING FINAL TEST EVALUATION\n" + "=" * 30)

    test_features, test_masks = get_random_batch(test_loader, DEVICE)
    mean_probs, class_map, var_map, ent_map, mi_map = get_decoder_output_maps(trained_model,
                                                                              test_features[0].unsqueeze(0))

    # Quantitative Metrics
    conf_map = torch.max(mean_probs, dim=1)[0].squeeze().cpu().numpy()
    miou, acc, avg_unc, ece = evaluate_metrics(class_map, test_masks[0].squeeze().cpu().numpy(), conf_map)

    log_msg(f"FINAL TEST RESULTS: mIoU: {miou:.4f} | Acc: {acc:.4f} | ECE: {ece:.4f}")

    # 5. Final Evaluation Metrics
    log_msg("\n" + "=" * 30 + "\nSAVING VISUALISATIONS\n" + "=" * 30)

    # Flatten tensors for reliability binning
    conf_flat = conf_map.flatten()
    gt_flat = test_masks[0].squeeze().cpu().numpy().flatten()
    pred_flat = class_map.flatten()

    # Create 10 bins for Reliability Diagram
    bin_boundaries = np.linspace(0, 1, 11)
    bin_accs = []
    bin_counts = []

    for i in range(10):
        mask = (conf_flat > bin_boundaries[i]) & (conf_flat <= bin_boundaries[i + 1])
        if mask.any():
            acc = (pred_flat[mask] == gt_flat[mask]).mean()
            bin_accs.append(acc)
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_counts.append(0)

    # 1. Save 5-pane figure to results/metrics/
    visualise_all_metrics(
        class_map=class_map,
        variance_map=var_map,
        total_entropy=ent_map,
        mi_map=mi_map,
        ground_truth=test_masks[0].squeeze().cpu().numpy(),
        hide_unlabelled=args.hide_unlabelled_pixels,
        save_name="spatial_analysis"
    )

    # 2. Save Reliability Diagram to results/reliability/
    plot_reliability_diagram(
        bin_accs_all=bin_accs,
        bin_counts=bin_counts,
        save_name="reliability_calib"
    )

    log_msg(f"Visualisations complete. Check {args.out_dir}/metrics and {args.out_dir}/reliability")
    """

    # Plot loss curves
    #loss_history_path = os.path.join(args.out_dir, "loss_history.json")
    #plot_loss_curves(loss_history_path, save_name="AS1_loss_curves")

    loss_history_path = os.path.join(os.getenv("OUT_DIR", "results"), "runs",
                                     f"{run_name}_loss_history.json")
    plot_loss_curves(loss_history_path, save_name=f"{run_name}_loss_curves")

    # --- 5. Final Test Evaluation ---
    #results = evaluate_test_set(
    #    trained_model, test_loader,
    #    nn.CrossEntropyLoss(ignore_index=0),
    #    args,
    #    run_name="AS1_baseline"
    #)
    results = evaluate_test_set(
        trained_model, test_loader,
        nn.CrossEntropyLoss(ignore_index=0),
        args,
        run_name=run_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder Training Script")
    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the embeddings folder")
    parser.add_argument("--out_dir", type=str, default="./results")
    # Data Params (must match extraction)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    # Model Params
    parser.add_argument("--decoder_in_channels", type=int, default=1024)
    parser.add_argument("--decoder_embed_dim", type=int, default=256)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=25)
    # Training Params
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lam", type=float, default=0.1, help="Lambda diversity")
    parser.add_argument("--enforce_diversity", action="store_true")
    parser.add_argument("--hide_unlabelled_pixels", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default="run")

    args = parser.parse_args()
    run_training(args)