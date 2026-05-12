import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse
import random

from utils.dataset import BakedFeatureDataset  # Your debugged dataset class
from utils.training import full_decoder_training_run, log_msg
from utils.misc import get_random_batch, visualise_all_metrics, get_decoder_output_maps, evaluate_metrics

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

    # 2. Split Discovery
    # We find all .pt files and split them into Train/Val/Test
    all_files = sorted(list(embedding_dir.glob("*_embeddings.pt")))
    random.seed(42)  # Keep splits consistent across different decoder experiments
    random.shuffle(all_files)

    n = len(all_files)
    train_files = all_files[:int(n * 0.7)]
    val_files = all_files[int(n * 0.7):int(n * 0.85)]
    test_files = all_files[int(n * 0.85):]

    log_msg(f"Split: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

    # 3. Data Loaders (The "Safe" Config)
    # We pass the list of files to the Dataset
    train_ds = BakedFeatureDataset(train_files)
    val_ds = BakedFeatureDataset(val_files)
    test_ds = BakedFeatureDataset(test_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # 4. Model Training
    # Pack parameters into your input_dict for the training function
    input_dict = vars(args)
    input_dict["lambda_div"] = args.lam  # Mapping arg name to dict key

    start_time = time.time()
    trained_model = full_decoder_training_run(input_dict, train_loader, val_loader)

    log_msg(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    # 5. Final Evaluation
    log_msg("\n" + "=" * 30 + "\nSTARTING FINAL TEST EVALUATION\n" + "=" * 30)

    test_features, test_masks = get_random_batch(test_loader, DEVICE)
    mean_probs, class_map, var_map, ent_map, mi_map = get_decoder_output_maps(trained_model,
                                                                              test_features[0].unsqueeze(0))

    # Quantitative Metrics
    conf_map = torch.max(mean_probs, dim=1)[0].squeeze().cpu().numpy()
    miou, acc, avg_unc, ece = evaluate_metrics(class_map, test_masks[0].squeeze().cpu().numpy(), conf_map)

    log_msg(f"FINAL TEST RESULTS: mIoU: {miou:.4f} | Acc: {acc:.4f} | ECE: {ece:.4f}")


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

    args = parser.parse_args()
    run_training(args)