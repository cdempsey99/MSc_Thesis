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
    embedding_dir = Path(args.data_dir) / "embeddings" / "fbp" / "clay_v1" / version_str

    meta_path = embedding_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata at {meta_path}. Did you run extract_embeddings.py?")

    with open(meta_path, "r") as f:
        meta = json.load(f)
        if meta["patch_size"] != args.patch_size:
            raise ValueError(f"Metadata mismatch! Data was extracted with patch {meta['patch_size']}")

    if args.resume:
        run_name = args.run_name
    else:
        run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M')}"
    log_msg(f"Run name: {run_name}")

    # 2. Split Discovery
    # We find all .pt files and split them into Train/Val/Test
    all_files = sorted(list(embedding_dir.glob("*_embeddings.pt")))
    if args.max_images is not None:
        all_files = all_files[:args.max_images]
        log_msg(f"Limited to {args.max_images} images")
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
    train_ds = BakedFeatureDataset(train_files, augment=True)
    log_msg("Initialising valid dataset...")
    val_ds = BakedFeatureDataset(val_files, augment=False)
    log_msg("Initialising test dataset...")
    test_ds = BakedFeatureDataset(test_files, augment=False)
    log_msg("All datasets initialised")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Training
    # Pack parameters into your input_dict for the training function
    input_dict = vars(args)
    #input_dict["diversity_methods"] = args.diversity_methods
    #input_dict["lam_jsd"] = args.lam_jsd
    #input_dict["lam_pearson"] = args.lam_pearson
    #input_dict["lam_orth"] = args.lam_orth
    #input_dict["lambda_div"] = args.lam  # Mapping arg name to dict key
    input_dict["in_channels"] = args.decoder_in_channels
    input_dict["embed_dim"] = args.decoder_embed_dim
    input_dict["device"] = DEVICE
    input_dict["learning_rate"] = args.lr
    input_dict["run_name"] = run_name
    #input_dict["checkpoint_path"] = args.checkpoint_path


    start_time = time.time()
    trained_model, student_model = full_decoder_training_run(input_dict, train_loader, val_loader)

    log_msg(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    # Plot loss curves
    loss_history_path = os.path.join(os.getenv("OUT_DIR", "results"), "runs",
                                     f"{run_name}_loss_history.json")
    plot_loss_curves(loss_history_path, save_name=f"{run_name}_loss_curves")

    # Load best model for evaluation instead of final model
    best_model_path = Path(os.getenv("OUT_DIR", "results")) / "checkpoints" / f"{run_name}_best_model.pth"

    if best_model_path.exists():
        log_msg(f"Loading best model from {best_model_path} for evaluation")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        log_msg("Best model loaded successfully")
    else:
        log_msg("No best model found, evaluating final model")

    results = evaluate_test_set(
        trained_model, test_loader,
        nn.CrossEntropyLoss(ignore_index=0),
        args,
        run_name=run_name
    )
    evaluate_error_localization(
        lambda feats: ensemble_uncertainty_and_pred(trained_model(feats), args.num_classes),
        test_loader, args, run_name=run_name, who="teacher"
    )

    if args.train_student and student_model is not None:
        best_student_path = Path(os.getenv("OUT_DIR", "results")) / "checkpoints" / f"{run_name}_best_student.pth"
        if best_student_path.exists():
            log_msg(f"Loading best student checkpoint from {best_student_path}")
            ckpt = torch.load(best_student_path, map_location=DEVICE)
            student_model.load_state_dict(ckpt['model_state_dict'])
        evaluate_student_test_set(student_model, test_loader, args, run_name=f"{run_name}_student")
        evaluate_uncertainty_correlation(trained_model, student_model, test_loader, args, run_name=f"{run_name}_student")
        evaluate_error_localization(
            lambda feats: dirichlet_uncertainty_and_pred(student_model(feats)),
            test_loader, args, run_name=run_name, who="student"
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
    parser.add_argument("--hide_unlabelled_pixels", action="store_true")
    #parser.add_argument("--lam", type=float, default=0.1, help="Lambda diversity")
    #parser.add_argument("--enforce_diversity", action="store_true")
    parser.add_argument("--diversity_methods", type=str, nargs="+", default=[], choices=["jsd", "pearson", "orthogonality"],
                        help="One or more diversity methods to combine")
    parser.add_argument("--lam_jsd", type=float, default=0.)
    parser.add_argument("--lam_pearson", type=float, default=0.)
    parser.add_argument("--lam_orth", type=float, default=0.0)
    # Misc
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss instead of cross-entropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--use_dice_loss", action="store_true", help="Use Dice loss instead of cross-entropy")
    parser.add_argument("--train_student", action="store_true", help="Train AS4 StudentHead in parallel with teacher ensemble")
    parser.add_argument("--student_warmup_epochs", type=int, default=10, help="Epochs to train teacher only before student starts updating")
    parser.add_argument("--student_T_start", type=float, default=4.0, help="Initial temperature for EnDD teacher softmax annealing, linearly anneals to 1.0 over the student's training window")

    args = parser.parse_args()
    run_training(args)