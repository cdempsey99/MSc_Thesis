from utils.training import *
from utils.dataset import *
from models.encoder import *
import time
import argparse
import shutil
import random
import gc

# TODO : Can we somehow take outputs from different layers our from the clay encoder? Instead of just the last one?
# TODO : Change learning rate as epochs go on? Also mess around with different optimizers potentially
# TODO : Try only turning on/increasing Diversity after a few epochs ?
# TODO : Use more assert statements
# TODO : Use dropout for regularisation ?
# TODO : write unit tests?
# TODO : Check second timer

# Current unsolved issues:
# Need to investigate the diversity enforcement more, size of lambda - other papers
# Scale up to all (or most) FBP images

# TODO: Will need to update code to expect files in the respective subdirs in the data dir?

# TODO : Add a timestamp to every log .lg.out style

#random.seed(None)

if __name__ == "__main__":

    # Parse cmd line args
    parser = argparse.ArgumentParser(description="training_run")

    # Set the defaults to the values we had locally on Windows
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--decoder_in_channels", type=int, default=1024)
    parser.add_argument("--decoder_embed_dim", type=int, default=256)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--enforce_diversity", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hide_unlabelled_pixels", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint")
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    # Based on previous tinkering, choose lambda_diversity of 0.01
    # After changing the diversity to be calculated based on softmaxxed probs rather than preds, lambda needs to be much higher
    # This change still appears to be a more rigorous choice but makes the diversity optimisation a bit more annoying
    # Now this (1.0) seems too large, moving down
    #lam = 0.1


    # Pass all variables defining a training run thourgh this dict
    # Populate dict with the values parsed from the cmd line
    input_dict = {
        "in_channels": args.decoder_in_channels,
        "embed_dim": args.decoder_embed_dim,
        "ensemble_size": args.ensemble_size,
        "num_classes": args.num_classes,
        "num_epochs": args.num_epochs,
        "lambda_div": args.lam,
        "enforce_diversity": args.enforce_diversity,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "hide_unlabelled_pixels" : args.hide_unlabelled_pixels,
        "out_dir" : args.out_dir,
        "resume" : args.resume
    }

    # Get paths for our initial test image and mask
    #image_filepaths = [os.path.join(args.data_dir, "GF2_PMS1__L1A0000962382-MSS1.tif")]
    #mask_filepaths = [os.path.join(args.data_dir, "GF2_PMS1__L1A0000962382-MSS1_24label.png")]


    """
    
    # Ingest all images available in the data dir
    data_dir = Path(args.data_dir)
    image_filepaths = sorted(data_dir.glob("*.tif"))
    if args.max_images is not None:
        image_filepaths = image_filepaths[:args.max_images]
    mask_filepaths = [data_dir / (p.stem + "_24label.png") for p in image_filepaths]
    
    for img, mask in zip(image_filepaths, mask_filepaths):
        assert mask.exists(), f"Missing mask for {img.name}"
    
    print(f"Found {len(image_filepaths)} image/mask pairs")
    
    # Instantiate an object of the Dataset class
    is_hpc = os.getenv("OUT_DIR") is not None
    use_preload = True if is_hpc else False
    
    # For tinkering, use a much smaller dataset
    small_dataset = False
    sample_limit = 100 if small_dataset else None
    
    if small_dataset:
        print(f"Using SMALL dataset of only {sample_limit} samples")
    
    # Define specific paths for the baked data in the scratch space
    feature_dir = BASE_OUT / "features"
    mask_dir = BASE_OUT / "masks_tensors"
    
    # Count existing files
    existing_files = list(feature_dir.glob("*.pt")) if feature_dir.exists() else []
    
    # Trigger bake if folder is missing, empty, or doesn't match our intended run size
    if not feature_dir.exists() or len(existing_files) == 0 or (not small_dataset and len(existing_files) < 200):
        print(f"--> Triggering fresh bake. Reason: {len(existing_files)} patches found, but larger run requested.")
    
        # CLEANUP: Remove old directories to ensure a clean index from 0 to N
        if feature_dir.exists():
            print(f"Cleaning old features at {feature_dir}...")
            shutil.rmtree(feature_dir)
        if mask_dir.exists():
            print(f"Cleaning old masks at {mask_dir}...")
            shutil.rmtree(mask_dir)
    
        # Use your original dataset/loader just for the extraction phase
        raw_dataset = FBPPatchDataset(
            image_filepaths,
            mask_filepaths,
            patch_size=224,
            stride=112,
            preload=False,  # Don't need to preload pixels into RAM for this
            max_samples=sample_limit
        )
        # Batch size 1 is safest for the large Transformer extraction
        extract_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False)
    
        # Record time for baking features
        bake_start = time.time()
    
        # Load encoder once to bake
        encoder_model = initialize_clay_encoder()
        encoder_model.to(DEVICE)
    
        # Call the new function in encoder.py
        bake_features(extract_loader, encoder_model)
    
        bake_end = time.time()
        bake_duration = (bake_end - bake_start) / 60
        print(f"✅ Baking of features complete. Total time: {bake_duration:.2f} minutes for {len(extract_loader)} patches.")
    
        # Clean up encoder to free 30GB+ of VRAM for training
        del encoder_model
        torch.cuda.empty_cache()
    else:
        print(f"--> Found existing baked features at {feature_dir}. Skipping extraction.")
    
    # 3. Instantiate the FAST dataset and loader
    # This loads [1024, 28, 28] tensors directly
    baked_dataset = BakedFeatureDataset(feature_dir, mask_dir)
    
    train_loader = DataLoader(
        baked_dataset,
        batch_size=input_dict["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    
    print(f"Total baked patches found: {len(baked_dataset)}")
    print(f"Total batches to run per epoch: {len(train_loader)}")
    
    # Record time
    start_time = time.time()
    
    """

    # --- 1. Dataset Discovery & Splitting ---
    data_dir = Path(args.data_dir)
    # Gaofen-2 images are usually .tif; masks are .png
    all_tifs = sorted(list(data_dir.glob("*.tif")))

    if args.max_images is not None:
        all_tifs = all_tifs[:args.max_images]

    all_image_filepaths = all_tifs
    all_mask_filepaths = [data_dir / (p.stem + "_24label.png") for p in all_image_filepaths]

    for img, mask in zip(all_image_filepaths, all_mask_filepaths):
        assert mask.exists(), f"Missing mask for {img.name}"

    # Reproducible Shuffle
    random.seed(42)
    combined = list(zip(all_image_filepaths, all_mask_filepaths))
    random.shuffle(combined)

    # Partition (e.g., 100 Train, 20 Val, 30 Test for the full 150)
    num_total = len(combined)
    train_idx = int(num_total * 0.66)
    val_idx = int(num_total * 0.80)

    splits = {
        "train": combined[:train_idx],
        "val": combined[train_idx:val_idx],
        "test": combined[val_idx:]
    }

    log_msg(f"Split: {len(splits['train'])} Train | {len(splits['val'])} Val | {len(splits['test'])} Test")

    """
    #Updating to save pre baked features in a single file per image rather than per tile:
    
    # --- 2. Mandatory Fresh Bake ---
    # Point this to your Beegfs scratch path via args.out_dir
    BAKED_ROOT = Path(args.out_dir) / "baked_data"

    if BAKED_ROOT.exists():
        print(f"Cleaning old baked data at {BAKED_ROOT}...")
        shutil.rmtree(BAKED_ROOT)

    bake_start = time.time()
    encoder_model = initialize_clay_encoder().to(DEVICE)

    for split_name, file_list in splits.items():
        if not file_list: continue

        print(f"Baking {split_name} split...")
        split_feat_dir = BAKED_ROOT / split_name / "features"
        split_mask_dir = BAKED_ROOT / split_name / "masks_tensors"
        split_feat_dir.mkdir(parents=True)
        split_mask_dir.mkdir(parents=True)

        imgs, msks = zip(*file_list)
        raw_dataset = FBPPatchDataset(list(imgs), list(msks), patch_size=224, stride=112, preload=False)
        # Batch size 1 for memory safety during Transformer extraction
        extract_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False)

        # UPDATED: Passes specific split paths to the bake function
        bake_features(extract_loader, encoder_model, split_feat_dir, split_mask_dir)

    del encoder_model
    torch.cuda.empty_cache()
    print(f"Baking complete. Total time: {(time.time() - bake_start) / 60:.2f}m")
    """

    # --- 2. Mandatory Fresh Bake (Optimized Option A) ---
    BAKED_ROOT = Path(args.out_dir) / "baked_data"
    if BAKED_ROOT.exists():
        shutil.rmtree(BAKED_ROOT)

    encoder_model = initialize_clay_encoder().to(DEVICE)
    encoder_model.eval()

    for split_name, file_list in splits.items():
        if not file_list: continue

        split_dir = BAKED_ROOT / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        log_msg(f"Baking {split_name} split image-by-image...")

        for img_p, mask_p in file_list:
            image_id = img_p.stem

            # Create dataset for ONLY this image to preserve your MIN_LABELLED_PIXELS logic
            single_img_ds = FBPPatchDataset([img_p], [mask_p], patch_size=224, stride=112, preload=False)

            if len(single_img_ds) == 0:
                continue

            # Use a higher batch size for baking to speed up the GPU
            loader = DataLoader(single_img_ds, batch_size=32, shuffle=False)

            # Accumulate all patches for THIS image
            all_feats = []
            all_masks = []
            for batch_imgs, batch_msks in loader:
                feats = get_encoder_representation(batch_imgs.to(DEVICE), encoder_model)
                all_feats.append(feats.cpu())
                all_masks.append(batch_msks.cpu())

            # Save as ONE packed file per image
            torch.save({
                'features': torch.cat(all_feats, dim=0),
                'masks': torch.cat(all_masks, dim=0)
            }, split_dir / f"{image_id}_packed.pt")

    del encoder_model
    gc.collect()
    torch.cuda.empty_cache()
    log_msg("Baking memory fully purged. Proceeding to Training phase...")

    # --- 3. Fast Training Loaders ---
    train_baked = BakedFeatureDataset(BAKED_ROOT / "train/")
    val_baked = BakedFeatureDataset(BAKED_ROOT / "val/")

    train_loader = DataLoader(
        train_baked,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_baked,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    start_time = time.time()

    """
    # ======  RUN  =======
    # Instantiating, training and evaluating the trained decoder ensemble visually and with metrics
    # getting the encoder rep is packaged into this fn as well
    miou, overall_accuracy, avg_unc, ece = full_decoder_training_run(input_dict, train_loader)
    
    end_time = time.time()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    print("-" * 30)
    print(f"Total Execution Time for full_decoder_training_run function: {minutes}m {seconds}s")
    print("-" * 30)
    
    # Keep in mind this timer will only finish when i manually exit out of the graphs
    """


    # --- 4. Execution (Training) ---
    trained_model = full_decoder_training_run(input_dict, train_loader, val_loader)

    end_time = time.time()
    log_msg(f"Time for training: {end_time - start_time}")

    # --- 5. Final Test Evaluation (Outside) ---
    log_msg("\n" + "="*30)
    log_msg("STARTING FINAL TEST EVALUATION")
    log_msg("="*30)

    # Instantiate Test Loader
    test_baked = BakedFeatureDataset(BAKED_ROOT / "test")
    test_loader = DataLoader(
        test_baked,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    ) # BS=1 is best for visualization

    # Get a random batch for visual check (from the unseen test set!)
    test_features, test_masks = get_random_batch(test_loader, DEVICE)

    # Get maps (using your existing utility)
    mean_probs, class_map, variance_map, total_entropy, mutual_info = get_decoder_output_maps(
        trained_model, test_features[0].unsqueeze(0)
    )

    # Visualize & Calculate Final Metrics
    visualise_all_metrics(
        class_map, variance_map, total_entropy, mutual_info,
        test_masks[0], input_dict["hide_unlabelled_pixels"]
    )

    # Quantitative Metrics
    conf_tensor, _ = torch.max(mean_probs, dim=1)
    confidence_map_2d = conf_tensor.squeeze().cpu().numpy()
    ground_truth_2d = test_masks[0].squeeze().cpu().numpy()

    miou, acc, avg_unc, ece = evaluate_metrics(
        class_map, ground_truth_2d, confidence_map_2d
    )

    log_msg(f"FINAL TEST RESULTS: mIoU: {miou:.4f} | Acc: {acc:.4f} | ECE: {ece:.4f}")




"""
# Ok now start enforcing diversity, steps:
# (i) Add diversity code
#  - Need to try and pick the best, or at least a good, lambda value
#  - Then choose a selection of lambda values, apply whole training process, and record accuracy and unc
# This begs the question, how exactly to measure accuracy ? mIoU?

Ok done above and we find that a very small lambda (about 0.001) looks best for now, in terms of accuracy
Uncertainty just went up monotonically for higher lambda


# (ii) Then change the functions to accept dicts
# (iii) inside this dict have options for (i) one decoder, (ii) M decoders (iii) M diverse decoders

Ok now i have done (iii) but strangely now the way the behaviour changes with lambda has changed. 
Now the lambda seems to need to be much higher to make any difference 
Maybe I need to apply softmax to the raw logits? 
This probably makes sense either way I guess?

Ok even if the behaviour is poorer now, thi is still very much in the rough stage. So I could just 
move on with building out the structure of the project code and then come back and start fixing the diversity penalty later.
I mean this is the completely 'wrong' penalty term anyway. Ok then. Maybe proper uncertainty metrics?


# Then add metrics for measuring the calibration of accuracy?
# Or would be expect that to be improve by the diversity of ensemble?

Ok latest run - Despite the change in behaviour since Friday, we don't seem to have nau fundamental issue i.e 
the different heads do predict different classes and at least two heads predicted more than one class on the given eval tile. 
This meant that the averaged final prediction was not just one class. Although ofc the miou is still very low, as we only used
20 sample tiles for 10 epochs
But I think fundamentally this means we can move on

After a small run (5 samples, 5 epochs) we recovered something more similar to the original behaviour,
 i.e. lambda of 0.01 was best and accuracy pretty steadily got worse for increasing lambda

Next Steps:
 
 - what about the softmax issue from before? in the class map output-- and diversity--
 - look at uncertainty metrics in more depth (including unc-- and then unc calibration--)
 - come up with a few better candidates for diversity penalty term-- (JS divergence)
 - plan for training of decoders to allow us to properly train student? Overnight? 
 
Uncertainty measures:
- average variance (what we have now)
- predictive entropy (shannon entropy that we saw before)
- mutual information = Total Entropy - Average entropy of individual heads




"""



