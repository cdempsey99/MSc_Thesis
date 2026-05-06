from utils.training import *
from utils.dataset import *
from models.encoder import *
import time
import argparse
import os
import shutil

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



#random.seed(None)

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



