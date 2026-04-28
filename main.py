from utils.training import *
from utils.dataset import *
import time
import argparse
import os

# TODO : Can we somehow take outputs from different layers our from the clay encoder? Instead of just the last one?
# TODO : Change learning rate as epochs go on? Also mess around with different optimizers potentially
# TODO : Try only turning on/increasing Diversity after a few epochs ?
# TODO : Use more assert statements
# TODO : Use dropout for regularisation ?

#random.seed(None)

# Parse cmd line args
parser = argparse.ArgumentParser(description="training_run")


# We set the defaults to your "Windows/Local" values
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--num_epochs", type=int, default=2)  # Local default
parser.add_argument("--lam", type=float, default=0.1)     # Local default
parser.add_argument("--out_dir", type=str, default="./results")

args = parser.parse_args()


# Based on previous tinkering, choose lambda_diversity of 0.01
# After changing the diversity to be calculated based on softmaxxed probs rather than preds, lambda needs to be much higher
# This change still appears to be a more rigorous choice but makes the diversity optimisation a bit more annoying
# Now this (1.0) seems too large, moving down
lam = 0.1

# Path for dev input test image and ground truth mask
#image_filepaths = ["data/GF2_PMS1__L1A0000962382-MSS1.tif"]
#mask_filepaths = ["data/GF2_PMS1__L1A0000962382-MSS1_24label.png"]

# Define input for a training run
input_dict = {
    "in_channels": DECODER_IN_CHANNELS,
    "embed_dim": DECODER_EMBED_DIM,
    "ensemble_size": ENSEMBLE_SIZE,
    "num_classes": NUM_CLASSES,
    "num_epochs": args.num_epochs,
    "lambda_div": args.lam,
    "enforce_diversity": True,
    "learning_rate": LEARNING_RATE,
    "batch_size": 2,
    "hide_unlabelled_pixels" : True,
    "out_dir" : args.out_dir
}

# Build paths using args.data_dir
image_filepaths = [os.path.join(args.data_dir, "GF2_PMS1__L1A0000962382-MSS1.tif")]
mask_filepaths = [os.path.join(args.data_dir, "GF2_PMS1__L1A0000962382-MSS1_24label.png")]


# Instantiate an object of the Dataset class
full_image_dataset = FBPPatchDataset(image_filepaths, mask_filepaths, patch_size=224, stride=112)

# For tinkering, use a much smaller dataset
small_dataset = True
small_samples = 2

if small_dataset:
    print(f"Using SMALL dataset of only {small_samples} samples")
    full_image_dataset.samples = full_image_dataset.samples[:small_samples]

# Create DataLoader
train_loader = DataLoader(full_image_dataset, batch_size=input_dict["batch_size"], shuffle=False)

print(f"Total patches found: {len(full_image_dataset)}")
print(f"Total batches to run per epoch : {len(train_loader)}")

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

# This timer will only finish when i manually exit out of the graphs


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



