import torch
import gc
import os
from configs.config import *

### SHOULD JUST NEED TO RUN THIS ONCE AT THE BEGINNING AND NEVER AGAIN (if still using Clay)

print("Starting pruning... this might take a minute on CPU.")

# 1. Load the full checkpoint dictionary on CPU
# 'map_location' is key for your 500MB VRAM situation
full_checkpoint = torch.load(CHECKPOINT_DIR / "clay-v1.5.ckpt", map_location="cpu")

# 2. Extract the weights (the 'state_dict')
state_dict = full_checkpoint["state_dict"]

# 3. Filter for ONLY the encoder layers
# These are the 'brains' that your Hydra decoders will read from
encoder_weights = {k: v for k, v in state_dict.items() if "model.encoder" in k}

# 4. Save the lightweight version
torch.save(encoder_weights, CHECKPOINT_DIR / "clay_encoder_only.pth")

# 5. FORCE RAM CLEANUP
# This is vital for your 16GB limit!
del full_checkpoint
del state_dict
gc.collect()

print(f"Success! Saved pruned weights to 'clay_encoder_only.pth'.")
print(f"Old file size: {os.path.getsize(CHECKPOINT_DIR / 'clay-v1.5.ckpt') / 1e9:.2f} GB")
print(f"New file size: {os.path.getsize(CHECKPOINT_DIR / 'clay_encoder_only.pth') / 1e6:.2f} MB")


