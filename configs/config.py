import torch
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Check for the HPC environment variable
hpc_out_dir = os.getenv("OUT_DIR")

# If on HPC, BASE_OUT is the scratch space; otherwise, it's your local ROOT_DIR
BASE_OUT = Path(hpc_out_dir) if hpc_out_dir else ROOT_DIR

CHECKPOINT_DIR = BASE_OUT / "checkpoints"

DATA_DIR = ROOT_DIR / "data"
UTILS_DIR = ROOT_DIR / "utils"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"

# Global constant for the specific file path
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pth"

# Ensure the checkpoint directory exists
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Change device to cuda or gpu if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model and training parameters
NUM_CLASSES=25
ENSEMBLE_SIZE=5
DECODER_IN_CHANNELS=1024
DECODER_EMBED_DIM=256
LEARNING_RATE=1e-4
IMAGE_SIZE=224
MIN_LABELLED_PIXELS=10000

FBP_CLASSES = [
    "Unlabeled", "Industrial", "Paddy Field", "Irrigated Field", "Dry Cropland",
    "Garden Land", "Arbor Forest", "Shrub Forest", "Park", "Natural Meadow",
    "Artificial Meadow", "River", "Urban Residential", "Lake", "Pond",
    "Fish Pond", "Snow", "Bareland", "Rural Residential", "Stadium",
    "Square", "Road", "Overpass", "Railway Station", "Airport"
]

# Create the dictionary: {0: "Unlabeled", 1: "Industrial", ...}
FBP_CLASSES_DICT = {i: name for i, name in enumerate(FBP_CLASSES)}

# Kaggle API token
KAGGLE_API_TOKEN="REDACTED"

