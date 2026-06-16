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

# reBEN — CORINE Land Cover 19-class nomenclature (same as BigEarthNet)
# Index 0 = no data / unlabelled in the pixel-level reference map
# TODO: verify exact class index assignments from reference map when data arrives
REBEN_CLASSES = [
    "No Data",
    "Urban Fabric",
    "Industrial or Commercial Units",
    "Arable Land",
    "Permanent Crops",
    "Pastures",
    "Complex Cultivation Patterns",
    "Land Principally Occupied by Agriculture",
    "Agro-Forestry Areas",
    "Broad-Leaved Forest",
    "Coniferous Forest",
    "Mixed Forest",
    "Natural Grassland and Sparsely Vegetated Areas",
    "Moors, Heathland and Sclerophyllous Vegetation",
    "Transitional Woodland and Shrub",
    "Beaches, Dunes and Sands",
    "Inland Wetlands",
    "Coastal Wetlands",
    "Inland Waters",
    "Marine Waters",
]

REBEN_CLASSES_DICT = {i: name for i, name in enumerate(REBEN_CLASSES)}

# Kaggle API token
KAGGLE_API_TOKEN="REDACTED"

