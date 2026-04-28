from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = ROOT_DIR / "data"
UTILS_DIR = ROOT_DIR / "utils"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"

# Change device to cuda or gpu if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

