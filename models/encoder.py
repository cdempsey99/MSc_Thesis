from claymodel.module import ClayMAEModule
from configs.config import *
from utils.dataset import *
from tqdm import tqdm

def initialize_clay_encoder():
    """Initializes the Clay model and loads pruned weights once."""
    print("Initializing Clay Encoder and loading pruned weights...")

    # 1. Setup
    model = ClayMAEModule(model_size="large")

    # 2. Load Weights
    try:
        pruned_weights = torch.load("checkpoints/clay_encoder_only.pth", map_location=DEVICE)
        model.load_state_dict(pruned_weights, strict=False)
        model.eval()
        model.to(DEVICE)
        print("----------------------------------------")
        print("Success: Pruned backbone loaded into memory.")
        print("----------------------------------------")
    except FileNotFoundError:
        print("Error: checkpoints/clay_encoder_only.pth not found.")
        return None

    return model

def get_encoder_representation(input_tensor, encoder_model):
    """Processes a batch using the provided encoder model object."""
    # Define waves for GF2
    waves = torch.tensor([842.0, 665.0, 560.0, 490.0], dtype=torch.float32).to(DEVICE)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        # A. Dynamic Patch Embedding
        patches, _ = encoder_model.model.encoder.to_patch_embed(input_tensor, waves)

        # B. Add Position Embeddings
        if hasattr(encoder_model.model.encoder, "pos_embed"):
            pos_embed = encoder_model.model.encoder.pos_embed
            if pos_embed.shape[1] > patches.shape[1]:
                patches = patches + pos_embed[:, 1:, :]
            else:
                patches = patches + pos_embed

        # C. Pass through the Transformer Blocks
        features = encoder_model.model.encoder.transformer(patches)

        # D. Final Normalization
        if hasattr(encoder_model.model.encoder.transformer, "norm"):
            features = encoder_model.model.encoder.transformer.norm(features)
        elif hasattr(encoder_model.model.encoder, "norm"):
            features = encoder_model.model.encoder.norm(features)

    # 5. Reshape for Batches
    batch_size = features.shape[0]
    grid_features = features.transpose(1, 2).reshape(batch_size, 1024, 28, 28)

    return grid_features


def bake_features(loader, encoder_model, feature_dir, mask_dir):
    """
    Runs the full dataset through the frozen encoder once and saves
    the resulting grid features to disk to avoid redundant computation.
    """

    #feature_dir = BASE_OUT / "features"
    #mask_dir = BASE_OUT / "masks_tensors"

    # Ensure directories exist
    feature_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting feature extraction. Saving to: {feature_dir}")
    encoder_model.eval()

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader)):
            # Move images to GPU for the encoder
            images = images.to(DEVICE)

            # grid_features shape: [Batch, 1024, 28, 28]
            grid_features = get_encoder_representation(images, encoder_model)

            # Save batch items individually
            for j in range(grid_features.size(0)):
                # Calculate unique index: (current batch index * batch size) + item index in batch
                patch_idx = i * loader.batch_size + j

                # Save as CPU tensors (.cpu()) to save VRAM and make them portable
                torch.save(grid_features[j].cpu(), feature_dir / f"feat_{patch_idx}.pt")
                torch.save(masks[j].cpu(), mask_dir / f"mask_{patch_idx}.pt")

    print(f"Extraction complete. {len(loader.dataset)} patches baked.")