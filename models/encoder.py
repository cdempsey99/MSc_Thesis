from claymodel.module import ClayMAEModule
from configs.config import *

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
