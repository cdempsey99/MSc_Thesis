from claymodel.module import ClayMAEModule
from configs.config import *
from utils.dataset import *
from utils.misc import *
from tqdm import tqdm

def initialize_clay_encoder():
    """Initializes the Clay model and loads pruned weights once."""
    log_msg("Initializing Clay Encoder and loading pruned weights...")

    # 1. Setup
    model = ClayMAEModule(model_size="large")

    # 2. Load Weights
    try:
        pruned_weights = torch.load("encoder_checkpoints/clay_encoder_only.pth", map_location=DEVICE)
        model.load_state_dict(pruned_weights, strict=False)
        model.eval()
        model.to(DEVICE)
        log_msg("----------------------------------------")
        log_msg("Success: Pruned backbone loaded into memory.")
        log_msg("----------------------------------------")
    except FileNotFoundError:
        log_msg("Error: checkpoints/clay_encoder_only.pth not found.")
        return None

    return model

def get_encoder_representation(input_tensor, encoder_model):
    """Processes a batch using the provided encoder model object."""
    # Define waves for GF2
    waves = torch.tensor([0.842, 0.665, 0.560], dtype=torch.float32).to(DEVICE)
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


def initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=2):
    """
    Loads Clay encoder, freezes all parameters, then unfreezes the last
    n_unfrozen_blocks transformer blocks and the final LayerNorm.
    More stable than LoRA on small datasets — pretrained features are
    mostly preserved, only the top layers adapt.
    """
    encoder_model = initialize_clay_encoder()
    if encoder_model is None:
        return None

    for param in encoder_model.parameters():
        param.requires_grad = False

    transformer = encoder_model.model.encoder.transformer
    total_blocks = len(transformer.layers)

    for block in transformer.layers[total_blocks - n_unfrozen_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in transformer.norm.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder_model.parameters())
    log_msg(f"Partial unfreeze: last {n_unfrozen_blocks}/{total_blocks} blocks + norm | "
            f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    return encoder_model


def get_encoder_representation_partial(input_tensor, encoder_model, waves=None):
    """
    Forward pass through the partially unfrozen encoder with gradients.
    Patch embedding and positional encoding run under no_grad (always frozen).
    The transformer runs normally — unfrozen blocks accumulate gradients,
    frozen blocks do not (requires_grad=False params are skipped by autograd).
    waves: band centre wavelengths in μm. Defaults to GF2 RGB (FBP dataset).
           Pass ReBENRawDataset.WAVELENGTHS for Sentinel-2 B04/B03/B02.
    """
    if waves is None:
        waves = torch.tensor([0.842, 0.665, 0.560], dtype=torch.float32)
    waves = waves.to(DEVICE)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        patches, _ = encoder_model.model.encoder.to_patch_embed(input_tensor, waves)
        if hasattr(encoder_model.model.encoder, "pos_embed"):
            pos_embed = encoder_model.model.encoder.pos_embed
            if pos_embed.shape[1] > patches.shape[1]:
                patches = patches + pos_embed[:, 1:, :]
            else:
                patches = patches + pos_embed

    features = encoder_model.model.encoder.transformer(patches)

    if hasattr(encoder_model.model.encoder.transformer, "norm"):
        features = encoder_model.model.encoder.transformer.norm(features)
    elif hasattr(encoder_model.model.encoder, "norm"):
        features = encoder_model.model.encoder.norm(features)

    batch_size = features.shape[0]
    return features.transpose(1, 2).reshape(batch_size, 1024, 28, 28)


def bake_features(loader, encoder_model, mask_dir, image_name):
    """
    Runs the full dataset through the frozen encoder once and saves
    the resulting grid features to disk to avoid redundant computation.
    """

    mask_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_masks = []

    log_msg(f"Baking {image_name}...")
    encoder_model.eval()

    with torch.no_grad():

        for images, masks in tqdm(loader):

            # Move images to GPU for the encoder
            images = images.to(DEVICE)

            # grid_features shape: [Batch, 1024, 28, 28]
            grid_features = get_encoder_representation(images, encoder_model)

            all_features.append(grid_features.cpu())
            all_masks.append(masks.cpu())


    # Concatenate all batches into two large tensors
    # Resulting shape : [total_patches, 1024, 28, 28]
    stacked_features = torch.cat(all_features, dim=0)
    stacked_masks = torch.cat(all_masks, dim=0)

    save_path = mask_dir / f"{image_name}_embeddings.pt"
    torch.save({
        'features' : stacked_features,
        'masks' : stacked_masks
    }, save_path)

    log_msg(f"Saved {stacked_features.size(0)} patches to {save_path}")



