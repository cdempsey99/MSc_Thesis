from claymodel.module import ClayMAEModule
from configs.config import *
from utils.dataset import *
from utils.misc import *
from tqdm import tqdm

# FBP band wavelengths (um). The 8-bit FBP tifs hold GF-2 bands 1-3 = blue/green/red (band 4 is an
# alpha channel; checked 23/10 from per-class band means). LEGACY is what every existing FBP
# embedding was extracted with (it labels them NIR/red/green); GF2_BGR is the correct labelling.
FBP_WAVES_LEGACY = [0.842, 0.665, 0.560]
FBP_WAVES_GF2_BGR = [0.485, 0.555, 0.660]
# Ground sampling distance (m) of the 224x224 encoder input, for Clay's position encoding:
# FBP is 4 m GF-2 imagery cut at native resolution; reBEN S2/S1 patches are 120 px at 10 m
# resampled to 224 px.
FBP_GSD = 4.0
REBEN_GSD = 10.0 * 120 / 224


def initialize_clay_encoder(single_norm=False, native_forward=False, gsd=None):
    """Initializes the Clay model and loads pruned weights once.
    single_norm: Clay's Transformer.forward already ends with its own final LayerNorm, so the
    extra transformer.norm call in the feature functions below applies that norm a second time.
    True skips the extra call (Clay's own forward); False (default) keeps the double norm that
    every existing embedding/checkpoint was produced with, so they still evaluate correctly.
    native_forward: run the encoder as Clay's own Encoder.forward does (minus MAE masking): add
    Clay's GSD-scaled 2D sin-cos position encoding (time/lat-lon unknown, so zeros), prepend the
    cls token, drop it from the output, single final norm. gsd (m) is required with it. False
    (default) keeps the original no-position-encoding path used by every existing model."""
    if native_forward:
        assert gsd is not None, "native_forward needs the input's ground sampling distance (gsd)"
    log_msg("Initializing Clay Encoder and loading pruned weights...")

    # 1. Setup
    model = ClayMAEModule(model_size="large")

    # 2. Load Weights
    try:
        pruned_weights = torch.load("encoder_checkpoints/clay_encoder_only.pth", map_location=DEVICE)
        model.load_state_dict(pruned_weights, strict=False)
        model.clay_single_norm = single_norm or native_forward
        model.clay_native_forward = native_forward
        model.clay_gsd = gsd
        if native_forward:
            log_msg(f"Clay native forward (--clay_native_forward): GSD-scaled position encoding (gsd={gsd:.3f} m), "
                    f"cls token, single final norm")
        elif single_norm:
            log_msg("Clay single final LayerNorm (--clay_single_norm): extra transformer.norm call skipped")
        model.eval()
        model.to(DEVICE)
        log_msg("----------------------------------------")
        log_msg("Success: Pruned backbone loaded into memory.")
        log_msg("----------------------------------------")
    except FileNotFoundError:
        log_msg("Error: checkpoints/clay_encoder_only.pth not found.")
        return None

    return model

def _add_position_and_cls(encoder_model, patches):
    """Position information for the patch tokens [B, L, D]. Native forward: Clay's own
    add_encodings (GSD-scaled 2D sin-cos + 8 time/lat-lon dims, zeros as they're unknown) and the
    cls token prepended, as in Clay's Encoder.forward. Otherwise the original path: a pos_embed
    attribute if the encoder has one - Clay's Encoder has none, so in practice no position info."""
    enc = encoder_model.model.encoder
    if getattr(encoder_model, "clay_native_forward", False):
        B = patches.shape[0]
        unknown = torch.zeros(B, 4, device=patches.device)
        gsd = torch.tensor(encoder_model.clay_gsd, dtype=torch.float32)
        patches = enc.add_encodings(patches, unknown, unknown, gsd)
        cls_tokens = enc.cls_token.expand(B, -1, -1).to(patches.dtype)
        return torch.cat((cls_tokens, patches), dim=1)
    if hasattr(enc, "pos_embed"):
        pos_embed = enc.pos_embed
        if pos_embed.shape[1] > patches.shape[1]:
            patches = patches + pos_embed[:, 1:, :]
        else:
            patches = patches + pos_embed
    return patches


def get_encoder_representation(input_tensor, encoder_model, waves=None):
    """Processes a batch using the provided encoder model object.
    waves: band wavelengths (um); defaults to FBP_WAVES_LEGACY (what existing FBP embeddings used)."""
    waves = torch.tensor(FBP_WAVES_LEGACY if waves is None else waves, dtype=torch.float32).to(DEVICE)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        # A. Dynamic Patch Embedding
        patches, _ = encoder_model.model.encoder.to_patch_embed(input_tensor, waves)

        # B. Position information (+ cls token with native forward)
        patches = _add_position_and_cls(encoder_model, patches)

        # C. Pass through the Transformer Blocks
        features = encoder_model.model.encoder.transformer(patches)
        if getattr(encoder_model, "clay_native_forward", False):
            features = features[:, 1:, :]  # drop the cls token

        # D. Final Normalization - Clay's transformer already applied its final norm above; this
        # second application is kept only for back-compatibility (see initialize_clay_encoder)
        if not getattr(encoder_model, "clay_single_norm", False):
            if hasattr(encoder_model.model.encoder.transformer, "norm"):
                features = encoder_model.model.encoder.transformer.norm(features)
            elif hasattr(encoder_model.model.encoder, "norm"):
                features = encoder_model.model.encoder.norm(features)

    # 5. Reshape for Batches
    batch_size = features.shape[0]
    grid_features = features.transpose(1, 2).reshape(batch_size, 1024, 28, 28)

    return grid_features


def initialize_clay_encoder_partial_unfreeze(n_unfrozen_blocks=2, single_norm=False, native_forward=False, gsd=None):
    """
    Loads Clay encoder, freezes all parameters, then unfreezes the last
    n_unfrozen_blocks transformer blocks and the final LayerNorm.
    More stable than LoRA on small datasets — pretrained features are
    mostly preserved, only the top layers adapt.
    """
    encoder_model = initialize_clay_encoder(single_norm=single_norm, native_forward=native_forward, gsd=gsd)
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
        patches = _add_position_and_cls(encoder_model, patches)

    features = encoder_model.model.encoder.transformer(patches)
    if getattr(encoder_model, "clay_native_forward", False):
        features = features[:, 1:, :]  # drop the cls token

    # Second final norm kept only for back-compatibility (see initialize_clay_encoder)
    if not getattr(encoder_model, "clay_single_norm", False):
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



