import torch
import torch.nn as nn
import torch.nn.functional as F
from claymodel.module import ClayMAEModule

from configs.config import *

class SegFormerDecoderHead(nn.Module):

    def __init__(self, in_channels=1024, embed_dim=256, num_classes=NUM_CLASSES):
        super().__init__()

        # 1024 -> 256 using a 2dconv as Pointwise MLP
        self.linear_fusion = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.activation1 = nn.GELU()

        # 256 -> 24, this is the classifier
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x starts as [1, 1024, 28, 28]
        x = self.linear_fusion(x)
        x = self.activation1(x)
        x = self.classifier(x)

        # Upsampling, we stretch the 24 outputs back to the 224 x 224 of the original image
        # TODO : Change this to learnable weights?
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return x

class DecoderEnsemble(nn.Module):

    def __init__(self, M=5, in_channels=1024, embed_dim=256, num_classes=NUM_CLASSES):
        super().__init__()

        self.M = M

        # Create M individual SegFormerDecoderHead instances
        self.heads = nn.ModuleList([
            SegFormerDecoderHead(in_channels, embed_dim, num_classes) for _ in range(M)
        ])

    def forward(self, x):
        # Run encoder features through each of the M heads
        # Output is a list of tensors [[1, 24, 224, 224], [1, 24, 224, 224], ...]
        head_outputs = [head(x) for head in self.heads]

        # Stack the outputs along a new 'member' dimension
        # so [Member, Batch, Class, H, W] = [5, ?, 24, 224, 224]
        return torch.stack(head_outputs)

# ENCODER Section of this file now:

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
