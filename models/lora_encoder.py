import torch
import torch.nn.functional as F
from pathlib import Path
from peft import LoraConfig, get_peft_model
from configs.config import *
from utils.misc import log_msg
from models.encoder import initialize_clay_encoder


def initialize_clay_encoder_with_lora(r=16, lora_alpha=32, lora_dropout=0.1):
    """
    Initialises Clay encoder with LoRA adapters injected into attention layers.
    Base weights are frozen, only LoRA parameters are trainable.
    Returns base_encoder (full ClayMAEModule) and lora_encoder (LoRA-wrapped transformer).
    """
    log_msg(f"Initialising Clay encoder with LoRA (r={r}, alpha={lora_alpha})...")

    base_encoder = initialize_clay_encoder()
    if base_encoder is None:
        return None, None

    # Freeze all base encoder parameters
    for param in base_encoder.parameters():
        param.requires_grad = False

    # Inject LoRA adapters into transformer blocks only
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["to_qkv", "to_out"],
        lora_dropout=lora_dropout,
        bias="none"
    )

    lora_encoder = get_peft_model(base_encoder.model.encoder, lora_config)
    lora_encoder.print_trainable_parameters()

    return base_encoder, lora_encoder


def get_encoder_representation_grad(input_tensor, base_encoder, lora_encoder):
    """
    End-to-end forward pass through LoRA encoder WITH gradients.
    Used during training so LoRA parameters can be updated.
    """
    waves = torch.tensor([0.842, 0.665, 0.560], dtype=torch.float32).to(DEVICE)
    input_tensor = input_tensor.to(DEVICE)

    # Patch embedding and positional encoding — no LoRA here, no grad needed
    with torch.no_grad():
        patches, _ = base_encoder.model.encoder.to_patch_embed(input_tensor, waves)
        if hasattr(base_encoder.model.encoder, "pos_embed"):
            pos_embed = base_encoder.model.encoder.pos_embed
            if pos_embed.shape[1] > patches.shape[1]:
                patches = patches + pos_embed[:, 1:, :]
            else:
                patches = patches + pos_embed

    # LoRA transformer — gradients flow here
    features = lora_encoder.transformer(patches)

    # Final normalisation
    if hasattr(lora_encoder.transformer, "norm"):
        features = lora_encoder.transformer.norm(features)
    elif hasattr(lora_encoder, "norm"):
        features = lora_encoder.norm(features)

    batch_size = features.shape[0]
    grid_features = features.transpose(1, 2).reshape(batch_size, 1024, 28, 28)

    return grid_features


def save_lora_weights(lora_encoder, save_path):
    """Saves only the LoRA adapter weights."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    lora_encoder.save_pretrained(save_path)
    log_msg(f"LoRA weights saved to {save_path}")



def load_lora_weights(base_encoder, lora_weights_path, r=16, lora_alpha=32, lora_dropout=0.1):
    """Loads saved LoRA weights back into a fresh LoRA encoder."""
    from peft import PeftModel
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["to_qkv", "to_out"],
        lora_dropout=lora_dropout,
        bias="none"
    )
    lora_encoder = get_peft_model(base_encoder.model.encoder, lora_config)
    lora_encoder.load_adapter(lora_weights_path, adapter_name="default")
    log_msg(f"LoRA weights loaded from {lora_weights_path}")
    return lora_encoder